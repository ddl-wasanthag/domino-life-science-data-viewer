import streamlit as st
import os
import pandas as pd
import pyreadstat
import numpy as np
from typing import List, Dict, Any, Optional
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import requests
import urllib3

# Suppress SSL warnings for internal Domino communication
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Matplotlib Domino colour palette ─────────────────────────────────────────
plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=[
        "#3B3BD3", "#0070CC", "#28A464", "#CCB718",
        "#FF6543", "#E835A7", "#2EDCC4",
    ]),
    "axes.facecolor":    "#FFFFFF",
    "figure.facecolor":  "#FAFAFA",
    "axes.edgecolor":    "#E0E0E0",
    "axes.labelcolor":   "#2E2E38",
    "xtick.color":       "#65657B",
    "ytick.color":       "#65657B",
    "grid.color":        "#E0E0E0",
    "grid.alpha":        0.5,
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.titleweight":  "600",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Add DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    st.warning("pydicom not installed. Install with: pip install pydicom")

# ── Domino constants ───────────────────────────────────────────────────────────
# DOMINO_API_HOST confirmed as http://nucleus-frontend.domino-platform:80
DOMINO_API_HOST = os.getenv("DOMINO_API_HOST", "http://nucleus-frontend.domino-platform:80")


# ── Identity propagation helpers ───────────────────────────────────────────────

def get_viewer_headers() -> dict:
    """
    Return incoming HTTP headers from the viewer's request.
    Primary:  st.context.headers  (Streamlit >= 1.37)
    Fallback: _get_websocket_headers() (older Streamlit)
    """
    try:
        return dict(st.context.headers)
    except AttributeError:
        pass
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        return dict(_get_websocket_headers() or {})
    except Exception:
        return {}


def get_viewer_username() -> Optional[str]:
    """Basic identity — domino-username header, display only."""
    headers = get_viewer_headers()
    return headers.get("domino-username") or headers.get("Domino-Username") or None


def get_viewer_api_token() -> Optional[str]:
    """
    Extended identity — viewer's Domino access token injected via
    Authorization: Bearer <token> header by Domino Extended Identity Propagation.
    Per guidelines: re-acquire on every call as the token expires quickly.
    Falls back to DOMINO_API_PROXY ephemeral token for non-Extension contexts.
    """
    # Extended identity path — Bearer token from header
    headers = get_viewer_headers()
    auth = headers.get("Authorization") or headers.get("authorization")
    if auth and auth.startswith("Bearer "):
        return auth[7:].strip()

    # Fallback: ephemeral token from API proxy (works in workspace/job context)
    # The proxy returns a raw JWT with no "Bearer " prefix
    try:
        resp = requests.get("http://localhost:8899/access-token", timeout=5)
        token = resp.text.strip()
        # Strip "Bearer " if present (shouldn't be, but defensive)
        return token[7:] if token.startswith("Bearer ") else token
    except Exception:
        return None


def domino_get(path: str, token: str) -> dict:
    """Authenticated GET against the Domino API."""
    resp = requests.get(
        f"{DOMINO_API_HOST}{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=15, verify=False)
    resp.raise_for_status()
    return resp.json()


def resolve_project(project_id: str, token: str) -> Optional[Dict]:
    """Resolve projectId (injected by Extensions) to project name/owner."""
    try:
        data = domino_get(f"/v4/projects/{project_id}", token)
        return {"id": data.get("id", project_id),
                "name": data.get("name", ""),
                "ownerUsername": data.get("ownerUsername", "")}
    except Exception:
        return None


def list_project_datasets(project_id: str, token: str) -> list:
    """
    List datasets for a project via Domino API.
    GET /v4/datasetrw/datasets-v2?projectIdsToInclude=<projectId>
    Returns list of datasetRwDto objects.
    Key fields: id, name, readWriteSnapshotId
    """
    try:
        data = domino_get(
            f"/v4/datasetrw/datasets-v2?projectIdsToInclude={project_id}", token)
        if isinstance(data, list):
            return [item.get("datasetRwDto", item) for item in data]
        return []
    except Exception as e:
        st.error(f"Dataset list error: {e}")
        return []


def list_snapshot_files(snapshot_id: str, token: str,
                        path: str = "", depth: int = 0) -> List[Dict]:
    """
    Recursively list all supported files in a dataset snapshot via API.
    Confirmed working endpoint:
      GET /v4/datasetrw/files/{snapshotId}?path=<path>
    path must be empty string for root (not "/" or ".").
    Response: {rows: [{name: {fileName, isDirectory, label}, size: {sizeInBytes}}]}
    Returns list of dicts: {path, name, size_bytes}
    """
    SUPPORTED_EXTS = (".parquet", ".xpt", ".dcm", ".dicom", ".dic", ".ima",
                      ".nii", ".nii.gz",
                      ".fastq", ".fastq.gz", ".fq", ".fq.gz",
                      ".fasta", ".fa", ".fna", ".ffn",
                      ".vcf", ".vcf.gz")
    if depth > 4:
        return []
    try:
        # path="" for root, "subdir" for subdirectory (no leading slash)
        endpoint = f"/v4/datasetrw/files/{snapshot_id}?path={requests.utils.quote(path, safe='')}"
        data = domino_get(endpoint, token)
        rows = data.get("rows", []) if isinstance(data, dict) else []
    except Exception:
        return []

    files = []
    for row in rows:
        entry = row.get("name", {})
        fname = entry.get("fileName") or entry.get("label", "")
        is_dir = entry.get("isDirectory", False) or entry.get("isDir", False)
        full_path = f"{path}/{fname}".lstrip("/") if path else fname
        if is_dir:
            files.extend(list_snapshot_files(
                snapshot_id, token, full_path, depth + 1))
        elif any(fname.lower().endswith(ext) for ext in SUPPORTED_EXTS):
            size = (row.get("size", {}).get("sizeInBytes") or
                    entry.get("sizeInBytes") or 0)
            files.append({
                "path": full_path,
                "name": fname,
                "size_bytes": size,
            })
    return files


def download_snapshot_file(snapshot_id: str, file_path: str,
                           token: str) -> Optional[bytes]:
    """
    Download raw file content from a dataset snapshot via API.
    Confirmed working endpoint (same base as file listing):
      GET /v4/datasetrw/snapshot/{snapshotId}/file/raw?path=<path>&download=true
    No filesystem mount required — works for any dataset the viewer can access.
    """
    try:
        url = f"{DOMINO_API_HOST}/v4/datasetrw/snapshot/{snapshot_id}/file/raw"
        resp = requests.get(
            url,
            params={"path": file_path, "download": "true"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=120, verify=False)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        st.error(f"Download error: {e}")
        return None


def get_dataset_files(base_path: str) -> List[Dict]:
    """
    Filesystem fallback: list supported files under a local path.
    Used only when no project context is available.
    """
    SUPPORTED_EXTS = (".parquet", ".xpt", ".dcm", ".dicom", ".dic", ".ima",
                      ".nii", ".nii.gz",
                      ".fastq", ".fastq.gz", ".fq", ".fq.gz",
                      ".fasta", ".fa", ".fna", ".ffn",
                      ".vcf", ".vcf.gz")
    files = []
    if not os.path.isdir(base_path):
        return files
    for dirpath, _, filenames in os.walk(base_path):
        for fn in filenames:
            if any(fn.lower().endswith(ext) for ext in SUPPORTED_EXTS):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, base_path)
                files.append({"path": rel, "full_path": full,
                               "name": fn, "size": os.path.getsize(full)})
    return files

def parse_fastq(content: str, max_reads: int = 50000):
    """Parse FASTQ: 4 lines per record (@header / seq / + / qual)."""
    records = []
    lines = content.splitlines()
    i = 0
    while i + 3 < len(lines) and len(records) < max_reads:
        header = lines[i].lstrip("@").strip()
        seq    = lines[i+1].strip()
        qual   = lines[i+3].strip()
        if seq and qual and len(seq) == len(qual):
            gc = sum(1 for c in seq if c in "GCgc")
            records.append({
                "id":      header.split()[0],
                "length":  len(seq),
                "gc_pct":  round(gc / len(seq) * 100, 1),
                "quality": round(sum(ord(c)-33 for c in qual) / len(qual), 1),
                "sequence_preview": seq[:60] + ("…" if len(seq) > 60 else ""),
            })
        i += 4
    return records


def parse_fasta(content: str, max_seqs: int = 50000):
    """Parse FASTA: >header lines followed by sequence lines."""
    records = []
    current_id, current_seq = None, []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_id and len(records) < max_seqs:
                seq = "".join(current_seq)
                gc = sum(1 for c in seq if c in "GCgc")
                records.append({
                    "id":      current_id.split()[0],
                    "length":  len(seq),
                    "gc_pct":  round(gc / len(seq) * 100, 1) if seq else 0,
                    "sequence_preview": seq[:60] + ("…" if len(seq) > 60 else ""),
                })
            # Store previous record before starting new one
            if current_id and len(records) < max_seqs:
                seq = "".join(current_seq)
                gc = sum(1 for c in seq if c in "GCgc")
                records.append({
                    "id":      current_id.split()[0],
                    "length":  len(seq),
                    "gc_pct":  round(gc / len(seq) * 100, 1) if seq else 0,
                    "sequence_preview": seq[:60] + ("…" if len(seq) > 60 else ""),
                    "_full_seq": seq,
                })
            current_id, current_seq = line[1:], []
        elif current_id:
            current_seq.append(line)
    if current_id and len(records) < max_seqs:
        seq = "".join(current_seq)
        gc = sum(1 for c in seq if c in "GCgc")
        records.append({
            "id":      current_id.split()[0],
            "length":  len(seq),
            "gc_pct":  round(gc / len(seq) * 100, 1) if seq else 0,
            "sequence_preview": seq[:60] + ("…" if len(seq) > 60 else ""),
            "_full_seq": seq,  # kept for GC sliding window, excluded from display
        })
    return records


def parse_vcf(content: str, max_variants: int = 100000):
    """
    Parse VCF (Variant Call Format) — plain text, tab-delimited after headers.
    Header lines start with ##, column header starts with #CHROM.
    Returns (list of variant dicts, list of metadata strings)
    """
    import gzip
    metadata = []
    variants = []
    columns  = []

    for line in content.splitlines():
        if line.startswith("##"):
            metadata.append(line[2:])
            continue
        if line.startswith("#CHROM"):
            columns = line[1:].split("	")
            continue
        if not line.strip() or not columns:
            continue
        parts = line.split("	")
        row = dict(zip(columns, parts))
        if len(variants) >= max_variants:
            break

        # Parse INFO field into a dict for key metrics
        info_raw = row.get("INFO", ".")
        info = {}
        if info_raw != ".":
            for item in info_raw.split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    info[k] = v
                else:
                    info[item] = True

        # Determine variant type
        ref = row.get("REF", "")
        alt = row.get("ALT", "")
        alts = [a for a in alt.split(",") if a != "."]
        if all(len(a) == 1 == len(ref) for a in alts):
            vtype = "SNP"
        elif any(len(a) != len(ref) for a in alts):
            vtype = "INDEL"
        else:
            vtype = "OTHER"

        variants.append({
            "CHROM":  row.get("CHROM", ""),
            "POS":    int(row.get("POS", 0)),
            "ID":     row.get("ID", "."),
            "REF":    ref,
            "ALT":    alt,
            "QUAL":   row.get("QUAL", "."),
            "FILTER": row.get("FILTER", "."),
            "TYPE":   vtype,
            "AF":     info.get("AF", info.get("MAF", None)),
            "DP":     info.get("DP", None),
        })

    return variants, metadata


def render_vcf_viewer(file_bytes: bytes, fname: str):
    """
    Render a VCF file from raw bytes.
    Shows: variant summary, type breakdown, quality distribution,
    allele frequency distribution, filterable variant table.
    No external library — pure Python parsing.
    """
    import gzip

    is_gz = fname.lower().endswith(".gz")
    try:
        raw = (gzip.decompress(file_bytes) if is_gz else file_bytes
               ).decode("utf-8", errors="replace")
    except Exception as e:
        st.error(f"Could not decode VCF: {e}")
        return

    MAX_VARIANTS = 100000
    with st.spinner("Parsing VCF…"):
        variants, metadata = parse_vcf(raw, MAX_VARIANTS)

    if not variants:
        st.warning("No variant records found. Check the file format.")
        return

    df = pd.DataFrame(variants)
    truncated = len(variants) == MAX_VARIANTS

    # Convert numeric columns
    df["POS"]  = pd.to_numeric(df["POS"],  errors="coerce")
    df["QUAL"] = pd.to_numeric(df["QUAL"], errors="coerce")
    df["AF"]   = pd.to_numeric(df["AF"],   errors="coerce")
    df["DP"]   = pd.to_numeric(df["DP"],   errors="coerce")

    pass_count = (df["FILTER"] == "PASS").sum()
    snp_count  = (df["TYPE"] == "SNP").sum()
    indel_count= (df["TYPE"] == "INDEL").sum()
    chroms     = df["CHROM"].nunique()

    # ── KPIs ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total variants", f"{len(df):,}" + (" (trunc.)" if truncated else ""))
    c2.metric("PASS variants",  f"{pass_count:,}")
    c3.metric("SNPs",           f"{snp_count:,}")
    c4.metric("INDELs",         f"{indel_count:,}")
    c5.metric("Chromosomes",    f"{chroms}")

    tabs = st.tabs(["📊 Summary", "🧬 Variants", "ℹ️ Header"])

    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            # Variant type breakdown
            type_counts = df["TYPE"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = ["#3B3BD3", "#28A464", "#CCB718"]
            bars = ax.bar(type_counts.index, type_counts.values,
                          color=colors[:len(type_counts)], edgecolor="none", width=0.5)
            for bar, val in zip(bars, type_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(type_counts.values)*0.01,
                        f"{val:,}", ha="center", va="bottom",
                        fontsize=10, fontweight="600", color="#2E2E38")
            ax.set_title("Variant types")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            # FILTER status pie
            filter_counts = df["FILTER"].value_counts().head(5)
            fig, ax = plt.subplots(figsize=(5, 4))
            pie_colors = ["#28A464", "#C20A29", "#CCB718", "#0070CC", "#8F8FA3"]
            ax.pie(filter_counts.values,
                   labels=filter_counts.index,
                   colors=pie_colors[:len(filter_counts)],
                   startangle=90,
                   wedgeprops={"edgecolor": "white", "linewidth": 2},
                   textprops={"fontsize": 11, "color": "#2E2E38"})
            ax.set_title("Filter status")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Quality score distribution
        qual_data = df["QUAL"].dropna()
        if len(qual_data) > 0:
            st.subheader("Quality score distribution")
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.hist(qual_data, bins=50, edgecolor="none",
                    alpha=0.85, color="#3B3BD3")
            ax.axvline(qual_data.median(), color="#C20A29", linestyle="--",
                       linewidth=1.5, label=f"Median Q{qual_data.median():.0f}")
            ax.set_xlabel("QUAL score"); ax.set_ylabel("Count")
            ax.set_title("Variant quality scores")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Allele frequency distribution
        af_data = df["AF"].dropna()
        if len(af_data) > 0:
            st.subheader("Allele frequency distribution")
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.hist(af_data.astype(float), bins=50, edgecolor="none",
                    alpha=0.85, color="#28A464")
            ax.set_xlabel("Allele frequency (AF)"); ax.set_ylabel("Count")
            ax.set_title("Variant allele frequencies")
            ax.set_xlim(0, 1)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Variants per chromosome
        st.subheader("Variants per chromosome")
        chrom_counts = df["CHROM"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(range(len(chrom_counts)), chrom_counts.values,
               color="#0070CC", edgecolor="none", alpha=0.85)
        ax.set_xticks(range(len(chrom_counts)))
        ax.set_xticklabels(chrom_counts.index, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Variant count")
        ax.set_title("Variant distribution across chromosomes")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tabs[1]:
        # Filterable variant table
        st.subheader(f"Variants ({len(df):,} total)")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            chrom_filter = st.multiselect(
                "Filter by chromosome",
                options=sorted(df["CHROM"].unique()),
                default=[])
        with filter_col2:
            type_filter = st.multiselect(
                "Filter by type",
                options=df["TYPE"].unique().tolist(),
                default=[])

        display = df.copy()
        if chrom_filter:
            display = display[display["CHROM"].isin(chrom_filter)]
        if type_filter:
            display = display[display["TYPE"].isin(type_filter)]

        st.caption(f"Showing {len(display):,} of {len(df):,} variants")
        st.dataframe(display, use_container_width=True, height=500)
        st.download_button(
            "Download filtered VCF table (CSV)",
            display.to_csv(index=False).encode(),
            f"{os.path.basename(fname)}_variants.csv", "text/csv")

    with tabs[2]:
        st.subheader("VCF header metadata")
        if metadata:
            # Show key metadata groups
            for prefix in ["fileformat", "reference", "contig", "INFO", "FILTER", "FORMAT"]:
                group = [m for m in metadata if m.startswith(prefix)]
                if group:
                    with st.expander(f"{prefix} ({len(group)})"):
                        for m in group[:20]:
                            st.text(m)
        else:
            st.info("No header metadata found.")


def render_sequence_viewer(file_bytes: bytes, fname: str):
    """
    Render FASTQ or FASTA from raw bytes — no external library needed.
    Shows: read count, length distribution, GC%, quality scores (FASTQ only).
    """
    import gzip

    is_gz = fname.lower().endswith(".gz")
    try:
        raw = (gzip.decompress(file_bytes) if is_gz else file_bytes
               ).decode("utf-8", errors="replace")
    except Exception as e:
        st.error(f"Could not decode file: {e}")
        return

    is_fastq = any(fname.lower().endswith(ext)
                   for ext in [".fastq", ".fastq.gz", ".fq", ".fq.gz"])

    with st.spinner("Parsing sequences…"):
        records = parse_fastq(raw) if is_fastq else parse_fasta(raw)

    if not records:
        st.warning("No sequence records found. Check the file format.")
        return

    df = pd.DataFrame(records).drop(columns=["_full_seq"], errors="ignore")
    fmt = "FASTQ" if is_fastq else "FASTA"

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{fmt} reads", f"{len(df):,}")
    c2.metric("Median length", f"{int(df['length'].median()):,} bp")
    c3.metric("Mean GC%", f"{df['gc_pct'].mean():.1f}%")
    if is_fastq:
        c4.metric("Mean quality", f"Q{df['quality'].mean():.1f}")
    else:
        c4.metric("Total bases", f"{df['length'].sum():,}")

    # For a single genome FASTA, compute GC sliding window from raw sequence
    # For multi-sequence FASTA/FASTQ, use per-record GC distribution
    is_single_genome = (not is_fastq and len(records) == 1)

    tab_names = ["📊 Composition"]
    if is_single_genome:
        tab_names.append("🧬 GC Landscape")
    else:
        tab_names.append("📊 Distributions")
    tab_names.append("🔬 Records")
    if is_fastq:
        tab_names.append("📈 Quality")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    with tabs[tab_idx]:
        tab_idx += 1
        if is_single_genome:
            # Single genome — show nucleotide composition bar chart
            seq = records[0].get("_full_seq", "")
            total = len(seq) if seq else 1
            nuc_counts = {
                "A": seq.count("A") + seq.count("a"),
                "T": seq.count("T") + seq.count("t"),
                "G": seq.count("G") + seq.count("g"),
                "C": seq.count("C") + seq.count("c"),
            }
            nuc_pcts = {k: round(v / total * 100, 2) for k, v in nuc_counts.items()}

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Nucleotide composition
            colors = ["#3B3BD3", "#0070CC", "#28A464", "#CCB718"]
            bars = ax1.bar(nuc_pcts.keys(), nuc_pcts.values(),
                           color=colors, edgecolor="none", width=0.5)
            ax1.set_ylabel("Percentage (%)")
            ax1.set_title("Nucleotide composition")
            ax1.set_ylim(0, max(nuc_pcts.values()) * 1.2)
            for bar, (nuc, pct) in zip(bars, nuc_pcts.items()):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f"{pct:.1f}%", ha="center", va="bottom",
                         fontsize=11, fontweight="600", color="#2E2E38")

            # AT vs GC pie
            at_pct = nuc_pcts["A"] + nuc_pcts["T"]
            gc_pct = nuc_pcts["G"] + nuc_pcts["C"]
            ax2.pie([at_pct, gc_pct],
                    labels=[f"AT  {at_pct:.1f}%", f"GC  {gc_pct:.1f}%"],
                    colors=["#3B3BD3", "#28A464"],
                    startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
                    textprops={"fontsize": 12, "color": "#2E2E38"})
            ax2.set_title("AT / GC content")

            plt.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            # Multi-sequence — show GC distribution histogram
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(df["gc_pct"], bins=50, edgecolor="none", alpha=0.85, color="#28A464")
            ax.axvline(df["gc_pct"].mean(), color="#C20A29", linestyle="--", linewidth=1.5,
                       label=f"Mean {df['gc_pct'].mean():.1f}%")
            ax.set_xlabel("GC%"); ax.set_ylabel("Count")
            ax.set_title("GC content distribution"); ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    with tabs[tab_idx]:
        tab_idx += 1
        if is_single_genome:
            # GC sliding window across the genome
            seq = records[0].get("_full_seq", "")
            if seq:
                window = st.slider("Window size (bp)", 100, 2000, 500, 100)
                step = window // 2

                positions, gc_values = [], []
                for i in range(0, len(seq) - window, step):
                    chunk = seq[i:i + window]
                    gc = sum(1 for c in chunk if c in "GCgc")
                    positions.append((i + window // 2) / 1000)  # kb
                    gc_values.append(gc / window * 100)

                mean_gc = sum(gc_values) / len(gc_values) if gc_values else 0

                fig, ax = plt.subplots(figsize=(14, 4))
                ax.fill_between(positions, gc_values, alpha=0.2, color="#28A464")
                ax.plot(positions, gc_values, color="#28A464", linewidth=1.2,
                        label=f"GC% ({window}bp window)")
                ax.axhline(mean_gc, color="#C20A29", linestyle="--", linewidth=1.2,
                           label=f"Mean {mean_gc:.1f}%")
                ax.set_xlabel("Genomic position (kb)")
                ax.set_ylabel("GC%")
                ax.set_title(f"GC content across genome  —  {len(seq)/1000:.1f} kb total")
                ax.legend(loc="upper right")
                ax.set_ylim(max(0, mean_gc - 20), min(100, mean_gc + 20))
                plt.tight_layout()
                st.pyplot(fig); plt.close()
                st.caption(
                    "GC-rich regions often correspond to coding sequences and regulatory elements. "
                    "GC-poor regions may indicate intergenic or repetitive sequence.")
            else:
                st.info("Full sequence not available for sliding window analysis.")
        else:
            # Multi-sequence — length distribution
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(df["length"], bins=50, edgecolor="none", alpha=0.85, color="#3B3BD3")
            ax.set_xlabel("Sequence length (bp)"); ax.set_ylabel("Count")
            ax.set_title("Length distribution")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    with tabs[tab_idx]:
        tab_idx += 1
        st.dataframe(df.head(1000), use_container_width=True, height=500)
        st.download_button("Download summary CSV",
                           df.to_csv(index=False).encode(),
                           f"{os.path.basename(fname)}_summary.csv", "text/csv")

    if is_fastq and tab_idx < len(tabs):
        with tabs[tab_idx]:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(df["quality"], bins=40, edgecolor="none", alpha=0.85, color="#0070CC")
            ax.axvline(20, color="#CCB718", linestyle="--", linewidth=1.5,
                       label="Q20 (99% accuracy)")
            ax.axvline(30, color="#C20A29", linestyle="--", linewidth=1.5,
                       label="Q30 (99.9% accuracy)")
            ax.set_xlabel("Mean Phred quality score"); ax.set_ylabel("Count")
            ax.set_title("Quality score distribution"); ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.caption("Q20 = 1 error per 100 bases · Q30 = 1 error per 1,000 bases")


def render_dicom_viewer_inline(file_bytes: bytes):
    """
    Render a DICOM image from raw bytes.

    Fixes vs original:
    - file_bytes stored in session_state so preset buttons don't lose the image
    - Uses st.image() instead of matplotlib for browser-native zoom/pan
    - Window sliders update in real time without losing the loaded file
    - Preset buttons update windowing without triggering a full file reload
    """
    if not DICOM_AVAILABLE:
        st.error("pydicom not available.")
        return

    # Store bytes in session state so preset buttons / slider reruns
    # don't lose the downloaded file
    if "dicom_bytes" not in st.session_state or        st.session_state.get("dicom_bytes_id") != id(file_bytes):
        st.session_state.dicom_bytes = file_bytes
        st.session_state.dicom_bytes_id = id(file_bytes)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(st.session_state.dicom_bytes)
        tmp_path = tmp.name

    try:
        ds = pydicom.dcmread(tmp_path)
        if not hasattr(ds, "pixel_array"):
            st.warning("This DICOM file has no pixel data.")
            return

        arr = ds.pixel_array
        if arr.ndim == 3:
            arr = arr[0] if arr.shape[0] < arr.shape[1]                   else np.mean(arr, axis=2)

        # Default windowing from DICOM tags or image stats
        wc_raw = getattr(ds, "WindowCenter", np.mean(arr))
        ww_raw = getattr(ds, "WindowWidth",  np.std(arr) * 4)
        d_center = int(wc_raw[0] if isinstance(wc_raw, (list, tuple)) else wc_raw)
        d_width  = int(ww_raw[0] if isinstance(ww_raw, (list, tuple)) else ww_raw)

        arr_min, arr_max = int(arr.min()), int(arr.max())

        # ── Controls (left) + Image (right) ──────────────────────────────────
        col_ctrl, col_img = st.columns([1, 3])

        with col_ctrl:
            st.markdown("**Window presets**")
            presets = {
                "Default":      (d_center, d_width),
                "Soft Tissue":  (40,   400),
                "Lung":         (-600, 1200),
                "Bone":         (300,  1500),
                "Brain":        (40,   80),
                "Abdomen":      (60,   400),
            }
            for pname, (pc, pw) in presets.items():
                if st.button(pname, key=f"preset_{pname}", use_container_width=True):
                    st.session_state.window_center = pc
                    st.session_state.window_width  = pw
                    # No st.rerun() — button click already triggers rerun
                    # and dicom_bytes is in session_state so image persists

            st.divider()
            st.markdown("**Fine control**")
            wc = st.slider(
                "Window center",
                arr_min, arr_max,
                st.session_state.get("window_center", d_center),
                key="window_center")
            ww = st.slider(
                "Window width",
                1, max(1, arr_max - arr_min),
                st.session_state.get("window_width", d_width),
                key="window_width")

            st.divider()
            st.markdown(f"**Shape:** {arr.shape}")
            st.markdown(f"**Range:** {arr_min} – {arr_max}")
            modality = getattr(ds, "Modality", "")
            if modality:
                st.markdown(f"**Modality:** {modality}")
            study_desc = getattr(ds, "StudyDescription", "")
            if study_desc:
                st.markdown(f"**Study:** {study_desc}")

        with col_img:
            # Apply windowing
            lo = wc - ww // 2
            hi = wc + ww // 2
            windowed = ((np.clip(arr, lo, hi) - lo) /
                        max(hi - lo, 1) * 255).astype(np.uint8)

            # Convert to PIL Image and display via st.image()
            # st.image supports browser-native zoom (click to expand)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(windowed).convert("RGB")
            st.image(pil_img, use_container_width=True,
                     caption=f"W:{ww} C:{wc}")

    finally:
        os.unlink(tmp_path)


def get_all_subdirectories(base_path):
    """
    Recursively collect all subdirectories under base_path.
    Returns a sorted list of paths relative to base_path.
    The base directory itself is represented as '.'.
    """
    subdirs = {'.'}
    for dirpath, dirnames, _ in os.walk(base_path):
        for d in dirnames:
            rel = os.path.relpath(os.path.join(dirpath, d), base_path)
            subdirs.add(rel)
    return sorted(subdirs)

def get_data_files(root_dir):
    """
    Walk root_dir and return list of (relative_path, full_path, file_type)
    for supported life sciences file formats.
    """
    files = []
    dicom_extensions = {'.dcm', '.dicom', '.dic', '.ima'}
    fastq_extensions = {'.fastq', '.fastq.gz', '.fq', '.fq.gz'}
    fasta_extensions = {'.fasta', '.fa', '.fna', '.ffn'}
    vcf_extensions   = {'.vcf', '.vcf.gz'}

    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            lower = fn.lower()
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root_dir)

            if lower.endswith('.parquet') or lower.endswith('.xpt'):
                files.append((rel, full, 'data'))
            elif any(lower.endswith(ext) for ext in dicom_extensions):
                files.append((rel, full, 'dicom'))
            elif lower.endswith('.nii.gz') or lower.endswith('.nii'):
                files.append((rel, full, 'nifti'))
            elif any(lower.endswith(ext) for ext in fastq_extensions):
                files.append((rel, full, 'fastq'))
            elif any(lower.endswith(ext) for ext in fasta_extensions):
                files.append((rel, full, 'fasta'))
            elif any(lower.endswith(ext) for ext in vcf_extensions):
                files.append((rel, full, 'vcf'))
            elif '.' not in fn and DICOM_AVAILABLE:
                try:
                    pydicom.dcmread(full, stop_before_pixels=True)
                    files.append((rel, full, 'dicom'))
                except:
                    pass

    return files

def is_dicom_file(file_path):
    """Check if a file is a valid DICOM file"""
    if not DICOM_AVAILABLE:
        return False
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except:
        return False

def load_dicom_image(file_path):
    """Load DICOM image and return image array and metadata"""
    try:
        dicom_data = pydicom.dcmread(file_path)
        
        # Extract image array
        if hasattr(dicom_data, 'pixel_array'):
            image_array = dicom_data.pixel_array
            
            # Handle different image types
            if len(image_array.shape) == 3:
                # Multi-frame or color image
                if image_array.shape[0] < image_array.shape[1]:
                    # Likely multi-frame, use first frame
                    image_array = image_array[0]
                else:
                    # Color image, convert to grayscale
                    image_array = np.mean(image_array, axis=2)
            
            return image_array, dicom_data
        else:
            return None, dicom_data
            
    except Exception as e:
        st.error(f"Error loading DICOM file: {str(e)}")
        return None, None

def apply_windowing(image_array, window_center, window_width):
    """Apply windowing (contrast/brightness) to medical image"""
    if image_array is None:
        return None
    
    # Calculate window bounds
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2
    
    # Apply windowing
    windowed = np.clip(image_array, window_min, window_max)
    
    # Normalize to 0-255 for display
    windowed = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
    
    return windowed

def get_dicom_metadata(dicom_data):
    """Extract key DICOM metadata for display"""
    metadata = {}
    
    # Common DICOM tags
    tags_to_extract = {
        'PatientName': 'Patient Name',
        'PatientID': 'Patient ID',
        'StudyDate': 'Study Date',
        'StudyTime': 'Study Time',
        'Modality': 'Modality',
        'StudyDescription': 'Study Description',
        'SeriesDescription': 'Series Description',
        'ImageType': 'Image Type',
        'Rows': 'Image Height',
        'Columns': 'Image Width',
        'PixelSpacing': 'Pixel Spacing',
        'SliceThickness': 'Slice Thickness',
        'WindowCenter': 'Window Center',
        'WindowWidth': 'Window Width',
        'RescaleIntercept': 'Rescale Intercept',
        'RescaleSlope': 'Rescale Slope'
    }
    
    for tag, display_name in tags_to_extract.items():
        if hasattr(dicom_data, tag):
            value = getattr(dicom_data, tag)
            if value is not None and str(value).strip():
                metadata[display_name] = str(value)
    
    return metadata

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    for col, filter_config in filters.items():
        if col not in df.columns:
            continue
            
        filter_type = filter_config.get('type')
        filter_value = filter_config.get('value')
        
        if filter_type == 'equals' and filter_value:
            filtered_df = filtered_df[filtered_df[col] == filter_value]
        elif filter_type == 'contains' and filter_value:
            if df[col].dtype == 'object':
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(filter_value), na=False, case=False)]
        elif filter_type == 'range' and filter_value:
            min_val, max_val = filter_value
            if min_val is not None:
                filtered_df = filtered_df[filtered_df[col] >= min_val]
            if max_val is not None:
                filtered_df = filtered_df[filtered_df[col] <= max_val]
    
    return filtered_df

def parse_query(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """Parse and execute basic query operations"""
    if not query.strip():
        return df
    
    try:
        # Replace column names with proper pandas syntax
        query_processed = query
        for col in df.columns:
            # Handle column names with spaces or special characters
            safe_col = f"df['{col}']"
            query_processed = re.sub(rf'\b{re.escape(col)}\b', safe_col, query_processed)
        
        # Replace operators
        query_processed = query_processed.replace(' AND ', ' & ')
        query_processed = query_processed.replace(' OR ', ' | ')
        query_processed = query_processed.replace(' NOT ', ' ~ ')
        query_processed = query_processed.replace(' <> ', ' != ')
        
        # Handle LIKE operator (simple contains)
        like_pattern = r"df\['([^']+)'\]\s+LIKE\s+'([^']+)'"
        query_processed = re.sub(like_pattern, r"df['\1'].astype(str).str.contains('\2', na=False, case=False)", query_processed)
        
        # Execute query
        mask = eval(query_processed)
        return df[mask]
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        return df

def get_basic_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Get basic statistics for a column"""
    stats = {}
    
    if df[column].dtype in ['int64', 'float64', 'int32', 'float32']:
        stats.update({
            'Count': len(df[column].dropna()),
            'Mean': df[column].mean(),
            'Median': df[column].median(),
            'Std Dev': df[column].std(),
            'Min': df[column].min(),
            'Max': df[column].max(),
            'Missing': df[column].isna().sum()
        })
    else:
        stats.update({
            'Count': len(df[column].dropna()),
            'Unique Values': df[column].nunique(),
            'Most Common': df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A',
            'Missing': df[column].isna().sum()
        })
    
    return stats

def display_frequency_table(df: pd.DataFrame, column: str, max_categories: int = 20):
    """Display frequency table for categorical data"""
    if df[column].dtype == 'object' or df[column].nunique() <= max_categories:
        freq_table = df[column].value_counts().head(max_categories)
        st.write(f"**Top {min(max_categories, len(freq_table))} values:**")
        
        freq_df = pd.DataFrame({
            'Value': freq_table.index,
            'Count': freq_table.values,
            'Percentage': (freq_table.values / len(df) * 100).round(2)
        })
        st.dataframe(freq_df)
    else:
        st.write("Too many unique values to display frequency table")

# Initialize session state
st.set_page_config(
    page_title="Life Sciences Data Viewer",
    page_icon="🔬",
    layout="wide",
)

# ── Domino Design System theme ────────────────────────────────────────────────
# Colors and typography from domino-apps-guidelines
# Primary: #3B3BD3  Text: #2E2E38  Secondary bg: #FAFAFA  Border: #E0E0E0
# Font: Inter (Domino's primary typeface)
st.markdown("""
<style>
/* ── Inter font — Domino's primary typeface ─────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Lato', 'Helvetica Neue', Arial, sans-serif;
}

/* ── Top navigation bar ─────────────────────────────────────────────── */
/* Height: 44px, Background: #2E2E38 (neutralDark700) per guidelines    */
header[data-testid="stHeader"] {
    background-color: #2E2E38;
    height: 44px;
}
header[data-testid="stHeader"]::before {
    content: "🔬  Life Sciences Data Viewer";
    color: #FFFFFF;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 600;
    padding: 0 24px;
    display: flex;
    align-items: center;
    height: 44px;
    line-height: 44px;
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E0E0E0;
}
section[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif;
}

/* ── Main content area ──────────────────────────────────────────────── */
.main .block-container {
    padding-top: 24px;
    padding-bottom: 40px;
    max-width: 1200px;
}

/* ── Typography ─────────────────────────────────────────────────────── */
h1 { font-size: 24px; font-weight: 600; color: #2E2E38; }
h2 { font-size: 18px; font-weight: 600; color: #2E2E38; }
h3 { font-size: 14px; font-weight: 600; color: #2E2E38; }
p, li { color: #2E2E38; font-size: 14px; }

/* ── Primary button — Domino purple ─────────────────────────────────── */
button[kind="primary"], .stButton > button[data-baseweb="button"] {
    background-color: #3B3BD3 !important;
    border-color: #3B3BD3 !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    padding: 6px 16px !important;
}
button[kind="primary"]:hover,
.stButton > button[data-baseweb="button"]:hover {
    background-color: #3123B1 !important;
    border-color: #3123B1 !important;
}

/* ── Secondary buttons (presets, etc.) ──────────────────────────────── */
.stButton > button:not([data-baseweb="button"]) {
    background-color: #EDECFB !important;
    border: 1px solid #C9C5F2 !important;
    color: #1820A0 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 4px !important;
    box-shadow: none !important;
}
.stButton > button:not([data-baseweb="button"]):hover {
    background-color: #C9C5F2 !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 16px;
}
[data-testid="stMetricLabel"] { color: #65657B; font-size: 12px; font-weight: 500; }
[data-testid="stMetricValue"] { color: #2E2E38; font-size: 24px; font-weight: 600; }

/* ── Selectbox & inputs ─────────────────────────────────────────────── */
[data-baseweb="select"] {
    border-radius: 4px !important;
    border-color: #E0E0E0 !important;
}
[data-baseweb="input"] {
    border-radius: 4px !important;
    border-color: #E0E0E0 !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    border-bottom: 2px solid #E0E0E0;
    gap: 0;
}
[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #65657B !important;
    padding: 8px 16px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #3B3BD3 !important;
    border-bottom: 2px solid #3B3BD3 !important;
}

/* ── Success/info/warning callouts ─────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 4px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
}

/* ── Dataframe ─────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #E0E0E0;
    border-radius: 4px;
}

/* ── Caption / helper text ──────────────────────────────────────────── */
[data-testid="stCaptionContainer"] {
    color: #65657B;
    font-size: 12px;
}

/* ── Expanders ──────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E0E0E0;
    border-radius: 4px;
}

/* ── Divider ────────────────────────────────────────────────────────── */
hr { border-color: #E0E0E0; }

/* ── Hide sidebar collapse button (prevents accidental hide during demo) */
button[data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

if 'filters' not in st.session_state:
    st.session_state.filters = {}
if 'sort_column' not in st.session_state:
    st.session_state.sort_column = None
if 'sort_ascending' not in st.session_state:
    st.session_state.sort_ascending = True
if 'hidden_columns' not in st.session_state:
    st.session_state.hidden_columns = set()
if 'frozen_columns' not in st.session_state:
    st.session_state.frozen_columns = []

# ── Identity & Extension context ─────────────────────────────────────────────
username  = get_viewer_username()
api_token = get_viewer_api_token()

# Extension URL params — all injected by Domino Extensions framework:
#   projectId       — user clicked Extension from a project sidebar
#   datasetId       — user clicked Extension from a dataset context
#   datasetSnapshotId + filePath — user right-clicked a specific file
#   mountPointType  — tells us which context we were launched from
project_id          = st.query_params.get("projectId", None)
dataset_id          = st.query_params.get("datasetId", None)
dataset_snapshot_id = st.query_params.get("datasetSnapshotId", None)
file_path_param     = st.query_params.get("filePath", None)
mount_point_type    = st.query_params.get("mountPointType", None)

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [("project_info", None), ("datasets", None), ("refresh_flag", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# Resolve project info once per session
if project_id and st.session_state.project_info is None and api_token:
    st.session_state.project_info = resolve_project(project_id, api_token)
project_info = st.session_state.project_info

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Life Sciences Data Viewer")
    display_name = username or ""
    if display_name:
        st.markdown(f"👤 **{display_name}**")
    if project_info:
        st.markdown(f"📁 **Project:** {project_info['name']}")
        st.markdown(f"👥 **Owner:** {project_info['ownerUsername']}")
    elif project_id:
        st.markdown(f"📁 **Project ID:** `{project_id}`")
        st.caption("Could not resolve — check token")
    if dataset_id:
        st.markdown(f"🗄️ **Dataset:** `{dataset_id[:12]}…`")
    if file_path_param:
        st.markdown(f"📄 **File:** `{os.path.basename(file_path_param)}`")
    if st.button("🔄 Refresh"):
        st.session_state.datasets = None
        st.rerun()

st.title("🔬 Life Sciences Data Viewer")

# ── Welcome banner — shown when no file context is active ─────────────────────
if not (dataset_id and file_path_param) and not project_id:
    # Standalone mode with no file loaded yet — show onboarding
    _show_welcome = True
elif project_id and not st.session_state.get("loaded_file_bytes"):
    _show_welcome = True
else:
    _show_welcome = False

if _show_welcome:
    st.markdown("""
<div style="
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-left: 4px solid #3B3BD3;
    border-radius: 8px;
    padding: 24px 28px;
    margin-bottom: 24px;
">
<p style="margin:0 0 8px 0; font-size:15px; font-weight:600; color:#2E2E38;">
    About this Extension
</p>
<p style="margin:0 0 16px 0; font-size:14px; color:#65657B; line-height:1.6;">
    <strong>Life Sciences Data Viewer</strong> lets you explore and visualise life sciences 
    datasets stored in Domino — directly from any project sidebar. Select a dataset below 
    to browse its files, then click <strong>Load File</strong> to open it.
</p>
<p style="margin:0 0 10px 0; font-size:13px; font-weight:600; color:#2E2E38;">Supported formats</p>
<div style="display:flex; flex-wrap:wrap; gap:8px;">
    <span style="background:#EDECFB; color:#1820A0; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500;">📊 Parquet</span>
    <span style="background:#EDECFB; color:#1820A0; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500;">📊 XPT (SAS Transport)</span>
    <span style="background:#E8F5EE; color:#1A6B3E; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500;">🩻 DICOM</span>
    <span style="background:#E8F5EE; color:#1A6B3E; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500;">🧠 NIfTI</span>
    <span style="background:#FFF8E1; color:#7B5B00; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500;">🧬 FASTQ / FASTA</span>
    <span style="background:#FFF8E1; color:#7B5B00; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500;">🧬 VCF Variants</span>
</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DIRECT FILE MODE
# Launched from datasetFileContext mount point with a specific file.
# Download and render directly without any browsing UI.
# ══════════════════════════════════════════════════════════════════════════════
if dataset_id and file_path_param and api_token:
    st.caption(f"Viewing file from dataset — `{file_path_param}`")
    fname = os.path.basename(file_path_param).lower()

    with st.spinner(f"Loading `{os.path.basename(file_path_param)}` from dataset…"):
        file_bytes = download_dataset_file(dataset_id, file_path_param, api_token)

    if file_bytes is None:
        st.error(f"Could not download `{file_path_param}` from dataset `{dataset_id}`. "
                 f"Check that you have access to this dataset.")
        st.stop()

    # Route to the right viewer based on extension
    if fname.endswith(".parquet"):
        import pyarrow.parquet as pq
        df = pd.read_parquet(io.BytesIO(file_bytes))
        st.success(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")
        st.dataframe(df.head(500), use_container_width=True, height=600)
    elif fname.endswith(".xpt"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xpt", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        df, _ = pyreadstat.read_xport(tmp_path)
        os.unlink(tmp_path)
        st.success(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")
        st.dataframe(df.head(500), use_container_width=True, height=600)
    elif any(fname.endswith(ext) for ext in [".dcm", ".dicom", ".dic", ".ima"]):
        if DICOM_AVAILABLE:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            dicom_file_map = {os.path.basename(file_path_param): tmp_path}
            os.unlink(tmp_path)
            # Re-write bytes to temp for pydicom
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            dicom_file_map = {os.path.basename(file_path_param): tmp_path}
        else:
            st.error("pydicom not available in this environment.")
            st.stop()
    elif fname.endswith(".nii") or fname.endswith(".nii.gz"):
        st.warning("NIfTI direct viewing from dataset requires nibabel — "
                   "use the dataset browser mode instead.")
        st.stop()
    else:
        st.warning(f"File type not supported for direct viewing: `{fname}`")
        st.stop()

    # For DICOM direct file mode, render the viewer
    if any(fname.endswith(ext) for ext in [".dcm", ".dicom", ".dic", ".ima"]) and DICOM_AVAILABLE:
        render_dicom_viewer_inline(file_bytes)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# DATASET BROWSER MODE
# Launched from projectSidebar or datasetContext — show dataset list then files.
# Uses the viewer's token so they only see their own accessible datasets.
# ══════════════════════════════════════════════════════════════════════════════

if not api_token:
    st.warning("No Domino token available. This app requires Extended Identity "
               "Propagation to be enabled. For local testing, ensure the "
               "app is running inside a Domino environment.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MODE: API dataset browser (Extension context — projectId present)
# ══════════════════════════════════════════════════════════════════════════════
if project_id:
    if st.session_state.datasets is None:
        with st.spinner("Loading datasets…"):
            st.session_state.datasets = list_project_datasets(project_id, api_token)
    datasets = st.session_state.datasets

    # preselected_dataset_id: set when launched from datasetContext mount point
    preselected_dataset_id = dataset_id

    if not datasets:
        st.info("No datasets found for this project. "
                "Make sure at least one dataset is mounted to the project.")
        st.stop()

    # Build selector — prefer preselected dataset if we have one
    ds_options = {d.get("name", d.get("id", str(i))): d
                  for i, d in enumerate(datasets)}
    default_idx = 0
    if preselected_dataset_id:
        for i, d in enumerate(datasets):
            if d.get("id") == preselected_dataset_id:
                default_idx = i
                break

    selected_ds_name = st.selectbox(
        "Select dataset", list(ds_options.keys()), index=default_idx)
    selected_ds = ds_options[selected_ds_name]
    snapshot_id = selected_ds.get("readWriteSnapshotId", "")

    if not snapshot_id:
        st.warning("This dataset has no snapshot yet.")
        st.stop()

    # List files via API — no filesystem mount needed
    with st.spinner("Listing files…"):
        files = list_snapshot_files(snapshot_id, api_token)

    if not files:
        st.markdown("""
<div style="
    background:#FAFAFA; border:1px solid #E0E0E0; border-radius:8px;
    padding:24px 28px; text-align:center;
">
<p style="font-size:32px; margin:0 0 8px 0;">📂</p>
<p style="font-size:15px; font-weight:600; color:#2E2E38; margin:0 0 8px 0;">
    No supported files in this dataset
</p>
<p style="font-size:13px; color:#65657B; margin:0 0 16px 0;">
    Add files to this dataset to view them here. Supported formats:
</p>
<div style="display:flex; flex-wrap:wrap; gap:6px; justify-content:center;">
    <span style="background:#EDECFB;color:#1820A0;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;">Parquet</span>
    <span style="background:#EDECFB;color:#1820A0;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;">XPT</span>
    <span style="background:#E8F5EE;color:#1A6B3E;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;">DICOM</span>
    <span style="background:#E8F5EE;color:#1A6B3E;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;">NIfTI</span>
    <span style="background:#FFF8E1;color:#7B5B00;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;">FASTQ</span>
    <span style="background:#FFF8E1;color:#7B5B00;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;">FASTA</span>
    <span style="background:#FFF8E1;color:#7B5B00;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;">VCF</span>
</div>
</div>
""", unsafe_allow_html=True)
        st.stop()

    file_options = {f["path"]: f for f in files}
    selected_file_path = st.selectbox(
        f"Select file ({len(files)} supported)", list(file_options.keys()))

    selected_file_entry = file_options[selected_file_path]
    fname = selected_file_path.lower()
    size_kb = selected_file_entry["size_bytes"] // 1024
    st.caption(f"📄 `{selected_file_path}` — {size_kb:,} KB")

    # Clear cached file if user selects a different file
    if st.session_state.get("loaded_file_path") != selected_file_path:
        for k in ["loaded_file_bytes", "loaded_file_path", "dicom_bytes", "dicom_bytes_id"]:
            st.session_state.pop(k, None)

    if st.button("Load File", type="primary"):
        with st.spinner(f"Downloading via API…"):
            file_bytes = download_snapshot_file(snapshot_id, selected_file_path, api_token)

        if not file_bytes:
            st.error("Download failed. Check that you have access to this dataset.")
            st.stop()

        # Store in session state so reruns (windowing sliders/presets) don't re-download
        st.session_state.loaded_file_bytes = file_bytes
        st.session_state.loaded_file_path  = selected_file_path
        st.success(f"Downloaded {len(file_bytes):,} bytes")

    # Use cached bytes if already downloaded
    file_bytes = st.session_state.get("loaded_file_bytes")
    if not file_bytes:
        st.stop()

    # Parse and render based on file type
    if fname.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(file_bytes))
        st.subheader(f"{len(df):,} rows × {len(df.columns):,} columns")
        tab1, tab2, tab3 = st.tabs(["📊 Data", "🔍 Filter & Query", "📈 Analysis"])
        with tab1:
            rpp = st.slider("Rows per page", 10, 500, 50, 10)
            total = max(1, (len(df)-1)//rpp+1)
            _, cm, _ = st.columns([1,2,1])
            with cm:
                pg = st.number_input("Page", 1, total, 1)
            start = (pg-1)*rpp
            st.dataframe(df.iloc[start:start+rpp], use_container_width=True, height=600)
            st.download_button("Download CSV", df.to_csv(index=False).encode(),
                               "export.csv", "text/csv")
        with tab2:
            qtext = st.text_area("SQL-like query (e.g. AGE > 50 AND SEX == 'M')", height=80)
            if st.button("Run") and qtext:
                result = parse_query(qtext, df)
                st.success(f"{len(result):,} rows")
                st.dataframe(result.head(100), use_container_width=True)
        with tab3:
            col = st.selectbox("Analyse column", df.columns)
            if col:
                for k, v in get_basic_stats(df, col).items():
                    st.write(f"**{k}:** {v:.2f}" if isinstance(v, float) else f"**{k}:** {v}")

    elif fname.endswith(".xpt"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xpt", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            df, _ = pyreadstat.read_xport(tmp_path)
        finally:
            os.unlink(tmp_path)
        st.subheader(f"{len(df):,} rows × {len(df.columns):,} columns")
        tab1, tab2 = st.tabs(["📊 Data", "📈 Analysis"])
        with tab1:
            rpp = st.slider("Rows per page", 10, 500, 50, 10)
            total = max(1, (len(df)-1)//rpp+1)
            _, cm, _ = st.columns([1,2,1])
            with cm:
                pg = st.number_input("Page", 1, total, 1)
            start = (pg-1)*rpp
            st.dataframe(df.iloc[start:start+rpp], use_container_width=True, height=600)
        with tab2:
            col = st.selectbox("Analyse column", df.columns)
            if col:
                for k, v in get_basic_stats(df, col).items():
                    st.write(f"**{k}:** {v:.2f}" if isinstance(v, float) else f"**{k}:** {v}")

    elif any(fname.endswith(ext) for ext in [".dcm", ".dicom", ".dic", ".ima"]):
        render_dicom_viewer_inline(file_bytes)

    elif fname.endswith(".nii") or fname.endswith(".nii.gz"):
        if NIFTI_AVAILABLE:
            import tempfile, nibabel as nib
            suffix = ".nii.gz" if fname.endswith(".nii.gz") else ".nii"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                img = nib.load(tmp_path)
                data = img.get_fdata()
                st.success(f"Shape: {data.shape}, voxel: {img.header.get_zooms()}")
                vol = data[:,:,:,0] if data.ndim == 4 else data
                c1,c2,c3 = st.columns(3)
                z = c1.slider("Axial",    0, vol.shape[2]-1, vol.shape[2]//2)
                y = c2.slider("Coronal",  0, vol.shape[1]-1, vol.shape[1]//2)
                x = c3.slider("Sagittal", 0, vol.shape[0]-1, vol.shape[0]//2)
                fig, axes = plt.subplots(1,3,figsize=(15,5))
                axes[0].imshow(np.rot90(vol[:,:,z]),  cmap="gray"); axes[0].set_title(f"Axial z={z}")
                axes[1].imshow(np.rot90(vol[:,y,:]),  cmap="gray"); axes[1].set_title(f"Coronal y={y}")
                axes[2].imshow(np.rot90(vol[x,:,:]),  cmap="gray"); axes[2].set_title(f"Sagittal x={x}")
                for ax in axes: ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            finally:
                os.unlink(tmp_path)
        else:
            st.error("nibabel not installed in this environment.")

    elif any(fname.endswith(ext) for ext in
             [".fastq", ".fastq.gz", ".fq", ".fq.gz",
              ".fasta", ".fa", ".fna", ".ffn"]):
        render_sequence_viewer(file_bytes, selected_file_path)

    elif fname.endswith(".vcf") or fname.endswith(".vcf.gz"):
        render_vcf_viewer(file_bytes, selected_file_path)

    st.stop()  # Don't fall through to filesystem mode when in API mode

# ══════════════════════════════════════════════════════════════════════════════
# MODE: Filesystem browser (no projectId — standalone / owner's project)
# Original file viewer tabs — full featured with filters, analysis, settings
# ══════════════════════════════════════════════════════════════════════════════
if not project_id:
    # Filesystem browser — select folder and file
    is_git = os.environ.get("DOMINO_IS_GIT_BASED", "false").lower() == "true"
    base_dir = "/mnt" if is_git else "/domino/datasets"
    if st.button("Refresh Directories"):
        st.session_state.refresh_flag = not st.session_state.refresh_flag
    subdirs = get_all_subdirectories(base_dir)
    if not subdirs:
        st.error("No subdirectories found.")
        st.stop()
    selected_folder = st.selectbox("Select folder", subdirs)
    target_path = base_dir if selected_folder == "." else os.path.join(base_dir, selected_folder)
    st.write(f"Browsing: `{target_path}`")
    data_files = get_data_files(target_path)
    if not data_files:
        st.info("No supported files found here. "
                "Supported: .parquet, .xpt, .dcm, .dicom, .nii, .nii.gz, "
                ".fastq, .fastq.gz, .fq, .fq.gz, .fasta, .fa, .fna, .ffn")
        st.stop()
    data_files_only  = [(rel, full) for rel, full, ftype in data_files if ftype == 'data']
    dicom_files_only = [(rel, full) for rel, full, ftype in data_files if ftype == 'dicom']
    nifti_files_only = [(rel, full) for rel, full, ftype in data_files if ftype == 'nifti']
    fastq_files_only = [(rel, full) for rel, full, ftype in data_files if ftype == 'fastq']
    fasta_files_only = [(rel, full) for rel, full, ftype in data_files if ftype == 'fasta']
    vcf_files_only   = [(rel, full) for rel, full, ftype in data_files if ftype == 'vcf']

    # Build file type options based on what's actually present
    type_options = []
    if data_files_only:  type_options.append("📊 Data Files (Parquet/XPT)")
    if dicom_files_only: type_options.append("🩻 DICOM Images")
    if nifti_files_only: type_options.append("🧠 NIfTI Images")
    if fastq_files_only: type_options.append("🧬 FASTQ")
    if fasta_files_only: type_options.append("🧬 FASTA")
    if vcf_files_only:   type_options.append("🧬 VCF Variants")

    file_type = st.radio("File Type", type_options, horizontal=True)

    if file_type == "📊 Data Files (Parquet/XPT)":
        file_map = {rel: full for rel, full in data_files_only}
        selected_file = st.selectbox("Select a data file", list(file_map.keys()))

    elif file_type in ("🧬 FASTQ", "🧬 FASTA"):
        seq_files = fastq_files_only if file_type == "🧬 FASTQ" else fasta_files_only
        seq_map = {rel: full for rel, full in seq_files}
        selected_seq = st.selectbox("Select file", list(seq_map.keys()))
        with open(seq_map[selected_seq], "rb") as f:
            render_sequence_viewer(f.read(), selected_seq)
        st.stop()

    elif file_type == "🧬 VCF Variants":
        vcf_map = {rel: full for rel, full in vcf_files_only}
        selected_vcf = st.selectbox("Select VCF file", list(vcf_map.keys()))
        with open(vcf_map[selected_vcf], "rb") as f:
            render_vcf_viewer(f.read(), selected_vcf)
        st.stop()

    # Load the data (Data Files path only from here)
    file_path = file_map[selected_file]
    try:
        if selected_file.lower().endswith('.parquet'):
            original_df = pd.read_parquet(file_path)
        else:
            original_df, _ = pyreadstat.read_xport(file_path)

        st.success(f"Successfully loaded **{selected_file}** ({original_df.shape[0]} rows × {original_df.shape[1]} columns)")

    except Exception as e:
        st.error(f"Failed to load `{selected_file}`: {e}")
        st.stop()

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data View", "🔍 Filters & Query", "📈 Quick Analysis", "⚙️ Settings", "ℹ️ Info"])
    
    with tab1:
        st.header("Data View")
        
        # Apply filters
        df = apply_filters(original_df, st.session_state.filters)
        
        # Apply sorting
        if st.session_state.sort_column and st.session_state.sort_column in df.columns:
            df = df.sort_values(by=st.session_state.sort_column, ascending=st.session_state.sort_ascending)
        
        # Filter out hidden columns
        visible_columns = [col for col in df.columns if col not in st.session_state.hidden_columns]
        display_df = df[visible_columns]
        
        # Display row count
        st.write(f"**Showing {len(display_df)} rows** (filtered from {len(original_df)} total rows)")
        
        # Pagination controls
        rows_per_page = st.slider("Rows per page", min_value=10, max_value=500, value=50, step=10)
        
        total_pages = (len(display_df) - 1) // rows_per_page + 1
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1)
        
        # Calculate pagination
        start_idx = (page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(display_df))
        
        # Display dataframe
        st.dataframe(
            display_df.iloc[start_idx:end_idx],
            use_container_width=True,
            height=600
        )
        
        # Download options
        st.subheader("Download Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name=f"{selected_file.replace('.xpt', '').replace('.parquet', '')}_filtered.csv",
                mime="text/csv"
            )
        
        with col2:
            # Convert to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                display_df.to_excel(writer, index=False, sheet_name='Data')
            excel_data = output.getvalue()
            
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name=f"{selected_file.replace('.xpt', '').replace('.parquet', '')}_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with tab2:
        st.header("Filters & Query")
        
        # Column filters
        st.subheader("Column Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            filter_column = st.selectbox("Select column to filter", [""] + list(original_df.columns))
        
        if filter_column:
            with col2:
                col_dtype = original_df[filter_column].dtype
                
                if col_dtype in ['int64', 'float64', 'int32', 'float32']:
                    filter_type = st.selectbox("Filter type", ["range", "equals"])
                else:
                    filter_type = st.selectbox("Filter type", ["contains", "equals"])
            
            # Filter value input
            if filter_type == "range":
                min_val = float(original_df[filter_column].min())
                max_val = float(original_df[filter_column].max())
                
                range_values = st.slider(
                    f"Select range for {filter_column}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                
                if st.button("Apply Range Filter"):
                    st.session_state.filters[filter_column] = {
                        'type': 'range',
                        'value': range_values
                    }
                    st.success(f"Filter applied to {filter_column}")
                    st.rerun()
            
            elif filter_type == "equals":
                unique_values = original_df[filter_column].dropna().unique()
                filter_value = st.selectbox(f"Select value for {filter_column}", unique_values)
                
                if st.button("Apply Equals Filter"):
                    st.session_state.filters[filter_column] = {
                        'type': 'equals',
                        'value': filter_value
                    }
                    st.success(f"Filter applied to {filter_column}")
                    st.rerun()
            
            elif filter_type == "contains":
                filter_value = st.text_input(f"Text to search in {filter_column}")
                
                if st.button("Apply Contains Filter"):
                    st.session_state.filters[filter_column] = {
                        'type': 'contains',
                        'value': filter_value
                    }
                    st.success(f"Filter applied to {filter_column}")
                    st.rerun()
        
        # Show active filters
        if st.session_state.filters:
            st.subheader("Active Filters")
            for col, filter_info in st.session_state.filters.items():
                filter_type = filter_info['type']
                filter_value = filter_info['value']
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    if filter_type == 'range':
                        st.write(f"**{col}**: {filter_value[0]} to {filter_value[1]}")
                    else:
                        st.write(f"**{col}** {filter_type}: {filter_value}")
                
                with col2:
                    if st.button(f"Remove", key=f"remove_{col}"):
                        del st.session_state.filters[col]
                        st.rerun()
            
            if st.button("Clear All Filters"):
                st.session_state.filters = {}
                st.rerun()
        
        # SQL-like query
        st.subheader("Advanced Query")
        st.write("Use SQL-like syntax. Example: `AGE > 50 AND SEX == 'M'`")
        
        query_text = st.text_area("Enter query", height=100)
        
        if st.button("Execute Query"):
            if query_text:
                try:
                    filtered_df = parse_query(query_text, original_df)
                    st.success(f"Query returned {len(filtered_df)} rows")
                    st.dataframe(filtered_df.head(100), use_container_width=True)
                except Exception as e:
                    st.error(f"Query error: {str(e)}")
    
    with tab3:
        st.header("Quick Analysis")
        
        analysis_column = st.selectbox("Select column to analyze", original_df.columns)
        
        if analysis_column:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Statistics")
                stats = get_basic_stats(original_df, analysis_column)
                
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, (int, float)):
                        st.write(f"**{stat_name}:** {stat_value:.2f}")
                    else:
                        st.write(f"**{stat_name}:** {stat_value}")
            
            with col2:
                st.subheader("Frequency Distribution")
                display_frequency_table(original_df, analysis_column)
            
            # Visualization
            st.subheader("Visualization")
            
            if original_df[analysis_column].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Numeric column - histogram
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(original_df[analysis_column].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax.set_xlabel(analysis_column)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Distribution of {analysis_column}")
                st.pyplot(fig)
                plt.close()
            else:
                # Categorical column - bar chart
                value_counts = original_df[analysis_column].value_counts().head(15)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                value_counts.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.7)
                ax.set_xlabel(analysis_column)
                ax.set_ylabel("Count")
                ax.set_title(f"Top 15 values in {analysis_column}")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab4:
        st.header("Settings")
        
        # Sorting
        st.subheader("Sorting")
        col1, col2 = st.columns(2)
        
        with col1:
            sort_col = st.selectbox("Sort by column", ["None"] + list(original_df.columns))
        
        with col2:
            sort_order = st.radio("Sort order", ["Ascending", "Descending"])
        
        if st.button("Apply Sorting"):
            if sort_col != "None":
                st.session_state.sort_column = sort_col
                st.session_state.sort_ascending = (sort_order == "Ascending")
                st.success(f"Sorting applied: {sort_col} ({sort_order})")
                st.rerun()
            else:
                st.session_state.sort_column = None
                st.success("Sorting cleared")
                st.rerun()
        
        # Column visibility
        st.subheader("Column Visibility")
        
        cols_to_hide = st.multiselect(
            "Select columns to hide",
            options=list(original_df.columns),
            default=list(st.session_state.hidden_columns)
        )
        
        if st.button("Update Column Visibility"):
            st.session_state.hidden_columns = set(cols_to_hide)
            st.success("Column visibility updated")
            st.rerun()
    
    with tab5:
        st.header("Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overview")
            st.write(f"**File:** {selected_file}")
            st.write(f"**Rows:** {original_df.shape[0]:,}")
            st.write(f"**Columns:** {original_df.shape[1]:,}")
            st.write(f"**Memory Usage:** {original_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with col2:
            st.subheader("Data Types")
            dtype_counts = original_df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"**{dtype}:** {count} columns")
        
        st.subheader("Column Details")
        
        col_info = []
        for col in original_df.columns:
            col_info.append({
                'Column': col,
                'Type': str(original_df[col].dtype),
                'Non-Null': original_df[col].notna().sum(),
                'Null': original_df[col].isna().sum(),
                'Unique': original_df[col].nunique()
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True, height=400)
        
        # Missing data visualization
        st.subheader("Missing Data")
        missing_data = original_df.isna().sum()
        missing_pct = (missing_data / len(original_df) * 100).round(2)
        
        cols_with_missing = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(cols_with_missing) > 0:
            fig, ax = plt.subplots(figsize=(10, max(4, len(cols_with_missing) * 0.3)))
            cols_with_missing.plot(kind='barh', ax=ax, color='salmon')
            ax.set_xlabel("Number of Missing Values")
            ax.set_ylabel("Column")
            ax.set_title("Missing Values by Column")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.write("✅ No missing values in this dataset!")

else:  # DICOM Images
    if not DICOM_AVAILABLE:
        st.error("DICOM support not available. Please install pydicom: pip install pydicom")
        st.stop()
    
    if not dicom_files_only:
        st.info("No DICOM files found in this directory. Try selecting a different folder.")
        st.stop()
    
    dicom_file_map = {rel: full for rel, full in dicom_files_only}
    selected_dicom = st.selectbox("Select a DICOM file", list(dicom_file_map.keys()))
    
    # Load DICOM file
    dicom_path = dicom_file_map[selected_dicom]
    image_array, dicom_data = load_dicom_image(dicom_path)
    
    if image_array is not None:
        st.success(f"Successfully loaded DICOM image: **{selected_dicom}**")
        
        # Create tabs for DICOM viewing
        dicom_tab1, dicom_tab2, dicom_tab3 = st.tabs(["🖼️ Image Viewer", "🔧 Image Controls", "ℹ️ DICOM Info"])
        
        with dicom_tab1:
            st.header("DICOM Image Viewer")
            
            # Get default windowing values
            default_center = int(np.mean(image_array))
            default_width = int(np.std(image_array) * 4)
            
            # Check if DICOM has windowing info
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                if isinstance(dicom_data.WindowCenter, (list, tuple)):
                    default_center = int(dicom_data.WindowCenter[0])
                else:
                    default_center = int(dicom_data.WindowCenter)
                
                if isinstance(dicom_data.WindowWidth, (list, tuple)):
                    default_width = int(dicom_data.WindowWidth[0])
                else:
                    default_width = int(dicom_data.WindowWidth)
            
            # Display image with current settings
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Get windowing values from session state or defaults
                window_center = st.session_state.get('window_center', default_center)
                window_width = st.session_state.get('window_width', default_width)
                
                # Apply windowing
                windowed_image = apply_windowing(image_array, window_center, window_width)
                
                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(windowed_image, cmap='gray')
                ax.set_title(f"{selected_dicom}")
                ax.axis('off')
                
                # Display image
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Quick Info")
                st.write(f"**Dimensions:** {image_array.shape}")
                st.write(f"**Data Type:** {image_array.dtype}")
                st.write(f"**Min Value:** {image_array.min()}")
                st.write(f"**Max Value:** {image_array.max()}")
                st.write(f"**Mean:** {image_array.mean():.1f}")
                
                # Preset windowing options
                st.subheader("Window Presets")
                presets = {
                    "Soft Tissue": (40, 400),
                    "Lung": (-600, 1200),
                    "Bone": (300, 1500),
                    "Brain": (40, 80),
                    "Abdomen": (60, 400)
                }
                
                for preset_name, (center, width) in presets.items():
                    if st.button(preset_name):
                        st.session_state.window_center = center
                        st.session_state.window_width = width
                        st.rerun()
        
        with dicom_tab2:
            st.header("Image Controls")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Windowing")
                
                # Window center and width controls
                new_center = st.slider(
                    "Window Center", 
                    int(image_array.min()), 
                    int(image_array.max()),
                    value=st.session_state.get('window_center', default_center),
                    key='window_center_slider'
                )
                
                new_width = st.slider(
                    "Window Width", 
                    1, 
                    int(image_array.max() - image_array.min()),
                    value=st.session_state.get('window_width', default_width),
                    key='window_width_slider'
                )
                
                # Update session state
                st.session_state.window_center = new_center
                st.session_state.window_width = new_width
                
                if st.button("Reset to Defaults"):
                    st.session_state.window_center = default_center
                    st.session_state.window_width = default_width
                    st.rerun()
            
            with col2:
                st.subheader("Image Statistics")
                
                # Histogram
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(image_array.flatten(), bins=50, alpha=0.7, color='blue')
                ax.axvline(new_center, color='red', linestyle='--', label=f'Window Center: {new_center}')
                ax.axvline(new_center - new_width//2, color='orange', linestyle='--', alpha=0.7, label=f'Window Min')
                ax.axvline(new_center + new_width//2, color='orange', linestyle='--', alpha=0.7, label=f'Window Max')
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                ax.set_title('Pixel Value Histogram')
                ax.legend()
                st.pyplot(fig)
                plt.close()
        
        with dicom_tab3:
            st.header("DICOM Metadata")
            
            # Display DICOM metadata
            metadata = get_dicom_metadata(dicom_data)
            
            if metadata:
                col1, col2 = st.columns(2)
                
                # Split metadata into two columns
                items = list(metadata.items())
                mid_point = len(items) // 2
                
                with col1:
                    for key, value in items[:mid_point]:
                        st.write(f"**{key}:** {value}")
                
                with col2:
                    for key, value in items[mid_point:]:
                        st.write(f"**{key}:** {value}")
            else:
                st.write("No metadata available")
            
            # Raw DICOM data explorer (expandable)
            with st.expander("Raw DICOM Tags (Advanced)"):
                st.write("**Available DICOM tags:**")
                
                # Display first 50 DICOM elements
                dicom_dict = {}
                count = 0
                for elem in dicom_data:
                    if count >= 50:  # Limit display
                        break
                    if elem.tag != (0x7fe0, 0x0010):  # Skip pixel data
                        dicom_dict[str(elem.tag)] = f"{elem.keyword}: {str(elem.value)[:100]}"
                        count += 1
                
                st.json(dicom_dict)
    else:
        st.error(f"Could not load DICOM image from {selected_dicom}")