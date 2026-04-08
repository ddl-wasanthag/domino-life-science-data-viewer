# 🔬 Life Sciences Data Viewer

A Domino Extension that brings life sciences file formats — tabular clinical data, medical imaging, and genomics sequences — directly into the Domino project sidebar. Each user sees only the datasets they are authorised to access, using Domino's Extended Identity Propagation.

---

## Supported File Formats

### 📊 Tabular Data

#### Parquet (`.parquet`)
Apache Parquet is a columnar storage format widely used for large analytical datasets. In life sciences it appears as the output format for processed clinical trial data, model training datasets, and pipeline results.

**What the viewer shows:**
- Paginated data table (configurable rows per page)
- Column filters — range, equals, contains
- SQL-like query syntax (`AGE > 50 AND SEX == 'M'`)
- Quick analysis tab — statistics (mean, median, std dev, min, max) and frequency distribution per column
- Missing data visualisation — horizontal bar chart of null counts by column
- Column visibility and sort controls
- Download as CSV or Excel

#### SAS Transport (`.xpt`)
XPT is the SAS transport format mandated by the FDA for clinical trial data submissions (CDISC/ADaM). It is the standard exchange format between pharmaceutical sponsors, CROs, and regulatory agencies.

**What the viewer shows:**
- Same full-featured viewer as Parquet (paginated table, filters, analysis, download)
- Reads variable labels and formats stored in the XPT metadata

---

### 🩻 Medical Imaging

#### DICOM (`.dcm`, `.dicom`, `.dic`, `.ima`)
DICOM (Digital Imaging and Communications in Medicine) is the universal standard for medical imaging. Every CT scan, MRI, X-ray, ultrasound, and PET scan produced in a clinical or research setting is stored as DICOM. Files contain both pixel data and rich metadata — patient information, scanner parameters, acquisition protocol, and embedded windowing recommendations.

**What the viewer shows:**
- Full image render using clinical windowing
- **Window presets** — one-click contrast optimisation for different tissue types:

  | Preset | Window Center | Window Width | Purpose |
  |---|---|---|---|
  | Default | From DICOM tags | From DICOM tags | Scanner-recommended view |
  | Soft Tissue | 40 HU | 400 HU | Muscles, organs, fat |
  | Lung | −600 HU | 1200 HU | Airways, lung nodules |
  | Bone | 300 HU | 1500 HU | Cortical and cancellous bone |
  | Brain | 40 HU | 80 HU | Grey/white matter differentiation |
  | Abdomen | 60 HU | 400 HU | Abdominal organs |

- **Fine windowing sliders** — manual control over Window Center and Window Width
- Image metadata panel — modality, study description, shape, pixel range
- Browser-native zoom (click to expand the image)

> **Note on windowing:** CT scanners measure tissue density in Hounsfield Units (HU). Air = −1000 HU, water = 0 HU, soft tissue = 20–80 HU, bone = 400–1000 HU. Windowing "stretches" a chosen HU range across the available greyscale, making structures in that range visible. The clinical presets above are optimised for CT. MRI images use arbitrary signal intensities rather than HU, so the scanner-embedded Default preset is generally best for MRI.

#### NIfTI (`.nii`, `.nii.gz`)
NIfTI (Neuroimaging Informatics Technology Initiative) is the standard format for research neuroimaging — fMRI, structural MRI, and diffusion tensor imaging. While DICOM is used in clinical/radiology workflows, NIfTI is preferred in research pipelines (FSL, SPM, FreeSurfer, ANTs).

**What the viewer shows:**
- Three orthogonal slice views — Axial, Coronal, Sagittal — with independent sliders
- 4D volume support — time-point selector for fMRI data
- Voxel size and image dimensions
- Pixel value range

---

### 🧬 Genomics Sequences

#### FASTA (`.fasta`, `.fa`, `.fna`, `.ffn`)
FASTA is the foundational plain-text format for biological sequences. Each record contains a header line (starting with `>`) followed by the nucleotide or amino acid sequence. Reference genomes, gene sequences, and protein databases are distributed in FASTA format.

**Example record:**
```
>NC_045512.2 Severe acute respiratory syndrome coronavirus 2, complete genome
ATTAAAGGTTTATACCTTCCCAGGTAACAAACCAACCAACTTTCGATCTCTTGTAGATCT...
```

**What the viewer shows:**
- **KPI metrics** — sequence count, median length, mean GC%, total bases
- **Length distribution** histogram
- **GC content** distribution with mean line
- Records table — sequence ID, length, GC%, sequence preview (first 60 bases)
- CSV download of the summary table

#### FASTQ (`.fastq`, `.fastq.gz`, `.fq`, `.fq.gz`)
FASTQ extends FASTA by adding per-base quality scores from the sequencer. It is the primary output format of next-generation sequencing (NGS) instruments — Illumina, Oxford Nanopore, PacBio. Every sequencing run produces FASTQ files as its raw data. Gzip-compressed `.fastq.gz` files are handled natively.

**Example record (4 lines per read):**
```
@SRR000001.1 read identifier
ACGTACGTACGTACGT...    ← nucleotide sequence
+
IIIIIIIIIIIIIIII...    ← Phred quality scores (ASCII-encoded)
```

**What the viewer shows:**
- **KPI metrics** — read count, median length, mean GC%, mean Phred quality score
- **Length distribution** histogram
- **GC content** distribution
- **Quality score distribution** with Q20 and Q30 reference lines
  - Q20 = 99% base call accuracy (1 error per 100 bases)
  - Q30 = 99.9% base call accuracy (1 error per 1,000 bases)
- Records table with per-read quality scores
- CSV download of the summary table

---

## How It Works as a Domino Extension

The app is deployed as a Domino App and registered as an Extension, which causes it to appear in the sidebar of every project. When a user clicks it from their project:

1. Domino injects the `projectId` as a URL query parameter (`?projectId=<uuid>`)
2. The app reads the viewer's Bearer token via Extended Identity Propagation
3. It calls `GET /v4/datasetrw/datasets-v2?projectIdsToInclude=<projectId>` using the **viewer's own token** — so each user sees only the datasets they are authorised to access
4. Files are listed via `GET /v4/datasetrw/files/{snapshotId}?path=` and downloaded via `GET /v4/datasetrw/snapshot/{snapshotId}/file/raw`
5. No filesystem mounts are required — everything is streamed via the Domino Dataset API

This means the Extension works correctly across projects — a user browsing from any project sidebar sees their own datasets, not the app owner's.

---

## Setup

### Requirements

All dependencies must be installed in the Domino compute environment:

```
streamlit>=1.32.0
requests>=2.31.0
pandas>=2.0.0
urllib3>=2.0.0
pyreadstat
openpyxl
pydicom
nibabel
```

FASTQ and FASTA parsing uses Python built-ins only (`gzip`, string processing) — no additional packages needed.

### App startup script (`app.sh`)

```bash
#!/bin/bash
streamlit run app.py --server.port=8888 --server.address=0.0.0.0
```

### Enabling as a Domino Extension

1. Ensure the following are enabled on the Domino deployment:
   - Feature flag: `SecureIdentityPropagationToAppsEnabled`
   - Central config: `com.cerebro.domino.apps.extendedIdentityPropagationToAppsEnabled=true`
   - Feature flag: `EnableDominoUIExtensions`
2. Publish the app with **Extended Identity** propagation enabled
3. Admin goes to the app's three-dot menu → **Create Extension**
4. The viewer appears in all project sidebars automatically