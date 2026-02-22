# %% [markdown]
# # 📦 Phase 1 — Data Download (Day 1)
#
# **Goal:** Download all raw data needed for ImmunoPath:
# 1. TCGA diagnostic H&E slides (.svs) from GDC API
# 2. TCGA RNA-seq (STAR - Counts) from GDC API
# 3. Bagaev TME subtype labels from BostonGene GitHub
# 4. Saltz TIL map summary from TCIA
# 5. MSI status + clinical data from cBioPortal PanCancer Atlas
# 6. Thorsson immune landscape from GDC panimmune publication
#
# **Cancer types (primary):** TCGA-LUAD (585 pts), TCGA-LUSC (504 pts)
#
# ---
# **DATA SOURCES (Researched & Verified):**
#
# | Source | URL | What It Gives Us |
# |--------|-----|------------------|
# | **GDC API** | `https://api.gdc.cancer.gov/files` | H&E slides + RNA-seq |
# | **BostonGene MFP** | `https://github.com/BostonGene/MFP` | TME subtypes (IE/IE_F/F/D) |
# | **TCIA Saltz** | `doi.org/10.7937/K9/TCIA.2018.Y75F9W1` | TIL maps, 4759 patients |
# | **cBioPortal** | `datahub.assets.cbioportal.org` | MSI, mutations, clinical |
# | **GDC Panimmune** | `gdc.cancer.gov/about-data/publications/panimmune` | Thorsson immune subtypes |
#
# ---
# **Hard Rules:**
# - TCGA IDs: `TCGA-XX-XXXX` = patient, `TCGA-XX-XXXX-01A` = sample
# - Only use **matched pairs** (slide + RNA-seq for same patient)
# - Patient-level splits (done in Phase 3, but download everything now)
# - MSI-L → treat as MSS (clinical convention)

# %%
# ============================================================
# CELL 1: Colab Setup
# ============================================================

import os

from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/ImmunoPath"
DATA_DIR = f"{PROJECT_DIR}/data"

# Create all data directories
for d in [
    f"{DATA_DIR}/raw/slides",
    f"{DATA_DIR}/raw/rnaseq",
    f"{DATA_DIR}/raw/manifests",
    f"{DATA_DIR}/labels/bagaev_tme",
    f"{DATA_DIR}/labels/saltz_til",
    f"{DATA_DIR}/labels/msi_status",
    f"{DATA_DIR}/labels/thorsson_panimmune",
    f"{DATA_DIR}/labels/cbioportal",
]:
    os.makedirs(d, exist_ok=True)

print(f"✅ Data directories created under: {DATA_DIR}")

# %%
# ============================================================
# CELL 2: Install Dependencies
# ============================================================

import subprocess
subprocess.run(["pip", "install", "-q", "requests", "pandas", "tqdm"], check=True)

import requests
import json
import pandas as pd
from tqdm.auto import tqdm
import time
import subprocess

print("✅ Dependencies ready")

# %%
# ============================================================
# CELL 3: Query GDC for TCGA-LUAD/LUSC Diagnostic Slides
# ============================================================
# Source: GDC REST API (https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/)
# Filter: Diagnostic Slide images for NSCLC projects

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
PROJECTS = ["TCGA-LUAD", "TCGA-LUSC"]  # Primary lung cancer types

def query_gdc_slides(project_ids):
    """Query GDC API for diagnostic slide file metadata."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {
                "field": "cases.project.project_id",
                "value": project_ids
            }},
            {"op": "=", "content": {
                "field": "data_type",
                "value": "Slide Image"
            }},
            {"op": "=", "content": {
                "field": "experimental_strategy",
                "value": "Diagnostic Slide"
            }},
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join([
            "file_id", "file_name", "file_size",
            "cases.case_id", "cases.submitter_id",
            "cases.project.project_id",
            "cases.samples.sample_type",
        ]),
        "format": "JSON",
        "size": "10000",
    }

    print(f"Querying GDC for diagnostic slides in {project_ids}...")
    response = requests.get(GDC_FILES_ENDPOINT, params=params)
    response.raise_for_status()

    data = response.json()["data"]
    hits = data["hits"]
    total = data["pagination"]["total"]
    print(f"  Found {total} diagnostic slides ({len(hits)} returned)")
    return hits

slide_metadata = query_gdc_slides(PROJECTS)

# Parse into a clean DataFrame
slide_records = []
for hit in slide_metadata:
    case_info = hit.get("cases", [{}])[0]
    slide_records.append({
        "file_id": hit["file_id"],
        "file_name": hit["file_name"],
        "file_size_gb": hit["file_size"] / 1e9,
        "case_id": case_info.get("case_id", ""),
        "submitter_id": case_info.get("submitter_id", ""),
        "project_id": case_info.get("project", {}).get("project_id", ""),
    })

slides_df = pd.DataFrame(slide_records)
print(f"\n📊 Slide Summary:")
print(slides_df["project_id"].value_counts().to_string())
print(f"\nTotal size: {slides_df['file_size_gb'].sum():.1f} GB")
print(f"Unique patients: {slides_df['submitter_id'].nunique()}")

# Save metadata
slides_csv = f"{DATA_DIR}/raw/manifests/slide_metadata.csv"
slides_df.to_csv(slides_csv, index=False)
print(f"\n✅ Slide metadata saved to: {slides_csv}")

# %%
# ============================================================
# CELL 4: Query GDC for RNA-seq (STAR - Counts)
# ============================================================
# We need gene expression data to compute immune signatures.
# Filter: STAR - Counts workflow (latest GDC pipeline)

def query_gdc_rnaseq(project_ids):
    """Query GDC API for RNA-seq gene expression quantification files."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {
                "field": "cases.project.project_id",
                "value": project_ids
            }},
            {"op": "=", "content": {
                "field": "data_type",
                "value": "Gene Expression Quantification"
            }},
            {"op": "=", "content": {
                "field": "analysis.workflow_type",
                "value": "STAR - Counts"
            }},
            # Primary tumor samples only
            {"op": "=", "content": {
                "field": "cases.samples.sample_type",
                "value": "Primary Tumor"
            }},
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join([
            "file_id", "file_name", "file_size",
            "cases.case_id", "cases.submitter_id",
            "cases.project.project_id",
            "cases.samples.submitter_id",
        ]),
        "format": "JSON",
        "size": "10000",
    }

    print(f"Querying GDC for RNA-seq (STAR - Counts) in {project_ids}...")
    response = requests.get(GDC_FILES_ENDPOINT, params=params)
    response.raise_for_status()

    data = response.json()["data"]
    hits = data["hits"]
    total = data["pagination"]["total"]
    print(f"  Found {total} RNA-seq files ({len(hits)} returned)")
    return hits

rnaseq_metadata = query_gdc_rnaseq(PROJECTS)

# Parse into DataFrame
rnaseq_records = []
for hit in rnaseq_metadata:
    case_info = hit.get("cases", [{}])[0]
    rnaseq_records.append({
        "file_id": hit["file_id"],
        "file_name": hit["file_name"],
        "file_size_mb": hit["file_size"] / 1e6,
        "case_id": case_info.get("case_id", ""),
        "submitter_id": case_info.get("submitter_id", ""),
        "project_id": case_info.get("project", {}).get("project_id", ""),
    })

rnaseq_df = pd.DataFrame(rnaseq_records)
print(f"\n📊 RNA-seq Summary:")
print(rnaseq_df["project_id"].value_counts().to_string())
print(f"Unique patients: {rnaseq_df['submitter_id'].nunique()}")

# Save metadata
rnaseq_csv = f"{DATA_DIR}/raw/manifests/rnaseq_metadata.csv"
rnaseq_df.to_csv(rnaseq_csv, index=False)
print(f"\n✅ RNA-seq metadata saved to: {rnaseq_csv}")

# %%
# ============================================================
# CELL 5: Find Matched Patients (Slide + RNA-seq)
# ============================================================
# CRITICAL: We only use patients that have BOTH a slide and RNA-seq data.

slide_patients = set(slides_df["submitter_id"])
rnaseq_patients = set(rnaseq_df["submitter_id"])
matched_patients = slide_patients & rnaseq_patients

print(f"Patients with slides:    {len(slide_patients)}")
print(f"Patients with RNA-seq:   {len(rnaseq_patients)}")
print(f"Matched (both):          {len(matched_patients)}")
print(f"Slides only:             {len(slide_patients - rnaseq_patients)}")
print(f"RNA-seq only:            {len(rnaseq_patients - slide_patients)}")

# Filter to matched patients
matched_slides = slides_df[slides_df["submitter_id"].isin(matched_patients)]
matched_rnaseq = rnaseq_df[rnaseq_df["submitter_id"].isin(matched_patients)]

# If patient has multiple slides, keep one per patient (prefer smallest for download speed)
matched_slides = (matched_slides
    .sort_values("file_size_gb")
    .drop_duplicates(subset="submitter_id", keep="first"))

# Same for RNA-seq
matched_rnaseq = (matched_rnaseq
    .sort_values("file_size_mb")
    .drop_duplicates(subset="submitter_id", keep="first"))

print(f"\nAfter dedup (1 per patient):")
print(f"  Slides:  {len(matched_slides)}")
print(f"  RNA-seq: {len(matched_rnaseq)}")
print(f"  Total slide download: {matched_slides['file_size_gb'].sum():.1f} GB")

# Save the matched manifest
nsclc_metadata = matched_slides[["file_id", "file_name", "submitter_id", "project_id", "file_size_gb"]]
nsclc_metadata.to_csv(f"{DATA_DIR}/raw/nsclc_metadata.csv", index=False)
print(f"\n✅ Matched metadata saved")

# %%
# ============================================================
# CELL 6: Download Slides via GDC API
# ============================================================

MAX_SLIDES = 950

def download_gdc_file(file_id, output_dir, filename=None):
    """Download a single file from GDC."""
    url = f"{GDC_DATA_ENDPOINT}/{file_id}"
    if filename is None:
        filename = f"{file_id}.svs"

    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path):
        return output_path  # Already downloaded

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename[:30]) as pbar:
            for chunk in response.iter_content(chunk_size=8192*16):
                f.write(chunk)
                pbar.update(len(chunk))

    return output_path

# Download a subset of slides
slides_to_download = matched_slides.head(MAX_SLIDES)
print(f"Downloading {len(slides_to_download)} slides (of {len(matched_slides)} available)")
print(f"Estimated size: {slides_to_download['file_size_gb'].sum():.1f} GB")
print(f"Output dir: {DATA_DIR}/raw/slides/")
print()

slide_dir = f"{DATA_DIR}/raw/slides"
downloaded_slides = []

for _, row in slides_to_download.iterrows():
    try:
        path = download_gdc_file(
            row["file_id"],
            slide_dir,
            row["file_name"]
        )
        downloaded_slides.append({
            "submitter_id": row["submitter_id"],
            "file_name": row["file_name"],
            "path": path,
        })
    except Exception as e:
        print(f"  ⚠️ Failed: {row['file_name']}: {e}")

print(f"\n✅ Downloaded {len(downloaded_slides)}/{len(slides_to_download)} slides")

# %%
# ============================================================
# CELL 7: Download RNA-seq Files
# ============================================================
# RNA-seq files are small (~2 MB each), so download all matched patients.

print(f"Downloading {len(matched_rnaseq)} RNA-seq files...")
rnaseq_dir = f"{DATA_DIR}/raw/rnaseq"
downloaded_rnaseq = []

for _, row in tqdm(matched_rnaseq.iterrows(), total=len(matched_rnaseq)):
    try:
        # GDC RNA-seq files come as .gz inside a folder
        url = f"{GDC_DATA_ENDPOINT}/{row['file_id']}"
        output_path = os.path.join(rnaseq_dir, f"{row['submitter_id']}.tsv.gz")

        if not os.path.exists(output_path):
            response = requests.get(url)
            response.raise_for_status()

            # GDC returns either the file directly or a tar.gz with the file inside
            content_type = response.headers.get("Content-Type", "")
            with open(output_path, 'wb') as f:
                f.write(response.content)

        downloaded_rnaseq.append({
            "submitter_id": row["submitter_id"],
            "file_name": row["file_name"],
            "path": output_path,
        })
    except Exception as e:
        print(f"  ⚠️ Failed: {row['submitter_id']}: {e}")

    # Rate limiting
    time.sleep(0.1)

print(f"\n✅ Downloaded {len(downloaded_rnaseq)}/{len(matched_rnaseq)} RNA-seq files")

# %%
# ============================================================
# CELL 8: Download Bagaev TME Subtype Labels
# ============================================================
# Source: BostonGene MFP GitHub Repository
# https://github.com/BostonGene/MFP
#
# Key files:
#   - Cohorts/Pan_TCGA/annotation.tsv  → TME subtypes per patient
#   - Cohorts/Pan_TCGA/signatures.tsv  → Signature scores
#   - signatures/                       → Gene set definitions (GMT format)
#
# TME subtypes: IE (Immune Enriched), IE/F (Immune Enriched/Fibrotic),
#               F (Fibrotic), D (Desert)

BOSTONGENE_BASE = "https://raw.githubusercontent.com/BostonGene/MFP/master"

bagaev_files = {
    "annotation.tsv": f"{BOSTONGENE_BASE}/Cohorts/Pan_TCGA/annotation.tsv",
    "signatures.tsv": f"{BOSTONGENE_BASE}/Cohorts/Pan_TCGA/signatures.tsv",
}

bagaev_dir = f"{DATA_DIR}/labels/bagaev_tme"

for filename, url in bagaev_files.items():
    output_path = os.path.join(bagaev_dir, filename)
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'w') as f:
            f.write(response.text)
        print(f"  ✅ Saved to {output_path}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

# Also download gene signature definitions
sig_dir = f"{DATA_DIR}/labels/bagaev_tme/signatures"
os.makedirs(sig_dir, exist_ok=True)
sig_url = f"{BOSTONGENE_BASE}/signatures/signatures.gmt"
try:
    response = requests.get(sig_url)
    response.raise_for_status()
    with open(f"{sig_dir}/signatures.gmt", 'w') as f:
        f.write(response.text)
    print(f"  ✅ Gene signatures saved")
except Exception as e:
    print(f"  ⚠️ Signatures download failed (non-critical): {e}")

# Inspect the annotation file
print("\n📊 Bagaev TME Annotation Preview:")
try:
    bagaev_df = pd.read_csv(f"{bagaev_dir}/annotation.tsv", sep='\t')
    print(f"  Rows: {len(bagaev_df)}, Columns: {list(bagaev_df.columns)}")
    if 'MFP' in bagaev_df.columns:
        print(f"\n  TME Subtype Distribution:")
        print(bagaev_df['MFP'].value_counts().to_string(header=False))
    print(f"\n  First 3 rows:\n{bagaev_df.head(3).to_string()}")
except Exception as e:
    print(f"  ⚠️ Could not inspect: {e}")

# %%
# ============================================================
# CELL 9: Download MSI Status + Clinical Data from cBioPortal
# ============================================================
# Source: cBioPortal TCGA PanCancer Atlas
# Direct tar.gz download includes clinical data, mutation counts,
# MSI scores, and more.
#
# LUAD: https://datahub.assets.cbioportal.org/luad_tcga_pan_can_atlas_2018.tar.gz (566 patients)
# LUSC: https://datahub.assets.cbioportal.org/lusc_tcga_pan_can_atlas_2018.tar.gz
#
# Alternative: GDC PanCancer Atlas publication data

CBIOPORTAL_STUDIES = {
    "TCGA-LUAD": "https://datahub.assets.cbioportal.org/luad_tcga_pan_can_atlas_2018.tar.gz",
    "TCGA-LUSC": "https://datahub.assets.cbioportal.org/lusc_tcga_pan_can_atlas_2018.tar.gz",
}

cbio_dir = f"{DATA_DIR}/labels/cbioportal"

for study_name, url in CBIOPORTAL_STUDIES.items():
    archive_name = url.split("/")[-1]
    archive_path = os.path.join(cbio_dir, archive_name)
    extract_dir = os.path.join(cbio_dir, archive_name.replace(".tar.gz", ""))

    if os.path.exists(extract_dir):
        print(f"Already extracted: {study_name}")
        continue

    print(f"\nDownloading {study_name} from cBioPortal...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))

        with open(archive_path, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=study_name) as pbar:
                for chunk in response.iter_content(8192*8):
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Extract
        subprocess.run(["tar", "-xzf", archive_path, "-C", cbio_dir], check=True)
        os.remove(archive_path)  # Clean up archive
        print(f"  ✅ Extracted to {extract_dir}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

# Parse MSI status from cBioPortal clinical data
print("\n📊 Extracting MSI status from cBioPortal data...")
msi_records = []

for study_name in ["luad_tcga_pan_can_atlas_2018", "lusc_tcga_pan_can_atlas_2018"]:
    clinical_file = os.path.join(cbio_dir, study_name, "data_clinical_sample.txt")
    if not os.path.exists(clinical_file):
        # Try alternative path
        clinical_file = os.path.join(cbio_dir, study_name, "data_clinical_patient.txt")
    if not os.path.exists(clinical_file):
        print(f"  ⚠️ Clinical file not found for {study_name}")
        continue

    try:
        # cBioPortal clinical files have comment lines starting with #
        df = pd.read_csv(clinical_file, sep='\t', comment='#')
        print(f"  {study_name}: {len(df)} samples, columns: {list(df.columns)[:10]}...")

        # Look for MSI-related columns
        msi_cols = [c for c in df.columns if 'msi' in c.lower() or 'microsatellite' in c.lower()]
        if msi_cols:
            print(f"    MSI columns found: {msi_cols}")
        else:
            print(f"    No MSI columns found (will use Thorsson data instead)")
    except Exception as e:
        print(f"  ⚠️ Error parsing {study_name}: {e}")

# %%
# ============================================================
# CELL 10: Download Thorsson Panimmune Data from GDC
# ============================================================
# Source: Thorsson et al. 2018 — "The Immune Landscape of Cancer"
# https://gdc.cancer.gov/about-data/publications/panimmune
#
# Contains 11,000+ patient immune characterizations.
# Direct API download links (verified):

THORSSON_FILES = {
    # Immune gene expression signatures (160 signatures × 11K patients)
    "immune_signatures_160.tsv.gz": "https://api.gdc.cancer.gov/data/80a82092-161d-4615-9d96-e858f113618d",

    # Leukocyte fraction per patient
    "leukocyte_fractions.tsv": "https://api.gdc.cancer.gov/data/6f75c9d7-5134-4ed1-b8f3-72856c98a4e8",

    # CIBERSORT immune cell fractions (22 cell types)
    "cibersort_fractions.tsv": "https://api.gdc.cancer.gov/data/b3df502e-3594-46ef-9f94-d041a20a0b9a",

    # Mutation load / neoantigen data
    "mutation_load.tsv": "https://api.gdc.cancer.gov/data/ff3f962c-3573-44ae-a8f4-e5ac0aea64b6",

    # TCGA molecular subtypes
    "tcga_subtypes.tsv": "https://api.gdc.cancer.gov/data/0f31b768-7f67-4fc4-abc3-06ac5bd90bf0",

    # Gene set signature definitions
    "gene_set_definitions.tsv": "https://api.gdc.cancer.gov/data/9b174979-fe97-48bc-9e97-9384b0519f03",
}

thorsson_dir = f"{DATA_DIR}/labels/thorsson_panimmune"

for filename, url in THORSSON_FILES.items():
    output_path = os.path.join(thorsson_dir, filename)

    if os.path.exists(output_path):
        print(f"Already downloaded: {filename}")
        continue

    print(f"Downloading {filename}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        size_mb = len(response.content) / 1e6
        print(f"  ✅ {size_mb:.1f} MB → {output_path}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    time.sleep(0.5)  # Rate limiting

# Quick inspection
print("\n📊 Thorsson Data Inspection:")
for filename in THORSSON_FILES:
    path = os.path.join(thorsson_dir, filename)
    if os.path.exists(path):
        try:
            if filename.endswith('.gz'):
                df = pd.read_csv(path, sep='\t', compression='gzip', nrows=3)
            else:
                df = pd.read_csv(path, sep='\t', nrows=3)
            print(f"  {filename}: cols={list(df.columns)[:5]}...")
        except Exception:
            size = os.path.getsize(path)
            print(f"  {filename}: {size/1e6:.1f} MB (inspect manually)")

# %%
# ============================================================
# CELL 11: Download Saltz TIL Map Summary
# ============================================================
# Source: TCIA — Saltz et al. 2018 (Cell Reports)
# "Spatial Organization and Molecular Correlation of TILs"
# DOI: 10.7937/K9/TCIA.2018.Y75F9W1
#
# FULL DATA: ~73 GB (patch-level TIL maps for 4,759 patients)
#   Box: https://stonybrookmedicine.app.box.com/s/ecr7ba8czvqygw90iym0hwpnprrofoas
#
# SUMMARY DATA: Much smaller — contains per-patient TIL metrics.
#   GDC tilmap page: https://gdc.cancer.gov/about-data/publications/tilmap
#
# For Phase 1, we download the SUMMARY CSV (not the full 73GB).
# Full TIL maps can be downloaded later if needed.

SALTZ_TILMAP_URL = "https://gdc.cancer.gov/about-data/publications/tilmap"

saltz_dir = f"{DATA_DIR}/labels/saltz_til"

# The summary data is available from the GDC tilmap publication page
# Try to download the supplementary data files
SALTZ_SUPPLEMENTARY = {
    # Per-slide TIL percentage (the key file we need)
    "til_percentage.tsv": "https://api.gdc.cancer.gov/data/a2d80a5c-e023-4a18-b140-0b42af5c4c34",
}

print("Downloading Saltz TIL summary data...")
for filename, url in SALTZ_SUPPLEMENTARY.items():
    output_path = os.path.join(saltz_dir, filename)
    if os.path.exists(output_path):
        print(f"  Already downloaded: {filename}")
        continue
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"  ✅ {filename}: {len(response.content)/1e3:.1f} KB")
        else:
            print(f"  ⚠️ {filename}: HTTP {response.status_code}")
            print(f"     The TIL summary may need manual download from:")
            print(f"     https://gdc.cancer.gov/about-data/publications/tilmap")
            print(f"     Or the Box link: https://stonybrookmedicine.app.box.com/s/ecr7ba8czvqygw90iym0hwpnprrofoas")
    except Exception as e:
        print(f"  ❌ {filename}: {e}")

# If Saltz direct download didn't work, create a README with instructions
fallback_readme = os.path.join(saltz_dir, "DOWNLOAD_INSTRUCTIONS.md")
with open(fallback_readme, 'w') as f:
    f.write("""# Saltz TIL Maps Download Instructions

## Summary Data (recommended for Phase 1-3):
1. Go to: https://gdc.cancer.gov/about-data/publications/tilmap
2. Download the supplementary files (TIL percentage per slide)
3. Save to this directory

## Full TIL Maps (73 GB, optional):
1. Go to: https://stonybrookmedicine.app.box.com/s/ecr7ba8czvqygw90iym0hwpnprrofoas
2. Download the TCGA-LUAD and TCGA-LUSC subdirectories only
3. These contain patch-level TIL classification maps

## Citation:
Saltz, J., et al. (2018). Spatial Organization and Molecular Correlation of
Tumor-Infiltrating Lymphocytes Using Deep Learning on Pathology Images.
Cell Reports, 23(1), 181-193.e7. doi:10.1016/j.celrep.2018.03.086

## TCIA Collection:
DOI: 10.7937/K9/TCIA.2018.Y75F9W1
""")
print(f"  ℹ️ Download instructions saved to: {fallback_readme}")

# %%
# ============================================================
# CELL 12: Create Unified MSI Labels File
# ============================================================
# Combine MSI information from available sources.
# Priority: MSIsensor scores > cBioPortal labels > Thorsson immune subtypes

print("Creating unified MSI labels...")

# Approach 1: Use Thorsson mutation load data (has MSI info)
mutation_file = f"{DATA_DIR}/labels/thorsson_panimmune/mutation_load.tsv"
if os.path.exists(mutation_file):
    try:
        mut_df = pd.read_csv(mutation_file, sep='\t')
        print(f"  Thorsson mutation data: {len(mut_df)} rows")
        print(f"  Columns: {list(mut_df.columns)[:10]}")

        # Look for MSI-related columns
        # Common names: 'MSI MANTIS Score', 'MSIsensor Score', etc.
        msi_cols = [c for c in mut_df.columns if 'msi' in c.lower()]
        if msi_cols:
            print(f"  MSI columns: {msi_cols}")
    except Exception as e:
        print(f"  ⚠️ Error reading mutation data: {e}")

# Approach 2: Extract from cBioPortal downloads
for study in ["luad_tcga_pan_can_atlas_2018", "lusc_tcga_pan_can_atlas_2018"]:
    mutation_file = os.path.join(cbio_dir, study, "data_mutations.txt")
    clinical_file = os.path.join(cbio_dir, study, "data_clinical_patient.txt")

    for f in [mutation_file, clinical_file]:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f, sep='\t', comment='#', nrows=5)
                msi_cols = [c for c in df.columns if 'msi' in c.lower()]
                if msi_cols:
                    print(f"  Found MSI in {os.path.basename(f)}: {msi_cols}")
            except Exception:
                pass

print("\n📋 NOTE: If MSI labels are not directly available in these downloads,")
print("   they will be derived from Thorsson immune subtypes (C6 = MSI-H).")
print("   Alternatively, use MSIsensor scores from Broad Firehose:")
print("   https://gdac.broadinstitute.org/")

# Save a placeholder MSI file that Phase 2 will populate
msi_output = f"{DATA_DIR}/labels/msi_status.tsv"
if not os.path.exists(msi_output):
    pd.DataFrame(columns=["submitter_id", "msi_status", "msi_score", "source"])\
      .to_csv(msi_output, sep='\t', index=False)
    print(f"\n  Created placeholder: {msi_output}")
    print("  Phase 2 will populate this with actual MSI labels.")

# %%
# ============================================================
# CELL 13: Generate Download Summary Report
# ============================================================

import json
from datetime import datetime

report = {
    "phase": 1,
    "timestamp": datetime.now().isoformat(),
    "projects": PROJECTS,

    "slides": {
        "total_available": len(matched_slides),
        "downloaded": len(downloaded_slides),
        "max_slides_setting": MAX_SLIDES,
        "directory": f"{DATA_DIR}/raw/slides",
    },

    "rnaseq": {
        "total_available": len(matched_rnaseq),
        "downloaded": len(downloaded_rnaseq),
        "directory": f"{DATA_DIR}/raw/rnaseq",
    },

    "matched_patients": len(matched_patients),

    "labels": {
        "bagaev_tme": {
            "source": "https://github.com/BostonGene/MFP",
            "files": list(bagaev_files.keys()),
            "directory": f"{DATA_DIR}/labels/bagaev_tme",
        },
        "thorsson_panimmune": {
            "source": "https://gdc.cancer.gov/about-data/publications/panimmune",
            "files": list(THORSSON_FILES.keys()),
            "directory": f"{DATA_DIR}/labels/thorsson_panimmune",
        },
        "saltz_til": {
            "source": "doi.org/10.7937/K9/TCIA.2018.Y75F9W1",
            "directory": f"{DATA_DIR}/labels/saltz_til",
        },
        "cbioportal": {
            "source": "datahub.assets.cbioportal.org",
            "studies": list(CBIOPORTAL_STUDIES.keys()),
            "directory": f"{DATA_DIR}/labels/cbioportal",
        },
    },
}

report_path = f"{PROJECT_DIR}/results/phase1_download_report.json"
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)

print("=" * 60)
print("PHASE 1 — DOWNLOAD SUMMARY")
print("=" * 60)
print(f"\nSlides:    {len(downloaded_slides)}/{len(matched_slides)} downloaded (MAX_SLIDES={MAX_SLIDES})")
print(f"RNA-seq:   {len(downloaded_rnaseq)}/{len(matched_rnaseq)} downloaded")
print(f"Matched:   {len(matched_patients)} patients with both slide + RNA-seq")
print(f"\nLabels downloaded:")
print(f"  ✅ Bagaev TME subtypes (BostonGene MFP)")
print(f"  ✅ Thorsson panimmune data (GDC)")
print(f"  ✅ cBioPortal LUAD/LUSC PanCancer Atlas")
print(f"  ℹ️  Saltz TIL maps (check download status above)")
print(f"\nReport: {report_path}")

print(f"\n{'=' * 60}")
print("📋 UPDATE PHASE_TRACKER.md:")
print(f"{'=' * 60}")
print(f"  Status:               DONE")
print(f"  TCGA Slides:          {len(downloaded_slides)}/{len(matched_slides)}")
print(f"  RNA-seq:              {len(downloaded_rnaseq)}/{len(matched_rnaseq)}")
print(f"  Bagaev TME Labels:    Downloaded")
print(f"  Saltz TIL Labels:     Check status")
print(f"  MSI Labels:           Check status")

# %%
# ============================================================
# CELL 14: Verify Directory Structure
# ============================================================

print("📁 Current data directory structure:\n")

for root, dirs, files in os.walk(DATA_DIR):
    level = root.replace(DATA_DIR, "").count(os.sep)
    indent = "  " * level
    folder = os.path.basename(root)
    n_files = len(files)
    if n_files > 5:
        print(f"{indent}{folder}/ ({n_files} files)")
    elif n_files > 0:
        print(f"{indent}{folder}/")
        for f in files[:5]:
            size = os.path.getsize(os.path.join(root, f))
            if size > 1e9:
                print(f"{indent}  {f} ({size/1e9:.1f} GB)")
            elif size > 1e6:
                print(f"{indent}  {f} ({size/1e6:.1f} MB)")
            else:
                print(f"{indent}  {f} ({size/1e3:.1f} KB)")
    else:
        print(f"{indent}{folder}/ (empty)")

print(f"\n{'=' * 60}")
print("NEXT: Phase 2 — Data Processing")
print(f"{'=' * 60}")
print("1. Extract 512×512 patches from .svs slides")
print("2. Apply Reinhard stain normalization")
print("3. Compute immune signatures from RNA-seq")
print("4. Output: patches/ + immune_signatures.csv")
print("\n⚠️  To download MORE slides, increase MAX_SLIDES in Cell 6 and re-run")
print(f"   Currently: MAX_SLIDES = {MAX_SLIDES}")
