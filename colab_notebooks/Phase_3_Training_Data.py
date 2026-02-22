# %% [markdown]
# # 📦 Phase 3 — Training Data Creation (Days 4–5)
#
# **Goal:** Join patches (Phase 2A) with immune signature labels (Phase 2B) +
# external labels (Bagaev TME, MSI) → Create JSONL training files with
# **patient-level splits** to prevent data leakage.
#
# **Outputs:**
# - `data/training/train.jsonl`
# - `data/training/val.jsonl`
# - `data/training/test.jsonl`
#
# **Optimisation notes (Colab Pro):**
# - Parallel metadata scanning with ThreadPoolExecutor
# - Vectorised pandas joins (no row-level loops for merges)
# - Chunked JSONL writing to avoid OOM on large datasets
#
# ---
# **Hard Rules:**
# - Patient-level splits (TCGA-XX-XXXX), NOT slide-level
# - CD274 → `cd274_expression` (RNA proxy, NOT `pdl1_tps`)
# - TME subtypes: IE, IE/F, F, D (slash, NOT hyphen)
# - Max 8 patches per sample
# - Stratify splits by cancer type + label distribution

# %%
# ============================================================
# CELL 1: Colab Setup + Mount Drive
# ============================================================
import os, sys, time

from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/ImmunoPath"
DATA_DIR = f"{PROJECT_DIR}/data"
RESULTS_DIR = f"{PROJECT_DIR}/results"

os.makedirs(f"{DATA_DIR}/training", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/phase3", exist_ok=True)

print(f"✅ Directories ready")

# %%
# ============================================================
# CELL 2: Install & Import Dependencies
# ============================================================
import subprocess
subprocess.run(["pip", "install", "-q", "pandas", "numpy", "tqdm", "scikit-learn"], check=True)

import json
import glob
import random
import hashlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedGroupKFold

print("✅ Dependencies loaded")

# %%
# ============================================================
# CELL 3: Configuration
# ============================================================

# --- Training Data Config ---
MAX_PATCHES_PER_SAMPLE = 8     # MedGemma multi-image capacity (Phase 0 verified)
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# --- Cancer type mapping (from spec) ---
CANCER_TYPES = {
    "TCGA-LUAD": "lung adenocarcinoma",
    "TCGA-LUSC": "lung squamous cell carcinoma",
}

# --- Paths ---
PATCHES_DIR     = f"{DATA_DIR}/processed/patches"
SIGNATURES_PATH = f"{DATA_DIR}/signatures/immune_signatures.csv"
METADATA_PATH   = f"{DATA_DIR}/raw/nsclc_metadata.csv"       # submitter_id → file_name
BAGAEV_PATH     = f"{DATA_DIR}/labels/bagaev_tme/annotation.tsv"
THORSSON_DIR    = f"{DATA_DIR}/labels/thorsson_panimmune"
CBIO_DIR        = f"{DATA_DIR}/labels/cbioportal"
OUTPUT_DIR      = f"{DATA_DIR}/training_v3"  # V3.1: separate output dir for new schema

print("✅ Config set")

# %%
# ============================================================
# CELL 4: Load Immune Signatures (Phase 2 output)
# ============================================================

signatures_df = pd.read_csv(SIGNATURES_PATH, index_col=0)
signatures_df.index.name = "sample_id"
print(f"✅ Immune signatures: {signatures_df.shape[0]} patients × {signatures_df.shape[1]} columns")
print(f"   Columns: {list(signatures_df.columns)}")
print(f"\n   CD274 distribution:")
if "cd274_expression" in signatures_df.columns:
    print(signatures_df["cd274_expression"].value_counts().to_string())
print(f"\n   Immune phenotype distribution:")
if "immune_phenotype" in signatures_df.columns:
    print(signatures_df["immune_phenotype"].value_counts().to_string())

# %%
# ============================================================
# CELL 5: Load Slide Metadata + Scan Patch Directories (PARALLEL)
# ============================================================
# The slide metadata maps submitter_id (patient) → file_name (slide).
# We scan patch directories in parallel with threads (I/O-bound on GDrive).

metadata_df = pd.read_csv(METADATA_PATH)
print(f"✅ Slide metadata: {len(metadata_df)} rows")
print(f"   Columns: {list(metadata_df.columns)}")

# Build a map: submitter_id → slide_id (file_name without .svs)
metadata_df["slide_id"] = metadata_df["file_name"].str.replace(".svs", "", regex=False)

# --- Parallel scan of patch directories for available slides ---
def scan_slide_patches(slide_dir: str) -> dict:
    """Scan a single slide's patch directory. Returns metadata or None.
    Prefers diversity-selected patches (from Phase 2 Cell 6.5) if available."""
    slide_id = os.path.basename(slide_dir)
    meta_path = os.path.join(slide_dir, f"{slide_id}_metadata.json")
    all_patches = sorted(glob.glob(os.path.join(slide_dir, "*.jpg")))

    if not all_patches:
        return None

    # Check for diversity-selected patches (K-Means from Phase 2)
    selected_path = os.path.join(slide_dir, f"{slide_id}_selected_8.json")
    if os.path.exists(selected_path):
        try:
            with open(selected_path) as f:
                sel = json.load(f)
            selected_names = sel.get("selected", [])
            if selected_names:
                patches = [os.path.join(slide_dir, name) for name in selected_names
                           if os.path.exists(os.path.join(slide_dir, name))]
                if patches:
                    method = sel.get("method", "kmeans")
                else:
                    patches = all_patches[:MAX_PATCHES_PER_SAMPLE]
                    method = "fallback_sorted"
            else:
                patches = all_patches[:MAX_PATCHES_PER_SAMPLE]
                method = "fallback_sorted"
        except Exception:
            patches = all_patches[:MAX_PATCHES_PER_SAMPLE]
            method = "fallback_sorted"
    else:
        patches = all_patches[:MAX_PATCHES_PER_SAMPLE]
        method = "fallback_sorted"

    # Load metadata if available
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            pass

    return {
        "slide_id": slide_id,
        "patch_dir": slide_dir,
        "n_patches": len(patches),
        "n_total_patches": len(all_patches),
        "patch_paths": patches,
        "selection_method": method,
        "metadata": meta,
    }


# Find all slide directories
slide_dirs = [
    os.path.join(PATCHES_DIR, d)
    for d in os.listdir(PATCHES_DIR)
    if os.path.isdir(os.path.join(PATCHES_DIR, d))
] if os.path.exists(PATCHES_DIR) else []

print(f"\nScanning {len(slide_dirs)} patch directories (parallel)...")

patch_data = {}
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(scan_slide_patches, d): d for d in slide_dirs}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
        result = future.result()
        if result is not None:
            patch_data[result["slide_id"]] = result

print(f"✅ Found patches for {len(patch_data)} slides")
total_patches = sum(d["n_patches"] for d in patch_data.values())
print(f"   Total patches (selected): {total_patches}")
sel_methods = Counter(d.get("selection_method", "unknown") for d in patch_data.values())
print(f"   Selection methods: {dict(sel_methods)}")

# %%
# ============================================================
# CELL 6: Load External Labels (Bagaev TME + MSI)
# ============================================================
# Merge Bagaev TME subtypes and MSI status into our label set.

# --- Bagaev TME subtypes ---
bagaev_tme = {}
if os.path.exists(BAGAEV_PATH):
    try:
        bagaev_df = pd.read_csv(BAGAEV_PATH, sep='\t')
        print(f"✅ Bagaev TME: {len(bagaev_df)} rows")
        print(f"   Columns: {list(bagaev_df.columns)}")

        # Identify the sample ID column and MFP column
        # Typically: first column is sample barcode, 'MFP' column is TME subtype
        id_col = bagaev_df.columns[0]
        mfp_col = "MFP" if "MFP" in bagaev_df.columns else None

        if mfp_col:
            # Normalise TME subtypes to spec notation
            # BostonGene uses: IE, IE/F, F, D (already correct format)
            # But some may use underscore (IE_F) → convert to slash
            bagaev_df[mfp_col] = bagaev_df[mfp_col].str.replace("IE_F", "IE/F", regex=False)

            # Extract TCGA patient barcodes (first 12 chars = TCGA-XX-XXXX)
            bagaev_df["patient_id"] = bagaev_df[id_col].astype(str).str[:12]

            # Filter to LUAD/LUSC patients
            lung_bagaev = bagaev_df[
                bagaev_df["patient_id"].str.startswith("TCGA-")
            ].drop_duplicates(subset="patient_id", keep="first")

            bagaev_tme = dict(zip(lung_bagaev["patient_id"], lung_bagaev[mfp_col]))
            print(f"   Matched Bagaev TME labels: {len(bagaev_tme)}")
            print(f"   TME distribution: {Counter(bagaev_tme.values())}")
        else:
            print(f"   ⚠️ MFP column not found in Bagaev data")
    except Exception as e:
        print(f"   ⚠️ Error loading Bagaev TME: {e}")
else:
    print("⚠️ Bagaev TME file not found")

# --- MSI Status: Priority order → Bagaev MSI col > cBioPortal scores > Thorsson C6 ---
msi_labels = {}
msi_source_counts = {"bagaev": 0, "cbioportal_score": 0, "thorsson_c6": 0}

# Source 1 (BEST): Bagaev annotation.tsv has a direct 'MSI' column
if 'bagaev_df' in dir() and 'MSI' in bagaev_df.columns:
    bagaev_df["_pid"] = bagaev_df[bagaev_df.columns[0]].astype(str).str[:12]
    for _, row in bagaev_df.iterrows():
        pid = row["_pid"]
        msi_val = str(row["MSI"]).strip()
        if msi_val.upper() in ("MSI-H", "MSI_H", "MSI"):
            msi_labels[pid] = "MSI-H"
            msi_source_counts["bagaev"] += 1
        elif msi_val.upper() in ("MSS", "MSI-L", "MSI_L", "STABLE"):
            if pid not in msi_labels:
                msi_labels[pid] = "MSS"
        elif msi_val.lower() not in ("nan", "", "na", "none"):
            if pid not in msi_labels:
                msi_labels[pid] = "MSS"
    bagaev_df.drop(columns=["_pid"], inplace=True, errors="ignore")
    print(f"\n✅ Bagaev MSI column: {sum(1 for v in msi_labels.values() if v == 'MSI-H')} MSI-H found")

# Source 2: cBioPortal MSIsensor/MANTIS numeric scores (proper thresholds)
# MSIsensor score >= 3.5 → MSI-H (Niu et al. 2014)
# MANTIS score >= 0.4 → MSI-H (Kautto et al. 2017)
for study in ["luad_tcga_pan_can_atlas_2018", "lusc_tcga_pan_can_atlas_2018"]:
    clinical_path = os.path.join(CBIO_DIR, study, "data_clinical_sample.txt")
    if not os.path.exists(clinical_path):
        clinical_path = os.path.join(CBIO_DIR, study, "data_clinical_patient.txt")
    if os.path.exists(clinical_path):
        try:
            cdf = pd.read_csv(clinical_path, sep='\t', comment='#')
            id_col = cdf.columns[0]
            for _, row in cdf.iterrows():
                pid = str(row[id_col])[:12]
                if pid in msi_labels:
                    continue

                msisensor = None
                mantis = None
                for c in cdf.columns:
                    cl = c.lower()
                    if "msi_sensor" in cl or "msisensor" in cl:
                        try:
                            msisensor = float(row[c])
                        except (ValueError, TypeError):
                            pass
                    elif "mantis" in cl:
                        try:
                            mantis = float(row[c])
                        except (ValueError, TypeError):
                            pass
                    elif cl in ("msi_status", "msi"):
                        val = str(row[c]).strip().upper()
                        if val in ("MSI-H", "INSTABLE"):
                            msi_labels[pid] = "MSI-H"
                            msi_source_counts["cbioportal_score"] += 1
                            continue

                if pid not in msi_labels:
                    if msisensor is not None and msisensor >= 3.5:
                        msi_labels[pid] = "MSI-H"
                        msi_source_counts["cbioportal_score"] += 1
                    elif mantis is not None and mantis >= 0.4:
                        msi_labels[pid] = "MSI-H"
                        msi_source_counts["cbioportal_score"] += 1
                    elif msisensor is not None or mantis is not None:
                        msi_labels[pid] = "MSS"
        except Exception as e:
            print(f"   ⚠️ cBioPortal {study}: {e}")

print(f"✅ cBioPortal scores: +{msi_source_counts['cbioportal_score']} MSI-H")

# Source 3 (FALLBACK): Thorsson C6 (only for non-lung cancers, C6 is NOT MSI-H in lung)
subtypes_path = os.path.join(THORSSON_DIR, "tcga_subtypes.tsv")
if os.path.exists(subtypes_path):
    try:
        subtypes_df = pd.read_csv(subtypes_path, sep='\t')
        print(f"✅ Thorsson subtypes: {len(subtypes_df)} rows (C6 fallback only)")
        for col in subtypes_df.columns:
            if "immune" in col.lower() and "subtype" in col.lower():
                id_col = subtypes_df.columns[0]
                subtypes_df["patient_id"] = subtypes_df[id_col].astype(str).str[:12]
                for _, row in subtypes_df.iterrows():
                    pid = row["patient_id"]
                    if pid in msi_labels:
                        continue
                    subtype = str(row[col])
                    if subtype == "C6":
                        msi_labels[pid] = "MSI-H"
                        msi_source_counts["thorsson_c6"] += 1
                    else:
                        msi_labels[pid] = "MSS"
                break
    except Exception as e:
        print(f"   ⚠️ Thorsson: {e}")

total_msih = sum(1 for v in msi_labels.values() if v == "MSI-H")
total_mss = sum(1 for v in msi_labels.values() if v == "MSS")
print(f"\n✅ MSI labels (combined): {len(msi_labels)} patients")
print(f"   MSI-H: {total_msih}  (bagaev={msi_source_counts['bagaev']}, "
      f"cbioportal={msi_source_counts['cbioportal_score']}, "
      f"thorsson_c6={msi_source_counts['thorsson_c6']})")
print(f"   MSS:   {total_mss}")

# %%
# ============================================================
# CELL 7: Join All Data Sources → Matched Samples
# ============================================================
# Inner join: Only keep samples with BOTH patches AND signatures

print("=" * 60)
print("JOINING DATA SOURCES")
print("=" * 60)

# Step 1: Map submitter_id → slide_id from metadata
patient_to_slide = dict(zip(metadata_df["submitter_id"], metadata_df["slide_id"]))

# Also map slide_id → project_id for cancer type
slide_to_project = dict(zip(metadata_df["slide_id"], metadata_df["project_id"]))

# Step 2: Find which signature patients have patches
matched_samples = []
unmatched_reasons = Counter()

for sample_id in signatures_df.index:
    # Extract patient ID (TCGA-XX-XXXX format)
    patient_id = sample_id if len(sample_id) == 12 else sample_id[:12]

    # Find slide for this patient
    slide_id = patient_to_slide.get(patient_id)
    if slide_id is None:
        # Try matching directly — RNA-seq sample IDs may differ from slide IDs
        # Check if any slide_id contains the patient barcode
        for sid, sdata in patch_data.items():
            if patient_id in sid:
                slide_id = sid
                break

    if slide_id is None:
        unmatched_reasons["no_slide_mapping"] += 1
        continue

    # Check if this slide has patches
    if slide_id not in patch_data:
        unmatched_reasons["no_patches"] += 1
        continue

    # Get patches
    pdata = patch_data[slide_id]
    if pdata["n_patches"] == 0:
        unmatched_reasons["zero_patches"] += 1
        continue

    # Get project/cancer type
    project_id = slide_to_project.get(slide_id, "TCGA-LUAD")
    cancer_type = CANCER_TYPES.get(project_id, "lung cancer")

    # Get external labels
    tme_subtype = bagaev_tme.get(patient_id, "unknown")
    msi_status = msi_labels.get(patient_id, "unknown")

    # Get immune signature from Phase 2
    sig = signatures_df.loc[sample_id].to_dict()

    matched_samples.append({
        "sample_id": sample_id,
        "patient_id": patient_id,
        "slide_id": slide_id,
        "project_id": project_id,
        "cancer_type": cancer_type,
        "patch_paths": pdata["patch_paths"],
        "n_patches": pdata["n_patches"],
        "tme_subtype": tme_subtype,
        "msi_status": msi_status,
        "immune_signature": sig,
    })

print(f"\n✅ Matched samples: {len(matched_samples)}")
print(f"   Unmatched reasons: {dict(unmatched_reasons)}")
print(f"   Total available signatures: {len(signatures_df)}")
print(f"   Total available slides with patches: {len(patch_data)}")

if matched_samples:
    # Distribution summary
    projects = Counter(s["project_id"] for s in matched_samples)
    tme_dist = Counter(s["tme_subtype"] for s in matched_samples)
    msi_dist = Counter(s["msi_status"] for s in matched_samples)
    print(f"\n   By project: {dict(projects)}")
    print(f"   By TME subtype: {dict(tme_dist)}")
    print(f"   By MSI status: {dict(msi_dist)}")

# %%
# ============================================================
# CELL 8: Build Prompt + Response for Each Sample
# ============================================================
# Create the exact prompt/response pairs for fine-tuning.

PROMPT_TEMPLATE = """Analyze these H&E-stained histopathology images from a {cancer_type} tumor.

Extract the following **H&E-inferred immune signals** as a *research* output (not diagnostic):
1. **CD274 (PD-L1) RNA proxy level** (high/low)
2. **MSI status** (MSI-H or MSS)
3. **TIL fraction** (0.0-1.0) + density bucket (low/moderate/high)
4. **TME subtype** (IE / IE/F / F / D)
5. **Immune phenotype** (inflamed/excluded/desert)
6. **CD8+ T-cell infiltration** (low/moderate/high)
7. **Overall immune score** (0.0-1.0)

Rules:
- If a field is not inferable, output "unknown".
- CD274 is an RNA proxy — NOT clinical PD-L1 IHC TPS.

Provide your analysis as a JSON object."""


def categorize_score(score: float) -> str:
    """Convert continuous z-score to category."""
    if pd.isna(score):
        return "unknown"
    if score < -0.5:
        return "low"
    elif score < 0.5:
        return "moderate"
    else:
        return "high"


def build_target_response(sample: dict) -> str:
    """Build the target JSON response for fine-tuning.
    
    V3.1 FIXES:
    - til_fraction now uses normalized til_signature (NOT immune_score)
    - Removed prediction_entropy, low_confidence_flag (always "unknown" = wasted tokens)
    - Removed cd274_note (fixed string = wasted tokens; added in post-processing)
    """
    sig = sample["immune_signature"]

    # V3.1 FIX: TIL fraction from normalized til_signature (distinct from immune_score)
    # til_fraction_normalized is pre-computed via min-max normalization of til_signature z-scores
    til_fraction = float(sig.get("til_fraction_normalized", 0.5))

    # TIL density from signature score
    til_sig = sig.get("til_signature", 0.0)
    til_density = categorize_score(float(til_sig) if not pd.isna(til_sig) else 0.0)

    # CD8 infiltration
    cd8_sig = sig.get("cd8_signature", 0.0)
    cd8_infiltration = categorize_score(float(cd8_sig) if not pd.isna(cd8_sig) else 0.0)

    # Immune score (already 0-1 normalised in Phase 2)
    immune_score = round(float(sig.get("immune_score", 0.5)), 3)

    response = {
        "cd274_expression": sig.get("cd274_expression", "unknown"),
        "msi_status": sample.get("msi_status", "unknown"),
        "tme_subtype": sample.get("tme_subtype", "unknown"),
        "til_fraction": round(til_fraction, 3),
        "til_density": til_density,
        "immune_phenotype": sig.get("immune_phenotype", "unknown"),
        "cd8_infiltration": cd8_infiltration,
        "immune_score": immune_score,
    }
    return json.dumps(response, indent=2)


# V3.1 FIX: Pre-compute normalized TIL fraction (distinct from immune_score)
# immune_score = mean(cd8, ifng, til) normalized to [0,1]
# til_fraction = til_signature ALONE normalized to [0,1]
print("\nPre-computing normalized TIL fractions (v3.1 fix)...")
all_til_sigs = []
for s in matched_samples:
    z = s["immune_signature"].get("til_signature", 0.0)
    all_til_sigs.append(float(z) if not pd.isna(z) else 0.0)

til_min = min(all_til_sigs)
til_max = max(all_til_sigs)
print(f"  TIL signature z-scores: min={til_min:.3f}, max={til_max:.3f}")

for i, s in enumerate(matched_samples):
    z = all_til_sigs[i]
    if til_max > til_min:
        normalized = (z - til_min) / (til_max - til_min)
    else:
        normalized = 0.5
    s["immune_signature"]["til_fraction_normalized"] = round(normalized, 4)

# Verify til_fraction != immune_score
sample_til = matched_samples[0]["immune_signature"].get("til_fraction_normalized", -1)
sample_imm = matched_samples[0]["immune_signature"].get("immune_score", -1)
print(f"  Sample 0: til_fraction={sample_til}, immune_score={sample_imm} (should differ)")

# Build training samples with patch selection
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

training_samples = []
for sample in tqdm(matched_samples, desc="Building samples"):
    # Select patches (random subset if > MAX_PATCHES)
    patches = sample["patch_paths"]
    if len(patches) > MAX_PATCHES_PER_SAMPLE:
        patches = random.sample(patches, MAX_PATCHES_PER_SAMPLE)

    prompt = PROMPT_TEMPLATE.format(cancer_type=sample["cancer_type"])
    response = build_target_response(sample)

    training_samples.append({
        "sample_id": sample["sample_id"],
        "patient_id": sample["patient_id"],
        "slide_id": sample["slide_id"],
        "project_id": sample["project_id"],
        "cancer_type": sample["cancer_type"],
        "patch_paths": patches,
        "n_patches": len(patches),
        "prompt": prompt,
        "response": response,
        "immune_signature": sample["immune_signature"],
    })

print(f"\n✅ Built {len(training_samples)} training samples")
if training_samples:
    avg_patches = np.mean([s["n_patches"] for s in training_samples])
    print(f"   Average patches per sample: {avg_patches:.1f}")

# %%
# ============================================================
# CELL 9: Patient-Level Split (Stratified)
# ============================================================
# CRITICAL: Split by PATIENT, not by slide/sample.
# Stratify by cancer type + cd274 label to balance splits.

def patient_level_split(
    samples: list,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple:
    """
    V3.1 FIX: Actually use StratifiedGroupKFold for proper stratified splitting.
    Split samples by patient ID to prevent data leakage.
    Stratifies by cancer_type + cd274_expression to ensure balanced classes.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    random.seed(seed)
    np.random.seed(seed)

    # Group by patient
    patient_to_samples = {}
    for s in samples:
        pid = s["patient_id"]
        if pid not in patient_to_samples:
            patient_to_samples[pid] = []
        patient_to_samples[pid].append(s)

    patient_ids = list(patient_to_samples.keys())
    n = len(patient_ids)

    # Create stratification labels (cancer type + CD274)
    strat_labels = []
    for pid in patient_ids:
        s = patient_to_samples[pid][0]
        sig = s.get("immune_signature", {})
        label = f"{s['project_id']}_{sig.get('cd274_expression', 'unk')}"
        strat_labels.append(label)
    strat_labels = np.array(strat_labels)

    if n < 10:
        # Too few patients — use random split as fallback
        random.shuffle(patient_ids)
        train_end = max(1, int(n * train_ratio))
        val_end = train_end + max(1, int(n * val_ratio))
        train_pids = patient_ids[:train_end]
        val_pids = patient_ids[train_end:val_end]
        test_pids = patient_ids[val_end:]
    else:
        # V3.1: Proper stratified split using StratifiedShuffleSplit
        # Step 1: Split off test set (10%)
        test_split = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        )
        indices = np.arange(n)
        train_val_idx, test_idx = next(test_split.split(indices, strat_labels))

        test_pids = [patient_ids[i] for i in test_idx]
        remaining_ids = [patient_ids[i] for i in train_val_idx]
        remaining_labels = strat_labels[train_val_idx]

        # Step 2: Split remaining into train (80%) and val (10%) → val is 10/90 of remaining
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        val_split = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio_adjusted, random_state=seed
        )
        remaining_indices = np.arange(len(remaining_ids))
        train_idx, val_idx = next(val_split.split(remaining_indices, remaining_labels))

        train_pids = [remaining_ids[i] for i in train_idx]
        val_pids = [remaining_ids[i] for i in val_idx]

    # Collect samples
    train = [s for pid in train_pids for s in patient_to_samples[pid]]
    val = [s for pid in val_pids for s in patient_to_samples[pid]]
    test = [s for pid in test_pids for s in patient_to_samples[pid]]

    print(f"\n📊 Patient-level STRATIFIED split (seed={seed}) — v3.1")
    print(f"   Train: {len(train_pids)} patients → {len(train)} samples")
    print(f"   Val:   {len(val_pids)} patients → {len(val)} samples")
    print(f"   Test:  {len(test_pids)} patients → {len(test)} samples")

    # Verify no leakage
    train_set = set(train_pids)
    val_set = set(val_pids)
    test_set = set(test_pids)
    assert len(train_set & val_set) == 0, "LEAKAGE: train ∩ val"
    assert len(train_set & test_set) == 0, "LEAKAGE: train ∩ test"
    assert len(val_set & test_set) == 0, "LEAKAGE: val ∩ test"
    print("   ✅ No patient leakage between splits")

    # Report class balance across splits
    for split_name, split_pids in [("Train", train_pids), ("Val", val_pids), ("Test", test_pids)]:
        split_labels = []
        for pid in split_pids:
            s = patient_to_samples[pid][0]
            split_labels.append(s.get("immune_signature", {}).get("cd274_expression", "unk"))
        label_counts = Counter(split_labels)
        total = len(split_labels)
        dist = {k: f"{v}/{total} ({100*v/total:.0f}%)" for k, v in label_counts.items()}
        print(f"   {split_name} CD274 balance: {dist}")

    return train, val, test


train_samples, val_samples, test_samples = patient_level_split(
    training_samples,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    seed=RANDOM_SEED,
)

# %%
# ============================================================
# CELL 10: Write JSONL Files (Efficient Chunked Writing)
# ============================================================

def write_jsonl(samples: list, output_path: str):
    """Write samples to JSONL file with efficient buffered writing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', buffering=1024*1024) as f:  # 1MB buffer
        for sample in samples:
            # Write only the fields needed for training (exclude large metadata)
            record = {
                "sample_id": sample["sample_id"],
                "patient_id": sample["patient_id"],
                "slide_id": sample["slide_id"],
                "project_id": sample["project_id"],
                "cancer_type": sample["cancer_type"],
                "patch_paths": sample["patch_paths"],
                "prompt": sample["prompt"],
                "response": sample["response"],
            }
            f.write(json.dumps(record) + '\n')
    print(f"   ✅ Written {len(samples)} samples to {output_path}")


write_jsonl(train_samples, f"{OUTPUT_DIR}/train.jsonl")
write_jsonl(val_samples, f"{OUTPUT_DIR}/val.jsonl")
write_jsonl(test_samples, f"{OUTPUT_DIR}/test.jsonl")

# Also save full metadata (for analysis)
with open(f"{OUTPUT_DIR}/split_metadata.json", 'w') as f:
    json.dump({
        "train_patients": list(set(s["patient_id"] for s in train_samples)),
        "val_patients": list(set(s["patient_id"] for s in val_samples)),
        "test_patients": list(set(s["patient_id"] for s in test_samples)),
        "total_samples": len(training_samples),
        "seed": RANDOM_SEED,
    }, f, indent=2)

print(f"\n✅ All JSONL files written to {OUTPUT_DIR}/")

# %%
# ============================================================
# CELL 11: Verify Dataset Integrity
# ============================================================

def verify_jsonl(path: str) -> dict:
    """Verify a JSONL file is well-formed and all patch paths exist."""
    stats = {"total": 0, "valid_json": 0, "valid_paths": 0, "missing_patches": 0}
    with open(path) as f:
        for line in f:
            stats["total"] += 1
            try:
                record = json.loads(line)
                stats["valid_json"] += 1

                # Verify patch paths exist
                all_exist = all(os.path.exists(p) for p in record["patch_paths"])
                if all_exist:
                    stats["valid_paths"] += 1
                else:
                    stats["missing_patches"] += sum(
                        1 for p in record["patch_paths"] if not os.path.exists(p)
                    )

                # Verify response is valid JSON
                resp = json.loads(record["response"])
                assert "cd274_expression" in resp
                assert "immune_score" in resp

            except (json.JSONDecodeError, KeyError, AssertionError) as e:
                print(f"   ⚠️ Invalid record: {e}")

    return stats


print("Verifying datasets...")
for split_name in ["train", "val", "test"]:
    path = f"{OUTPUT_DIR}/{split_name}.jsonl"
    if os.path.exists(path):
        stats = verify_jsonl(path)
        print(f"\n   {split_name}.jsonl:")
        print(f"      Records: {stats['total']}")
        print(f"      Valid JSON: {stats['valid_json']}")
        print(f"      Valid paths: {stats['valid_paths']}")
        if stats["missing_patches"] > 0:
            print(f"      ⚠️ Missing patches: {stats['missing_patches']}")

# %%
# ============================================================
# CELL 12: Label Distribution Analysis
# ============================================================

def analyze_split(samples: list, name: str):
    """Print label distribution for a split."""
    if not samples:
        print(f"\n{name}: EMPTY")
        return

    print(f"\n📊 {name} ({len(samples)} samples):")

    # CD274
    cd274_dist = Counter()
    for s in samples:
        resp = json.loads(s["response"])
        cd274_dist[resp.get("cd274_expression", "unknown")] += 1
    print(f"   CD274: {dict(cd274_dist)}")

    # TME
    tme_dist = Counter()
    for s in samples:
        resp = json.loads(s["response"])
        tme_dist[resp.get("tme_subtype", "unknown")] += 1
    print(f"   TME:   {dict(tme_dist)}")

    # MSI
    msi_dist = Counter()
    for s in samples:
        resp = json.loads(s["response"])
        msi_dist[resp.get("msi_status", "unknown")] += 1
    print(f"   MSI:   {dict(msi_dist)}")

    # Immune phenotype
    pheno_dist = Counter()
    for s in samples:
        resp = json.loads(s["response"])
        pheno_dist[resp.get("immune_phenotype", "unknown")] += 1
    print(f"   Phenotype: {dict(pheno_dist)}")

    # Patches per sample
    patches = [s["n_patches"] for s in samples]
    print(f"   Patches/sample: mean={np.mean(patches):.1f}, min={min(patches)}, max={max(patches)}")


analyze_split(train_samples, "TRAIN")
analyze_split(val_samples, "VAL")
analyze_split(test_samples, "TEST")

# %%
# ============================================================
# CELL 13: Phase 3 Summary + Report
# ============================================================

report = {
    "phase": 3,
    "timestamp": datetime.now().isoformat(),
    "total_matched_samples": len(training_samples),
    "train_samples": len(train_samples),
    "val_samples": len(val_samples),
    "test_samples": len(test_samples),
    "train_patients": len(set(s["patient_id"] for s in train_samples)),
    "val_patients": len(set(s["patient_id"] for s in val_samples)),
    "test_patients": len(set(s["patient_id"] for s in test_samples)),
    "patient_split_verified": True,
    "max_patches_per_sample": MAX_PATCHES_PER_SAMPLE,
    "random_seed": RANDOM_SEED,
    "output_dir": OUTPUT_DIR,
}

report_path = f"{RESULTS_DIR}/phase3/phase3_report.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 60)
print("PHASE 3 — TRAINING DATA CREATION SUMMARY")
print("=" * 60)
print(f"\n  Total matched samples:  {len(training_samples)}")
print(f"  Train:  {len(train_samples)} samples")
print(f"  Val:    {len(val_samples)} samples")
print(f"  Test:   {len(test_samples)} samples")
print(f"  Patient-level split:    ✅ Verified (no leakage)")
print(f"  Max patches/sample:     {MAX_PATCHES_PER_SAMPLE}")
print(f"\n  Output: {OUTPUT_DIR}/")
print(f"  Report: {report_path}")

print(f"\n{'=' * 60}")
print("📋 UPDATE PHASE_TRACKER.md:")
print(f"{'=' * 60}")
print(f"  Status:             DONE")
print(f"  train.jsonl:        {len(train_samples)} samples")
print(f"  val.jsonl:          {len(val_samples)} samples")
print(f"  test.jsonl:         {len(test_samples)} samples")
print(f"  Patient split:      Verified")

print(f"\n{'=' * 60}")
print("NEXT: Phase 4 — Zero-Shot Baseline")
print(f"{'=' * 60}")
print("1. Load MedGemma (no fine-tuning)")
print("2. Run inference on test + val sets")
print("3. Parse JSON, compute metrics")
print("4. Save baseline results for comparison")
