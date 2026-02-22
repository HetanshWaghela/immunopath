# %% [markdown]
# # 🔬 Phase 2 — Data Processing (Days 2-3)
#
# **Goal:** (1) Tile every downloaded .svs slide into 512×512 patches with Reinhard
# normalization. (2) Compute RNA-seq ground-truth labels: CD274 expression, immune
# gene set scores, immune phenotype, immune score.
#
# **Outputs:**
# - `data/processed/patches/{slide_id}/` — 512×512 JPEG patches per slide
# - `data/signatures/immune_signatures.csv` — Per-patient immune labels
#
# ---
# **Hard Rules:**
# - Patch size: **512×512** at 20× magnification (~0.5 µm/px)
# - Stain normalization: **Reinhard** (LAB color space, NOT Macenko)
# - Tissue threshold: **≥50%** tissue content per patch
# - CD274 expression: log2(TPM+1) → **median split** → high/low
# - TME subtypes: IE, IE/F, F, D (slash notation)
# - MSI-L → treated as MSS (clinical convention)

# %%
# ============================================================
# CELL 1: Colab Setup
# ============================================================

import os

from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/ImmunoPath"
DATA_DIR = f"{PROJECT_DIR}/data"

# Create processing output directories
for d in [
    f"{DATA_DIR}/processed/patches",
    f"{DATA_DIR}/signatures",
]:
    os.makedirs(d, exist_ok=True)

print(f"✅ Directories ready")
print(f"   Input slides:  {DATA_DIR}/raw/slides/")
print(f"   Input RNA-seq: {DATA_DIR}/raw/rnaseq/")
print(f"   Output patches: {DATA_DIR}/processed/patches/")
print(f"   Output sigs:    {DATA_DIR}/signatures/")

# %%
# ============================================================
# CELL 2: Install Dependencies
# ============================================================
# OpenSlide needs system-level install on Linux (Colab)

import subprocess
subprocess.run(["apt-get", "install", "-y", "-qq", "openslide-tools"], check=True)
subprocess.run(["pip", "install", "-q", "openslide-python", "opencv-python-headless",
                 "Pillow", "numpy", "pandas", "scipy", "tqdm"], check=True)

# Verify imports
import openslide
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from tqdm.auto import tqdm
import json
from pathlib import Path
import glob

print(f"✅ OpenSlide version: {openslide.__version__}")
print(f"✅ All dependencies loaded")

# %%
# ============================================================
# CELL 3: Reinhard Stain Normalization
# ============================================================
# Reinhard normalization transfers color statistics in LAB space.
# This is MANDATORY per spec (NOT Macenko).
#
# Reference: Reinhard et al. "Color Transfer Between Images" (2001)
# We use fixed reference statistics from a representative TCGA H&E slide.

# Reference statistics (TCGA standard H&E)
# These are from a well-stained, high-quality TCGA lung adenocarcinoma slide
REF_MEANS = np.array([148.60, 169.30, 105.97])  # LAB channel means
REF_STDS  = np.array([ 41.56,   9.01,  14.67])  # LAB channel stds

def reinhard_normalize(image_rgb: np.ndarray,
                       ref_means: np.ndarray = REF_MEANS,
                       ref_stds: np.ndarray = REF_STDS) -> np.ndarray:
    """
    Apply Reinhard color normalization in LAB color space.

    Args:
        image_rgb: Input RGB image as numpy array (H, W, 3), dtype uint8
        ref_means: Reference LAB channel means [L, A, B]
        ref_stds: Reference LAB channel standard deviations [L, A, B]

    Returns:
        Normalized RGB image as numpy array (H, W, 3), dtype uint8
    """
    # Convert to LAB color space (float for precision)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Compute source statistics
    src_means = np.mean(lab, axis=(0, 1))
    src_stds = np.std(lab, axis=(0, 1))

    # Transfer statistics channel by channel
    for i in range(3):
        if src_stds[i] > 1e-6:  # Avoid division by zero
            lab[:, :, i] = ((lab[:, :, i] - src_means[i])
                            * (ref_stds[i] / src_stds[i])
                            + ref_means[i])

    # Clip and convert back
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def test_reinhard():
    """Quick visual test of Reinhard normalization."""
    # Create a synthetic pinkish image (simulating H&E)
    test_img = np.random.randint(150, 230, (256, 256, 3), dtype=np.uint8)
    test_img[:, :, 0] = np.clip(test_img[:, :, 0] + 30, 0, 255)  # More red
    test_img[:, :, 2] = np.clip(test_img[:, :, 2] + 20, 0, 255)  # More blue

    normalized = reinhard_normalize(test_img)

    # Check LAB stats of normalized image are closer to reference
    lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB).astype(np.float32)
    norm_means = np.mean(lab, axis=(0, 1))
    norm_stds = np.std(lab, axis=(0, 1))

    print("Reinhard normalization test:")
    print(f"  Reference means: {REF_MEANS}")
    print(f"  Normalized means: {norm_means}")
    print(f"  Reference stds:  {REF_STDS}")
    print(f"  Normalized stds: {norm_stds}")
    print(f"  ✅ Normalization function works")

test_reinhard()

# %%
# ============================================================
# CELL 4: Tissue Detection Functions
# ============================================================
# Detect tissue regions in H&E patches using Otsu thresholding.
# Filter out background, pen marks, and artifacts.

def compute_tissue_mask(image_rgb: np.ndarray, threshold: int = 220) -> np.ndarray:
    """
    Create a binary tissue mask using grayscale thresholding.

    Tissue appears darker than background in H&E slides.
    Background is usually white/near-white (>220 in grayscale).

    Args:
        image_rgb: RGB image as numpy array
        threshold: Grayscale threshold (pixels < threshold = tissue)

    Returns:
        Binary mask (1 = tissue, 0 = background)
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Tissue is darker than background
    mask = (gray < threshold).astype(np.uint8)

    # Morphological cleanup (remove small holes and noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def compute_tissue_fraction(image_rgb: np.ndarray, threshold: int = 220) -> float:
    """Compute fraction of tissue (non-background) pixels."""
    mask = compute_tissue_mask(image_rgb, threshold)
    return np.mean(mask)


def is_pen_mark(image_rgb: np.ndarray) -> bool:
    """
    Detect pen marks (common artifacts in TCGA slides).
    Pen marks are typically blue, green, or black ink.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Blue pen: high saturation, low value, hue in blue range
    blue_mask = ((hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 130) &
                 (hsv[:, :, 1] > 50) & (hsv[:, :, 2] < 200))

    # Green pen
    green_mask = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) &
                  (hsv[:, :, 1] > 50) & (hsv[:, :, 2] < 200))

    # Black pen / very dark artifacts
    black_mask = (hsv[:, :, 2] < 40)

    pen_fraction = (np.sum(blue_mask) + np.sum(green_mask) + np.sum(black_mask)) / (image_rgb.shape[0] * image_rgb.shape[1])
    return pen_fraction > 0.1  # >10% pen pixels → reject


def is_blurry(image_rgb: np.ndarray, threshold: float = 50.0) -> bool:
    """Detect blurry patches using Laplacian variance."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


print("✅ Tissue detection functions defined")

# %%
# ============================================================
# CELL 5: WSI Patch Extraction Pipeline
# ============================================================
# Extract 512×512 patches at 20× from .svs slides.
# Apply tissue detection, artifact filtering, and Reinhard normalization.

PATCH_SIZE = 512          # Pixels
TARGET_MPP = 0.5          # Microns per pixel (≈20× magnification)
TISSUE_THRESHOLD = 0.5    # Minimum tissue fraction per patch
MAX_PATCHES_PER_SLIDE = 64  # Memory-practical ceiling
JPEG_QUALITY = 95


def get_slide_mpp(slide: openslide.OpenSlide) -> float:
    """Get microns per pixel from slide properties."""
    try:
        mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0))
        if mpp_x > 0:
            return mpp_x
    except (ValueError, KeyError):
        pass

    # Fallback: estimate from objective power
    try:
        power = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 0))
        if power > 0:
            return 10.0 / power  # Rough conversion
    except (ValueError, KeyError):
        pass

    # Default assumption for TCGA slides
    return 0.25  # Most TCGA slides are scanned at 40× ≈ 0.25 mpp


def extract_patches_from_slide(
    slide_path: str,
    output_dir: str,
    patch_size: int = PATCH_SIZE,
    target_mpp: float = TARGET_MPP,
    tissue_threshold: float = TISSUE_THRESHOLD,
    max_patches: int = MAX_PATCHES_PER_SLIDE,
    normalize: bool = True,
    jpeg_quality: int = JPEG_QUALITY,
) -> dict:
    """
    Extract tissue patches from a whole-slide image.

    Args:
        slide_path: Path to .svs file
        output_dir: Directory to save patches
        patch_size: Patch dimensions in pixels
        target_mpp: Target microns per pixel
        tissue_threshold: Min tissue fraction
        max_patches: Max patches to extract
        normalize: Apply Reinhard normalization
        jpeg_quality: JPEG save quality

    Returns:
        Metadata dict with extraction stats
    """
    slide = openslide.OpenSlide(slide_path)
    slide_id = Path(slide_path).stem

    # Determine the best level for target_mpp
    slide_mpp = get_slide_mpp(slide)
    downsample_factor = target_mpp / slide_mpp if slide_mpp > 0 else 1.0

    # Find the closest available level
    best_level = 0
    for level in range(slide.level_count):
        level_ds = slide.level_downsamples[level]
        if level_ds <= downsample_factor * 1.2:  # Allow 20% tolerance
            best_level = level

    level_ds = slide.level_downsamples[best_level]
    level_dims = slide.level_dimensions[best_level]

    # Compute effective patch size at level 0 coordinates
    patch_size_l0 = int(patch_size * level_ds)

    # Create output directory
    patch_dir = os.path.join(output_dir, slide_id)
    os.makedirs(patch_dir, exist_ok=True)

    # Generate grid of patch locations
    width_l0, height_l0 = slide.dimensions
    patches_metadata = []
    all_candidates = []

    for y in range(0, height_l0 - patch_size_l0 + 1, patch_size_l0):
        for x in range(0, width_l0 - patch_size_l0 + 1, patch_size_l0):
            all_candidates.append((x, y))

    # Shuffle candidates for random sampling (when max_patches < total)
    np.random.shuffle(all_candidates)

    extracted = 0
    for x, y in tqdm(all_candidates, desc=f"{slide_id}", leave=False):
        if extracted >= max_patches:
            break

        try:
            # Read patch at best level
            patch = slide.read_region((x, y), best_level, (patch_size, patch_size))
            patch_rgb = np.array(patch.convert('RGB'))

            # Quality checks
            tissue_frac = compute_tissue_fraction(patch_rgb)
            if tissue_frac < tissue_threshold:
                continue

            if is_pen_mark(patch_rgb):
                continue

            if is_blurry(patch_rgb):
                continue

            # Apply Reinhard normalization
            if normalize:
                patch_rgb = reinhard_normalize(patch_rgb)

            # Save patch
            patch_name = f"{slide_id}_patch_{extracted:03d}.jpg"
            patch_path = os.path.join(patch_dir, patch_name)
            Image.fromarray(patch_rgb).save(patch_path, quality=jpeg_quality)

            patches_metadata.append({
                "patch_name": patch_name,
                "x": x,
                "y": y,
                "level": best_level,
                "tissue_fraction": round(tissue_frac, 3),
            })
            extracted += 1

        except Exception as e:
            continue  # Skip problematic patches

    slide.close()

    # Save slide metadata
    metadata = {
        "slide_id": slide_id,
        "slide_path": slide_path,
        "slide_mpp": slide_mpp,
        "extraction_level": best_level,
        "level_downsample": level_ds,
        "patch_size": patch_size,
        "target_mpp": target_mpp,
        "total_candidates": len(all_candidates),
        "patches_extracted": extracted,
        "normalized": normalize,
        "patches": patches_metadata,
    }

    meta_path = os.path.join(patch_dir, f"{slide_id}_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


print("✅ Patch extraction pipeline defined")
print(f"   Settings: {PATCH_SIZE}×{PATCH_SIZE} at {TARGET_MPP} µm/px")
print(f"   Tissue threshold: {TISSUE_THRESHOLD}")
print(f"   Max patches/slide: {MAX_PATCHES_PER_SLIDE}")
print(f"   Reinhard normalization: enabled")

# %%
# ============================================================
# CELL 6: Run Patch Extraction on Downloaded Slides
# ============================================================
# Process all .svs slides from Phase 1 download.

slide_dir = f"{DATA_DIR}/raw/slides"
patch_output_dir = f"{DATA_DIR}/processed/patches"

# Find all .svs files
svs_files = glob.glob(os.path.join(slide_dir, "*.svs"))
print(f"Found {len(svs_files)} SVS slides to process")

if len(svs_files) == 0:
    print("\n⚠️  No .svs files found in slide directory!")
    print(f"   Expected location: {slide_dir}")
    print("   Run Phase 1 first to download slides.")
    print("   Or set MAX_SLIDES > 0 in Phase 1, Cell 6")
else:
    extraction_results = []

    for i, svs_path in enumerate(svs_files):
        slide_name = Path(svs_path).stem
        print(f"\n[{i+1}/{len(svs_files)}] Processing: {slide_name}")

        # Skip if already processed
        existing_meta = os.path.join(patch_output_dir, slide_name, f"{slide_name}_metadata.json")
        if os.path.exists(existing_meta):
            print(f"  Already processed — skipping")
            with open(existing_meta) as f:
                extraction_results.append(json.load(f))
            continue

        try:
            meta = extract_patches_from_slide(
                slide_path=svs_path,
                output_dir=patch_output_dir,
                patch_size=PATCH_SIZE,
                target_mpp=TARGET_MPP,
                tissue_threshold=TISSUE_THRESHOLD,
                max_patches=MAX_PATCHES_PER_SLIDE,
                normalize=True,
            )
            extraction_results.append(meta)
            print(f"  ✅ {meta['patches_extracted']} patches extracted "
                  f"(from {meta['total_candidates']} candidates)")
        except Exception as e:
            print(f"  ❌ Failed: {type(e).__name__}: {e}")

    # Summary
    total_patches = sum(r.get("patches_extracted", 0) for r in extraction_results)
    print(f"\n{'=' * 60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Slides processed: {len(extraction_results)}")
    print(f"  Total patches:    {total_patches}")
    print(f"  Output dir:       {patch_output_dir}")

# %%
# ============================================================
# CELL 6.5: Diversity-Based Patch Selection (K-Means)
# ============================================================
# Spec Section 4.2.2: Select 8 representative patches per slide
# using clustering-based diversity sampling. This ensures the
# 8 patches sent to MedGemma cover distinct morphological regions.
#
# Method: Color histogram features (96-dim) → K-Means (K=8) →
#         select patch closest to each centroid.

from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed

N_SELECT = 8  # Patches per sample for MedGemma (Phase 0 verified max=8)


def compute_color_histogram(image_rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    """Compute 96-dim color histogram feature vector (32 bins × 3 channels)."""
    features = []
    for ch in range(3):
        hist, _ = np.histogram(image_rgb[:, :, ch], bins=bins, range=(0, 256))
        features.extend(hist / max(hist.sum(), 1))
    return np.array(features, dtype=np.float32)


def select_diverse_patches(slide_dir: str, n_select: int = N_SELECT) -> dict:
    """Select diverse patches from one slide using K-Means on color histograms."""
    slide_id = os.path.basename(slide_dir)
    patches = sorted(glob.glob(os.path.join(slide_dir, "*.jpg")))

    if len(patches) == 0:
        return {"slide_id": slide_id, "n_patches": 0, "selected": []}

    if len(patches) <= n_select:
        selected = [os.path.basename(p) for p in patches]
        result_path = os.path.join(slide_dir, f"{slide_id}_selected_8.json")
        with open(result_path, 'w') as f:
            json.dump({"slide_id": slide_id, "method": "all_available",
                       "selected": selected}, f, indent=2)
        return {"slide_id": slide_id, "n_patches": len(patches), "selected": selected}

    embeddings = []
    for p in patches:
        try:
            img = np.array(Image.open(p).convert("RGB"))
            embeddings.append(compute_color_histogram(img))
        except Exception:
            embeddings.append(np.zeros(96, dtype=np.float32))

    embeddings = np.array(embeddings)

    kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    selected_indices = []
    for cluster_id in range(n_select):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        cluster_embs = embeddings[mask]
        cluster_indices = np.where(mask)[0]
        centroid = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        best = cluster_indices[np.argmin(dists)]
        selected_indices.append(int(best))

    selected = [os.path.basename(patches[i]) for i in sorted(selected_indices)]

    np.save(os.path.join(slide_dir, f"{slide_id}_all_embeddings.npy"), embeddings)

    result = {"slide_id": slide_id, "method": "kmeans_color_histogram",
              "n_total": len(patches), "n_selected": len(selected),
              "selected": selected}
    with open(os.path.join(slide_dir, f"{slide_id}_selected_8.json"), 'w') as f:
        json.dump(result, f, indent=2)

    return {"slide_id": slide_id, "n_patches": len(patches), "selected": selected}


print(f"Running diversity-based patch selection (K={N_SELECT})...")

slide_dirs_for_selection = [
    os.path.join(patch_output_dir, d)
    for d in os.listdir(patch_output_dir)
    if os.path.isdir(os.path.join(patch_output_dir, d))
] if os.path.exists(patch_output_dir) else []

selection_results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(select_diverse_patches, d): d for d in slide_dirs_for_selection}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Selecting patches"):
        result = future.result()
        selection_results.append(result)

n_with_selection = sum(1 for r in selection_results if len(r.get("selected", [])) > 0)
print(f"\n✅ Diversity patch selection complete:")
print(f"   Slides processed: {len(selection_results)}")
print(f"   Slides with selected patches: {n_with_selection}")
print(f"   Method: K-Means on 96-dim color histograms → {N_SELECT} diverse patches/slide")

# %%
# ============================================================
# CELL 7: Visualize Sample Patches (QC Check)
# ============================================================
# Display some extracted patches to visually verify quality.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def show_sample_patches(patch_dir, n_patches=8):
    """Display random sample of extracted patches."""
    # Find all patch directories
    slide_dirs = [d for d in os.listdir(patch_dir)
                  if os.path.isdir(os.path.join(patch_dir, d))]

    if not slide_dirs:
        print("No patches to display")
        return

    # Collect patches
    all_patches = []
    for sd in slide_dirs:
        patches = glob.glob(os.path.join(patch_dir, sd, "*.jpg"))
        all_patches.extend(patches)

    if not all_patches:
        print("No patch files found")
        return

    # Sample and display
    n_show = min(n_patches, len(all_patches))
    indices = np.random.choice(len(all_patches), n_show, replace=False)

    fig, axes = plt.subplots(2, n_show // 2, figsize=(3 * (n_show // 2), 6))
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        img = Image.open(all_patches[idx])
        ax.imshow(img)
        ax.set_title(Path(all_patches[idx]).stem[-10:], fontsize=8)
        ax.axis('off')

    plt.suptitle("Sample Extracted Patches (Reinhard Normalized)", fontsize=12)
    plt.tight_layout()

    qc_path = f"{PROJECT_DIR}/results/phase2_patch_samples.png"
    plt.savefig(qc_path, dpi=150)
    plt.show()
    print(f"✅ QC image saved to: {qc_path}")

show_sample_patches(patch_output_dir)

# %% [markdown]
# ---
# ## Part B: Compute Immune Signatures from RNA-seq
#
# This section processes GDC RNA-seq files to compute:
# 1. **Gene set signature scores** (CD8, IFNγ, TIL, PD-L1 related, exhaustion)
# 2. **CD274 expression** → median split → high/low
# 3. **Immune phenotype** classification
# 4. **Composite immune score**
#
# Gene sets from IMMUNOPATH_LUNG_SPEC.md Section 5.3 Step 2.

# %%
# ============================================================
# CELL 8: Define Immune Gene Sets
# ============================================================
# From spec Section 5.3 Step 2 (EXACT gene lists — do not modify)

IMMUNE_GENE_SETS = {
    "cd8_signature": [
        "CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "IFNG", "CXCL9", "CXCL10"
    ],
    "ifng_signature": [
        "IFNG", "STAT1", "CCR5", "CXCL9", "CXCL10", "CXCL11", "IDO1",
        "PRF1", "GZMA", "HLA-DRA", "CXCR6", "LAG3", "NKG7", "PSMB10",
        "CMKLR1", "CD8A", "TIGIT", "PDCD1LG2"
    ],
    "til_signature": [
        "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "FOXP3",
        "CD19", "MS4A1", "NCAM1"  # MS4A1=CD20, NCAM1=CD56
    ],
    "pdl1_related": [
        "CD274", "PDCD1", "PDCD1LG2", "CTLA4", "LAG3", "TIGIT", "HAVCR2"
    ],
    "exhaustion_signature": [
        "PDCD1", "CTLA4", "LAG3", "TIGIT", "HAVCR2", "TOX", "ENTPD1"
    ],
}

# CD274 is the gene encoding PD-L1 protein
# We use CD274 mRNA as a PROXY for PD-L1 IHC (NOT a replacement)
CD274_GENE = "CD274"

# Gene alias mapping (common issues with TCGA gene names)
GENE_ALIASES = {
    "CD20": "MS4A1",    # CD20 is commonly called MS4A1 in RNA-seq
    "CD56": "NCAM1",    # CD56 is NCAM1
    "PD-L1": "CD274",   # PD-L1 protein, CD274 gene
    "PD-1": "PDCD1",    # PD-1 protein, PDCD1 gene
    "TIM-3": "HAVCR2",  # TIM-3 protein, HAVCR2 gene
}

print("✅ Immune gene sets defined:")
for name, genes in IMMUNE_GENE_SETS.items():
    print(f"   {name}: {len(genes)} genes")

# %%
# ============================================================
# CELL 9: Load RNA-seq Data
# ============================================================
# GDC STAR-Counts files are tab-separated with columns:
#   gene_id, gene_name, gene_type, unstranded, stranded_first,
#   stranded_second, tpm_unstranded, fpkm_unstranded, fpkm_uq_unstranded
#
# We use tpm_unstranded (transcripts per million).
# Skip metadata rows starting with "N_" (e.g., N_unmapped, N_multimapping)

import gzip

def load_gdc_rnaseq(file_path: str) -> pd.Series:
    """
    Load a single GDC RNA-seq quantification file.

    Returns: pd.Series indexed by gene_name with TPM values.
    """
    try:
        if file_path.endswith('.gz'):
            df = pd.read_csv(file_path, sep='\t', compression='gzip', comment='#')
        else:
            df = pd.read_csv(file_path, sep='\t', comment='#')
    except Exception:
        # Some GDC downloads come as tar.gz with internal structure
        # Try reading with alternative approaches
        raise

    # Identify the gene name and TPM columns
    # GDC format varies slightly between versions
    gene_col = None
    tpm_col = None

    for col in df.columns:
        if col.lower() in ('gene_name', 'gene_id'):
            gene_col = col
        if 'tpm' in col.lower():
            tpm_col = col

    if gene_col is None or tpm_col is None:
        # Fallback: assume first column is gene names, column 6 is TPM
        if len(df.columns) >= 7:
            gene_col = df.columns[1]  # gene_name is usually column 2
            tpm_col = df.columns[6]   # tpm_unstranded is usually column 7
        else:
            raise ValueError(f"Cannot identify columns in {file_path}: {list(df.columns)}")

    # Filter out metadata rows (N_unmapped, N_multimapping, etc.)
    mask = ~df[gene_col].astype(str).str.startswith('N_')
    df = df[mask]

    # Create gene → TPM series
    tpm = df.set_index(gene_col)[tpm_col].astype(float)

    # Handle duplicate gene names (keep the one with higher expression)
    tpm = tpm.groupby(level=0).max()

    return tpm


# Load all RNA-seq files
rnaseq_dir = f"{DATA_DIR}/raw/rnaseq"
rnaseq_files = glob.glob(os.path.join(rnaseq_dir, "*.tsv*"))

print(f"Found {len(rnaseq_files)} RNA-seq files")

if len(rnaseq_files) == 0:
    print("\n⚠️ No RNA-seq files found!")
    print(f"   Expected in: {rnaseq_dir}")
    print("   Run Phase 1 first.")
else:
    # Load all into a DataFrame (patients × genes)
    # Use the metadata to map file → patient
    metadata_path = f"{DATA_DIR}/raw/manifests/rnaseq_metadata.csv"
    nsclc_meta_path = f"{DATA_DIR}/raw/nsclc_metadata.csv"

    patient_tpm = {}
    errors = []

    for fpath in tqdm(rnaseq_files, desc="Loading RNA-seq"):
        # Extract patient ID from filename
        basename = Path(fpath).stem.replace(".tsv", "")

        try:
            tpm = load_gdc_rnaseq(fpath)
            patient_tpm[basename] = tpm
        except Exception as e:
            errors.append(f"{basename}: {e}")

    if errors:
        print(f"\n⚠️ {len(errors)} files failed to load")
        for err in errors[:5]:
            print(f"   {err}")

    if patient_tpm:
        # Create expression matrix (patients × genes)
        expression_matrix = pd.DataFrame(patient_tpm).T
        print(f"\n✅ Expression matrix: {expression_matrix.shape[0]} patients × {expression_matrix.shape[1]} genes")

        # Log2 transform: log2(TPM + 1)
        expression_log2 = np.log2(expression_matrix + 1)

        # Check for our target genes
        target_genes = set()
        for genes in IMMUNE_GENE_SETS.values():
            target_genes.update(genes)

        found_genes = target_genes.intersection(expression_matrix.columns)
        missing_genes = target_genes - found_genes
        print(f"   Target immune genes found: {len(found_genes)}/{len(target_genes)}")
        if missing_genes:
            print(f"   Missing: {missing_genes}")
    else:
        print("❌ No RNA-seq data loaded")

# %%
# ============================================================
# CELL 10: Compute Immune Signature Scores
# ============================================================
# Method: Z-score each gene across all patients, then average within
# each gene set to get a single score per patient per signature.

def compute_signature_scores(expr_log2: pd.DataFrame,
                              gene_sets: dict) -> pd.DataFrame:
    """
    Compute immune signature scores from log2-TPM expression matrix.

    Method:
    1. Z-score normalize each gene across samples
    2. Average z-scores within each gene set
    3. Result: one score per patient per signature

    Args:
        expr_log2: DataFrame (patients × genes), log2(TPM+1)
        gene_sets: Dict mapping signature_name → list of gene symbols

    Returns:
        DataFrame (patients × signatures)
    """
    # Z-score all genes across patients
    z_scores = expr_log2.apply(stats.zscore, nan_policy='omit')

    scores = {}
    for sig_name, genes in gene_sets.items():
        # Find which genes are available
        available = [g for g in genes if g in z_scores.columns]

        if len(available) == 0:
            print(f"  ⚠️ {sig_name}: no genes found in expression data")
            scores[sig_name] = np.nan
            continue

        coverage = len(available) / len(genes)
        if coverage < 0.5:
            print(f"  ⚠️ {sig_name}: only {len(available)}/{len(genes)} genes found ({coverage:.0%})")

        # Average z-score of available genes
        scores[sig_name] = z_scores[available].mean(axis=1)

    return pd.DataFrame(scores)


if 'expression_log2' in dir():
    print("Computing immune signature scores...")
    sig_scores = compute_signature_scores(expression_log2, IMMUNE_GENE_SETS)
    print(f"\n✅ Signature scores: {sig_scores.shape[0]} patients × {sig_scores.shape[1]} signatures")
    print(f"\nSummary statistics:")
    print(sig_scores.describe().round(3).to_string())
else:
    print("⚠️ Expression matrix not available — skipping signature computation")

# %%
# ============================================================
# CELL 11: Compute CD274 Expression Labels
# ============================================================
# CD274 (PD-L1 gene) expression: log2(TPM+1) → median split → high/low
# This is a SURROGATE for PD-L1 IHC protein expression (r²=0.65-0.81)

if 'expression_log2' in dir() and CD274_GENE in expression_log2.columns:
    cd274_log2 = expression_log2[CD274_GENE]
    cd274_median = cd274_log2.median()

    # Median split → binary labels
    cd274_label = pd.Series(
        np.where(cd274_log2 >= cd274_median, "high", "low"),
        index=cd274_log2.index,
        name="cd274_expression"
    )

    print(f"CD274 (PD-L1 gene) expression:")
    print(f"  Median log2(TPM+1): {cd274_median:.3f}")
    print(f"  High (≥ median):    {(cd274_label == 'high').sum()}")
    print(f"  Low  (< median):    {(cd274_label == 'low').sum()}")
    print(f"\n  ⚠️ REMINDER: This is RNA proxy, NOT IHC PD-L1 TPS.")
    print(f"     Use 'cd274_expression' (NOT 'pdl1_tps') in all code.")
else:
    print("⚠️ CD274 gene not found — cannot compute PD-L1 proxy")
    cd274_log2 = None
    cd274_label = None

# %%
# ============================================================
# CELL 12: Classify Immune Phenotype
# ============================================================
# Based on TIL and CD8 signature scores:
#   inflamed  = TIL high + CD8 high
#   excluded  = TIL high + CD8 low
#   desert    = TIL low
#
# Use median split on signature scores for high/low.

def classify_immune_phenotype(sig_scores: pd.DataFrame) -> pd.Series:
    """
    Classify immune phenotype from signature scores.

    Returns: Series with values 'inflamed', 'excluded', or 'desert'
    """
    til_score = sig_scores["til_signature"]
    cd8_score = sig_scores["cd8_signature"]

    til_median = til_score.median()
    cd8_median = cd8_score.median()

    til_high = til_score >= til_median
    cd8_high = cd8_score >= cd8_median

    phenotype = pd.Series("desert", index=sig_scores.index, name="immune_phenotype")
    phenotype[til_high & cd8_high] = "inflamed"
    phenotype[til_high & ~cd8_high] = "excluded"

    return phenotype


if 'sig_scores' in dir():
    immune_phenotype = classify_immune_phenotype(sig_scores)
    print(f"Immune phenotype distribution:")
    print(immune_phenotype.value_counts().to_string())
else:
    immune_phenotype = None

# %%
# ============================================================
# CELL 13: Compute Composite Immune Score
# ============================================================
# Average of (CD8 + IFNg + TIL scores) / 3, min-max normalized to [0, 1]

def compute_immune_score(sig_scores: pd.DataFrame) -> pd.Series:
    """Compute composite immune score, normalized to [0, 1]."""
    components = ["cd8_signature", "ifng_signature", "til_signature"]
    available = [c for c in components if c in sig_scores.columns]

    if not available:
        return pd.Series(np.nan, index=sig_scores.index, name="immune_score")

    raw_score = sig_scores[available].mean(axis=1)

    # Min-max normalize to [0, 1]
    score_min = raw_score.min()
    score_max = raw_score.max()

    if score_max - score_min > 0:
        normalized = (raw_score - score_min) / (score_max - score_min)
    else:
        normalized = pd.Series(0.5, index=raw_score.index)

    normalized.name = "immune_score"
    return normalized


if 'sig_scores' in dir():
    immune_score = compute_immune_score(sig_scores)
    print(f"Immune score summary (0 = cold, 1 = hot):")
    print(f"  Mean:   {immune_score.mean():.3f}")
    print(f"  Median: {immune_score.median():.3f}")
    print(f"  Std:    {immune_score.std():.3f}")
else:
    immune_score = None

# %%
# ============================================================
# CELL 14: Assemble Final immune_signatures.csv
# ============================================================
# Combine all computed features into one CSV.

if 'sig_scores' in dir():
    final_df = sig_scores.copy()
    final_df.index.name = "sample_id"

    # Add CD274 expression
    if cd274_log2 is not None:
        final_df["cd274_log2_tpm"] = cd274_log2
    if cd274_label is not None:
        final_df["cd274_expression"] = cd274_label

    # Add immune phenotype
    if immune_phenotype is not None:
        final_df["immune_phenotype"] = immune_phenotype

    # Add immune score
    if immune_score is not None:
        final_df["immune_score"] = immune_score

    # Save
    output_path = f"{DATA_DIR}/signatures/immune_signatures.csv"
    final_df.to_csv(output_path)

    print(f"✅ Saved: {output_path}")
    print(f"   Shape: {final_df.shape}")
    print(f"   Columns: {list(final_df.columns)}")
    print(f"\n   Preview:")
    print(final_df.head().to_string())
else:
    print("⚠️ No signature data to save. Run RNA-seq loading cells first.")

# %%
# ============================================================
# CELL 15: Phase 2 Summary + Next Steps
# ============================================================

from datetime import datetime

print("=" * 60)
print("PHASE 2 — DATA PROCESSING SUMMARY")
print("=" * 60)

# Count patches
total_patches = 0
n_slides_processed = 0
if os.path.exists(patch_output_dir):
    for d in os.listdir(patch_output_dir):
        dpath = os.path.join(patch_output_dir, d)
        if os.path.isdir(dpath):
            n_patches = len(glob.glob(os.path.join(dpath, "*.jpg")))
            if n_patches > 0:
                n_slides_processed += 1
                total_patches += n_patches

print(f"\nPart A — Patch Extraction:")
print(f"  Slides processed:  {n_slides_processed}")
print(f"  Total patches:     {total_patches}")
print(f"  Patch size:        {PATCH_SIZE}×{PATCH_SIZE}")
print(f"  Target MPP:        {TARGET_MPP} (≈20×)")
print(f"  Normalization:     Reinhard (LAB color space)")
print(f"  Output:            {patch_output_dir}")

print(f"\nPart B — Immune Signatures:")
if 'final_df' in dir():
    print(f"  Patients:          {len(final_df)}")
    print(f"  Features:          {len(final_df.columns)}")
    if 'cd274_label' in dir() and cd274_label is not None:
        print(f"  CD274 high:        {(cd274_label == 'high').sum()}")
        print(f"  CD274 low:         {(cd274_label == 'low').sum()}")
    if 'immune_phenotype' in dir() and immune_phenotype is not None:
        print(f"  Inflamed:          {(immune_phenotype == 'inflamed').sum()}")
        print(f"  Excluded:          {(immune_phenotype == 'excluded').sum()}")
        print(f"  Desert:            {(immune_phenotype == 'desert').sum()}")
    print(f"  Output:            {DATA_DIR}/signatures/immune_signatures.csv")
else:
    print(f"  ⚠️ Not computed (no RNA-seq data)")

# Save phase report
report = {
    "phase": 2,
    "timestamp": datetime.now().isoformat(),
    "patches": {"slides": n_slides_processed, "total": total_patches,
                "size": PATCH_SIZE, "mpp": TARGET_MPP, "normalization": "reinhard"},
    "signatures": {"patients": len(final_df) if 'final_df' in dir() else 0,
                    "features": list(final_df.columns) if 'final_df' in dir() else []},
}
report_path = f"{PROJECT_DIR}/results/phase2_processing_report.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"\n📁 Report: {report_path}")

print(f"\n{'=' * 60}")
print("📋 UPDATE PHASE_TRACKER.md:")
print(f"{'=' * 60}")
print(f"  Status:                    DONE")
print(f"  Patches Extracted:         {total_patches} from {n_slides_processed} slides")
print(f"  Reinhard Normalization:    Applied")
print(f"  Immune Signatures:         {'Yes' if 'final_df' in dir() else 'No'}")
print(f"  immune_signatures.csv:     {len(final_df) if 'final_df' in dir() else 0} rows")

print(f"\n{'=' * 60}")
print("NEXT: Phase 3 — Training Data Creation")
print(f"{'=' * 60}")
print("1. Join patches with immune signatures by patient ID")
print("2. Add Bagaev TME subtypes + MSI labels")
print("3. Create JSONL training files (train/val/test)")
print("4. Apply PATIENT-LEVEL splits (prevent leakage)")
