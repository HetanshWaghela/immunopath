# ImmunoPath — Phase Tracker

> **PURPOSE:** Update this file after completing each phase. When you start a new AI chat
> (VSCode Copilot, Colab AI, ChatGPT, etc.), paste the **MASTER_CONTEXT.md** + this tracker
> so the AI instantly knows where you are. No wasted prompts.
>
> **HOW TO USE:** After finishing a phase, update its status, fill in decisions/outputs/notes.
> Before starting the next phase, paste the relevant context into your AI.

---

## Current State

| Field | Value |
|-------|-------|
| **Current Phase** | Phase 3 — Re-run on FULL dataset (then 4 → 5 → 6) |
| **Last Updated** | 2026-02-16 |
| **Days Remaining** | ~8 (deadline: Feb 24, 2026 11:59 PM UTC) |
| **Colab Runtime** | Google Colab Pro (A100 40GB) |
| **Repo** | `/Users/Hetansh/Github/medgemma-impact-challenge` |
| **Data Status** | ✅ FULL DATASET READY — 950 slides downloaded, 949 tiled, 950 signatures |

---

## Phase Status

### Phase 0 — Test MedGemma
| Field | Value |
|-------|-------|
| **Status** | `COMPLETE` |
| **Colab Notebook** | `colab_run_notebooks/phase_0_colab.ipynb` |
| **GPU Used** | NVIDIA A100-SXM4-40GB |
| **Single Image Test** | **Pass** — 380 chars response, correct H&E analysis |
| **Multi-Image Test** | Max images that work: **8** (1, 2, 4, 8 all PASS) |
| **JSON Parse Rate** | **100%** over 10 trials (all parsed via markdown_block) |
| **Architecture Decision** | **Multi-image** — 4-8 patches per call |
| **VRAM Usage** | 8.60 GB allocated |
| **Notes** | Key names vary across trials (e.g. `cd274_proxy_level` vs `CD274_proxy`); fine-tuning will standardize. `transformers==5.1.0` used. |

---

### Phase 1 — Data Download
| Field | Value |
|-------|-------|
| **Status** | `COMPLETE` ✅ **FULL DATASET** |
| **Colab Notebook** | `colab_run_notebooks/phase_1_colab.ipynb` |
| **TCGA Slides Found** | 1053 (541 LUAD + 512 LUSC) across 956 patients |
| **Matched Patients** | 950 (with both slide + RNA-seq) |
| **TCGA Slides Downloaded** | **950/950** (ALL slides downloaded) |
| **RNA-seq Downloaded** | **950/950** (all successful) |
| **Bagaev TME Labels** | **Downloaded** — annotation.tsv, signatures.tsv, gene_signatures.gmt |
| **Thorsson Panimmune** | **Downloaded** — immune_signatures_160, leukocyte_fractions, cibersort, mutation_load, subtypes |
| **cBioPortal Data** | **Downloaded** — LUAD (566 samples) + LUSC PanCancer Atlas; MSI_SCORE_MANTIS + MSI_SENSOR_SCORE extracted |
| **Saltz TIL Labels** | **Downloaded** — summary archive |
| **Notes** | Data saved to Google Drive `/ImmunoPath/data/`. nsclc_metadata.csv created with matched patients. Total slide download: ~718.9 GB. |

---

### Phase 2 — Data Processing
| Field | Value |
|-------|-------|
| **Status** | `COMPLETE` ✅ **FULL DATASET** |
| **Colab Notebook** | `colab_run_notebooks/phase_2_colab.ipynb` |
| **GPU Used** | L4 (Colab) |
| **Slides Processed** | **949** (1 slide TCGA-44-7661 had 0 valid patches) |
| **Total Patches Extracted** | **60,396 patches** from 949 slides |
| **Patch Settings** | 512×512 at 0.5 µm/px (≈20×), max 64 patches/slide, tissue threshold 0.5 |
| **Reinhard Normalization** | **Applied** — LAB color space, vectorized, verified with ref stats |
| **Diversity Patch Selection** | **Done** — 949 slides selected (K-Means on xy+tissue; 8 per slide) |
| **Selection Methods** | kmeans_xy_tissue: 948, uniform_xy_tissue: 1, none: 111 (empty slides) |
| **Immune Signatures Computed** | **Yes** — 5 signatures from **950** RNA-seq samples |
| **CD274 Expression** | Median log2(TPM+1): 3.154 → **475 high / 475 low** |
| **Immune Phenotypes** | desert: 475, inflamed: 383, excluded: 92 |
| **Immune Score** | Mean: 0.483, Median: 0.477, Std: 0.189 (0-1 normalized) |
| **immune_signatures.csv** | **950 rows × 9 columns** |
| **Notes** | 108 slides were additionally downloaded via gdc-client during Phase 2 run. All 34/34 target immune genes found. QC image saved. |

---

### Phase 3 — Training Data Creation
| Field | Value |
|-------|-------|
| **Status** | `NEEDS RE-RUN` ⚠️ (ran on dev subset; full data now available) |
| **Colab Notebook** | `colab_run_notebooks/phase_3_colab.ipynb` |
| **Previous Run (dev)** | 19 matched → 15 train / 1 val / 3 test |
| **Expected Full Run** | ~949 matched → ~760 train / ~95 val / ~95 test |
| **Code Changes Needed** | **None** — code auto-discovers all slides with patches |
| **GPU Needed** | CPU only (no GPU required) |
| **Estimated Time** | ~5-10 minutes |
| **Notes** | Just re-run the entire notebook. The join logic will find all 949 slides now. Patient-level split verified. |

---

### Phase 4 — Zero-Shot Baseline
| Field | Value |
|-------|-------|
| **Status** | `NEEDS RE-RUN` ⚠️ (ran on 4 eval samples; need full test set) |
| **Colab Notebook** | `colab_run_notebooks/phase_4_colab.ipynb` |
| **Previous Run (dev)** | 4 eval samples — metrics unreliable |
| **Expected Full Run** | ~95 test + ~95 val = ~190 eval samples |
| **Code Changes Needed** | **None** — handles any number of eval samples |
| **GPU Needed** | L4 24GB (4-bit quantized inference) |
| **Estimated Time** | ~30-50 minutes for ~190 samples |
| **Notes** | Run AFTER Phase 3 re-run. Will produce meaningful metrics. |

---

### Phase 5 — Fine-Tuning MedGemma
| Field | Value |
|-------|-------|
| **Status** | `NEEDS RE-RUN` ⚠️ (ran on 15 train samples; need full training set) |
| **Colab Notebook** | `colab_run_notebooks/phase_5_colab.ipynb` |
| **Previous Run (dev)** | 15 samples, 2.1 min, loss 1.1828 |
| **Expected Full Run** | ~760 train, ~95 val, est. 1.5-3 hours on A100 |
| **Code Changes Applied** | `save_steps` fixed: 500 → 100 (ensures checkpoint during training) |
| **Config** | DoRA r=16, α=32, batch=1, grad_accum=8, lr=2e-4, epochs=3, max_patches=4 |
| **GPU Needed** | **A100 40GB** (ONLY phase needing A100) |
| **Estimated Steps** | ~285 total (95 steps/epoch × 3 epochs) |
| **Notes** | Run AFTER Phase 3 re-run. SSD pre-copy handles full dataset. Eval every 100 steps. |

---

### Phase 6 — Evaluation + Calibration
| Field | Value |
|-------|-------|
| **Status** | `NOT STARTED` |
| **Source Code** | `colab_notebooks/Phase_6_Evaluation.py` (ready — bug fixed) |
| **Bug Fix Applied** | `total_mem` → `total_memory` (line 77) |
| **GPU Needed** | L4 24GB (inference only) |
| **Expected Time** | ~30-50 min for inference + metrics |
| **Target Metrics** | CD274 AUC >0.70, MSI AUC >0.75, TME Acc >0.65, TIL ρ >0.60, ECE <0.10 |
| **Notes** | Convert .py to .ipynb or run cells manually. Loads Phase 4 zero-shot for comparison. |

---

### Phase 7 — Integration Pipeline
| Field | Value |
|-------|-------|
| **Status** | `NOT STARTED` |
| **Colab Notebook** | TBD |
| **Guideline Engine** | Not yet |
| **TxGemma Integration** | Not yet |
| **End-to-End Pipeline** | Not yet |
| **Notes** | Rule-based NCCN guidelines + TxGemma for drug explanations only |

---

### Phase 8 — Demo Application
| Field | Value |
|-------|-------|
| **Status** | `NOT STARTED` |
| **Colab Notebook** | TBD |
| **Gradio App** | Not yet |
| **HF Spaces Deployed** | No |
| **Notes** | |

---

### Phase 9 — Submission
| Field | Value |
|-------|-------|
| **Status** | `NOT STARTED` |
| **Writeup** | Not yet |
| **Video** | Not yet |
| **Model on HuggingFace** | Not yet |
| **Kaggle Submission** | Not yet |
| **Notes** | Deadline: Feb 24, 2026 11:59 PM UTC |

---

## Execution Order for Full Run

```
Phase 3 (CPU, ~5-10 min)     → Creates full JSONL files
    ↓
Phase 4 (L4, ~30-50 min)     → Full zero-shot baseline metrics
    ↓
Phase 5 (A100, ~1.5-3 hours) → Fine-tune on full training set
    ↓
Phase 6 (L4, ~30-50 min)     → Full evaluation + calibration
    ↓
Phase 7-9 (L4/CPU)           → Integration, demo, submission
```

**Total estimated compute: ~3-5 hours GPU time**

---

## Key Decisions Log

| Date | Phase | Decision | Reasoning |
|------|-------|----------|-----------|
| 2026-02-10 | Phase 0 | Architecture: **Multi-image** (4-8 patches/call) | All multi-image tests passed (1,2,4,8); 100% JSON parse rate |
| 2026-02-10 | Phase 1 | Data scope: **LUAD + LUSC** (950 matched patients) | Both lung cancer types available |
| 2026-02-10 | Phase 2 | Immune phenotype split: **desert/inflamed/excluded** | Based on TIL + CD8 signature z-scores |
| 2026-02-11 | Phase 5 | Quantization: **QDoRA 4-bit NF4** | Required for A100 40GB |
| 2026-02-11 | Phase 5 | Optimizer: **adamw_bnb_8bit** | Much lower VRAM than standard AdamW |
| 2026-02-12 | Phase 5 | grad_accum: **8**, lr: **2e-4** | Better throughput for full dataset |
| 2026-02-12 | All | Key normalization for model outputs | `normalize_prediction_keys()` maps to canonical schema |
| 2026-02-16 | Phase 5 | save_steps: **100** (was 500) | Ensures checkpoints saved during full training run |
| 2026-02-16 | Phase 6 | Bug fix: `total_mem` → `total_memory` | PyTorch API fix |

---

## Issues / Blockers

| Date | Phase | Issue | Resolution |
|------|-------|-------|------------|
| 2026-02-10 | Phase 2 | TCGA-44-7661 yielded 0 patches | Likely artifact/pen marks; slide excluded |
| 2026-02-12 | Phase 3 | MSI labels: 0 MSI-H found (Thorsson C6 wrong for lung) | Fixed: Bagaev MSI → cBioPortal MSIsensor/MANTIS → Thorsson C6 fallback |
| 2026-02-11 | Phase 4 | CD274 eval got 0 samples (key mismatch) | Fixed with `normalize_prediction_keys()` |
| 2026-02-16 | Phase 3-5 | Dev subset data used for training | **Action: Re-run Phase 3 → 4 → 5 on full dataset** |
