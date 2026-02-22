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
| **Current Phase** | Phase 5 v3 → Phase 6 v3 (ready to run on Colab) |
| **Last Updated** | 2026-02-18 |
| **Days Remaining** | ~6 (deadline: Feb 24, 2026 11:59 PM UTC) |
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
| **Status** | `COMPLETE` ✅ **FULL DATASET** |
| **Colab Notebook** | `colab_run_notebooks/phase_3_colab.ipynb` |
| **Matched Samples** | **939** (11 unmatched: no_patches) |
| **train.jsonl** | **751 samples** (CD274: 376 low / 375 high) |
| **val.jsonl** | **94 samples** (CD274: 47 high / 47 low) |
| **test.jsonl** | **94 samples** (CD274: 47 high / 47 low) |
| **Patient-level split** | **Verified** — no leakage, stratified by cancer_type + CD274 |
| **TME Distribution (train)** | D: 261, IE: 177, F: 168, IE/F: 124, unknown: 21 |
| **MSI Distribution (train)** | MSS: 745, MSI-H: 4, unknown: 2 |
| **Patches/sample** | mean=8.0, min=1, max=8 |
| **Notes** | By project: LUAD 470 / LUSC 469. All 939 JSONL records verified (valid JSON + valid paths). |

---

### Phase 4 — Zero-Shot Baseline
| Field | Value |
|-------|-------|
| **Status** | `COMPLETE` ✅ (inference done) — `NEEDS RE-RUN` ⚠️ (metrics all 0 due to key normalization bug) |
| **Colab Notebook** | `colab_run_notebooks/phase_4_colab.ipynb` |
| **GPU Used** | NVIDIA L4 (23.7 GB) |
| **Samples Evaluated** | **188** (94 test + 94 val) |
| **JSON Parse Rate** | **94%** (176/188) — good baseline |
| **Avg Inference Time** | 26.4s per sample (~83 min total) |
| **CD274 AUC** | **N/A** (bug: key normalization was case-sensitive, model outputs mixed-case keys) |
| **All Other Metrics** | **N/A** (same bug — 0 samples matched for every metric) |
| **Bug Found & Fixed** | `normalize_prediction_keys` was case-sensitive; model outputs `CD274_RNA_proxy_level`, `MSI_status`, `TIL_fraction` etc. Fixed: now lowercases all keys before mapping. |
| **Notes** | Re-run the METRICS CELL ONLY (cell 19+) after restarting runtime to pick up the fixed `normalize_prediction_keys`. The predictions JSONL is saved and valid — no need to re-run inference. |

---

### Phase 5 — Fine-Tuning MedGemma
| Field | Value |
|-------|-------|
| **Status** | `COMPLETE` ✅ |
| **Colab Notebook** | `colab_run_notebooks/phase_5_colab.ipynb` |
| **GPU Used** | NVIDIA A100-SXM4-40GB |
| **Train Samples** | 751 |
| **Val Samples** | 94 |
| **Total Steps** | 282 (94 steps/epoch × 3 epochs) |
| **Training Time** | 96.3 min (~1.6 hours) |
| **Final Train Loss** | **0.0925** |
| **Best Eval Loss** | **0.0922** (step 200) |
| **Eval Loss @ Step 100** | 0.0946 |
| **Eval Loss @ Step 200** | 0.0922 |
| **Config** | DoRA r=16, α=32, batch=1, grad_accum=8, lr=1e-4, epochs=3, max_patches=4 |
| **Quantization** | QDoRA 4-bit NF4 + double quant |
| **Optimizer** | adamw_bnb_8bit |
| **Gradient Checkpointing** | Yes (use_reentrant=True) |
| **Flash Attention** | Yes (flash_attention_2) |
| **VRAM Before Training** | 5.24 GB allocated, 9.22 GB reserved |
| **Adapters Saved** | `/ImmunoPath/models/immunopath-v1/lora_adapters` |
| **Inference Test** | ✅ Valid JSON with all 11 keys |
| **Overfitting** | None — train/eval loss nearly identical |
| **Notes** | Collator fix applied (unsqueeze 3D pixel_values). SSD pre-copy: 3375 patches in 71s. |

---

### Phase 6 — Evaluation + Calibration
| Field | Value |
|-------|-------|
| **Status** | `v2 COMPLETE — FAILED ALL TARGETS` ⚠️ → `v3 READY TO RUN` |
| **Source Code** | `colab_run_notebooks/phase_6_v3_colab.ipynb` (ready) |
| **v2 Results** | CD274 AUC=0.5638, MSI AUC=0.5000, TME Acc=0.2473, TIL ρ=-0.174, Immune MAE=0.3032 — ALL targets missed |
| **v2 Root Causes** | 1) Gradient signal dilution (9% useful), 2) 1 epoch insufficient, 3) LR 2e-4 too aggressive, 4) PEFT tied weight corruption, 5) MSI class imbalance, 6) No response-only masking |
| **v3 Fixes** | 1) Response-only loss masking (QLoRA paper +2-5%), 2) REMOVED modules_to_save (PEFT #1750/#2864), 3) 3 epochs + early stopping, 4) LR 1e-4, 5) MSI-H 5× oversampling |
| **v3 Speed Opts** | max_length 2048→512, batch 4→8, grad_accum 4→2, eval_batch 2→8, eval 1×/epoch |
| **GPU Needed** | L4 24GB (inference only) |
| **Expected Time** | ~30-50 min for inference + metrics |
| **Target Metrics** | CD274 AUC >0.70, MSI AUC >0.75, TME Acc >0.65, TIL ρ >0.60, ECE <0.10 |
| **Notes** | v3 loads from `models/immunopath-v3/lora_adapters`. No weight tying issues. |

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
Phase 5 (A100, ~20 min)       → Fine-tune on full training set (speed-optimized v3)
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
| 2026-02-18 | Phase 5 | v3: Response-only loss masking | QLoRA paper Table 10: +2-5% MMLU. Google Forum confirms default was train-on-all. |
| 2026-02-18 | Phase 5 | v3: REMOVED modules_to_save | PEFT #1750/#2864: passing both lm_head+embed_tokens UNTIES them. Maintainer recommends removal. |
| 2026-02-18 | Phase 5 | v3: 3 epochs + early stopping (patience=3) | v2 had only 94 steps — insufficient for value learning |
| 2026-02-18 | Phase 5 | v3: LR 1e-4 (was 2e-4) | Prevents catastrophic forgetting over 3 epochs |
| 2026-02-18 | Phase 5 | v3: MSI-H oversampling 5× | v2 predicted ALL MSS (AUC=0.50) due to 98% class imbalance |

---

## Issues / Blockers

| Date | Phase | Issue | Resolution |
|------|-------|-------|------------|
| 2026-02-10 | Phase 2 | TCGA-44-7661 yielded 0 patches | Likely artifact/pen marks; slide excluded |
| 2026-02-12 | Phase 3 | MSI labels: 0 MSI-H found (Thorsson C6 wrong for lung) | Fixed: Bagaev MSI → cBioPortal MSIsensor/MANTIS → Thorsson C6 fallback |
| 2026-02-11 | Phase 4 | CD274 eval got 0 samples (key mismatch) | Fixed with `normalize_prediction_keys()` |
| 2026-02-16 | Phase 3-5 | Dev subset data used for training | **Action: Re-run Phase 3 → 4 → 5 on full dataset** |
