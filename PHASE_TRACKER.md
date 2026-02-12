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
| **Current Phase** | Phase 6 — Evaluation + Calibration (next) |
| **Last Updated** | 2026-02-12 |
| **Days Remaining** | ~12 (deadline: Feb 24, 2026 11:59 PM UTC) |
| **Colab Runtime** | Google Colab Pro (A100 40GB) |
| **Repo** | `/Users/Hetansh/Github/medgemma-impact-challenge` |

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
| **Status** | `COMPLETE` |
| **Colab Notebook** | `colab_run_notebooks/phase_1_colab.ipynb` |
| **TCGA Slides Found** | 1053 (541 LUAD + 512 LUSC) across 956 patients |
| **Matched Patients** | 950 (with both slide + RNA-seq) |
| **TCGA Slides Downloaded** | 20/950 (subset for dev; full = 718.9 GB) |
| **RNA-seq Downloaded** | 948/950 (2 failed — GDC 500 errors) |
| **Bagaev TME Labels** | **Downloaded** — annotation.tsv, signatures.tsv, gene_signatures.gmt |
| **Thorsson Panimmune** | **Downloaded** — immune_signatures_160, leukocyte_fractions, cibersort, mutation_load, subtypes |
| **cBioPortal Data** | **Downloaded** — LUAD (566 samples) + LUSC PanCancer Atlas; MSI_SCORE_MANTIS + MSI_SENSOR_SCORE extracted |
| **Saltz TIL Labels** | **Downloaded** — summary archive |
| **MSI Labels** | Placeholder created; to be derived from Thorsson C6 subtypes in Phase 2 |
| **Notes** | Data saved to Google Drive `/ImmunoPath/data/`. nsclc_metadata.csv created with matched patients. |

---

### Phase 2 — Data Processing
| Field | Value |
|-------|-------|
| **Status** | `COMPLETE` |
| **Colab Notebook** | `colab_run_notebooks/phase_2_colab.ipynb` |
| **GPU Used** | L4 (Colab) |
| **Patches Extracted** | **1216 patches** from **19/20 slides** (1 slide had 0 valid patches after filtering) |
| **Patch Settings** | 512×512 at 0.5 µm/px (≈20×), max 64 patches/slide, tissue threshold 0.5 |
| **Reinhard Normalization** | **Applied** — LAB color space, verified with ref stats |
| **Immune Signatures Computed** | **Yes** — 5 signatures (cd8, ifng, til, pdl1, exhaustion) from 948 RNA-seq samples |
| **CD274 Expression** | Median log2(TPM+1): 3.154 → 474 high / 474 low |
| **Immune Phenotypes** | desert: 474, inflamed: 382, excluded: 92 |
| **Immune Score** | Mean: 0.483, Median: 0.477, Std: 0.190 (0-1 normalized) |
| **immune_signatures.csv rows** | **948** (9 columns) |
| **Notes** | 1 slide (TCGA-44-7661) produced 0 patches from 6144 candidates (likely artifact/pen marks). All 34/34 target immune genes found in expression matrix. QC image saved. |

---

### Phase 3 — Training Data Creation
| Field | Value |
|-------|-------|
| **Status** | `COMPLETE` |
| **Colab Notebook** | `colab_run_notebooks/phase_3_colab.ipynb` |
| **train.jsonl samples** | 15 (dev subset) |
| **val.jsonl samples** | 1 (dev subset) |
| **test.jsonl samples** | 3 (dev subset) |
| **Patient-level split** | Verified (no leakage) |
| **Label Distribution** | TME: D=3133, IE=1912, F=1718, IE/F=1261 (pan-TCGA Bagaev) |
| **MSI Labels** | MSI-H=0, MSS=1045 (NSCLC is mostly MSS) |
| **Split Ratios** | 80/10/10 (train/val/test) |
| **Notes** | Dev subset: 19 matched patients only. Full dataset will have ~760 train / ~95 val / ~95 test samples. |

---

### Phase 4 — Zero-Shot Baseline
| Field | Value |
|-------|-------|
| **Status** | `COMPLETE` |
| **Colab Notebook** | `colab_run_notebooks/phase_4_colab.ipynb` |
| **CD274 AUC (zero-shot)** | N/A (0 matched samples — model output used wrong key name) |
| **MSI Accuracy (zero-shot)** | 0.25 (4 samples) |
| **TME Accuracy (zero-shot)** | 0.25, Macro-F1=0.13 (4 samples) |
| **TIL Spearman ρ (zero-shot)** | -0.6325 (4 samples, p=0.37) |
| **JSON Parse Rate (zero-shot)** | 100% |
| **Notes** | CD274 eval got 0 samples because model outputs `cd274_rna_proxy_level` instead of `cd274_expression`. Fixed with key normalization for next run. MSI all MSS in ground truth so AUC N/A. Very small test set (4 samples) makes metrics unreliable. |

---

### Phase 5 — Fine-Tuning MedGemma
| Field | Value |
|-------|-------|
| **Status** | `COMPLETE` (dev subset) |
| **Colab Notebook** | `colab_run_notebooks/phase_5_colab.ipynb` |
| **GPU Used** | NVIDIA A100-SXM4-40GB |
| **VRAM Available** | 42.4 GB (10.29 GB allocated after model load) |
| **Quantization** | QDoRA (4-bit NF4 + double quant) |
| **Training Samples** | 15 |
| **Training Time** | 2.1 minutes |
| **Final Train Loss** | 1.1828 |
| **Final Val Loss** | N/A (eval disabled for OOM safety) |
| **Trainable Params** | 33,891,456 (0.78% of 4.33B) |
| **Checkpoint Path** | `/content/drive/MyDrive/ImmunoPath/models/immunopath-v1/` |
| **Notes** | OOM fixes applied: batch=1, grad_accum=8, max_patches=4, max_length=1536, 336x336 resize, adamw_bnb_8bit. Fine-tuned model produces valid JSON with ~correct keys. Local SSD pre-copy added for next run. |

---

### Phase 6 — Evaluation + Calibration
| Field | Value |
|-------|-------|
| **Status** | `NOT STARTED` |
| **Colab Notebook** | `colab_notebooks/Phase_6_Evaluation.ipynb` |
| **CD274 AUC (fine-tuned)** | ___ (vs zero-shot: ___) |
| **MSI AUC (fine-tuned)** | ___ (vs zero-shot: ___) |
| **TME Accuracy (fine-tuned)** | ___ (vs zero-shot: ___) |
| **TIL Spearman ρ (fine-tuned)** | ___ (vs zero-shot: ___) |
| **ECE (before calibration)** | ___ |
| **ECE (after temp scaling)** | ___ |
| **Temperature T** | ___ |
| **Ablations Run** | Patches: ___ / Stain: ___ / DoRA vs LoRA: ___ |
| **Notes** | |

---

### Phase 7 — Integration Pipeline
| Field | Value |
|-------|-------|
| **Status** | `NOT STARTED` |
| **Colab Notebook** | `colab_notebooks/Phase_7_Integration.ipynb` |
| **Guideline Engine** | Working / Not yet |
| **TxGemma Integration** | Working / Skipped / Mock |
| **End-to-End Pipeline** | Working / Not yet |
| **Guideline Concordance** | ___% |
| **Notes** | |

---

### Phase 8 — Demo Application
| Field | Value |
|-------|-------|
| **Status** | `NOT STARTED` |
| **Colab Notebook** | `colab_notebooks/Phase_8_Demo.ipynb` |
| **Gradio App** | Working / Mock mode only |
| **HF Spaces Deployed** | Yes / No |
| **HF Spaces URL** | ___ |
| **Notes** | |

---

### Phase 9 — Submission
| Field | Value |
|-------|-------|
| **Status** | `NOT STARTED` |
| **Writeup** | Draft / Final / Submitted |
| **Video** | Recorded / Edited / Uploaded |
| **Model on HuggingFace** | Uploaded / Not yet |
| **Model Card** | Written / Not yet |
| **Kaggle Notebook** | Created / Not yet |
| **Submitted on Kaggle** | Yes / No |
| **Notes** | |

---

## Key Decisions Log

| Date | Phase | Decision | Reasoning |
|------|-------|----------|-----------|
| 2026-02-10 | Phase 0 | Architecture: **Multi-image** (4-8 patches/call) | All multi-image tests passed (1,2,4,8); 100% JSON parse rate |
| 2026-02-10 | Phase 1 | Data scope: **LUAD + LUSC** (950 matched patients) | Both lung cancer types available; 20 slides downloaded for dev |
| 2026-02-10 | Phase 2 | Immune phenotype split: **desert/inflamed/excluded** | Based on TIL + CD8 signature z-scores; balanced desert(474)/inflamed(382)/excluded(92) |
| 2026-02-11 | Phase 5 | Quantization: **QDoRA 4-bit NF4** | Required for A100 40GB; OOM with full bf16 fine-tuning |
| 2026-02-11 | Phase 5 | Optimizer: **adamw_bnb_8bit** | Much lower VRAM than standard AdamW |
| 2026-02-12 | Phase 5 | grad_accum: **8** (was 16), lr: **2e-4** (was 1e-4) | Better throughput for full dataset; higher lr compensates smaller effective batch |
| 2026-02-12 | All | Key normalization for model outputs | Model produces variant key names; `normalize_prediction_keys()` maps to canonical schema |

---

## Issues / Blockers

| Date | Phase | Issue | Resolution |
|------|-------|-------|------------|
| 2026-02-10 | Phase 1 | 2/950 RNA-seq downloads failed (GDC 500 errors) | 948/950 succeeded; acceptable loss |
| 2026-02-10 | Phase 1 | MSI labels not directly in downloads | Placeholder created; derive from Thorsson C6 subtypes |
| 2026-02-10 | Phase 2 | TCGA-44-7661 yielded 0 patches from 6144 candidates | Likely artifact/pen marks; slide excluded |
| 2026-02-10 | Phase 2 | Only 20/950 slides tiled (dev subset) | Need to download + tile remaining slides for full training |
| 2026-02-12 | Phase 2 | Patch selection was random (no diversity) | Added K-Means clustering on color histograms → 8 diverse patches/slide |
| 2026-02-12 | Phase 3 | MSI labels: 0 MSI-H found (Thorsson C6 wrong for lung) | Fixed: Bagaev MSI column → cBioPortal MSIsensor/MANTIS scores → Thorsson C6 fallback |
| 2026-02-11 | Phase 4 | CD274 eval got 0 samples (key mismatch) | Model outputs `cd274_rna_proxy_level`; fixed with `normalize_prediction_keys()` |
| 2026-02-11 | Phase 5 | CUDA OOM on A100 with original config | Fixed: batch=1, grad_accum=8, max_patches=4, max_length=1536, 336x336, adamw_bnb_8bit |
| 2026-02-11 | Phase 5 | 2.1 min for 15 samples (GDrive I/O bottleneck) | Added local SSD pre-copy step for next run |
| 2026-02-11 | Phase 6 | .py loaded model without 4-bit quant (mismatch with training) | Fixed: added BitsAndBytesConfig to Phase 6 model loading |
