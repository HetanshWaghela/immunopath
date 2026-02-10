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
| **Current Phase** | Phase 3 — Training Data Creation (next) |
| **Last Updated** | 2026-02-10 |
| **Days Remaining** | ~14 (deadline: Feb 24, 2026 11:59 PM UTC) |
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
| **Status** | `NOT STARTED` |
| **Colab Notebook** | `colab_notebooks/Phase_3_Training_Data.ipynb` |
| **train.jsonl samples** | ___ |
| **val.jsonl samples** | ___ |
| **test.jsonl samples** | ___ |
| **Patient-level split** | Verified / Not yet |
| **Label Distribution** | TME: ___, MSI: ___, CD274: ___ |
| **Notes** | |

---

### Phase 4 — Zero-Shot Baseline
| Field | Value |
|-------|-------|
| **Status** | `NOT STARTED` |
| **Colab Notebook** | `colab_notebooks/Phase_4_Zero_Shot_Baseline.ipynb` |
| **CD274 AUC (zero-shot)** | ___ |
| **MSI AUC (zero-shot)** | ___ |
| **TME Accuracy (zero-shot)** | ___ |
| **TIL Spearman ρ (zero-shot)** | ___ |
| **JSON Parse Rate (zero-shot)** | ___% |
| **Notes** | |

---

### Phase 5 — Fine-Tuning MedGemma
| Field | Value |
|-------|-------|
| **Status** | `NOT STARTED` |
| **Colab Notebook** | `colab_notebooks/Phase_5_Fine_Tuning.ipynb` |
| **GPU Used** | A100 / V100 / T4 |
| **VRAM Available** | ___ GB |
| **Quantization** | None (bf16) / QDoRA (4-bit) |
| **Training Time** | ___ hours |
| **Final Train Loss** | ___ |
| **Final Val Loss** | ___ |
| **Checkpoint Path** | `models/immunopath-v1/` |
| **Notes** | |

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
| | Phase 5 | Quantization: ___ | GPU VRAM was ___ GB |

---

## Issues / Blockers

| Date | Phase | Issue | Resolution |
|------|-------|-------|------------|
| 2026-02-10 | Phase 1 | 2/950 RNA-seq downloads failed (GDC 500 errors) | 948/950 succeeded; acceptable loss |
| 2026-02-10 | Phase 1 | MSI labels not directly in downloads | Placeholder created; derive from Thorsson C6 subtypes |
| 2026-02-10 | Phase 2 | TCGA-44-7661 yielded 0 patches from 6144 candidates | Likely artifact/pen marks; slide excluded |
| 2026-02-10 | Phase 2 | Only 20/950 slides tiled (dev subset) | Need to download + tile remaining slides for full training |
