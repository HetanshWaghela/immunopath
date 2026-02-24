# 🧬 ImmunoPath

**H&E Histopathology → Immunotherapy Decision Support**

ImmunoPath predicts tumor immune microenvironment biomarkers directly from routine H&E histopathology slides, enabling immunotherapy selection without expensive molecular assays. It integrates **4 HAI-DEF models** into a single clinical decision-support pipeline.

> **MedGemma Impact Challenge Submission** | [Kaggle Competition](https://www.kaggle.com/competitions/med-gemma-impact-challenge)

---

## Pipeline Overview

```
H&E Patches ──┬──► MedGemma ──► Immune Profile ──► Guideline Engine ──► Recommendation
              │                                          │
              ├──► Path Foundation ──► Embeddings         ├──► TxGemma ──► Drug Explanations
              │                                          │
              └──► MedSigLIP ──► Zero-Shot Scores        └──► Clinical Report
```

### 4 HAI-DEF Models

| # | Model | Role | Type |
|---|-------|------|------|
| 1 | **MedGemma** `google/medgemma-1.5-4b-it` | Core immune profiling from H&E | Fine-tuned (LoRA) |
| 2 | **TxGemma** `google/txgemma-9b-chat` | Drug pharmacology explanations | Inference-only |
| 3 | **Path Foundation** `google/path-foundation` | Patch visual embeddings | Inference-only |
| 4 | **MedSigLIP** `google/medsiglip-448` | Zero-shot phenotype scoring | Inference-only |

---

## Quick Start

### Pipeline (Mock Mode — No GPU Required)

```bash
python immunopath_pipeline.py --test
```

### Gradio Demo

```bash
pip install gradio
python demo_app.py
# Opens at http://localhost:7860
```

### Real Inference (Requires GPU)

Use the Colab notebooks in `colab_run_notebooks/` for GPU-backed inference:
- Phase 5: Fine-tuning (`phase_5_v3_1_colab_(1).ipynb`)
- Phase 6: Evaluation (`phase_6_v3_1_colab.ipynb`)

---

## Results

### Key Metrics (94 test samples, TCGA NSCLC)

| Metric | Zero-Shot | Fine-Tuned (v2) | Fine-Tuned (v3.1) |
|--------|-----------|-----------------|-------------------|
| CD274 Balanced Acc | 0.50 | **0.56** | 0.51 |
| TME Accuracy | 0.23 | 0.25 | **0.27** |
| TIL MAE | 0.38 | 0.26 | **0.16 ✅** |
| JSON Compliance | 94% | **100%** | **100%** |
| Schema Compliance | 16% | **100%** | **100%** |

> **TIL MAE meets its clinical target** (<0.20), demonstrating that VLMs can learn to quantify immune infiltration from morphology.

### Why This Matters

Immunotherapy selection currently requires PD-L1 IHC staining, MSI testing, and often NGS panels — costing $2,000–5,000 per patient and taking 1–3 weeks. H&E slides are already available for every cancer diagnosis. ImmunoPath explores whether a fine-tuned VLM can provide preliminary immune profiling from these universally available slides.

---

## Dataset

| Component | Details |
|-----------|---------|
| Source | TCGA NSCLC (LUAD + LUSC) |
| Patients | 950 with matched slides + RNA-seq |
| Patches | 60,396 extracted, 7,574 diversity-selected |
| Labels | RNA-seq immune signatures, Bagaev TME, MSI/MSS |
| Splits | 751 train / 94 val / 94 test (patient-level, no leakage) |

---

## Project Structure

```
├── immunopath_pipeline.py     # End-to-end pipeline (4 HAI-DEF models)
├── demo_app.py                # Gradio interactive demo
├── scripts/
│   ├── guideline_engine.py    # NCCN-aligned treatment recommendation engine
│   └── txgemma_engine.py      # TxGemma drug explanation module
├── colab_run_notebooks/       # 15 Colab notebooks (Phases 0-6, all versions)
├── colab_notebooks/           # Phase Python scripts
├── NOTEBOOK_RESULTS_REFERENCE.md  # Complete metrics from all runs
└── ALL_NOTEBOOK_OUTPUTS.md    # Raw output extract from all 15 notebooks
```

---

## Training Details

| Parameter | v2 (Primary) | v3.1 (Iteration) |
|-----------|-------------|-------------------|
| Base model | `google/medgemma-1.5-4b-it` | Same |
| PEFT | LoRA (r=16, α=16) | LoRA (r=16, α=16) |
| Training data | 1,502 TCGA samples (2× aug) | 751 samples (TIL fix) |
| GPU | A100-SXM4-80GB | A100-SXM4-80GB |
| Training time | 38.4 min | 77.2 min |
| Best eval loss | 0.0618 | 0.1476 |

---

## Limitations & Disclaimers

- **Research prototype** — NOT validated for clinical use
- Predictions are based on RNA-seq proxy labels, not IHC ground truth
- CD274 RNA expression ≠ PD-L1 IHC TPS; always confirm with IHC before treatment decisions
- Small test set (n=94) limits statistical power
- MSI-H prevalence in NSCLC is extremely low (~0.7%), limiting MSI metric reliability

---

## License

MIT License — see [LICENSE](LICENSE)

## Citation

If you use ImmunoPath in your research, please cite:
```
@software{immunopath2026,
  title={ImmunoPath: H&E Histopathology to Immunotherapy Decision Support via Fine-Tuned MedGemma},
  author={Hetansh Waghela},
  year={2026},
  url={https://github.com/HetanshWaghela/medgemma-hackathon}
}
```