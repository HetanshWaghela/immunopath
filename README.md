# ImmunoPath

**Predicting immunotherapy response from routine H&E histopathology using fine-tuned MedGemma.**

ImmunoPath predicts 8 tumor immune microenvironment biomarkers directly from hematoxylin and eosin (H&E) stained tissue slides, providing a low-cost triage signal for immunotherapy eligibility without requiring PD-L1 immunohistochemistry, MSI testing, or NGS panels. It orchestrates 4 HAI-DEF (Health AI Developer Foundation) models into a single clinical decision-support pipeline.

Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) on Kaggle.

**Live demo:** [huggingface.co/spaces/hetanshwaghela/ImmunoPath](https://huggingface.co/spaces/hetanshwaghela/ImmunoPath) (no GPU required)

---

## The Problem

Identifying immunotherapy candidates in lung cancer requires PD-L1 IHC staining ($300–800), MSI testing ($200–500), or a full NGS panel ($2,000–5,000) — tests that take 1–3 weeks and are unavailable in most low- and middle-income country (LMIC) hospitals. H&E slides cost $5–15, are prepared same-day, and exist for every cancer patient.

ImmunoPath answers: **"Which patients should I send to the reference lab first?"**

---

## Pipeline

```
H&E patch (512x512px)
        |
        +---> MedGemma 4B (fine-tuned) -----> 8 immune biomarkers (JSON)
        |                                              |
        +---> Path Foundation -----> embeddings        +--> Guideline Engine --> recommendation
        |                                              |
        +---> MedSigLIP -----------> zero-shot score   +--> TxGemma --> drug pharmacology
```

| # | Model | Role | Mode |
|---|-------|------|------|
| 1 | MedGemma `google/medgemma-1.5-4b-it` | H&E to 8 immune biomarkers (structured JSON) | Fine-tuned (LoRA) |
| 2 | TxGemma `google/txgemma-9b-chat` | Drug mechanism, ADMET, toxicity | Inference |
| 3 | Path Foundation `google/path-foundation` | 384-dim patch visual embeddings | Inference |
| 4 | MedSigLIP `google/medsiglip-448` | Zero-shot immune phenotype scoring | Inference |

---

## Results

Evaluated on 94 held-out patients (TCGA NSCLC, patient-level split, no leakage):

| Metric | Zero-Shot | v2 | v3.1 | Target |
|--------|-----------|----|----- |--------|
| CD274 Balanced Accuracy | 0.50 | **0.56** | 0.51 | >0.70 |
| TME Accuracy | 0.23 | 0.25 | **0.27** | >0.65 |
| TIL Mean Absolute Error | 0.38 | 0.26 | **0.16** | <0.20 |
| TIL Spearman | 0.006 | -0.17 | **+0.12** | >0.60 |
| JSON Compliance | 94% | **100%** | **100%** | -- |
| Schema Compliance | 16% | **100%** | **100%** | -- |

v3.1 is the primary adapter. TIL MAE of 0.16 meets the clinical relevance threshold. The key change from v2 to v3.1 was response-only loss masking -- restricting gradient computation to only the JSON biomarker response tokens, which reduced TIL MAE by 38%.

---

## Fine-Tuned Adapters

Both adapters are open-weight on HuggingFace and trace directly to `google/medgemma-1.5-4b-it`:

| Adapter | Best metric | HuggingFace |
|---------|-------------|-------------|
| v3.1 (primary) | TIL MAE 0.16 | [hetanshwaghela/immunopath-medgemma-v3.1](https://huggingface.co/hetanshwaghela/immunopath-medgemma-v3.1) |
| v2 | CD274 Acc 0.56 | [hetanshwaghela/immunopath-medgemma-v2](https://huggingface.co/hetanshwaghela/immunopath-medgemma-v2) |

---

## Quick Start

**Local demo (no GPU):**

```bash
pip install gradio
python demo_app.py
# Opens at http://localhost:7860
```

**Pipeline test (mock mode):**

```bash
python immunopath_pipeline.py --test
```

**Real inference (GPU required):** Use the Colab notebooks in `colab_run_notebooks/`:
- Fine-tuning: `phase_5_v3_1_colab_(1).ipynb`
- Evaluation: `phase_6_v3_1_colab.ipynb`

---

## Training

| Parameter | v2 | v3.1 |
|-----------|----|------|
| Base model | `google/medgemma-1.5-4b-it` | `google/medgemma-1.5-4b-it` |
| PEFT method | LoRA (r=16, alpha=16) | LoRA (r=16, alpha=16) |
| Loss masking | Full sequence | Response-only (key change) |
| Trainable params | 46.56% (modules_to_save) | 2.37% |
| Training samples | 1,502 (2x augmentation) | 751 |
| GPU | A100-SXM4-80GB | A100-SXM4-80GB |
| Training time | 38.4 min | 77.2 min |

---

## Dataset

| Component | Details |
|-----------|---------|
| Source | TCGA NSCLC (470 LUAD + 469 LUSC) |
| Patients | 950 with matched H&E slides and RNA-seq |
| Patches | 60,396 extracted at 20x, 7,574 diversity-selected via K-Means |
| Labels | RNA-seq immune signatures: CD274 expression, 8-gene TIL signature, Bagaev TME subtypes, MSI status |
| Splits | 751 train / 94 val / 94 test, patient-level stratified, zero leakage |

---

## Project Structure

```
immunopath_pipeline.py      -- End-to-end 4-model pipeline
demo_app.py                 -- Gradio interactive demo
scripts/
    guideline_engine.py     -- NCCN-aligned treatment recommendation engine
    txgemma_engine.py       -- TxGemma drug explanation module
colab_run_notebooks/        -- Colab notebooks for all phases (0-6)
colab_notebooks/            -- Phase Python scripts
huggingface_space/          -- HuggingFace Space app source
submission/                 -- Kaggle writeup, technical writeup, model cards
```

---

## Limitations

- Research prototype. Not validated for clinical use.
- CD274 labels derive from bulk RNA-seq (median split), not PD-L1 IHC TPS. Always confirm with IHC before any treatment decision.
- Small test set (n=94) limits statistical power.
- MSI-H prevalence in NSCLC is ~0.7%; MSI metrics are statistically unreliable at this sample size.
- TxGemma 9B hits VRAM limits on free-tier T4 GPUs when running alongside MedGemma 4B; the pipeline falls back to a curated pharmacology knowledge base in that scenario.

---

## License

MIT -- see [LICENSE](LICENSE)

## Citation

```bibtex
@software{immunopath2026,
  title={ImmunoPath: H&E Histopathology to Immunotherapy Decision Support via Fine-Tuned MedGemma},
  author={Hetansh Waghela},
  year={2026},
  url={https://github.com/hetanshwaghela/immunopath}
}
```