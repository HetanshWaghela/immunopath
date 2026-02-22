# Model Card: ImmunoPath

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | ImmunoPath (v2 / v3.1) |
| **Base Model** | `google/medgemma-1.5-4b-it` |
| **Model Type** | Fine-tuned Vision-Language Model (VLM) |
| **PEFT Method** | LoRA (rank=16, alpha=16) |
| **Language** | English |
| **License** | MIT (adapter weights); base model per Google's terms |
| **Developer** | Hetansh Waghela |
| **Date** | February 2026 |

## Intended Use

**Primary use case:** Research prototype for predicting tumor immune microenvironment biomarkers from H&E histopathology patches in NSCLC (non-small cell lung cancer).

**Intended users:** Computational pathology researchers, AI-for-health researchers, competition evaluators.

**Out-of-scope uses:**
- ❌ Clinical decision-making without confirmatory molecular testing
- ❌ Any diagnostic or treatment decisions
- ❌ Deployment in clinical settings without prospective validation
- ❌ Use on cancer types other than NSCLC without retraining

## Training Data

| Component | Details |
|-----------|---------|
| Source | TCGA (The Cancer Genome Atlas) NSCLC |
| Cancer types | LUAD (adenocarcinoma) + LUSC (squamous cell) |
| Patients | 950 with matched diagnostic slides + RNA-seq |
| Patches | 7,574 diversity-selected from 60,396 extracted (512×512 px, 0.5 µm/px) |
| Train/Val/Test | 751 / 94 / 94 (patient-level stratified, no leakage) |
| Labels | RNA-seq immune signatures (CD274, TIL, immune score), Bagaev TME subtypes, MSI status |

### Label Caveats

- CD274 (PD-L1) labels are RNA expression proxy (median split), **NOT** PD-L1 IHC TPS
- MSI status mapped from Bagaev et al. and cBioPortal — only 7 MSI-H patients in NSCLC
- Immune phenotypes derived from signature thresholds, not manual pathologist annotation

## Evaluation Results

### v2 (Primary Model)
| Metric | Value | Target |
|--------|-------|--------|
| CD274 Balanced Accuracy | 0.5638 | >0.70 |
| MSI AUC | 0.50 | >0.75 |
| TME Accuracy | 0.2473 | >0.65 |
| TIL MAE | 0.2576 | <0.20 |
| JSON Compliance | 100% | — |

### v3.1 (Iteration — TIL Fix)
| Metric | Value | Target |
|--------|-------|--------|
| CD274 Balanced Accuracy | 0.5106 | >0.70 |
| TME Accuracy | 0.2747 | >0.65 |
| TIL MAE | **0.1615** ✅ | <0.20 |
| JSON Compliance | 100% | — |

## Limitations

1. **Not clinically validated** — TCGA data only, no prospective or external validation
2. **Label quality** — RNA-seq proxies, not IHC/flow cytometry ground truth
3. **Small test set** — 94 patients limits statistical power and confidence intervals
4. **Cancer scope** — Trained on NSCLC only; performance on other cancers unknown
5. **MSI-H rarity** — 0.7% prevalence in NSCLC makes MSI prediction statistically unreliable
6. **Calibration** — Model confidence (ECE) is not well-calibrated; temperature scaling helps but ECE remains above 0.10

## Ethical Considerations

- This is a **research prototype** and must NOT be used for clinical decisions
- H&E-based predictions should always be confirmed with molecular testing before treatment
- The model may not generalize across patient demographics or tissue preparation methods
- Potential for false positives/negatives that could lead to inappropriate treatment if used clinically

## Environmental Impact

- Fine-tuning: ~38–77 minutes on A100-80GB GPU (~0.05 kg CO₂e per run)
- Inference: ~5–12 seconds per sample on L4 GPU
