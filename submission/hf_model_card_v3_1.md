---
base_model: google/medgemma-1.5-4b-it
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:google/medgemma-1.5-4b-it
- lora
- sft
- transformers
- trl
- medical
- pathology
- immunotherapy
- TCGA
- NSCLC
---

# ImmunoPath v3.1: H&E Histopathology to Immune Biomarker Prediction

LoRA adapter that fine-tunes MedGemma to predict 8 tumor immune microenvironment biomarkers from routine H&E histopathology patches. Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) on Kaggle.

This is the **v3.1 adapter** (primary/recommended). It achieves the best TIL prediction (MAE 0.16, meeting the clinical target of <0.20) and the best TME accuracy (0.27). For the version with best CD274 balanced accuracy, see [immunopath-medgemma-v2](https://huggingface.co/hetanshwaghela/immunopath-medgemma-v2).

## Model Details

### Model Description

ImmunoPath v3.1 fine-tunes `google/medgemma-1.5-4b-it` using LoRA with **response-only loss masking** to predict immune biomarkers from H&E-stained tissue patches. The model takes a histopathology image and a structured prompt, then outputs a JSON object with 8 biomarker predictions: CD274 expression level, MSI status, TME subtype, TIL fraction, TIL density, immune phenotype, CD8 infiltration level, and a composite immune score.

Compared to v2, this version makes three targeted changes based on ablation analysis:

1. **Removed `modules_to_save`:** All learning flows through LoRA adapters only (2.37% trainable params vs. 46.56% in v2), resolving a PEFT weight-tying bug.
2. **Response-only loss masking:** The gradient focuses exclusively on the JSON biomarker output, not the instruction tokens. This concentrates the training signal 3.4x and was the single largest contributor to the TIL MAE improvement.
3. **Sigmoid-normalized TIL targets:** Raw RNA-seq z-scores (range [-2, +3]) are bounded to [0, 1] with sigmoid, matching the expected output range.

The training signal comes from RNA-seq-derived immune signatures (not manual pathologist annotation), making this a novel cross-modal supervision approach: teaching a vision-language model to infer molecular-level immune properties from tissue morphology.

- **Developed by:** Hetansh Waghela
- **Model type:** Vision-Language Model (VLM) with LoRA adapters
- **Language(s):** English
- **License:** MIT (adapter weights); base model under Google's MedGemma terms
- **Finetuned from model:** [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)

### Model Sources

- **Repository:** [github.com/hetanshwaghela/immunopath](https://github.com/hetanshwaghela/immunopath)
- **Demo:** [HuggingFace Space](https://huggingface.co/spaces/hetanshwaghela/ImmunoPath)

## Uses

### Direct Use

Load with PEFT and run inference on H&E histopathology patches (512x512 px, ideally at 20x magnification / 0.5 um/px) to get structured immune biomarker predictions as JSON. The model expects a specific prompt format (see code example below).

### Downstream Use

Plug into a clinical decision-support pipeline alongside a guideline engine, TxGemma for drug pharmacology, Path Foundation for embedding-based similarity, and MedSigLIP for zero-shot phenotype scoring. The full ImmunoPath pipeline is demonstrated in the Kaggle notebooks linked in the repository.

### Out-of-Scope Use

- **Clinical decision-making.** This is a research prototype. Do not use predictions for treatment decisions without confirmatory molecular testing (PD-L1 IHC, MSI PCR/NGS).
- **Non-NSCLC cancers.** Trained exclusively on TCGA NSCLC (LUAD + LUSC). Performance on other cancer types is unknown and expected to be poor.
- **Non-H&E stains.** The model was trained on standard H&E-stained tissue only. IHC, IF, or other staining methods are out of scope.
- **Regulatory or diagnostic contexts.** Not validated for any regulated use. Research Use Only.

## Bias, Risks, and Limitations

- **TCGA training bias.** TCGA overrepresents US academic medical centers and certain demographics. The model may not generalize to tissue from non-US populations, different scanners, or different tissue preparation protocols.
- **RNA-seq proxy labels.** CD274 labels are based on RNA expression (median split), NOT PD-L1 IHC TPS scoring. RNA expression and protein-level staining correlate imperfectly. Clinical decisions about immunotherapy eligibility are based on IHC, not RNA.
- **Small test set.** 94 patients in the held-out test split limits statistical power and confidence intervals.
- **MSI-H rarity.** Only 7 MSI-H patients across 939 matched samples (0.7% prevalence in NSCLC). The v3.1 test split contains 0 MSI-H patients, making MSI metrics statistically unreliable for this version.
- **Calibration.** ECE is 0.52 after temperature scaling (poor). Model confidence values should not be interpreted as calibrated probabilities. Proper calibration requires logit-based methods, not heuristic confidence scores.

### Recommendations

All predictions must be confirmed with molecular testing before any clinical action. This model should be treated as a research triage tool: it can help prioritize which patients to send for confirmatory testing, but it cannot replace those tests. Users should pay attention to the known calibration limitation and not treat the confidence field as a probability.

## How to Get Started with the Model

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

MODEL_ID = "google/medgemma-1.5-4b-it"
ADAPTER_REPO = "hetanshwaghela/immunopath-medgemma-v3.1"

# Load base model (4-bit quantized)
processor = AutoProcessor.from_pretrained(MODEL_ID)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
model.eval()

# Run inference on a histopathology patch
image = Image.open("patch_512x512.png").convert("RGB")

prompt = """Analyze this H&E histopathology patch from a lung cancer biopsy.
Predict the tumor immune microenvironment and return a JSON object with these fields:
- cd274_expression: "high" or "low"
- msi_status: "MSI-H" or "MSS"
- tme_subtype: one of "IE", "IE/F", "F", "D"
- til_fraction: float 0-1
- til_density: "high", "moderate", or "low"
- immune_phenotype: "inflamed", "excluded", or "desert"
- cd8_infiltration: "high", "moderate", or "low"
- immune_score: float 0-1"""

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)

response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)  # JSON with 8 biomarker predictions
```

## Training Details

### Training Data

- **Source:** TCGA NSCLC (LUAD + LUSC), accessed via GDC Data Portal
- **Patients:** 950 with matched diagnostic H&E slides and RNA-seq profiles
- **Patches:** 60,396 extracted at 0.5 um/px (20x), 512x512 px, Reinhard stain normalization. Diversity-selected to 7,574 via K-Means (K=8/slide).
- **Labels:** RNA-seq-derived immune signatures: CD274 gene expression (median split), 8-gene TIL signature (z-score then **sigmoid-normalized to [0,1]** in v3.1), Bagaev TME subtypes, MSI status from Bagaev + cBioPortal + Thorsson.
- **Splits:** 751 train / 94 val / 94 test (patient-level stratified by cancer type x CD274 status, zero leakage)
- **No augmentation** in v3.1 (751 training samples, unlike v2's 2x augmentation)

### Training Procedure

#### Preprocessing

Images resized and normalized per MedGemma's processor defaults. Prompts formatted as chat-template conversations with image tokens. **Response-only loss masking**: only tokens in the JSON response contribute to the loss; prompt and instruction tokens are masked out.

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision
- **PEFT method:** LoRA (rank=16, alpha=16, dropout=0.05)
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **modules_to_save:** None (key difference from v2)
- **Trainable parameters:** 2.37%
- **Loss masking:** Response-only (prompt tokens masked)
- **Optimizer:** AdamW
- **Learning rate:** 2e-4 with cosine schedule
- **Warmup:** 10% of steps
- **Weight decay:** 0.01
- **Batch size:** 1 (gradient accumulation 8, effective batch 8)
- **Epochs:** 3 with early stopping (patience 2)
- **Max sequence length:** 1536 tokens

#### Speeds, Sizes, Times

- **GPU:** NVIDIA A100-SXM4-80GB (Google Colab)
- **Training time:** 77.2 minutes
- **Best eval loss:** 0.1476

Note: eval loss is higher than v2's (0.0618) because response-only masking means the loss denominator is smaller (only response tokens). Despite higher scalar loss, the biological metric profile is substantially better. Lowest eval loss does not equal best clinical utility.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

94-patient held-out test set from TCGA NSCLC. Patient-level split ensures no patch from any test patient appeared during training.

#### Factors

Evaluation is disaggregated by biomarker type (binary classification for CD274, multi-class for TME, continuous for TIL/immune-score). MSI is not reliably evaluable in this version because the test split contains 0 MSI-H patients.

#### Metrics

- **CD274:** Balanced accuracy (accounts for class imbalance at median split)
- **TME:** Multi-class accuracy across 4 subtypes (IE, IE/F, F, D)
- **TIL:** Mean Absolute Error (lower is better) and Spearman rank correlation
- **Immune-score:** Mean Absolute Error
- **JSON/Schema compliance:** Percentage of outputs that parse as valid JSON matching the expected schema

### Results

| Metric | Value | Target | vs. v2 |
|--------|-------|--------|--------|
| CD274 Balanced Accuracy | 0.5106 | >0.70 | v2 better (0.56) |
| TME Accuracy | **0.2747** | >0.65 | Best across versions |
| TIL MAE | **0.1615** | <0.20 | **Meets target** (v2: 0.26) |
| TIL Spearman | **+0.1194** | >0.60 | Positive (v2: -0.17) |
| Immune-score MAE | **0.2071** | <0.20 | Near target (v2: 0.30) |
| ECE (after temp scaling) | 0.524 | <0.10 | v2 better (0.11) |
| JSON Compliance | **100%** | - | Same |
| Schema Compliance | **100%** | - | Same |

#### Summary

v3.1 is the recommended adapter for general use. TIL MAE of 0.16 meets the clinical target, with positive Spearman correlation (+0.12) confirming directional accuracy. TME accuracy is the best across all versions. The tradeoff is a slight CD274 regression (0.51 vs. 0.56) and worse calibration (ECE 0.52 vs. 0.11). Response-only loss masking was the single largest contributor to the TIL improvement, reducing MAE by 38% compared to v2.

## Environmental Impact

- **Hardware Type:** NVIDIA A100-SXM4-80GB
- **Hours used:** ~1.29
- **Cloud Provider:** Google Colab
- **Compute Region:** US
- **Carbon Emitted:** ~0.10 kg CO2eq (estimated via ML Impact Calculator)

## Technical Specifications

### Model Architecture and Objective

MedGemma 1.5 4B-IT is a Gemma-based vision-language model with a SigLIP vision encoder. LoRA adapters are applied to all attention and MLP projection layers. The training objective is causal language modeling loss (cross-entropy) with response-only masking (prompt tokens excluded from loss computation).

### Compute Infrastructure

#### Hardware

NVIDIA A100-SXM4-80GB (single GPU)

#### Software

- transformers 4.x
- peft 0.18.1
- trl (SFTTrainer with DataCollatorForCompletionOnlyLM for response-only masking)
- bitsandbytes (4-bit NF4 quantization)
- torch 2.x with CUDA
- Pillow <12.0 (12.x breaks torchvision ANTIALIAS)

## Citation

**BibTeX:**

```bibtex
@software{immunopath2026,
  title={ImmunoPath: H&E Histopathology to Immunotherapy Decision Support via Fine-Tuned MedGemma},
  author={Hetansh Waghela},
  year={2026},
  url={https://github.com/hetanshwaghela/immunopath}
}
```

## Model Card Authors

Hetansh Waghela

## Model Card Contact

Via GitHub issues at [github.com/hetanshwaghela/immunopath](https://github.com/hetanshwaghela/immunopath)

### Framework versions

- PEFT 0.18.1
