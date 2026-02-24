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

# ImmunoPath v2: H&E Histopathology to Immune Biomarker Prediction

LoRA adapter that fine-tunes MedGemma to predict 8 tumor immune microenvironment biomarkers from routine H&E histopathology patches. Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) on Kaggle.

This is the **v2 adapter**. It has the best CD274 (PD-L1) balanced accuracy (0.56) across all versions. For the best TIL prediction performance, see [immunopath-medgemma-v3.1](https://huggingface.co/hetanshwaghela/immunopath-medgemma-v3.1).

## Model Details

### Model Description

ImmunoPath v2 fine-tunes `google/medgemma-1.5-4b-it` using LoRA with `modules_to_save=['lm_head', 'embed_tokens']` to predict immune biomarkers from H&E-stained tissue patches. The model takes a histopathology image and a structured prompt, then outputs a JSON object with 8 biomarker predictions: CD274 expression level, MSI status, TME subtype, TIL fraction, TIL density, immune phenotype, CD8 infiltration level, and a composite immune score.

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
- **MSI-H rarity.** Only 7 MSI-H patients across 939 matched samples (0.7% prevalence in NSCLC). MSI predictions are statistically unreliable.
- **Weight-tying interaction.** This version uses `modules_to_save=['lm_head', 'embed_tokens']`, which causes 46.56% of parameters to become trainable and triggers a known PEFT weight-tying issue (see [peft#1750](https://github.com/huggingface/peft/issues/1750), [peft#2864](https://github.com/huggingface/peft/issues/2864)). This was resolved in v3.1 by removing `modules_to_save`.
- **Calibration.** Model confidence values are not well-calibrated. ECE improved from 0.37 to 0.11 after temperature scaling, but true probability calibration requires logit-based methods.

### Recommendations

All predictions must be confirmed with molecular testing before any clinical action. This model should be treated as a research triage tool: it can help prioritize which patients to send for confirmatory testing, but it cannot replace those tests.

## How to Get Started with the Model

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

MODEL_ID = "google/medgemma-1.5-4b-it"
ADAPTER_REPO = "hetanshwaghela/immunopath-medgemma-v2"

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
- **Labels:** RNA-seq-derived immune signatures: CD274 gene expression (median split), 8-gene TIL signature (raw z-scores, not normalized), Bagaev TME subtypes, MSI status from Bagaev + cBioPortal + Thorsson. Sigmoid normalization of TIL targets was introduced in v3.1.
- **Splits:** 751 train / 94 val / 94 test (patient-level stratified by cancer type x CD274 status, zero leakage)
- **Augmentation:** 2x augmentation (1,502 training samples for v2)

### Training Procedure

#### Preprocessing

Images resized and normalized per MedGemma's processor defaults. Prompts formatted as chat-template conversations with image tokens. Full-sequence loss (prompt + response tokens).

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision
- **PEFT method:** LoRA (rank=16, alpha=16, dropout=0.05)
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **modules_to_save:** lm_head, embed_tokens
- **Trainable parameters:** 46.56% (due to modules_to_save; see Limitations)
- **Optimizer:** AdamW
- **Learning rate:** 2e-4 with cosine schedule
- **Warmup:** 10% of steps
- **Weight decay:** 0.01
- **Batch size:** 1 (gradient accumulation 8, effective batch 8)
- **Epochs:** 1
- **Max sequence length:** 1536 tokens

#### Speeds, Sizes, Times

- **GPU:** NVIDIA A100-SXM4-80GB (Google Colab)
- **Training time:** 38.4 minutes
- **Best eval loss:** 0.0618

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

94-patient held-out test set from TCGA NSCLC. Patient-level split ensures no patch from any test patient appeared during training.

#### Factors

Evaluation is disaggregated by biomarker type (binary classification for CD274/MSI, multi-class for TME, continuous for TIL/immune-score).

#### Metrics

- **CD274:** Balanced accuracy (accounts for class imbalance at median split)
- **TME:** Multi-class accuracy across 4 subtypes (IE, IE/F, F, D)
- **TIL:** Mean Absolute Error (lower is better) and Spearman rank correlation
- **Immune-score:** Mean Absolute Error
- **JSON/Schema compliance:** Percentage of outputs that parse as valid JSON matching the expected schema

### Results

| Metric | Value | Target |
|--------|-------|--------|
| CD274 Balanced Accuracy | **0.5638** | >0.70 |
| MSI AUC | 0.50 | >0.75 |
| TME Accuracy | 0.2473 | >0.65 |
| TIL MAE | 0.2576 | <0.20 |
| TIL Spearman | -0.1740 | >0.60 |
| Immune-score MAE | 0.3032 | <0.20 |
| ECE (after temp scaling) | 0.1091 | <0.10 |
| JSON Compliance | **100%** | - |
| Schema Compliance | **100%** | - |

#### Summary

v2 achieves the best CD274 balanced accuracy (0.56) and strong calibration after temperature scaling (ECE 0.11). JSON and schema compliance both reach 100%, up from 94%/16% at zero-shot. TIL MAE (0.26) does not meet the clinical target; this was addressed in v3.1 via response-only loss masking.

## Environmental Impact

- **Hardware Type:** NVIDIA A100-SXM4-80GB
- **Hours used:** ~0.64
- **Cloud Provider:** Google Colab
- **Compute Region:** US
- **Carbon Emitted:** ~0.05 kg CO2eq (estimated via ML Impact Calculator)

## Technical Specifications

### Model Architecture and Objective

MedGemma 1.5 4B-IT is a Gemma-based vision-language model with a SigLIP vision encoder. LoRA adapters are applied to all attention and MLP projection layers. The training objective is standard causal language modeling loss (cross-entropy) over the full sequence (prompt + response).

### Compute Infrastructure

#### Hardware

NVIDIA A100-SXM4-80GB (single GPU)

#### Software

- transformers 4.x
- peft 0.18.1
- trl (SFTTrainer)
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
