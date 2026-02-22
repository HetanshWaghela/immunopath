# %% [markdown]
# # 🔍 Phase 4 — Zero-Shot Baseline (Day 6)
#
# **Goal:** Run MedGemma WITHOUT fine-tuning on our test set to establish
# baseline metrics. This is REQUIRED for the "novel task" claim — judges
# need to see the delta from fine-tuning.
#
# **Outputs:**
# - `results/phase4/zero_shot_predictions.jsonl`  — Raw model outputs
# - `results/phase4/zero_shot_metrics.json`        — All evaluation metrics
# - `results/phase4/zero_shot_report.md`           — Human-readable report
#
# **Optimisation notes (Colab Pro):**
# - Flash Attention 2 for 2× faster inference
# - KV cache enabled
# - bf16 precision (Gemma 3 requirement)
# - Greedy decoding (deterministic, faster than sampling)
# - Batched image loading with PIL + ThreadPoolExecutor
#
# ---
# **Hard Rules:**
# - Model: `google/medgemma-1.5-4b-it`
# - Class: `AutoModelForImageTextToText`
# - Precision: `torch.bfloat16` (NOT fp16)
# - No fine-tuning, no adapters — pure zero-shot

# %%
# ============================================================
# CELL 1: Colab Setup
# ============================================================
import os

from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/ImmunoPath"
DATA_DIR = f"{PROJECT_DIR}/data"
RESULTS_DIR = f"{PROJECT_DIR}/results/phase4"
TRAINING_DIR = f"{DATA_DIR}/training"

os.makedirs(RESULTS_DIR, exist_ok=True)

# HuggingFace login
from huggingface_hub import login
from google.colab import userdata
try:
    hf_token = userdata.get('HF_TOKEN')
    login(token=hf_token)
    print("✅ Logged in to HuggingFace")
except Exception:
    print("⚠️  Set HF_TOKEN in Colab Secrets")

# GPU check
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("❌ No GPU!")

# %%
# ============================================================
# CELL 2: Install Dependencies
# ============================================================
import subprocess
subprocess.run([
    "pip", "install", "-q", "--upgrade",
    "transformers>=4.50.0",
    "accelerate>=0.34.0",
    "pillow>=10.0.0",
    "pandas", "numpy", "scikit-learn", "scipy", "tqdm",
], check=True)

# Try to install flash-attention (may fail on some runtimes)
try:
    subprocess.run(["pip", "install", "-q", "flash-attn", "--no-build-isolation"], check=True)
    FLASH_ATTN_AVAILABLE = True
    print("✅ Flash Attention 2 installed")
except Exception:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️ Flash Attention not available (using default attention)")

import transformers
print(f"✅ transformers=={transformers.__version__}")

# %%
# ============================================================
# CELL 3: Load MedGemma (Zero-Shot — No Adapters)
# ============================================================
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

MODEL_ID = "google/medgemma-1.5-4b-it"

print(f"Loading {MODEL_ID} (zero-shot, no adapters)...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Use Flash Attention if available
attn_impl = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "eager"
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation=attn_impl,
)
model.eval()

# CUDA optimisations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True

allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
print(f"✅ Model loaded ({attn_impl})")
print(f"   VRAM: {allocated:.2f} GB")

# %%
# ============================================================
# CELL 4: Load Test + Val Data
# ============================================================
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import time

def load_jsonl(path: str) -> list:
    """Load JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples

# Load test and val sets
test_samples = load_jsonl(f"{TRAINING_DIR}/test.jsonl") if os.path.exists(f"{TRAINING_DIR}/test.jsonl") else []
val_samples = load_jsonl(f"{TRAINING_DIR}/val.jsonl") if os.path.exists(f"{TRAINING_DIR}/val.jsonl") else []

# Combine for baseline evaluation (we want metrics on both)
eval_samples = test_samples + val_samples
print(f"✅ Loaded {len(test_samples)} test + {len(val_samples)} val = {len(eval_samples)} eval samples")

# Also load ground truth signatures for metrics
SIGNATURES_PATH = f"{DATA_DIR}/signatures/immune_signatures.csv"
ground_truth_df = pd.read_csv(SIGNATURES_PATH, index_col=0)
print(f"✅ Ground truth: {len(ground_truth_df)} patients")

# %%
# ============================================================
# CELL 5: Efficient Image Loading
# ============================================================

def load_patches_parallel(patch_paths: list, max_patches: int = 8) -> list:
    """Load patch images in parallel with ThreadPoolExecutor."""
    paths = patch_paths[:max_patches]

    def _load_one(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    # Use threads for I/O-bound GDrive reads
    images = []
    with ThreadPoolExecutor(max_workers=min(4, len(paths))) as executor:
        results = list(executor.map(_load_one, paths))
    images = [img for img in results if img is not None]
    return images

# Quick test
if eval_samples:
    test_imgs = load_patches_parallel(eval_samples[0]["patch_paths"])
    print(f"✅ Image loading test: {len(test_imgs)} images loaded")
    if test_imgs:
        print(f"   Size: {test_imgs[0].size}, Mode: {test_imgs[0].mode}")

# %%
# ============================================================
# CELL 6: Zero-Shot Inference Function
# ============================================================

def run_zero_shot_inference(
    sample: dict,
    model,
    processor,
    max_new_tokens: int = 600,
) -> dict:
    """
    Run zero-shot MedGemma inference on a single sample.
    Returns the raw model output + parsed JSON (if valid).
    """
    # Load images
    images = load_patches_parallel(sample["patch_paths"])
    if not images:
        return {"sample_id": sample["sample_id"], "error": "no_images", "raw_output": ""}

    prompt = sample["prompt"]

    # Build message in MedGemma chat format
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    try:
        # Tokenise
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Move to device with correct dtypes
        inputs = {
            k: v.to(model.device, dtype=torch.bfloat16) if v.is_floating_point()
            else v.to(model.device)
            for k, v in inputs.items()
            if torch.is_tensor(v)
        }

        input_len = inputs["input_ids"].shape[-1]

        # Generate (greedy, deterministic)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # Decode only the generated tokens
        generated = output_ids[0][input_len:]
        raw_output = processor.decode(generated, skip_special_tokens=True).strip()

        # Try to parse JSON from output
        parsed_json = None
        try:
            # Handle markdown code blocks
            clean = raw_output
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0]
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0]
            # Find the JSON object
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start != -1 and end > start:
                parsed_json = json.loads(clean[start:end])
        except (json.JSONDecodeError, IndexError):
            pass

        return {
            "sample_id": sample["sample_id"],
            "patient_id": sample.get("patient_id", ""),
            "raw_output": raw_output,
            "parsed_json": parsed_json,
            "json_valid": parsed_json is not None,
            "input_tokens": input_len,
            "output_tokens": len(generated),
            "n_images": len(images),
            "error": None,
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"sample_id": sample["sample_id"], "error": "OOM", "raw_output": ""}
    except Exception as e:
        return {"sample_id": sample["sample_id"], "error": str(e), "raw_output": ""}


# Quick single-sample test
if eval_samples:
    print("Running single-sample test...")
    test_result = run_zero_shot_inference(eval_samples[0], model, processor)
    print(f"   JSON valid: {test_result['json_valid']}")
    print(f"   Output preview: {test_result['raw_output'][:200]}")
    if test_result['parsed_json']:
        print(f"   Keys: {list(test_result['parsed_json'].keys())}")

# %%
# ============================================================
# CELL 7: Run Zero-Shot on All Eval Samples
# ============================================================
# Process sequentially (GPU is the bottleneck, not CPU).
# Clear CUDA cache between samples to avoid OOM.

print(f"\n{'=' * 60}")
print(f"Running zero-shot baseline on {len(eval_samples)} samples")
print(f"{'=' * 60}")

all_predictions = []
inference_times = []

for i, sample in enumerate(tqdm(eval_samples, desc="Zero-shot inference")):
    start_time = time.time()
    result = run_zero_shot_inference(sample, model, processor)
    elapsed = time.time() - start_time
    inference_times.append(elapsed)

    result["inference_time_s"] = round(elapsed, 2)
    all_predictions.append(result)

    # Clear cache periodically
    if i % 5 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Progress update every 10 samples
    if (i + 1) % 10 == 0:
        valid = sum(1 for p in all_predictions if p["json_valid"])
        print(f"   [{i+1}/{len(eval_samples)}] JSON parse rate: {valid}/{i+1} "
              f"({valid/(i+1)*100:.0f}%), avg time: {np.mean(inference_times):.1f}s")

# Summary
n_valid = sum(1 for p in all_predictions if p["json_valid"])
n_errors = sum(1 for p in all_predictions if p.get("error"))
print(f"\n✅ Inference complete:")
print(f"   Total:       {len(all_predictions)}")
print(f"   JSON valid:  {n_valid}/{len(all_predictions)} ({n_valid/max(1,len(all_predictions))*100:.0f}%)")
print(f"   Errors:      {n_errors}")
print(f"   Avg time:    {np.mean(inference_times):.1f}s per sample")

# %%
# ============================================================
# CELL 8: Save Raw Predictions
# ============================================================

predictions_path = f"{RESULTS_DIR}/zero_shot_predictions.jsonl"
with open(predictions_path, 'w') as f:
    for pred in all_predictions:
        # Convert non-serialisable types
        record = {k: v for k, v in pred.items()}
        f.write(json.dumps(record, default=str) + '\n')

print(f"✅ Predictions saved to {predictions_path}")

# %%
# ============================================================
# CELL 9: Compute Zero-Shot Metrics
# ============================================================
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_absolute_error
from scipy.stats import spearmanr

def compute_zero_shot_metrics(predictions: list, ground_truth: pd.DataFrame) -> dict:
    """Compute all evaluation metrics for zero-shot baseline."""

    metrics = {
        "n_total": len(predictions),
        "n_json_valid": sum(1 for p in predictions if p["json_valid"]),
        "json_parse_rate": 0.0,
        "cd274": {},
        "msi": {},
        "tme": {},
        "til": {},
        "immune_score": {},
    }
    metrics["json_parse_rate"] = metrics["n_json_valid"] / max(1, metrics["n_total"])

    # Only use predictions with valid JSON
    valid_preds = [p for p in predictions if p["json_valid"] and p["parsed_json"]]

    # --- CD274 AUC ---
    cd274_true, cd274_pred = [], []
    for p in valid_preds:
        sid = p["sample_id"]
        if sid in ground_truth.index and "cd274_expression" in ground_truth.columns:
            true_label = str(ground_truth.loc[sid, "cd274_expression"]).lower()
            pred_label = str(p["parsed_json"].get("cd274_expression", "unknown")).lower()
            if true_label in ("high", "low") and pred_label in ("high", "low"):
                cd274_true.append(1 if true_label == "high" else 0)
                cd274_pred.append(1 if pred_label == "high" else 0)

    if len(cd274_true) >= 2 and len(set(cd274_true)) > 1:
        metrics["cd274"]["auc"] = round(roc_auc_score(cd274_true, cd274_pred), 4)
        metrics["cd274"]["accuracy"] = round(accuracy_score(cd274_true, cd274_pred), 4)
        metrics["cd274"]["f1"] = round(f1_score(cd274_true, cd274_pred, zero_division=0), 4)
    metrics["cd274"]["n_samples"] = len(cd274_true)

    # --- MSI Accuracy ---
    msi_true, msi_pred = [], []
    for p in valid_preds:
        sid = p["sample_id"]
        pj = p["parsed_json"]
        pred_msi = str(pj.get("msi_status", "unknown")).upper()
        # Ground truth MSI: check if we have it in response target
        gt_resp = None
        for es in eval_samples:
            if es["sample_id"] == sid:
                try:
                    gt_resp = json.loads(es["response"])
                except Exception:
                    pass
                break
        if gt_resp and gt_resp.get("msi_status") not in (None, "unknown"):
            true_msi = gt_resp["msi_status"].upper()
            if true_msi in ("MSI-H", "MSS") and pred_msi in ("MSI-H", "MSS"):
                msi_true.append(1 if true_msi == "MSI-H" else 0)
                msi_pred.append(1 if pred_msi == "MSI-H" else 0)

    if len(msi_true) >= 2 and len(set(msi_true)) > 1:
        metrics["msi"]["auc"] = round(roc_auc_score(msi_true, msi_pred), 4)
        metrics["msi"]["accuracy"] = round(accuracy_score(msi_true, msi_pred), 4)
    elif msi_true:
        metrics["msi"]["accuracy"] = round(accuracy_score(msi_true, msi_pred), 4)
    metrics["msi"]["n_samples"] = len(msi_true)

    # --- TME Subtype Accuracy ---
    tme_labels = {"IE": 0, "IE/F": 1, "F": 2, "D": 3}
    tme_true, tme_pred = [], []
    for p in valid_preds:
        sid = p["sample_id"]
        pj = p["parsed_json"]
        pred_tme = str(pj.get("tme_subtype", "unknown"))
        # Normalise: IE_F → IE/F
        pred_tme = pred_tme.replace("IE_F", "IE/F").replace("IE-F", "IE/F")

        gt_resp = None
        for es in eval_samples:
            if es["sample_id"] == sid:
                try:
                    gt_resp = json.loads(es["response"])
                except Exception:
                    pass
                break
        if gt_resp and gt_resp.get("tme_subtype") in tme_labels and pred_tme in tme_labels:
            tme_true.append(tme_labels[gt_resp["tme_subtype"]])
            tme_pred.append(tme_labels[pred_tme])

    if tme_true:
        metrics["tme"]["accuracy"] = round(accuracy_score(tme_true, tme_pred), 4)
        metrics["tme"]["macro_f1"] = round(f1_score(tme_true, tme_pred, average="macro", zero_division=0), 4)
    metrics["tme"]["n_samples"] = len(tme_true)

    # --- TIL Fraction Correlation ---
    til_true, til_pred = [], []
    for p in valid_preds:
        sid = p["sample_id"]
        pj = p["parsed_json"]
        pred_til = pj.get("til_fraction")
        if pred_til is not None and sid in ground_truth.index:
            try:
                pred_val = float(pred_til)
                true_val = float(ground_truth.loc[sid, "immune_score"])  # Use immune_score as proxy
                til_true.append(true_val)
                til_pred.append(pred_val)
            except (ValueError, TypeError):
                pass

    if len(til_true) >= 3:
        rho, pval = spearmanr(til_true, til_pred)
        metrics["til"]["spearman_rho"] = round(rho, 4)
        metrics["til"]["spearman_p"] = round(pval, 6)
        metrics["til"]["mae"] = round(mean_absolute_error(til_true, til_pred), 4)
    metrics["til"]["n_samples"] = len(til_true)

    # --- Immune Score ---
    iscore_true, iscore_pred = [], []
    for p in valid_preds:
        sid = p["sample_id"]
        pj = p["parsed_json"]
        pred_is = pj.get("immune_score")
        if pred_is is not None and sid in ground_truth.index:
            try:
                pred_val = float(pred_is)
                true_val = float(ground_truth.loc[sid, "immune_score"])
                iscore_true.append(true_val)
                iscore_pred.append(pred_val)
            except (ValueError, TypeError):
                pass

    if len(iscore_true) >= 3:
        rho, pval = spearmanr(iscore_true, iscore_pred)
        metrics["immune_score"]["spearman_rho"] = round(rho, 4)
        metrics["immune_score"]["mae"] = round(mean_absolute_error(iscore_true, iscore_pred), 4)
    metrics["immune_score"]["n_samples"] = len(iscore_true)

    return metrics


metrics = compute_zero_shot_metrics(all_predictions, ground_truth_df)

# Pretty print
print("\n" + "=" * 60)
print("ZERO-SHOT BASELINE METRICS")
print("=" * 60)
for task, task_metrics in metrics.items():
    if isinstance(task_metrics, dict):
        print(f"\n  {task}:")
        for k, v in task_metrics.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {task}: {task_metrics}")

# %%
# ============================================================
# CELL 10: Save Metrics + Report
# ============================================================
from datetime import datetime

# Save metrics JSON
metrics_path = f"{RESULTS_DIR}/zero_shot_metrics.json"
full_report = {
    "phase": 4,
    "timestamp": datetime.now().isoformat(),
    "model_id": MODEL_ID,
    "attention": attn_impl,
    "n_eval_samples": len(eval_samples),
    "avg_inference_time_s": round(np.mean(inference_times), 2) if inference_times else 0,
    "metrics": metrics,
}
with open(metrics_path, 'w') as f:
    json.dump(full_report, f, indent=2)
print(f"✅ Metrics saved to {metrics_path}")

# Generate markdown report
report_md = f"""# Phase 4 — Zero-Shot Baseline Report

**Model:** {MODEL_ID}
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Eval samples:** {len(eval_samples)}
**Avg inference time:** {np.mean(inference_times):.1f}s per sample

## Key Metrics

| Task | Metric | Value |
|------|--------|-------|
| JSON Parse Rate | % valid | {metrics['json_parse_rate']*100:.0f}% |
| CD274 (PD-L1 proxy) | AUC | {metrics['cd274'].get('auc', 'N/A')} |
| CD274 | Accuracy | {metrics['cd274'].get('accuracy', 'N/A')} |
| MSI Status | Accuracy | {metrics['msi'].get('accuracy', 'N/A')} |
| TME Subtype | Accuracy | {metrics['tme'].get('accuracy', 'N/A')} |
| TIL Fraction | Spearman ρ | {metrics['til'].get('spearman_rho', 'N/A')} |
| Immune Score | Spearman ρ | {metrics['immune_score'].get('spearman_rho', 'N/A')} |

## Notes
- Zero-shot = no fine-tuning, no adapters
- Metrics computed on test + val combined ({len(eval_samples)} samples)
- These are the BASELINE to beat in Phase 5 (fine-tuning)

## Next Steps
- Phase 5: Fine-tune with DoRA (r=16, alpha=32)
- Expected improvements: CD274 AUC +10-20%, TME accuracy +15-25%
"""

report_path = f"{RESULTS_DIR}/zero_shot_report.md"
with open(report_path, 'w') as f:
    f.write(report_md)
print(f"✅ Report saved to {report_path}")

# %%
# ============================================================
# CELL 11: Phase 4 Summary
# ============================================================

print("\n" + "=" * 60)
print("PHASE 4 — ZERO-SHOT BASELINE COMPLETE")
print("=" * 60)
print(f"\n  Model: {MODEL_ID}")
print(f"  Samples evaluated: {len(eval_samples)}")
print(f"  JSON parse rate:   {metrics['json_parse_rate']*100:.0f}%")
print(f"  CD274 AUC:         {metrics['cd274'].get('auc', 'N/A')}")
print(f"  MSI accuracy:      {metrics['msi'].get('accuracy', 'N/A')}")
print(f"  TME accuracy:      {metrics['tme'].get('accuracy', 'N/A')}")
print(f"  TIL Spearman ρ:    {metrics['til'].get('spearman_rho', 'N/A')}")

print(f"\n📋 UPDATE PHASE_TRACKER.md:")
print(f"  Status:             DONE")
print(f"  CD274 AUC (zero-shot): {metrics['cd274'].get('auc', '__')}")
print(f"  MSI AUC (zero-shot):   {metrics['msi'].get('auc', '__')}")
print(f"  TME Accuracy (zero-shot): {metrics['tme'].get('accuracy', '__')}")
print(f"  TIL Spearman ρ (zero-shot): {metrics['til'].get('spearman_rho', '__')}")
print(f"  JSON Parse Rate: {metrics['json_parse_rate']*100:.0f}%")

print(f"\n{'=' * 60}")
print("NEXT: Phase 5 — Fine-Tuning MedGemma with DoRA")
print(f"{'=' * 60}")
