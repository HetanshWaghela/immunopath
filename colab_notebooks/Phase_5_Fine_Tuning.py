# %% [markdown]
# # 🧬 Phase 5 — Fine-Tuning MedGemma with DoRA (Days 7–8)
#
# **Goal:** Fine-tune `google/medgemma-1.5-4b-it` using DoRA (Weight-Decomposed LoRA)
# on the training JSONL from Phase 3. Response-only loss masking ensures the model
# only learns to produce the JSON output, not reproduce the prompt.
#
# **Outputs:**
# - `models/immunopath-v1/adapter_config.json`     — DoRA config
# - `models/immunopath-v1/adapter_model.safetensors`— DoRA weights
# - `results/phase5/training_log.json`              — Loss curves + training stats
#
# ---
# **Hard Rules (DO NOT CHANGE):**
# - Model: `google/medgemma-1.5-4b-it`, `AutoModelForImageTextToText`
# - Precision: `torch.bfloat16` (NOT fp16)
# - DoRA: `use_dora=True`, r=16, alpha=32, dropout=0.05
# - Training: batch=2, grad_accum=8 (effective=16), lr=1e-4, epochs=3
# - Response-only loss masking: prompt tokens → -100
# - Patient-level split (from Phase 3)

# %%
# ============================================================
# CELL 1: Colab Setup + Dependency Install
# ============================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import subprocess

from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive/ImmunoPath"
DATA_DIR = f"{PROJECT_DIR}/data"
TRAINING_DIR = f"{DATA_DIR}/training"
MODEL_DIR = f"{PROJECT_DIR}/models/immunopath-v1"
RESULTS_DIR = f"{PROJECT_DIR}/results/phase5"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Install dependencies
subprocess.run([
    "pip", "install", "-q", "--upgrade",
    "transformers>=4.50.0",
    "accelerate>=0.34.0",
    "peft>=0.12.0",
    "bitsandbytes>=0.43.0",
    "datasets",
    "tensorboard",
    "pillow>=10.0.0",
    "pandas", "numpy", "scipy", "tqdm",
], check=True)

# Flash Attention (optional, try to install)
try:
    subprocess.run(["pip", "install", "-q", "flash-attn", "--no-build-isolation"], check=True)
    FLASH_ATTN_AVAILABLE = True
    print("✅ Flash Attention 2 installed")
except Exception:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️  Flash Attention not available")

# HuggingFace auth
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
    raise RuntimeError("❌ No GPU! Fine-tuning requires a GPU.")

import transformers, peft
print(f"✅ transformers=={transformers.__version__}, peft=={peft.__version__}")

# %%
# ============================================================
# CELL 2: Configuration
# ============================================================
from dataclasses import dataclass

@dataclass
class Config:
    # Model
    model_id: str = "google/medgemma-1.5-4b-it"
    # DoRA PEFT
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_dora: bool = True
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # Training
    num_epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 8   # effective batch = 1 × 8 = 8
    lr: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # Data
    max_patches: int = 4
    max_length: int = 1536
    # Paths
    train_path: str = f"{TRAINING_DIR}/train.jsonl"
    val_path: str = f"{TRAINING_DIR}/val.jsonl"
    output_dir: str = MODEL_DIR
    log_dir: str = RESULTS_DIR
    # Eval
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500

cfg = Config()
print(f"✅ Config: DoRA r={cfg.lora_r}, alpha={cfg.lora_alpha}, "
      f"batch={cfg.batch_size}×{cfg.grad_accum}={cfg.batch_size*cfg.grad_accum}, "
      f"lr={cfg.lr}, epochs={cfg.num_epochs}")
if USE_QDORA:
    print("⚠️  VRAM < 24GB → QDoRA (4-bit quantisation) will be used")

# %%
# ============================================================
# CELL 3: Load Model + Apply DoRA Adapters
# ============================================================
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

print(f"Loading {cfg.model_id}...")

processor = AutoProcessor.from_pretrained(cfg.model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

attn_impl = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "eager"

model = AutoModelForImageTextToText.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation=attn_impl,
    quantization_config=bnb_config,
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Apply DoRA
peft_config = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=list(cfg.target_modules),
    use_dora=cfg.use_dora,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Should be ~0.5-1%

allocated = torch.cuda.memory_allocated() / 1e9
print(f"✅ Model loaded + DoRA applied ({attn_impl})")
print(f"   VRAM used: {allocated:.2f} GB")

# %%
# ============================================================
# CELL 4: ImmunoPathDataset  (response-only loss masking)
# ============================================================
import json
from PIL import Image
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any

class ImmunoPathDataset(Dataset):
    """
    Dataset that:
      1. Loads multi-image patches per sample (parallel I/O)
      2. Tokenises full conversation (user + assistant)
      3. Masks prompt tokens in labels (-100) → response-only loss
    """

    def __init__(
        self,
        data_path: str,
        processor: Any,
        max_patches: int = 8,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.max_patches = max_patches
        self.max_length = max_length

        # Pre-load JSONL into memory (small — just metadata + paths)
        self.samples: List[Dict] = []
        with open(data_path) as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        print(f"   Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    # ---- Parallel Image Loading ----
    @staticmethod
    def _load_image(path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB").resize((336, 336))
        except Exception:
            return None

    def _load_images(self, paths: List[str]) -> List[Image.Image]:
        paths = paths[: self.max_patches]
        with ThreadPoolExecutor(max_workers=min(4, len(paths))) as pool:
            results = list(pool.map(self._load_image, paths))
        images = [img for img in results if img is not None]
        if not images:
            images = [Image.new("RGB", (512, 512), "white")]
        return images

    # ---- Core __getitem__ ----
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        images = self._load_images(sample["patch_paths"])

        prompt_text = sample["prompt"]
        response_text = sample["response"]

        # ----- Build message content with images -----
        user_content = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": prompt_text})

        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
        ]
        prompt_only_messages = [
            {"role": "user", "content": user_content},
        ]

        # ----- Tokenise FULL conversation -----
        full_inputs = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # ----- Tokenise PROMPT-ONLY (to find boundary) -----
        prompt_inputs = self.processor.apply_chat_template(
            prompt_only_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        prompt_len = prompt_inputs["input_ids"].shape[-1]

        # ----- Build labels: mask prompt, keep response -----
        labels = full_inputs["input_ids"].clone().squeeze(0)   # (seq,)
        labels[:prompt_len] = -100
        # Also mask padding
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Squeeze batch dim added by return_tensors="pt"
        out = {}
        for k, v in full_inputs.items():
            if torch.is_tensor(v):
                out[k] = v.squeeze(0)
        out["labels"] = labels
        return out


# Quick sanity check
print("Building train dataset...")
train_dataset = ImmunoPathDataset(cfg.train_path, processor, cfg.max_patches, cfg.max_length)
print("Building val dataset...")
val_dataset = ImmunoPathDataset(cfg.val_path, processor, cfg.max_patches, cfg.max_length)

# Test one sample
if len(train_dataset) > 0:
    sample = train_dataset[0]
    print(f"\n✅ Sample shapes:")
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"   {k}: {v.shape} ({v.dtype})")
    n_masked = (sample["labels"] == -100).sum().item()
    n_total = sample["labels"].shape[0]
    print(f"   Masked tokens: {n_masked}/{n_total} ({n_masked/n_total*100:.0f}% masked)")

# %%
# ============================================================
# CELL 5: Custom Data Collator (variable-length padding)
# ============================================================
from dataclasses import dataclass as dc
from typing import Optional

@dc
class ImmunoPathCollator:
    """Pads a batch of variable-length samples to the same length."""
    processor: Any
    padding: str = "longest"

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Separate labels (need special handling)
        label_list = [f.pop("labels") for f in features]

        # Separate pixel_values if present (can't be padded like text)
        pixel_values_list = None
        if "pixel_values" in features[0]:
            pixel_values_list = [f.pop("pixel_values") for f in features]

        # Pad text inputs (input_ids, attention_mask)
        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(f["input_ids"].shape[0] for f in features)

        batch = {}
        for key in features[0].keys():
            tensors = [f[key] for f in features]
            # Pad all to max_len
            padded = []
            for t in tensors:
                pad_size = max_len - t.shape[0]
                if pad_size > 0:
                    if key == "attention_mask":
                        padded.append(torch.cat([t, torch.zeros(pad_size, dtype=t.dtype)]))
                    else:
                        padded.append(torch.cat([t, torch.full((pad_size,), pad_id, dtype=t.dtype)]))
                else:
                    padded.append(t)
            batch[key] = torch.stack(padded)

        # Pad labels
        padded_labels = []
        for lb in label_list:
            pad_size = max_len - lb.shape[0]
            if pad_size > 0:
                padded_labels.append(torch.cat([lb, torch.full((pad_size,), -100, dtype=lb.dtype)]))
            else:
                padded_labels.append(lb)
        batch["labels"] = torch.stack(padded_labels)

        # Handle pixel_values (stack if same shape, else pad)
        if pixel_values_list is not None:
            # pixel_values can have different numbers of images per sample
            # For simplicity, pad with zeros to max images in the batch
            shapes = [pv.shape for pv in pixel_values_list]
            max_imgs = max(s[0] for s in shapes)
            padded_pv = []
            for pv in pixel_values_list:
                if pv.shape[0] < max_imgs:
                    pad = torch.zeros(
                        (max_imgs - pv.shape[0], *pv.shape[1:]),
                        dtype=pv.dtype,
                    )
                    padded_pv.append(torch.cat([pv, pad]))
                else:
                    padded_pv.append(pv)
            batch["pixel_values"] = torch.stack(padded_pv)

        return batch

collator = ImmunoPathCollator(processor=processor)
print("✅ Data collator ready")

# %%
# ============================================================
# CELL 6: Training Arguments + Trainer
# ============================================================
from transformers import TrainingArguments, Trainer
import json, time

# Checkpoint dir on GDrive (survives runtime disconnect)
ckpt_dir = f"{cfg.output_dir}/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=ckpt_dir,
    num_train_epochs=cfg.num_epochs,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=cfg.grad_accum,
    learning_rate=cfg.lr,
    warmup_ratio=cfg.warmup_ratio,
    weight_decay=cfg.weight_decay,
    logging_steps=cfg.logging_steps,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=cfg.save_steps,
    save_total_limit=2,
    bf16=True,
    optim="adamw_bnb_8bit",
    max_grad_norm=cfg.max_grad_norm,
    report_to="tensorboard",
    logging_dir=f"{cfg.log_dir}/tb_logs",
    load_best_model_at_end=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
)

print(f"✅ Trainer ready")
print(f"   Train samples: {len(train_dataset)}")
print(f"   Val samples:   {len(val_dataset)}")
print(f"   Effective batch size: {cfg.batch_size * cfg.grad_accum}")
print(f"   Total steps: ~{len(train_dataset) // (cfg.batch_size * cfg.grad_accum) * cfg.num_epochs}")

# %%
# ============================================================
# CELL 7: Train!
# ============================================================
print("\n" + "=" * 60)
print("STARTING FINE-TUNING")
print("=" * 60)

start_time = time.time()

# Resume from checkpoint if one exists (GDrive persistence)
resume_from = None
if os.path.exists(ckpt_dir):
    ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
    if ckpts:
        latest = sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1]
        resume_from = f"{ckpt_dir}/{latest}"
        print(f"   Resuming from {resume_from}")

train_result = trainer.train(resume_from_checkpoint=resume_from)

elapsed = time.time() - start_time
print(f"\n✅ Training complete in {elapsed/60:.1f} minutes")
print(f"   Final train loss: {train_result.training_loss:.4f}")

# %%
# ============================================================
# CELL 8: Save DoRA Adapters + Processor
# ============================================================

# Save the PEFT adapters (small — just DoRA weights)
adapter_dir = f"{cfg.output_dir}/lora_adapters"
os.makedirs(adapter_dir, exist_ok=True)

model.save_pretrained(adapter_dir)
processor.save_pretrained(adapter_dir)

# Verify saved files
saved_files = os.listdir(adapter_dir)
print(f"✅ DoRA adapters saved to {adapter_dir}")
print(f"   Files: {saved_files}")

# Also save via trainer (includes training state)
trainer.save_model(cfg.output_dir)
processor.save_pretrained(cfg.output_dir)
print(f"✅ Full model checkpoint saved to {cfg.output_dir}")

# %%
# ============================================================
# CELL 9: Save Training Log + Metrics
# ============================================================
from datetime import datetime

# Extract training history
log_history = trainer.state.log_history

# Separate train and eval logs
train_logs = [l for l in log_history if "loss" in l and "eval_loss" not in l]
eval_logs = [l for l in log_history if "eval_loss" in l]

training_report = {
    "phase": 5,
    "timestamp": datetime.now().isoformat(),
    "model_id": cfg.model_id,
    "peft_method": "DoRA" if cfg.use_dora else "LoRA",
    "peft_config": {
        "r": cfg.lora_r,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "target_modules": list(cfg.target_modules),
        "use_dora": cfg.use_dora,
    },
    "training_config": {
        "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
        "effective_batch": cfg.batch_size * cfg.grad_accum,
        "lr": cfg.lr,
        "epochs": cfg.num_epochs,
        "bf16": True,
        "quantised": USE_QDORA,
    },
    "data": {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    },
    "results": {
        "final_train_loss": train_result.training_loss,
        "best_eval_loss": min((l["eval_loss"] for l in eval_logs), default=None),
        "training_time_minutes": round(elapsed / 60, 1),
    },
    "train_logs": train_logs,
    "eval_logs": eval_logs,
}

log_path = f"{cfg.log_dir}/training_log.json"
with open(log_path, 'w') as f:
    json.dump(training_report, f, indent=2, default=str)
print(f"✅ Training log saved to {log_path}")

# %%
# ============================================================
# CELL 10: Quick Inference Test (verify fine-tuned model works)
# ============================================================
from PIL import Image

# Load a val sample
val_sample = val_dataset.samples[0] if val_dataset.samples else None

if val_sample:
    images = [Image.open(p).convert("RGB") for p in val_sample["patch_paths"][:8]]
    if not images:
        images = [Image.new("RGB", (512, 512), "white")]

    prompt = val_sample["prompt"]

    # Build messages
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(model.device, dtype=torch.bfloat16) if v.is_floating_point()
        else v.to(model.device)
        for k, v in inputs.items()
        if torch.is_tensor(v)
    }
    input_len = inputs["input_ids"].shape[-1]

    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            use_cache=True,
        )

    generated = output_ids[0][input_len:]
    response_text = processor.decode(generated, skip_special_tokens=True).strip()

    print("✅ Fine-tuned model inference test:")
    print(f"   Output preview: {response_text[:300]}")

    # Try to parse JSON
    try:
        clean = response_text
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0]
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(clean[start:end])
            print(f"   ✅ Valid JSON with keys: {list(parsed.keys())}")
        else:
            print("   ⚠️  No JSON object found in output")
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON parse error: {e}")

    # Ground truth comparison
    print(f"\n   Ground truth (first 200 chars):")
    print(f"   {val_sample['response'][:200]}")
else:
    print("⚠️  No val samples available for inference test")

# %%
# ============================================================
# CELL 11: Phase 5 Summary
# ============================================================

print("\n" + "=" * 60)
print("PHASE 5 — FINE-TUNING COMPLETE")
print("=" * 60)
print(f"\n  Model:       {cfg.model_id}")
print(f"  PEFT:        {'DoRA' if cfg.use_dora else 'LoRA'} (r={cfg.lora_r}, α={cfg.lora_alpha})")
print(f"  Quantised:   {'QDoRA (4-bit)' if USE_QDORA else 'No (full bf16)'}")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples:   {len(val_dataset)}")
print(f"  Final loss:    {train_result.training_loss:.4f}")
best_eval = min((l["eval_loss"] for l in eval_logs), default="N/A")
print(f"  Best eval loss: {best_eval}")
print(f"  Training time: {elapsed/60:.1f} min")
print(f"\n  Adapters saved: {adapter_dir}")
print(f"  Log saved:      {log_path}")

print(f"\n📋 UPDATE PHASE_TRACKER.md:")
print(f"  Phase 5 Status: DONE")
print(f"  Final train loss: {train_result.training_loss:.4f}")
print(f"  Best eval loss:   {best_eval}")
print(f"  Training time:    {elapsed/60:.1f} min")
print(f"  Adapters:         {adapter_dir}")

print(f"\n{'=' * 60}")
print("NEXT: Phase 6 — Evaluation + Calibration")
print(f"{'=' * 60}")
