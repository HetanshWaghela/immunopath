# %% [markdown]
# # 🧪 Phase 0 — Test MedGemma (Day 0)
#
# **Goal:** Verify MedGemma can: (1) load + process a single image, (2) handle
# multiple images in one prompt, (3) output valid JSON. Results determine the
# architecture path for the entire project.
#
# **Decision Matrix:**
# | Test Result | Architecture |
# |---|---|
# | 4+ images + JSON >80% | Multi-image input |
# | 4+ images + JSON <80% | Multi-image + JSON repair |
# | Only 1-2 images | 3×3 tile collage |
# | Nothing works | Debug setup → retry |
#
# ---
# **Hard Rules (apply to ALL phases):**
# - Model: `google/medgemma-1.5-4b-it`
# - Class: `AutoModelForImageTextToText` (NOT `AutoModelForCausalLM`)
# - Precision: `torch.bfloat16` (NOT fp16 — Gemma 3 requirement)
# - Pipeline type: `"image-text-to-text"`
# - TME subtypes: IE, IE/F, F, D (slash notation)
# - PD-L1 terminology: CD274 expression (RNA proxy, not IHC)

# %%
# ============================================================
# CELL 1: Colab Setup (run this FIRST)
# ============================================================

import os

# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Project directories ---
PROJECT_DIR = "/content/drive/MyDrive/ImmunoPath"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/data", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/models", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/results/phase0", exist_ok=True)

# --- Clone/update repo ---
import subprocess
REPO_DIR = "/content/immunopath"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "https://github.com/YOUR_USERNAME/medgemma-impact-challenge.git", REPO_DIR], check=False)
else:
    subprocess.run(["git", "pull"], cwd=REPO_DIR, check=False)

# --- HuggingFace login ---
from huggingface_hub import login
from google.colab import userdata
try:
    hf_token = userdata.get('HF_TOKEN')
    login(token=hf_token)
    print("✅ Logged in to HuggingFace via Colab secret")
except Exception:
    print("⚠️  Set HF_TOKEN in Colab Secrets (key icon in sidebar)")
    print("    Get token at: https://huggingface.co/settings/tokens")
    print("    You MUST accept the MedGemma license first:")
    print("    https://huggingface.co/google/medgemma-1.5-4b-it")

# --- GPU check ---
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"   bfloat16 support: {torch.cuda.is_bf16_supported()}")
else:
    print("❌ No GPU! Runtime → Change runtime type → GPU (T4 minimum, A100 ideal)")

# %%
# ============================================================
# CELL 2: Install Dependencies
# ============================================================

# Pin versions for reproducibility
# CRITICAL: MedGemma (Gemma 3) requires transformers >= 4.50.0
import subprocess
subprocess.run([
    "pip", "install", "-q", "--upgrade",
    "transformers>=4.50.0",   # Gemma 3 minimum
    "accelerate>=0.34.0",
    "bitsandbytes>=0.44.0",
    "pillow>=10.0.0",
], check=True)

import transformers
print(f"✅ Dependencies installed (transformers=={transformers.__version__})")

# %%
# ============================================================
# CELL 3: Create Synthetic H&E Test Images
# ============================================================
# We create synthetic pink-and-purple "H&E-like" patches so we don't need
# to download real slides for this initial test. Real slides come in Phase 1.

import numpy as np
from PIL import Image, ImageDraw
import random

def create_synthetic_he_patch(size=512, seed=None):
    """
    Create a synthetic H&E-like histopathology patch.
    Pink background (eosin) + purple dots (hematoxylin nuclei).
    This is NOT a real histopathology image — just for testing model I/O.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Pink eosin background with slight variation
    bg_r = np.random.randint(200, 240)
    bg_g = np.random.randint(160, 200)
    bg_b = np.random.randint(180, 220)
    img = Image.new('RGB', (size, size), (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    # Add tissue-like texture (lighter pink regions)
    for _ in range(30):
        x, y = random.randint(0, size), random.randint(0, size)
        rx, ry = random.randint(20, 80), random.randint(20, 80)
        color = (bg_r + random.randint(-20, 20),
                 bg_g + random.randint(-30, 10),
                 bg_b + random.randint(-20, 20))
        color = tuple(max(0, min(255, c)) for c in color)
        draw.ellipse([x-rx, y-ry, x+rx, y+ry], fill=color)

    # Purple nuclei (hematoxylin-stained)
    n_nuclei = random.randint(50, 300)
    for _ in range(n_nuclei):
        x, y = random.randint(5, size-5), random.randint(5, size-5)
        r = random.randint(3, 8)
        purple = (random.randint(80, 140),
                  random.randint(40, 100),
                  random.randint(120, 180))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=purple)

    # Occasional lymphocyte clusters (darker, smaller, denser)
    if random.random() > 0.5:
        cx, cy = random.randint(50, size-50), random.randint(50, size-50)
        for _ in range(random.randint(20, 60)):
            x = cx + random.randint(-30, 30)
            y = cy + random.randint(-30, 30)
            r = random.randint(2, 4)
            dark_purple = (random.randint(50, 90),
                           random.randint(20, 60),
                           random.randint(80, 130))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=dark_purple)

    return img

# Generate 8 test patches with different seeds (for reproducibility)
test_images = [create_synthetic_he_patch(512, seed=i) for i in range(8)]
print(f"✅ Created {len(test_images)} synthetic H&E patches (512×512)")

# Save one for visual inspection
test_images[0].save(f"{PROJECT_DIR}/results/phase0/test_patch_0.jpg", quality=95)
print(f"   Saved sample to: {PROJECT_DIR}/results/phase0/test_patch_0.jpg")

# Display the first patch
from IPython.display import display
display(test_images[0].resize((256, 256)))

# %%
# ============================================================
# CELL 4: Load MedGemma
# ============================================================
# CRITICAL: Use AutoModelForImageTextToText (NOT AutoModelForCausalLM)
# CRITICAL: Use torch.bfloat16 (NOT float16 — Gemma 3 requires bf16)

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

MODEL_ID = "google/medgemma-1.5-4b-it"

print(f"Loading {MODEL_ID}...")
print(f"  dtype: torch.bfloat16")
print(f"  device_map: auto")
print()

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    offload_buffers=True,
)
model.eval()

# CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.fp32_precision = "tf32"

# Print memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"✅ Model loaded successfully!")
    print(f"   VRAM allocated: {allocated:.2f} GB")
    print(f"   VRAM reserved:  {reserved:.2f} GB")

# %%
# ============================================================
# CELL 5: TEST 1 — Single Image Sanity Check
# ============================================================
# Goal: Verify model loads, processes one image, generates text.

SINGLE_IMAGE_PROMPT = """Analyze this H&E-stained histopathology image from a lung adenocarcinoma tumor.

Describe what you observe in terms of:
1. Tissue architecture
2. Nuclear morphology
3. Any signs of immune infiltration

Keep your answer concise (3-5 sentences)."""

def run_single_image_test(image, prompt):
    """Test MedGemma with a single image input."""
    # Build chat message — image object MUST be included in the message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Process inputs — ONE-CALL pattern (tokenize + return_dict in apply_chat_template)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device, dtype=torch.bfloat16) if v.is_floating_point()
              else v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}

    # Generate with inference_mode (more efficient than no_grad)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # deterministic for testing
        )

    # Decode using post_process_image_text_to_text
    response = processor.post_process_image_text_to_text(output_ids, skip_special_tokens=True)
    # Strip the input prompt from decoded output (returns list)
    return response[0].split("model\n")[-1].strip() if response else ""

print("=" * 60)
print("TEST 1: Single Image Sanity Check")
print("=" * 60)

try:
    response = run_single_image_test(test_images[0], SINGLE_IMAGE_PROMPT)
    print(f"\n✅ PASS — Model generated response ({len(response)} chars)")
    print(f"\nResponse:\n{response[:500]}")
    test1_pass = True
except Exception as e:
    print(f"\n❌ FAIL — {type(e).__name__}: {e}")
    test1_pass = False

# %%
# ============================================================
# CELL 6: TEST 2 — Multi-Image Capacity Test
# ============================================================
# Goal: Determine how many images MedGemma can handle in one prompt.
# This is the key architecture decision: multi-image vs collage.

MULTI_IMAGE_PROMPT = """Analyze these H&E-stained histopathology patches from a lung adenocarcinoma tumor.

These patches are from different regions of the same whole-slide image.
Provide a brief overall assessment of the tumor microenvironment.
Keep your answer concise."""

def run_multi_image_test(images, prompt):
    """Test MedGemma with multiple image inputs."""
    # Build content list: one {"type": "image", "image": img} per image, then text
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    # ONE-CALL pattern
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device, dtype=torch.bfloat16) if v.is_floating_point()
              else v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    response = processor.post_process_image_text_to_text(output_ids, skip_special_tokens=True)
    return response[0].split("model\n")[-1].strip() if response else ""

print("=" * 60)
print("TEST 2: Multi-Image Capacity Test")
print("=" * 60)

image_counts = [1, 2, 4, 8]
multi_image_results = {}

for n in image_counts:
    print(f"\n--- Testing {n} image(s) ---")
    try:
        imgs = test_images[:n]
        response = run_multi_image_test(imgs, MULTI_IMAGE_PROMPT)
        multi_image_results[n] = {
            "status": "PASS",
            "response_len": len(response),
            "response_preview": response[:200],
        }
        print(f"  ✅ PASS — {len(response)} chars")

        # Clear CUDA cache between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        multi_image_results[n] = {"status": "OOM"}
        print(f"  ❌ OOM — Out of GPU memory")
        torch.cuda.empty_cache()
        break  # No point testing more images

    except Exception as e:
        multi_image_results[n] = {"status": "FAIL", "error": str(e)}
        print(f"  ❌ FAIL — {type(e).__name__}: {str(e)[:100]}")

# Determine max working image count
max_images = 0
for n in image_counts:
    if n in multi_image_results and multi_image_results[n]["status"] == "PASS":
        max_images = n

print(f"\n{'=' * 60}")
print(f"RESULT: Max working images = {max_images}")
print(f"{'=' * 60}")

# %%
# ============================================================
# CELL 7: TEST 3 — JSON Output Reliability
# ============================================================
# Goal: Test if MedGemma can output valid JSON with our exact schema.
# Run 10 trials and measure parse rate.

import json

JSON_PROMPT = """Analyze these H&E-stained histopathology images from a lung adenocarcinoma tumor.

Extract the following H&E-inferred immune signals as a research output (not diagnostic):
1. CD274 (PD-L1) RNA proxy level (high/low)
2. MSI status (MSI-H or MSS) + probability
3. TIL fraction (0.0-1.0) + bucket (low/moderate/high)
4. TME subtype (IE / IE/F / F / D)
5. Immune phenotype (inflamed/excluded/desert)
6. CD8+ T-cell infiltration (low/moderate/high)
7. Overall immune score (0.0-1.0)

Provide your analysis as a JSON object only. No other text."""

# Use the max working image count from Test 2
n_images_for_json = min(max_images, 4) if max_images > 0 else 1
print(f"Using {n_images_for_json} image(s) for JSON test\n")

def extract_json_from_response(response_text):
    """Try to parse JSON from model response with fallback strategies."""
    text = response_text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text), "direct"
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip()), "markdown_block"
        except (json.JSONDecodeError, ValueError):
            pass

    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        try:
            return json.loads(text[start:end].strip()), "code_block"
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Find first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end]), "brace_extract"
        except json.JSONDecodeError:
            pass

    return None, "failed"

print("=" * 60)
print("TEST 3: JSON Output Reliability (10 trials)")
print("=" * 60)

N_TRIALS = 10
json_results = []

for trial in range(N_TRIALS):
    print(f"\n--- Trial {trial + 1}/{N_TRIALS} ---")
    try:
        # Use different image subsets for variety
        start_idx = trial % (len(test_images) - n_images_for_json + 1)
        imgs = test_images[start_idx : start_idx + n_images_for_json]

        # Use do_sample=True with temperature for variety across trials
        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": JSON_PROMPT})
        messages = [{"role": "user", "content": content}]

        # ONE-CALL pattern
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device, dtype=torch.bfloat16) if v.is_floating_point()
                  else v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=(trial > 0),  # deterministic first trial, random after
                temperature=0.7 if trial > 0 else 1.0,
            )

        response_list = processor.post_process_image_text_to_text(output_ids, skip_special_tokens=True)
        response = response_list[0].split("model\n")[-1].strip() if response_list else ""

        parsed, method = extract_json_from_response(response)

        if parsed is not None:
            print(f"  ✅ JSON parsed ({method})")
            print(f"     Keys: {list(parsed.keys())[:5]}...")
            json_results.append({"trial": trial, "success": True, "method": method,
                                 "data": parsed, "raw": response})
        else:
            print(f"  ❌ JSON parse failed")
            print(f"     Raw (first 200 chars): {response[:200]}")
            json_results.append({"trial": trial, "success": False, "raw": response})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ❌ Error: {type(e).__name__}: {str(e)[:100]}")
        json_results.append({"trial": trial, "success": False, "error": str(e)})

# Calculate parse rate
successes = sum(1 for r in json_results if r["success"])
parse_rate = (successes / N_TRIALS) * 100

print(f"\n{'=' * 60}")
print(f"JSON PARSE RATE: {successes}/{N_TRIALS} = {parse_rate:.0f}%")
print(f"{'=' * 60}")

# %%
# ============================================================
# CELL 8: Architecture Decision + Results Summary
# ============================================================
# Apply the decision matrix and save all results.

import json
from datetime import datetime

# --- Architecture Decision ---
if max_images >= 4 and parse_rate >= 80:
    architecture = "multi_image"
    arch_desc = "Multi-image input (4-8 patches per call)"
elif max_images >= 4 and parse_rate < 80:
    architecture = "multi_image_with_repair"
    arch_desc = "Multi-image + JSON post-processing repair"
elif max_images >= 1:
    architecture = "collage"
    arch_desc = "Single tile collage (3×3 mosaic of patches)"
else:
    architecture = "debug_required"
    arch_desc = "Setup needs debugging — model failed to load/generate"

print("=" * 60)
print("PHASE 0 RESULTS SUMMARY")
print("=" * 60)
print()
print(f"Test 1 (Single Image):   {'✅ PASS' if test1_pass else '❌ FAIL'}")
print(f"Test 2 (Multi-Image):    Max images = {max_images}")
print(f"Test 3 (JSON Parse):     {parse_rate:.0f}% ({successes}/{N_TRIALS})")
print()
print(f"🏗️  Architecture Decision: {arch_desc}")
print(f"    (Code: {architecture})")
print()

# --- Save results to Drive ---
results = {
    "phase": 0,
    "timestamp": datetime.now().isoformat(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    "model_id": "google/medgemma-1.5-4b-it",
    "test1_single_image": test1_pass,
    "test2_max_images": max_images,
    "test2_details": {str(k): v for k, v in multi_image_results.items()},
    "test3_json_parse_rate": parse_rate,
    "test3_successes": successes,
    "test3_total": N_TRIALS,
    "architecture_decision": architecture,
    "architecture_description": arch_desc,
}

results_path = f"{PROJECT_DIR}/results/phase0/phase0_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"📁 Results saved to: {results_path}")
print()

# --- Instructions for PHASE_TRACKER.md ---
print("=" * 60)
print("📋 UPDATE PHASE_TRACKER.md WITH THESE VALUES:")
print("=" * 60)
print(f"  Status:              DONE")
print(f"  Single Image Test:   {'Pass' if test1_pass else 'Fail'}")
print(f"  Multi-Image Test:    Max images = {max_images}")
print(f"  JSON Parse Rate:     {parse_rate:.0f}%")
print(f"  Architecture:        {architecture}")

# %%
# ============================================================
# CELL 9: Cleanup + Next Steps
# ============================================================

# Free GPU memory
del model, processor
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("🧹 Model unloaded, GPU memory freed")
print()
print("=" * 60)
print("NEXT STEPS")
print("=" * 60)
print()
print("1. Update PHASE_TRACKER.md with results above")
print("2. Proceed to Phase 1 — Data Download")
print("3. Open Phase_1_Data_Download.ipynb on Colab")
print()
print("If architecture = 'multi_image':")
print("   → Great! Phase 5 will use multi-patch MedGemma input")
print()
print("If architecture = 'collage':")
print("   → Phase 2 will include a create_mosaic() function to")
print("     assemble patches into a single image grid")
print()
print("If architecture = 'debug_required':")
print("   → Check: HF token accepted? MedGemma license agreed?")
print("   → Check: GPU available? (Runtime → Change runtime type)")
print("   → Check: torch.bfloat16 supported on your GPU?")
