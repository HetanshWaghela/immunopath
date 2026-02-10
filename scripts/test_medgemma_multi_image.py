#!/usr/bin/env python3
"""TEST 2: MedGemma multi-image GO/NO-GO test.

Run:
  python scripts/test_medgemma_multi_image.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL_ID = "google/medgemma-1.5-4b-it"
TEST_IMAGES = {
    "lung_squamous_histopathology": "https://commons.wikimedia.org/wiki/Special:FilePath/Histopathology%20of%20squamous-cell%20carcinoma%20of%20the%20lung.jpg",
    "small_cell_histopathology": "https://commons.wikimedia.org/wiki/Special:FilePath/Histopathology%20of%20small%20cell%20carcinoma.jpg",
    "squamous_cell_histopathology": "https://commons.wikimedia.org/wiki/Special:FilePath/Squamous%20cell%20carcinoma%202.jpg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedGemma multi-image capability checks.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Force device (default: auto).",
    )
    parser.add_argument(
        "--mps-mem-fraction",
        type=float,
        default=None,
        help="Limit per-process MPS memory usage (0-1). Default: unset (uses PyTorch default).",
    )
    parser.add_argument(
        "--tests",
        default="1,2,4,8",
        help="Comma-separated image counts to test (e.g. 1,2,4,8)",
    )
    parser.add_argument("--synthetic-patches", type=int, default=5, help="Number of synthetic patches")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Generation length")
    parser.add_argument("--min-new-tokens", type=int, default=24, help="Minimum generation length")
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable KV cache (faster, higher memory). Default: on for CUDA, off otherwise.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()

def resolve_runtime(requested_device: str) -> tuple[torch.device, torch.dtype]:
    if requested_device != "auto":
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda but CUDA is not available")
        if requested_device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested device=mps but MPS is not available")
        dtype = torch.bfloat16 if requested_device == "cuda" else torch.float16 if requested_device == "mps" else torch.float32
        return torch.device(requested_device), dtype

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def move_inputs_to_device(inputs: dict, device: torch.device, dtype: torch.dtype) -> dict:
    moved: dict = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def maybe_limit_mps_memory(device: torch.device, fraction: float | None) -> None:
    if device.type != "mps":
        return
    if fraction is None:
        return
    if not (0.0 < fraction <= 1.0):
        return
    if hasattr(torch.mps, "set_per_process_memory_fraction"):
        torch.mps.set_per_process_memory_fraction(fraction)


def print_mps_stats() -> None:
    if not hasattr(torch, "mps"):
        return
    try:
        rec = torch.mps.recommended_max_memory() / (1024**3)
        cur = torch.mps.current_allocated_memory() / (1024**3)
        drv = torch.mps.driver_allocated_memory() / (1024**3)
        print(f"MPS mem: recommended_max={rec:.2f}GiB current_alloc={cur:.2f}GiB driver_alloc={drv:.2f}GiB")
    except Exception:
        pass

def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if device.type == "cuda":
        torch.cuda.empty_cache()


def make_synthetic_patch(rng: np.random.Generator, n_nuclei: int = 500, size: int = 512) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(230, 180, 190))
    draw = ImageDraw.Draw(image)
    for _ in range(n_nuclei):
        x = int(rng.integers(10, size - 10))
        y = int(rng.integers(10, size - 10))
        r = int(rng.integers(2, 5))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(80, 40, 120))
    return image


def download_public_images() -> list[Image.Image]:
    images: list[Image.Image] = []
    for name, url in TEST_IMAGES.items():
        try:
            print(f"Downloading {name}...")
            response = requests.get(url, headers={"User-Agent": "immunopath-test"}, stream=True, timeout=60)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB").resize((512, 512))
            images.append(image)
            print(f"  OK {name}: {image.size}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}: {exc}")
    return images


def build_messages(images: list[Image.Image], n_imgs: int) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images[:n_imgs]]
    user_content.append(
        {
            "type": "text",
            "text": (
                "Analyze these H&E patches and provide your findings as JSON:\n"
                "{\n"
                '  "tme_subtype": "IE or IE/F or F or D",\n'
                '  "til_density": "low or moderate or high",\n'
                '  "immune_phenotype": "inflamed or excluded or desert",\n'
                '  "notes": "brief observation"\n'
                "}\n"
                "Respond ONLY with the JSON object."
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert pathologist."}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def extract_json(text: str) -> tuple[bool, str]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.replace("```json", "").replace("```", "").strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = clean[start : end + 1]
    else:
        candidate = clean
    try:
        json.loads(candidate)
        return True, candidate
    except json.JSONDecodeError:
        return False, candidate


def main() -> int:
    args = parse_args()
    if args.min_new_tokens > args.max_new_tokens:
        print("FAIL: --min-new-tokens cannot be greater than --max-new-tokens", file=sys.stderr)
        return 1
    tests = [int(x.strip()) for x in args.tests.split(",") if x.strip()]
    device, dtype = resolve_runtime(args.device)
    maybe_limit_mps_memory(device, args.mps_mem_fraction)
    use_cache = args.use_cache if args.use_cache is not None else (device.type == "cuda")

    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("TEST 2: MedGemma Multi-Image - GO/NO-GO Decision")
    print("=" * 60)
    print(f"Runtime: device={device}, dtype={dtype}, use_cache={use_cache}")
    if device.type == "mps":
        print_mps_stats()

    images = download_public_images()

    print("\nCreating synthetic H&E patches...")
    for idx in range(args.synthetic_patches):
        nuclei = int(rng.choice([200, 500, 1000, 1500, 2000]))
        images.append(make_synthetic_patch(rng, n_nuclei=nuclei))
        print(f"  OK synthetic_patch_{idx}: 512x512 ({nuclei} nuclei)")

    if not images:
        print("No test images available. Exiting.")
        return 1

    print(f"\nTotal available images: {len(images)}")

    print("\nLoading MedGemma...")
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if device.type == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["offload_buffers"] = True
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.fp32_precision = "tf32"
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    print("Model loaded.")

    best_success = 0
    for n_imgs in tests:
        print(f"\n{'=' * 40}\nTesting with {n_imgs} image(s)...\n{'=' * 40}")

        if n_imgs > len(images):
            print(f"  SKIP: only {len(images)} images available")
            continue

        messages = build_messages(images, n_imgs)
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = move_inputs_to_device(dict(inputs), device=device, dtype=dtype)

            input_len = inputs["input_ids"].shape[-1]
            print(f"  Input tokens: {input_len}")

            start = time.time()
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    use_cache=use_cache,
                )
            elapsed = time.time() - start
            new_tokens = generation[0][input_len:]
            print(f"  Generated tokens: {new_tokens.shape[-1]}")
            decoded_list = processor.post_process_image_text_to_text(generation, skip_special_tokens=True)
            decoded = decoded_list[0].split("model\n")[-1].strip() if decoded_list else ""
            if not decoded.strip():
                with torch.inference_mode():
                    generation = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=args.min_new_tokens,
                        do_sample=True,
                        temperature=0.2,
                        top_p=0.9,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        use_cache=use_cache,
                    )
                new_tokens = generation[0][input_len:]
                print(f"  Generated tokens (retry): {new_tokens.shape[-1]}")
                decoded_list = processor.post_process_image_text_to_text(generation, skip_special_tokens=True)
                decoded = decoded_list[0].split("model\n")[-1].strip() if decoded_list else ""

            valid_json, candidate = extract_json(decoded)
            print(f"  Inference time: {elapsed:.1f}s")
            print(f"  Response preview: {decoded[:400].strip()}")
            print(f"  JSON valid: {'YES' if valid_json else 'NO'}")
            if not valid_json:
                print(f"  JSON candidate preview: {candidate[:200].strip()}")

            if not decoded.strip():
                print(f"  FAIL: {n_imgs} image(s) returned empty output")
                continue
            best_success = max(best_success, n_imgs)
            print(f"  PASS: {n_imgs} image(s)")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL: {type(exc).__name__}: {exc}")
            if n_imgs > 1:
                print("  Fallback: use collage/grid or per-patch aggregation.")

    print("\n" + "=" * 60)
    print("DECISION SUMMARY")
    print("=" * 60)
    print(f"Best working image count: {best_success}")
    print(
        "If 4+ images worked: use multi-image architecture.\n"
        "If only 1-2 worked: use 2x2/3x3 collage fallback.\n"
        "If none worked: verify GPU, auth, and model access."
    )
    cleanup(device)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
