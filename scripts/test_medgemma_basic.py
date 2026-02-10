#!/usr/bin/env python3
"""TEST 1: MedGemma single-image sanity check.

Run:
  python scripts/test_medgemma_basic.py
"""

from __future__ import annotations

import argparse
import gc
import sys

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL_ID = "google/medgemma-1.5-4b-it"
DEFAULT_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a basic MedGemma single-image check.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL, help="Public image URL")
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
        "--image-max-side",
        type=int,
        default=1024,
        help="Downscale input image so its longest side is this many pixels.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Generation length")
    parser.add_argument("--min-new-tokens", type=int, default=24, help="Minimum generation length")
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable KV cache (faster, higher memory). Default: on for CUDA, off otherwise.",
    )
    return parser.parse_args()


def fetch_image(image_url: str) -> Image.Image:
    response = requests.get(image_url, headers={"User-Agent": "immunopath-test"}, stream=True, timeout=60)
    response.raise_for_status()
    return Image.open(response.raw).convert("RGB")


def make_synthetic_patch(size: int = 512, n_nuclei: int = 800) -> Image.Image:
    rng = np.random.default_rng(13)
    image = Image.new("RGB", (size, size), color=(230, 180, 190))
    draw = ImageDraw.Draw(image)
    for _ in range(n_nuclei):
        x = int(rng.integers(10, size - 10))
        y = int(rng.integers(10, size - 10))
        r = int(rng.integers(2, 5))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(80, 40, 120))
    return image


def resolve_runtime(requested_device: str) -> tuple[str, torch.dtype]:
    if requested_device != "auto":
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda but CUDA is not available")
        if requested_device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested device=mps but MPS is not available")
        dtype = torch.bfloat16 if requested_device == "cuda" else torch.float16 if requested_device == "mps" else torch.float32
        return requested_device, dtype

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def move_inputs_to_device(inputs: dict, device: str, dtype: torch.dtype) -> dict:
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


def maybe_limit_mps_memory(device: str, fraction: float | None) -> None:
    if device != "mps":
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

def cleanup(device: str) -> None:
    gc.collect()
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    if args.min_new_tokens > args.max_new_tokens:
        print("FAIL: --min-new-tokens cannot be greater than --max-new-tokens", file=sys.stderr)
        return 1
    device, dtype = resolve_runtime(args.device)
    maybe_limit_mps_memory(device, args.mps_mem_fraction)
    use_cache = args.use_cache if args.use_cache is not None else (device == "cuda")

    print("=" * 60)
    print("TEST 1: MedGemma Single Image - Basic Sanity Check")
    print("=" * 60)
    print(f"Runtime: device={device}, dtype={dtype}, use_cache={use_cache}")
    if device == "mps":
        print_mps_stats()

    print(f"\nDownloading test image: {args.image_url}")
    try:
        image = fetch_image(args.image_url)
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed ({type(exc).__name__}). Falling back to synthetic patch.")
        image = make_synthetic_patch()
    if args.image_max_side and max(image.size) > args.image_max_side:
        image.thumbnail((args.image_max_side, args.image_max_side))
    print(f"Image size: {image.size}, mode: {image.mode}")

    print("\nLoading MedGemma 1.5 4B (this may take 1-3 minutes)...")
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
    model_kwargs: dict[str, object] = {"dtype": dtype}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["offload_buffers"] = True
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.fp32_precision = "tf32"
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    print("Model loaded.")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert pathologist."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what you see in this medical image in 2-3 sentences."},
                {"type": "image", "image": image},
            ],
        },
    ]

    print("\nRunning inference...")
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(dict(inputs), device=device, dtype=dtype)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=use_cache,
        )
    new_tokens = generation[0][input_len:]
    print(f"Generated tokens: {new_tokens.shape[-1]}")
    decoded_list = processor.post_process_image_text_to_text(generation, skip_special_tokens=True)
    response = decoded_list[0].split("model\n")[-1].strip() if decoded_list else ""
    if not response:
        print("Empty output on first pass; retrying with light sampling...")
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
        print(f"Generated tokens (retry): {new_tokens.shape[-1]}")
        decoded_list = processor.post_process_image_text_to_text(generation, skip_special_tokens=True)
        response = decoded_list[0].split("model\n")[-1].strip() if decoded_list else ""

    print("\n--- MODEL RESPONSE ---")
    print(response)
    print("--- END ---")
    if not response:
        print("\nFAIL: Empty response generated.", file=sys.stderr)
        cleanup(device)
        return 1
    print("\nPASS: MedGemma can process a single image.")
    cleanup(device)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
