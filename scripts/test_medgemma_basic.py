#!/usr/bin/env python3
"""TEST 1: MedGemma single-image sanity check.

Run:
  python scripts/test_medgemma_basic.py
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Generation length")
    parser.add_argument("--min-new-tokens", type=int, default=24, help="Minimum generation length")
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


def resolve_runtime() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def main() -> int:
    args = parse_args()
    if args.min_new_tokens > args.max_new_tokens:
        print("FAIL: --min-new-tokens cannot be greater than --max-new-tokens", file=sys.stderr)
        return 1
    device, dtype = resolve_runtime()

    print("=" * 60)
    print("TEST 1: MedGemma Single Image - Basic Sanity Check")
    print("=" * 60)
    print(f"Runtime: device={device}, dtype={dtype}")

    print(f"\nDownloading test image: {args.image_url}")
    try:
        image = fetch_image(args.image_url)
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed ({type(exc).__name__}). Falling back to synthetic patch.")
        image = make_synthetic_patch()
    print(f"Image size: {image.size}, mode: {image.mode}")

    print("\nLoading MedGemma 1.5 4B (this may take 1-3 minutes)...")
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
    model_kwargs: dict[str, object] = {"dtype": dtype}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    if device != "cuda":
        model = model.to(device)
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
    ).to(device)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    new_tokens = generation[0][input_len:]
    print(f"Generated tokens: {new_tokens.shape[-1]}")
    response = processor.decode(new_tokens, skip_special_tokens=True).strip()
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
            )
        new_tokens = generation[0][input_len:]
        print(f"Generated tokens (retry): {new_tokens.shape[-1]}")
        response = processor.decode(new_tokens, skip_special_tokens=True).strip()

    print("\n--- MODEL RESPONSE ---")
    print(response)
    print("--- END ---")
    if not response:
        print("\nFAIL: Empty response generated.", file=sys.stderr)
        return 1
    print("\nPASS: MedGemma can process a single image.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
