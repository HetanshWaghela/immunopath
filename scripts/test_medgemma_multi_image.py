#!/usr/bin/env python3
"""TEST 2: MedGemma multi-image GO/NO-GO test.

Run:
  python scripts/test_medgemma_multi_image.py
"""

from __future__ import annotations

import argparse
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
        "--tests",
        default="1,2,4,8",
        help="Comma-separated image counts to test (e.g. 1,2,4,8)",
    )
    parser.add_argument("--synthetic-patches", type=int, default=5, help="Number of synthetic patches")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Generation length")
    parser.add_argument("--min-new-tokens", type=int, default=24, help="Minimum generation length")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()


def resolve_runtime() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


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
    device, dtype = resolve_runtime()

    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("TEST 2: MedGemma Multi-Image - GO/NO-GO Decision")
    print("=" * 60)
    print(f"Runtime: device={device}, dtype={dtype}")

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
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    if device.type != "cuda":
        model = model.to(device)
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
            ).to(device)

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
                )
            elapsed = time.time() - start
            new_tokens = generation[0][input_len:]
            print(f"  Generated tokens: {new_tokens.shape[-1]}")
            decoded = processor.decode(new_tokens, skip_special_tokens=True)
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
                    )
                new_tokens = generation[0][input_len:]
                print(f"  Generated tokens (retry): {new_tokens.shape[-1]}")
                decoded = processor.decode(new_tokens, skip_special_tokens=True)

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
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
