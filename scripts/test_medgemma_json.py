#!/usr/bin/env python3
"""TEST 3: MedGemma JSON output reliability.

Run:
  python scripts/test_medgemma_json.py
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL_ID = "google/medgemma-1.5-4b-it"
IMMUNOPATH_PROMPT = """Analyze these H&E-stained histopathology images from a lung adenocarcinoma tumor.

Extract the following H&E-inferred immune signals as a research output (not diagnostic):
1. CD274 (PD-L1) RNA proxy level (high/low)
2. MSI status (MSI-H or MSS) + probability
3. TIL fraction (0.0-1.0) + bucket (low/moderate/high)
4. TME subtype (IE / IE/F / F / D)
5. Immune phenotype (inflamed/excluded/desert)
6. CD8+ T-cell infiltration (low/moderate/high)
7. Overall immune score (0.0-1.0)

Provide your analysis as a JSON object only. No other text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure MedGemma JSON parse reliability.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--trials", type=int, default=10, help="Number of generation trials")
    parser.add_argument("--max-new-tokens", type=int, default=220, help="Generation length")
    parser.add_argument("--min-new-tokens", type=int, default=24, help="Minimum generation length")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    return parser.parse_args()


def resolve_runtime() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def make_patch(rng: np.random.Generator, n_nuclei: int = 800, size: int = 512) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(230, 180, 190))
    draw = ImageDraw.Draw(image)
    for _ in range(n_nuclei):
        x = int(rng.integers(10, size - 10))
        y = int(rng.integers(10, size - 10))
        r = int(rng.integers(2, 5))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(80, 40, 120))
    return image


def parse_json_candidate(decoded: str) -> tuple[bool, dict[str, object] | None, str]:
    clean = decoded.strip()
    if clean.startswith("```"):
        clean = clean.replace("```json", "").replace("```", "").strip()

    start = clean.find("{")
    end = clean.rfind("}")
    candidate = clean[start : end + 1] if start != -1 and end != -1 and end > start else clean

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return True, parsed, candidate
        return False, None, candidate
    except json.JSONDecodeError:
        return False, None, candidate


def main() -> int:
    args = parse_args()
    if args.min_new_tokens > args.max_new_tokens:
        print("FAIL: --min-new-tokens cannot be greater than --max-new-tokens", file=sys.stderr)
        return 1
    rng = np.random.default_rng(args.seed)
    device, dtype = resolve_runtime()

    print("=" * 60)
    print("TEST 3: MedGemma JSON Output Reliability")
    print("=" * 60)
    print(f"Runtime: device={device}, dtype={dtype}")

    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
    model_kwargs: dict[str, object] = {"dtype": dtype}
    if device.type == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    if device.type != "cuda":
        model = model.to(device)

    valid_json = 0
    for trial in range(args.trials):
        patch = make_patch(rng, n_nuclei=int(rng.integers(200, 2000)))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": patch},
                    {"type": "text", "text": IMMUNOPATH_PROMPT},
                ],
            }
        ]

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
        decoded = processor.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"Trial {trial + 1}: generated_tokens={new_tokens.shape[-1]}")
        if not decoded:
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
            decoded = processor.decode(new_tokens, skip_special_tokens=True).strip()
            print(f"Trial {trial + 1}: generated_tokens_retry={new_tokens.shape[-1]}")
        is_valid, parsed, _ = parse_json_candidate(decoded)
        if is_valid:
            valid_json += 1
            keys = list(parsed.keys()) if parsed else []
            print(f"Trial {trial + 1}: VALID JSON - keys: {keys}")
        else:
            print(f"Trial {trial + 1}: INVALID JSON - preview: {decoded[:180]}")

    rate = (valid_json / args.trials) * 100 if args.trials else 0.0
    print(f"\n{'=' * 40}")
    print(f"JSON Parse Rate: {valid_json}/{args.trials} ({rate:.0f}%)")
    print(f"{'=' * 40}")

    if rate > 80:
        decision = "JSON output is reliable; proceed with this schema."
    elif rate >= 50:
        decision = "Add JSON repair/post-processing or constrained decoding."
    else:
        decision = "Simplify schema and/or enforce stronger output constraints."

    print(f"Decision: {decision}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
