#!/usr/bin/env python3
"""TEST 3: MedGemma JSON output reliability.

Run:
  python scripts/test_medgemma_json.py
"""

from __future__ import annotations

import argparse
import gc
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
    parser.add_argument("--trials", type=int, default=10, help="Number of generation trials")
    parser.add_argument("--max-new-tokens", type=int, default=220, help="Generation length")
    parser.add_argument("--min-new-tokens", type=int, default=24, help="Minimum generation length")
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable KV cache (faster, higher memory). Default: on for CUDA, off otherwise.",
    )
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
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
    device, dtype = resolve_runtime(args.device)
    maybe_limit_mps_memory(device, args.mps_mem_fraction)
    use_cache = args.use_cache if args.use_cache is not None else (device.type == "cuda")

    print("=" * 60)
    print("TEST 3: MedGemma JSON Output Reliability")
    print("=" * 60)
    print(f"Runtime: device={device}, dtype={dtype}, use_cache={use_cache}")
    if device.type == "mps":
        print_mps_stats()

    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=False)
    model_kwargs: dict[str, object] = {"dtype": dtype}
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
        decoded_list = processor.post_process_image_text_to_text(generation, skip_special_tokens=True)
        decoded = decoded_list[0].split("model\n")[-1].strip() if decoded_list else ""
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
                    use_cache=use_cache,
                )
            new_tokens = generation[0][input_len:]
            decoded_list = processor.post_process_image_text_to_text(generation, skip_special_tokens=True)
            decoded = decoded_list[0].split("model\n")[-1].strip() if decoded_list else ""
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
    cleanup(device)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
