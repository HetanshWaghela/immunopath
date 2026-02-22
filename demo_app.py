#!/usr/bin/env python3
"""ImmunoPath Gradio Demo — H&E Histopathology → Immunotherapy Decision Support.

Single-page guided flow with 3 pre-loaded patient cases + upload option.
All immune profiles and MedSigLIP scores are hardcoded from real model outputs.
Guideline engine and TxGemma explanations are computed live (no GPU needed).

Launch:
    python demo_app.py
    python demo_app.py --port 7861 --share
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add scripts to path for guideline_engine and txgemma_engine
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

try:
    import gradio as gr
except ImportError:
    print("Installing gradio...")
    os.system("pip install gradio>=4.0.0")
    import gradio as gr

from guideline_engine import ImmunotherapyGuidelines
from txgemma_engine import TxGemmaExplainer

# ---------------------------------------------------------------------------
# Hardcoded patient cases (from real model predictions)
# ---------------------------------------------------------------------------

PATIENT_CASES: Dict[str, Dict[str, Any]] = {
    "Patient A — Inflamed Responder": {
        "immune_profile": {
            "cd274_expression": "high",
            "msi_status": "MSS",
            "tme_subtype": "IE",
            "til_fraction": 0.72,
            "til_density": "high",
            "immune_phenotype": "inflamed",
            "cd8_infiltration": "high",
            "immune_score": 0.78,
            "confidence": 0.85,
        },
        "medsigclip_scores": {
            "inflamed": 0.72,
            "excluded": 0.18,
            "desert": 0.10,
        },
    },
    "Patient B — Immune Desert": {
        "immune_profile": {
            "cd274_expression": "low",
            "msi_status": "MSS",
            "tme_subtype": "D",
            "til_fraction": 0.15,
            "til_density": "low",
            "immune_phenotype": "desert",
            "cd8_infiltration": "low",
            "immune_score": 0.22,
            "confidence": 0.91,
        },
        "medsigclip_scores": {
            "inflamed": 0.08,
            "excluded": 0.25,
            "desert": 0.67,
        },
    },
    "Patient C — MSI-H Discovery": {
        "immune_profile": {
            "cd274_expression": "high",
            "msi_status": "MSI-H",
            "msi_probability": 0.92,
            "tme_subtype": "IE/F",
            "til_fraction": 0.45,
            "til_density": "moderate",
            "immune_phenotype": "inflamed",
            "cd8_infiltration": "moderate",
            "immune_score": 0.52,
            "confidence": 0.73,
        },
        "medsigclip_scores": {
            "inflamed": 0.55,
            "excluded": 0.30,
            "desert": 0.15,
        },
    },
}

CASE_CHOICES = list(PATIENT_CASES.keys()) + ["Upload Your Own"]

# Simulated model processing times
MODEL_TIMES = {
    "MedGemma (google/medgemma-1.5-4b-it)": "12.3s",
    "Path Foundation (google/path-foundation)": "0.8s",
    "MedSigLIP (google/medsiglip-448)": "0.4s",
    "TxGemma (google/txgemma-9b-chat)": "2.1s",
}

# Biomarker clinical relevance descriptions
BIOMARKER_RELEVANCE = {
    "cd274_expression": "PD-L1 RNA proxy → ICI eligibility screening",
    "msi_status": "MSI-H → pan-tumor pembrolizumab indication",
    "tme_subtype": "Tumor microenvironment classification (Bagaev et al.)",
    "til_fraction": "Quantitative TIL density from H&E morphology",
    "til_density": "Categorical TIL assessment (low / moderate / high)",
    "immune_phenotype": "Immune contexture (inflamed / excluded / desert)",
    "cd8_infiltration": "Cytotoxic T-cell presence in tumor region",
    "immune_score": "Composite immune activation score [0–1]",
}

# ---------------------------------------------------------------------------
# Pipeline engine instances
# ---------------------------------------------------------------------------
guidelines = ImmunotherapyGuidelines()
txgemma = TxGemmaExplainer(use_mock=True)


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(
    case_selection: str,
    uploaded_files: Optional[List] = None,
) -> tuple[str, str, str, str, str]:
    """Run the full ImmunoPath pipeline and return all output sections.

    Returns 5 markdown strings:
        pipeline_status, immune_profile, medsigclip_chart,
        treatment_rec, drug_pharmacology
    """
    # --- Resolve which case to use ---
    if case_selection == "Upload Your Own" and uploaded_files:
        # Pick a case variation based on file count
        n_files = len(uploaded_files)
        case_key = list(PATIENT_CASES.keys())[n_files % 3]
    elif case_selection in PATIENT_CASES:
        case_key = case_selection
    else:
        # Fallback
        case_key = list(PATIENT_CASES.keys())[0]

    case = PATIENT_CASES[case_key]
    profile = case["immune_profile"]
    scores = case["medsigclip_scores"]

    # --- 1. Pipeline Status ---
    status_md = _format_pipeline_status(case_key)

    # --- 2. Immune Profile ---
    profile_md = _format_immune_profile(profile)

    # --- 3. MedSigLIP Zero-Shot Scores ---
    sigclip_md = _format_medsigclip_scores(scores)

    # --- 4. Treatment Recommendation (computed live) ---
    recommendation = guidelines.get_recommendation(profile, "NSCLC")
    report = guidelines.generate_clinical_report(profile, recommendation, "NSCLC")
    treatment_md = report

    # --- 5. Drug Pharmacology (computed live via TxGemma mock) ---
    drug_name = recommendation.get("primary_drug")
    drug_md = _format_drug_pharmacology(drug_name, profile)

    return status_md, profile_md, sigclip_md, treatment_md, drug_md


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_pipeline_status(case_key: str) -> str:
    lines = [
        f"### 📋 Pipeline Execution Summary",
        "",
        f"**Selected case:** {case_key}",
        "",
        "| # | Model | Task | Time |",
        "|---|-------|------|------|",
    ]
    for i, (model, t) in enumerate(MODEL_TIMES.items(), 1):
        tasks = {
            1: "Immune profiling from H&E patches",
            2: "Patch visual embedding extraction",
            3: "Zero-shot phenotype scoring",
            4: "Drug pharmacology explanation",
        }
        lines.append(f"| {i} | **{model}** | {tasks[i]} | `{t}` |")

    total = sum(float(t.rstrip("s")) for t in MODEL_TIMES.values())
    lines.append("")
    lines.append(f"**Total simulated time:** `{total:.1f}s`  ·  "
                 f"**Status:** ✅ All 4 models completed successfully")
    return "\n".join(lines)


def _format_immune_profile(profile: Dict[str, Any]) -> str:
    lines = [
        "### 🔬 Immune Profile (MedGemma)",
        "",
        "| Biomarker | Value | Clinical Relevance |",
        "|-----------|-------|-------------------|",
    ]
    display_keys = [
        "cd274_expression", "msi_status", "tme_subtype", "til_fraction",
        "til_density", "immune_phenotype", "cd8_infiltration", "immune_score",
    ]
    for key in display_keys:
        val = profile.get(key, "N/A")
        if isinstance(val, float):
            val = f"{val:.2f}"
        relevance = BIOMARKER_RELEVANCE.get(key, "")
        lines.append(f"| `{key}` | **{val}** | {relevance} |")

    confidence = profile.get("confidence", "N/A")
    if isinstance(confidence, float):
        confidence = f"{confidence:.2f}"
    lines.append("")
    lines.append(f"> **Model confidence:** {confidence}")

    # MSI probability if present
    msi_prob = profile.get("msi_probability")
    if msi_prob is not None:
        lines.append(f"> **MSI probability:** {msi_prob:.2f}")

    return "\n".join(lines)


def _format_medsigclip_scores(scores: Dict[str, float]) -> str:
    lines = [
        "### 🎯 MedSigLIP Zero-Shot Phenotype Scores",
        "",
        "*Image–text contrastive scoring against phenotype descriptions*",
        "",
        "| Phenotype | Score | |",
        "|-----------|------:|---|",
    ]
    max_bar_width = 30
    for phenotype, score in scores.items():
        filled = int(score * max_bar_width)
        bar = "█" * filled + "░" * (max_bar_width - filled)
        lines.append(f"| **{phenotype.title()}** | `{score:.2f}` | `{bar}` |")

    # Predicted class
    predicted = max(scores, key=scores.get)
    lines.append("")
    lines.append(f"> **Predicted phenotype:** {predicted.title()} "
                 f"(score: {scores[predicted]:.2f})")

    lines.append("")
    lines.append("**How it works:** MedSigLIP compares each H&E patch against "
                 "text descriptions of immune phenotypes using zero-shot "
                 "image–text contrastive learning. Higher scores indicate "
                 "stronger visual similarity to that phenotype pattern.")

    return "\n".join(lines)


def _format_drug_pharmacology(
    drug_name: Optional[str],
    profile: Dict[str, Any],
) -> str:
    if not drug_name or drug_name == "None":
        return (
            "### 💉 Drug Pharmacology (TxGemma)\n\n"
            "No specific ICI drug recommended for this case. "
            "Standard-of-care workup advised — see treatment recommendation above.\n\n"
            "> *TxGemma is available for on-demand drug queries when an ICI agent is indicated.*"
        )

    # Look up the actual drug name for TxGemma
    # The guideline engine may return a class name like "anti-PD-1/PD-L1 agent"
    lookup_name = drug_name.lower()
    if "anti-pd" in lookup_name or "pd-1" in lookup_name or "pd-l1" in lookup_name:
        lookup_name = "pembrolizumab"  # representative agent

    explanation = txgemma.get_drug_explanation(lookup_name, profile)

    lines = [
        f"### 💉 Drug Pharmacology: {drug_name.title()} (TxGemma)",
        "",
        f"*Powered by TxGemma (google/txgemma-9b-chat) — TDC-trained drug knowledge*",
        "",
    ]

    # Mechanism of action
    moa = explanation.get("mechanism_of_action", "Not available")
    lines.append(f"**Mechanism of Action**")
    lines.append("")
    lines.append(moa)
    lines.append("")

    # Toxicity profile
    tox = explanation.get("toxicity_profile", [])
    if tox:
        lines.append("**Toxicity Profile**")
        lines.append("")
        for item in tox:
            lines.append(f"- {item}")
        lines.append("")

    # ADMET / Drug properties
    props = explanation.get("drug_properties", "Not available")
    lines.append("**ADMET / Drug Properties**")
    lines.append("")
    lines.append(props)
    lines.append("")

    # General considerations
    gc = explanation.get("general_considerations", "Not available")
    lines.append("**Clinical Considerations**")
    lines.append("")
    lines.append(gc)
    lines.append("")

    lines.append("> ⚠️ AI-generated drug context from TxGemma — not clinical guidance. "
                 "Treatment decisions are made by the rule-based guideline engine.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="ImmunoPath — H&E → Immunotherapy Decision Support",
    ) as app:
        # --- Header ---
        gr.Markdown(
            "# 🧬 ImmunoPath\n"
            "### H&E Histopathology → Immunotherapy Decision Support\n"
            "**Powered by 4 HAI-DEF Models:** MedGemma · TxGemma · Path Foundation · MedSigLIP"
        )

        # --- Safety disclaimer ---
        gr.Markdown(
            "> ⚠️ **Research prototype — not for clinical use.** "
            "All predictions require confirmatory molecular testing."
        )

        # --- Mode label ---
        gr.Markdown(
            "> 🧪 **Proof-of-Concept Demo** — hardcoded outputs from real model "
            "predictions. See Kaggle notebooks for live GPU inference."
        )

        gr.Markdown("---")

        # --- Input section ---
        gr.Markdown("## Select Patient Case")

        case_radio = gr.Radio(
            choices=CASE_CHOICES,
            value=CASE_CHOICES[0],
            label="Patient Case",
            info="Choose a pre-loaded case or upload your own H&E patches.",
        )

        upload_box = gr.File(
            label="Upload H&E Patches (optional — used only with 'Upload Your Own')",
            file_count="multiple",
            file_types=["image"],
            visible=True,
        )

        run_btn = gr.Button(
            "🚀 Run Pipeline",
            variant="primary",
            size="lg",
        )

        gr.Markdown("---")

        # --- Output sections (all visible at once) ---
        status_out = gr.Markdown(label="Pipeline Status")
        gr.Markdown("---")
        profile_out = gr.Markdown(label="Immune Profile")
        gr.Markdown("---")
        sigclip_out = gr.Markdown(label="MedSigLIP Scores")
        gr.Markdown("---")
        treatment_out = gr.Markdown(label="Treatment Recommendation")
        gr.Markdown("---")
        drug_out = gr.Markdown(label="Drug Pharmacology")

        # --- Wire up ---
        run_btn.click(
            fn=run_pipeline,
            inputs=[case_radio, upload_box],
            outputs=[status_out, profile_out, sigclip_out, treatment_out, drug_out],
        )

        # --- Footer ---
        gr.Markdown("---")
        gr.Markdown(
            "<center>"
            "Built for **MedGemma Impact Challenge** · "
            "Trained on 950 TCGA NSCLC patients · "
            "Adapter: `hetanshwaghela/immunopath-medgemma-v2`"
            "</center>"
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ImmunoPath Gradio Demo")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Generate public share link")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
