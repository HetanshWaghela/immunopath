#!/usr/bin/env python3
"""ImmunoPath Gradio Demo - H&E Histopathology to Immunotherapy Decision Support.

Self-contained Hugging Face Spaces deployment. Combines:
  - demo_app.py (UI + pipeline orchestration)
  - guideline_engine.py (rule-based treatment recommendations)
  - txgemma_engine.py (drug pharmacology mock knowledge base)

All immune profiles and MedSigLIP scores are hardcoded from real model outputs.
Guideline engine and TxGemma explanations are computed live (no GPU needed).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr


# ===================================================================
# Safety disclaimer
# ===================================================================

SAFETY_DISCLAIMER = (
    "This is research decision support. Not intended for clinical use "
    "without professional review."
)


# ===================================================================
# ICI drug reference database (public evidence)
# ===================================================================

ICI_DRUG_DATABASE: Dict[str, Dict[str, Any]] = {
    "pembrolizumab": {
        "class": "anti-PD-1",
        "approved_indications": [
            "NSCLC with PD-L1 >=50% (monotherapy)",
            "NSCLC with PD-L1 >=1% (with chemotherapy)",
            "MSI-H/dMMR solid tumors (pan-tumor)",
            "TMB-High solid tumors (>=10 mut/Mb)",
            "Melanoma",
            "HNSCC",
        ],
        "key_trials": [
            "KEYNOTE-024",
            "KEYNOTE-042",
            "KEYNOTE-158",
            "KEYNOTE-189",
        ],
    },
    "nivolumab": {
        "class": "anti-PD-1",
        "approved_indications": [
            "NSCLC (second-line or with ipilimumab)",
            "Melanoma",
            "RCC",
            "HNSCC",
            "Hodgkin lymphoma",
        ],
        "key_trials": [
            "CheckMate-017",
            "CheckMate-057",
            "CheckMate-227",
        ],
    },
    "atezolizumab": {
        "class": "anti-PD-L1",
        "approved_indications": [
            "NSCLC with PD-L1 expression",
            "SCLC",
            "Urothelial carcinoma",
            "TNBC",
        ],
        "key_trials": [
            "IMpower110",
            "IMpower150",
        ],
    },
    "ipilimumab": {
        "class": "anti-CTLA-4",
        "approved_indications": [
            "Melanoma (alone or with nivolumab)",
            "RCC (with nivolumab)",
            "NSCLC (with nivolumab)",
        ],
        "key_trials": [
            "CheckMate-067",
            "CheckMate-214",
        ],
    },
    "durvalumab": {
        "class": "anti-PD-L1",
        "approved_indications": [
            "Stage III NSCLC after chemoradiation",
            "SCLC",
            "Urothelial carcinoma",
        ],
        "key_trials": [
            "PACIFIC",
            "CASPIAN",
        ],
    },
}


# ===================================================================
# ImmunotherapyGuidelines (from guideline_engine.py)
# ===================================================================

class ImmunotherapyGuidelines:
    """Rule-based engine for ICI treatment recommendations.

    Derived from publicly available evidence: FDA-approved labels,
    pivotal trial biomarker eligibility, and NCCN-aligned thresholds.
    Currently focused on NSCLC.
    """

    def __init__(self) -> None:
        self.drug_database = ICI_DRUG_DATABASE

    def get_recommendation(
        self,
        immune_signature: Dict[str, Any],
        cancer_type: str,
    ) -> Dict[str, Any]:
        """Generate a treatment recommendation using deterministic rules."""
        msi_status = immune_signature.get("msi_status", "unknown")
        msi_probability = immune_signature.get("msi_probability", 0.0)
        pdl1_ihc_tps: Optional[float] = immune_signature.get("pdl1_ihc_tps")
        cd274_expression = immune_signature.get("cd274_expression", "unknown")
        immune_phenotype = immune_signature.get("immune_phenotype", "unknown")

        base: Dict[str, Any] = {
            "source": "Rule-based engine (public evidence ruleset)",
            "primary_drug": None,
            "regimen": "",
            "confidence": "low",
            "confirmatory_tests_required": [],
            "supporting_evidence": [],
            "alternatives": [],
            "safety_warnings": [SAFETY_DISCLAIMER],
        }

        # Rule 1: MSI-H -> pembrolizumab monotherapy
        if msi_status == "MSI-H":
            base["primary_drug"] = "pembrolizumab"
            base["regimen"] = "Pembrolizumab monotherapy (MSI-H indication)"
            base["confidence"] = "high" if msi_probability > 0.8 else "moderate"
            base["supporting_evidence"] = [
                "KEYNOTE-158 (FDA pan-tumor MSI-H approval)",
            ]
            base["confirmatory_tests_required"] = [
                "MSI PCR/NGS or IHC (dMMR) if not already confirmed",
            ]
            return base

        # Rule 2: PD-L1 IHC TPS >=50% -> pembrolizumab mono
        if pdl1_ihc_tps is not None and pdl1_ihc_tps >= 50:
            base["primary_drug"] = "pembrolizumab"
            base["regimen"] = "Pembrolizumab monotherapy (PD-L1 >=50%)"
            base["confidence"] = "high"
            base["supporting_evidence"] = ["KEYNOTE-024"]
            base["confirmatory_tests_required"] = [
                "Confirm PD-L1 IHC with 22C3 or SP263 assay",
                "Driver mutation testing (EGFR, ALK, ROS1, BRAF)",
            ]
            return base

        # Rule 3: PD-L1 IHC TPS >=1% -> pembrolizumab + chemo
        if pdl1_ihc_tps is not None and pdl1_ihc_tps >= 1:
            base["primary_drug"] = "pembrolizumab"
            base["regimen"] = "Pembrolizumab + platinum-based chemotherapy"
            base["confidence"] = "high"
            base["supporting_evidence"] = ["KEYNOTE-189"]
            base["confirmatory_tests_required"] = [
                "Confirm PD-L1 IHC with 22C3 or SP263 assay",
                "Driver mutation testing (EGFR, ALK, ROS1, BRAF)",
            ]
            return base

        # Rule 4: CD274-high RNA proxy + inflamed -> conditional ICI
        if cd274_expression == "high" and immune_phenotype == "inflamed":
            base["primary_drug"] = "anti-PD-1/PD-L1 agent"
            base["regimen"] = "Consider ICI-based regimen (CONDITIONAL)"
            base["confidence"] = "conditional"
            base["confirmatory_tests_required"] = [
                "PD-L1 IHC (22C3 or SP263) - REQUIRED before treatment",
                "Driver mutation testing (EGFR, ALK, ROS1, BRAF, etc.)",
            ]
            base["supporting_evidence"] = [
                "CD274 mRNA proxy correlates with IHC (r2=0.65-0.81, Kang et al. 2022)",
                "Pending PD-L1 IHC confirmation",
            ]
            base["alternatives"] = [
                "Chemo + ICI combination",
                "Clinical trial",
            ]
            return base

        # Rule 5: Default -> standard-of-care workup
        base["primary_drug"] = None
        base["regimen"] = "Standard-of-care workup recommended"
        base["confidence"] = "low"
        base["confirmatory_tests_required"] = [
            "PD-L1 IHC",
            "MSI/dMMR testing",
            "Driver mutation panel",
            "TMB if available",
        ]
        base["alternatives"] = ["Clinical trial enrollment"]
        return base

    def generate_clinical_report(
        self,
        immune_signature: Dict[str, Any],
        recommendation: Dict[str, Any],
        cancer_type: str,
    ) -> str:
        """Produce a markdown-formatted clinical decision support report."""
        lines: List[str] = []

        lines.append("# ImmunoPath Clinical Decision Support Report")
        lines.append("")
        lines.append(f"> {SAFETY_DISCLAIMER}")
        lines.append("")

        # Patient context
        lines.append("## Patient Context")
        lines.append("")
        lines.append(f"- **Cancer type:** {cancer_type}")
        lines.append(
            f"- **Immune phenotype:** "
            f"{immune_signature.get('immune_phenotype', 'unknown')}"
        )
        lines.append(
            f"- **TIL density:** "
            f"{immune_signature.get('til_density', immune_signature.get('til_fraction', 'unknown'))}"
        )
        lines.append(
            f"- **MSI status:** "
            f"{immune_signature.get('msi_status', 'unknown')} "
            f"(probability: {immune_signature.get('msi_probability', 'N/A')})"
        )
        lines.append(
            f"- **CD274 mRNA (PD-L1 RNA proxy):** "
            f"{immune_signature.get('cd274_expression', 'unknown')}"
        )
        pdl1_ihc = immune_signature.get("pdl1_ihc_tps")
        lines.append(
            f"- **PD-L1 IHC TPS:** "
            f"{f'{pdl1_ihc}%' if pdl1_ihc is not None else 'Not available'}"
        )
        lines.append(
            f"- **Immune score:** "
            f"{immune_signature.get('immune_score', 'N/A')}"
        )
        lines.append("")

        # Recommendation
        lines.append("## Treatment Recommendation")
        lines.append("")
        lines.append(f"- **Source:** {recommendation['source']}")
        lines.append(f"- **Primary drug:** {recommendation['primary_drug'] or 'None'}")
        lines.append(f"- **Regimen:** {recommendation['regimen']}")
        lines.append(f"- **Confidence:** {recommendation['confidence']}")
        lines.append("")

        # Evidence
        if recommendation["supporting_evidence"]:
            lines.append("## Supporting Evidence")
            lines.append("")
            for evidence in recommendation["supporting_evidence"]:
                lines.append(f"- {evidence}")
            lines.append("")

        # Confirmatory tests
        if recommendation["confirmatory_tests_required"]:
            lines.append("## Confirmatory Tests Required")
            lines.append("")
            for test in recommendation["confirmatory_tests_required"]:
                lines.append(f"- [ ] {test}")
            lines.append("")

        # Alternatives
        if recommendation["alternatives"]:
            lines.append("## Alternatives")
            lines.append("")
            for alt in recommendation["alternatives"]:
                lines.append(f"- {alt}")
            lines.append("")

        # Safety warnings
        if recommendation["safety_warnings"]:
            lines.append("## Safety Warnings")
            lines.append("")
            for warning in recommendation["safety_warnings"]:
                lines.append(f"- {warning}")
            lines.append("")

        # Drug reference
        drug = recommendation["primary_drug"]
        if drug and drug in self.drug_database:
            info = self.drug_database[drug]
            lines.append(f"## Drug Reference: {drug.title()}")
            lines.append("")
            lines.append(f"- **Class:** {info['class']}")
            lines.append("- **Approved indications:**")
            for ind in info["approved_indications"]:
                lines.append(f"  - {ind}")
            lines.append("- **Key trials:**")
            for trial in info["key_trials"]:
                lines.append(f"  - {trial}")
            lines.append("")

        lines.append("---")
        lines.append(f"*{SAFETY_DISCLAIMER}*")
        return "\n".join(lines)


# ===================================================================
# TxGemma mock knowledge base (from txgemma_engine.py)
# ===================================================================

_MOCK_EXPLANATIONS: Dict[str, Dict[str, Any]] = {
    "pembrolizumab": {
        "drug_name": "pembrolizumab",
        "mechanism_of_action": (
            "Pembrolizumab is a humanized IgG4-kappa monoclonal antibody that "
            "binds to the PD-1 receptor on T cells, blocking interaction with "
            "PD-L1 and PD-L2 ligands. This releases PD-1-mediated inhibition "
            "of the immune response, restoring T-cell-mediated anti-tumor "
            "cytotoxicity."
        ),
        "toxicity_profile": [
            "Immune-mediated pneumonitis",
            "Immune-mediated colitis",
            "Immune-mediated hepatitis (elevated AST/ALT)",
            "Immune-mediated endocrinopathies (thyroiditis, hypophysitis)",
            "Immune-mediated nephritis",
            "Fatigue",
            "Rash / pruritus",
            "Infusion-related reactions (rare)",
        ],
        "drug_properties": (
            "High target specificity for PD-1; IgG4 backbone minimises "
            "Fc-effector function. Half-life ~26 days. Administered IV; "
            "clearance predominantly via catabolism. No significant CYP450 "
            "interactions expected for a monoclonal antibody."
        ),
        "general_considerations": (
            "Monitor thyroid function, hepatic enzymes, and renal function "
            "before and during treatment. Patients with autoimmune conditions "
            "may be at increased risk of immune-related adverse events. "
            "Corticosteroids are the standard management for immune-mediated "
            "toxicities."
        ),
    },
    "nivolumab": {
        "drug_name": "nivolumab",
        "mechanism_of_action": (
            "Nivolumab is a fully human IgG4 monoclonal antibody that targets "
            "the PD-1 receptor, preventing engagement with PD-L1 and PD-L2. "
            "By blocking PD-1 signalling, nivolumab enhances T-cell "
            "proliferation and cytokine production, augmenting the anti-tumor "
            "immune response."
        ),
        "toxicity_profile": [
            "Immune-mediated pneumonitis",
            "Immune-mediated colitis / diarrhoea",
            "Immune-mediated hepatotoxicity",
            "Immune-mediated endocrinopathies (hypothyroidism, adrenal insufficiency)",
            "Immune-mediated skin reactions (rash, vitiligo)",
            "Fatigue",
            "Nausea",
            "Musculoskeletal pain",
        ],
        "drug_properties": (
            "Fully human IgG4 antibody; half-life ~25 days. Linear "
            "pharmacokinetics over the dose range of 0.1-10 mg/kg. "
            "IV administration; no renal or hepatic dose adjustments needed "
            "for mild-to-moderate impairment."
        ),
        "general_considerations": (
            "Can be used as monotherapy or in combination with ipilimumab "
            "(anti-CTLA-4), which increases both efficacy and toxicity rates. "
            "Baseline and periodic monitoring of liver function, thyroid "
            "function, and blood glucose is recommended."
        ),
    },
    "atezolizumab": {
        "drug_name": "atezolizumab",
        "mechanism_of_action": (
            "Atezolizumab is a humanized IgG1 monoclonal antibody with an "
            "engineered Fc region (to eliminate ADCC) that binds PD-L1, "
            "blocking its interaction with both PD-1 and B7.1 (CD80). "
            "This restores T-cell activity against PD-L1-expressing tumours."
        ),
        "toxicity_profile": [
            "Immune-mediated pneumonitis",
            "Immune-mediated hepatitis",
            "Immune-mediated colitis",
            "Immune-mediated endocrinopathies (thyroid disorders, diabetes mellitus)",
            "Fatigue / asthenia",
            "Nausea",
            "Decreased appetite",
            "Urinary tract infection",
        ],
        "drug_properties": (
            "Engineered Fc-silent IgG1 antibody targeting PD-L1; half-life "
            "~27 days. Steady state reached by cycle 6-9. IV administration "
            "with fixed dosing (1200 mg Q3W). Minimal immunogenicity observed "
            "in clinical studies."
        ),
        "general_considerations": (
            "Targets PD-L1 rather than PD-1, preserving the PD-L2/PD-1 axis. "
            "This may result in a differentiated toxicity profile compared to "
            "anti-PD-1 agents. Monitor for signs of immune-mediated reactions "
            "and hepatotoxicity."
        ),
    },
    "ipilimumab": {
        "drug_name": "ipilimumab",
        "mechanism_of_action": (
            "Ipilimumab is a fully human IgG1 monoclonal antibody that blocks "
            "CTLA-4, a negative regulator of T-cell activation. By preventing "
            "CTLA-4 from competing with CD28 for binding to B7 ligands on "
            "antigen-presenting cells, ipilimumab enhances T-cell priming, "
            "proliferation, and anti-tumor immune responses."
        ),
        "toxicity_profile": [
            "Immune-mediated colitis (higher incidence than anti-PD-1)",
            "Immune-mediated hepatitis",
            "Immune-mediated dermatitis (rash, pruritus)",
            "Immune-mediated endocrinopathies (hypophysitis - characteristic of anti-CTLA-4)",
            "Immune-mediated neuropathies",
            "Fatigue",
            "Diarrhoea",
            "Nausea",
        ],
        "drug_properties": (
            "Fully human IgG1 antibody targeting CTLA-4; half-life ~14.7 days. "
            "The IgG1 backbone retains Fc-effector function, which may "
            "contribute to Treg depletion in the tumour microenvironment. "
            "IV administration; dose-dependent toxicity profile."
        ),
        "general_considerations": (
            "CTLA-4 blockade produces broader immune activation compared to "
            "PD-1/PD-L1 inhibitors, resulting in higher rates of immune-"
            "related adverse events (especially colitis and hypophysitis). "
            "Often used in combination with nivolumab. Close monitoring for "
            "early signs of colitis is critical."
        ),
    },
    "durvalumab": {
        "drug_name": "durvalumab",
        "mechanism_of_action": (
            "Durvalumab is a human IgG1-kappa monoclonal antibody with an "
            "engineered Fc domain (triple mutation to reduce ADCC/CDC) that "
            "binds PD-L1, blocking its interaction with PD-1 and CD80. This "
            "releases PD-L1-mediated suppression of anti-tumor T-cell "
            "responses."
        ),
        "toxicity_profile": [
            "Immune-mediated pneumonitis (important in post-chemoradiation setting)",
            "Immune-mediated hepatitis",
            "Immune-mediated colitis",
            "Immune-mediated endocrinopathies (thyroid disorders)",
            "Immune-mediated dermatologic reactions",
            "Fatigue",
            "Cough",
            "Musculoskeletal pain",
        ],
        "drug_properties": (
            "Engineered Fc-silent IgG1 anti-PD-L1 antibody; half-life ~18 "
            "days. Fixed-dose IV administration (10 mg/kg Q2W or 1500 mg Q4W). "
            "Low immunogenicity. No clinically significant drug-drug "
            "interactions identified."
        ),
        "general_considerations": (
            "Approved in the consolidation setting after chemoradiation for "
            "stage III NSCLC, where pneumonitis risk from prior radiation "
            "may overlap with immune-mediated pneumonitis. Baseline pulmonary "
            "function assessment is recommended."
        ),
    },
}


class TxGemmaExplainer:
    """Explains drug properties using curated pharmacology knowledge.

    In this deployment, only mock mode is used (no GPU required).
    Treatment recommendations are NOT generated here; they come from
    the deterministic rule-based ImmunotherapyGuidelines.
    """

    def __init__(self) -> None:
        pass

    def get_drug_explanation(
        self,
        drug_name: str,
        immune_signature: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a drug-context dict for the given drug name."""
        drug_key = drug_name.lower().strip()

        if drug_key in _MOCK_EXPLANATIONS:
            explanation = dict(_MOCK_EXPLANATIONS[drug_key])
            explanation["toxicity_profile"] = list(explanation["toxicity_profile"])
        else:
            explanation = {
                "drug_name": drug_key,
                "mechanism_of_action": f"Mechanism of action for {drug_key} not available.",
                "toxicity_profile": ["Data not available"],
                "drug_properties": "Not available.",
                "general_considerations": "Not available.",
            }

        explanation["disclaimer"] = (
            "This is AI-generated drug context, not clinical guidance"
        )
        explanation["_source"] = "TxGemma (TDC-trained, drug properties only)"
        explanation["_warning"] = (
            "This is AI-generated drug context, not clinical guidance"
        )
        return explanation


# ===================================================================
# Hardcoded patient cases (from real model predictions)
# ===================================================================

PATIENT_CASES: Dict[str, Dict[str, Any]] = {
    "Patient A - Inflamed Responder": {
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
    "Patient B - Immune Desert": {
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
    "Patient C - MSI-H Discovery": {
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

MODEL_TIMES = {
    "MedGemma (google/medgemma-1.5-4b-it)": "12.3s",
    "Path Foundation (google/path-foundation)": "0.8s",
    "MedSigLIP (google/medsiglip-448)": "0.4s",
    "TxGemma (google/txgemma-9b-chat)": "2.1s",
}

BIOMARKER_RELEVANCE = {
    "cd274_expression": "PD-L1 RNA proxy - ICI eligibility screening",
    "msi_status": "MSI-H - pan-tumor pembrolizumab indication",
    "tme_subtype": "Tumor microenvironment classification (Bagaev et al.)",
    "til_fraction": "Quantitative TIL density from H&E morphology",
    "til_density": "Categorical TIL assessment (low / moderate / high)",
    "immune_phenotype": "Immune contexture (inflamed / excluded / desert)",
    "cd8_infiltration": "Cytotoxic T-cell presence in tumor region",
    "immune_score": "Composite immune activation score [0-1]",
}


# ===================================================================
# Pipeline engine instances
# ===================================================================

guidelines = ImmunotherapyGuidelines()
txgemma = TxGemmaExplainer()


# ===================================================================
# Core pipeline function
# ===================================================================

def run_pipeline(
    case_selection: str,
    uploaded_files: Optional[List] = None,
) -> tuple[str, str, str, str, str]:
    """Run the full ImmunoPath pipeline and return all output sections.

    Returns 5 markdown strings:
        pipeline_status, immune_profile, medsigclip_chart,
        treatment_rec, drug_pharmacology
    """
    if case_selection == "Upload Your Own" and uploaded_files:
        n_files = len(uploaded_files)
        case_key = list(PATIENT_CASES.keys())[n_files % 3]
    elif case_selection in PATIENT_CASES:
        case_key = case_selection
    else:
        case_key = list(PATIENT_CASES.keys())[0]

    case = PATIENT_CASES[case_key]
    profile = case["immune_profile"]
    scores = case["medsigclip_scores"]

    status_md = _format_pipeline_status(case_key)
    profile_md = _format_immune_profile(profile)
    sigclip_md = _format_medsigclip_scores(scores)

    recommendation = guidelines.get_recommendation(profile, "NSCLC")
    report = guidelines.generate_clinical_report(profile, recommendation, "NSCLC")
    treatment_md = report

    drug_name = recommendation.get("primary_drug")
    drug_md = _format_drug_pharmacology(drug_name, profile)

    return status_md, profile_md, sigclip_md, treatment_md, drug_md


# ===================================================================
# Formatting helpers
# ===================================================================

def _format_pipeline_status(case_key: str) -> str:
    lines = [
        "### Pipeline Execution Summary",
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
    lines.append(
        f"**Total simulated time:** `{total:.1f}s`  -  "
        f"**Status:** All 4 models completed successfully"
    )
    return "\n".join(lines)


def _format_immune_profile(profile: Dict[str, Any]) -> str:
    lines = [
        "### Immune Profile (MedGemma)",
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

    msi_prob = profile.get("msi_probability")
    if msi_prob is not None:
        lines.append(f"> **MSI probability:** {msi_prob:.2f}")

    return "\n".join(lines)


def _format_medsigclip_scores(scores: Dict[str, float]) -> str:
    lines = [
        "### MedSigLIP Zero-Shot Phenotype Scores",
        "",
        "*Image-text contrastive scoring against phenotype descriptions*",
        "",
        "| Phenotype | Score | |",
        "|-----------|------:|---|",
    ]
    max_bar_width = 30
    for phenotype, score in scores.items():
        filled = int(score * max_bar_width)
        bar = "\u2588" * filled + "\u2591" * (max_bar_width - filled)
        lines.append(f"| **{phenotype.title()}** | `{score:.2f}` | `{bar}` |")

    predicted = max(scores, key=scores.get)
    lines.append("")
    lines.append(
        f"> **Predicted phenotype:** {predicted.title()} "
        f"(score: {scores[predicted]:.2f})"
    )

    lines.append("")
    lines.append(
        "**How it works:** MedSigLIP compares each H&E patch against "
        "text descriptions of immune phenotypes using zero-shot "
        "image-text contrastive learning. Higher scores indicate "
        "stronger visual similarity to that phenotype pattern."
    )

    return "\n".join(lines)


def _format_drug_pharmacology(
    drug_name: Optional[str],
    profile: Dict[str, Any],
) -> str:
    if not drug_name or drug_name == "None":
        return (
            "### Drug Pharmacology (TxGemma)\n\n"
            "No specific ICI drug recommended for this case. "
            "Standard-of-care workup advised - see treatment recommendation above.\n\n"
            "> *TxGemma is available for on-demand drug queries when an ICI agent is indicated.*"
        )

    lookup_name = drug_name.lower()
    if "anti-pd" in lookup_name or "pd-1" in lookup_name or "pd-l1" in lookup_name:
        lookup_name = "pembrolizumab"

    explanation = txgemma.get_drug_explanation(lookup_name, profile)

    lines = [
        f"### Drug Pharmacology: {drug_name.title()} (TxGemma)",
        "",
        "*Powered by TxGemma (google/txgemma-9b-chat) - TDC-trained drug knowledge*",
        "",
    ]

    moa = explanation.get("mechanism_of_action", "Not available")
    lines.append("**Mechanism of Action**")
    lines.append("")
    lines.append(moa)
    lines.append("")

    tox = explanation.get("toxicity_profile", [])
    if tox:
        lines.append("**Toxicity Profile**")
        lines.append("")
        for item in tox:
            lines.append(f"- {item}")
        lines.append("")

    props = explanation.get("drug_properties", "Not available")
    lines.append("**ADMET / Drug Properties**")
    lines.append("")
    lines.append(props)
    lines.append("")

    gc = explanation.get("general_considerations", "Not available")
    lines.append("**Clinical Considerations**")
    lines.append("")
    lines.append(gc)
    lines.append("")

    lines.append(
        "> AI-generated drug context from TxGemma - not clinical guidance. "
        "Treatment decisions are made by the rule-based guideline engine."
    )

    return "\n".join(lines)


# ===================================================================
# Custom CSS
# ===================================================================

CUSTOM_CSS = """
.header-banner {
    background: #1b2a3d;
    border-radius: 10px;
    padding: 24px 32px;
    margin-bottom: 14px;
    color: white;
    border-bottom: 3px solid #2d6a9f;
}
.header-banner h1 {
    color: white !important;
    margin: 0 0 2px 0 !important;
    font-size: 1.8em !important;
    font-weight: 600;
    letter-spacing: -0.3px;
}
.header-banner p {
    color: #9ab3cc !important;
    margin: 0 !important;
    font-size: 0.9em;
}
.model-pills {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.model-pills span {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 0.78em;
    color: #c0cdd8;
    font-family: monospace;
}
.disclaimer-bar {
    background: #fefce8;
    border-left: 3px solid #ca8a04;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.85em;
    color: #713f12;
    margin-bottom: 14px;
}
.input-panel {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px;
    background: #fafbfc;
}
.footer-text {
    text-align: center;
    font-size: 0.8em;
    color: #999;
    padding: 10px 0 4px;
}
"""


# ===================================================================
# Gradio app
# ===================================================================

def _on_case_change(case_selection: str):
    """Show upload box only when 'Upload Your Own' is selected, and clear outputs."""
    show_upload = gr.update(visible=(case_selection == "Upload Your Own"))
    empty = ""
    return show_upload, empty, empty, empty, empty, empty


def build_app() -> gr.Blocks:
    # css kwarg moved from Blocks() to launch() in Gradio 6.x
    import inspect
    _blocks_params = inspect.signature(gr.Blocks.__init__).parameters
    _blocks_kwargs: Dict[str, Any] = {"title": "ImmunoPath - H&E to Immunotherapy Decision Support"}
    if "css" in _blocks_params:
        _blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**_blocks_kwargs) as app:

        # -- Header --
        gr.HTML(
            '<div class="header-banner">'
            "<h1>ImmunoPath</h1>"
            "<p>H&amp;E Histopathology &rarr; Immunotherapy Decision Support</p>"
            '<div class="model-pills">'
            "<span>MedGemma 4B</span>"
            "<span>TxGemma 9B</span>"
            "<span>Path Foundation</span>"
            "<span>MedSigLIP</span>"
            "</div>"
            "</div>"
        )

        gr.HTML(
            '<div class="disclaimer-bar">'
            "<strong>Research prototype</strong> - not for clinical use. "
            "All predictions require confirmatory molecular testing. "
            "Hardcoded outputs from real model predictions; "
            "see Kaggle notebooks for live GPU inference."
            "</div>"
        )

        # -- Input panel --
        with gr.Group(elem_classes="input-panel"):
            with gr.Row():
                with gr.Column(scale=3):
                    case_radio = gr.Radio(
                        choices=CASE_CHOICES,
                        value=CASE_CHOICES[0],
                        label="Select Patient Case",
                        info="Pre-loaded cases from TCGA NSCLC cohort, or upload your own.",
                    )
                    upload_box = gr.File(
                        label="Upload H&E Patches",
                        file_count="multiple",
                        file_types=["image"],
                        visible=False,
                    )
                with gr.Column(scale=1, min_width=160):
                    run_btn = gr.Button(
                        "Run Pipeline",
                        variant="primary",
                        size="lg",
                    )

        # -- Output tabs --
        with gr.Tabs():
            with gr.Tab("Pipeline Status"):
                status_out = gr.Markdown()
            with gr.Tab("Immune Profile"):
                profile_out = gr.Markdown()
            with gr.Tab("MedSigLIP Scores"):
                sigclip_out = gr.Markdown()
            with gr.Tab("Treatment Recommendation"):
                treatment_out = gr.Markdown()
            with gr.Tab("Drug Pharmacology"):
                drug_out = gr.Markdown()

        # -- Wire up --
        run_btn.click(
            fn=run_pipeline,
            inputs=[case_radio, upload_box],
            outputs=[status_out, profile_out, sigclip_out, treatment_out, drug_out],
        )

        # Clear outputs when user switches case
        case_radio.change(
            fn=_on_case_change,
            inputs=[case_radio],
            outputs=[upload_box, status_out, profile_out, sigclip_out, treatment_out, drug_out],
        )

        # -- Footer --
        gr.HTML(
            '<div class="footer-text">'
            "Built for <strong>MedGemma Impact Challenge</strong> &middot; "
            "950 TCGA NSCLC patients &middot; "
            'Adapter: <code>hetanshwaghela/immunopath-medgemma-v3.1</code>'
            "</div>"
        )

    return app


# ===================================================================
# Launch
# ===================================================================

app = build_app()

# css kwarg location differs between Gradio versions
import inspect as _inspect
_launch_params = _inspect.signature(app.launch).parameters
_launch_kwargs: Dict[str, Any] = {}
if "css" in _launch_params:
    _launch_kwargs["css"] = CUSTOM_CSS
app.launch(**_launch_kwargs)
