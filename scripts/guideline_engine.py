#!/usr/bin/env python3
"""Rule-based immunotherapy guideline engine for ImmunoPath.

This module provides deterministic, transparent treatment recommendations
based on immune signatures predicted by MedGemma. All biomarker-to-treatment
logic uses publicly available evidence (FDA labels, pivotal trial eligibility
criteria, NCCN-aligned biomarker thresholds).

TxGemma is NOT used here - it provides optional drug explanations only
(see txgemma_integration.py).

Run:
  python scripts/guideline_engine.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Safety disclaimer (referenced throughout)
# ---------------------------------------------------------------------------
SAFETY_DISCLAIMER = (
    "This is research decision support. Not intended for clinical use "
    "without professional review."
)


# ---------------------------------------------------------------------------
# ICI drug reference database (public evidence)
# ---------------------------------------------------------------------------
ICI_DRUG_DATABASE: Dict[str, Dict[str, Any]] = {
    "pembrolizumab": {
        "class": "anti-PD-1",
        "approved_indications": [
            "NSCLC with PD-L1 ≥50% (monotherapy)",
            "NSCLC with PD-L1 ≥1% (with chemotherapy)",
            "MSI-H/dMMR solid tumors (pan-tumor)",
            "TMB-High solid tumors (≥10 mut/Mb)",
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


class ImmunotherapyGuidelines:
    """Rule-based engine for ICI treatment recommendations.

    This is the PRIMARY source of treatment recommendations in ImmunoPath.
    Rules are derived from publicly available evidence: FDA-approved labels,
    pivotal trial biomarker eligibility, and NCCN-aligned thresholds.

    Currently focused on NSCLC (LUAD + LUSC from TCGA).  The *cancer_type*
    parameter is accepted for future extensibility but does not yet alter
    the rule logic.
    """

    def __init__(self) -> None:
        self.drug_database = ICI_DRUG_DATABASE

    # ------------------------------------------------------------------
    # Core recommendation engine
    # ------------------------------------------------------------------
    def get_recommendation(
        self,
        immune_signature: Dict[str, Any],
        cancer_type: str,
    ) -> Dict[str, Any]:
        """Generate a treatment recommendation using deterministic rules.

        Parameters
        ----------
        immune_signature:
            Dict produced by MedGemma fine-tuned model.  Expected keys
            (all optional - missing keys trigger the default rule):
              - msi_status: "MSI-H" | "MSS" | "unknown"
              - msi_probability: float 0-1
              - pdl1_ihc_tps: float 0-100 (if clinical IHC available)
              - cd274_expression: "high" | "low" | "unknown"
              - immune_phenotype: "inflamed" | "excluded" | "desert" | "unknown"
        cancer_type:
            E.g. "NSCLC", "LUAD", "LUSC".  Reserved for future per-type rules.

        Returns
        -------
        dict with keys: source, primary_drug, regimen, confidence,
            confirmatory_tests_required, supporting_evidence, alternatives,
            safety_warnings.
        """
        msi_status = immune_signature.get("msi_status", "unknown")
        msi_probability = immune_signature.get("msi_probability", 0.0)
        pdl1_ihc_tps: Optional[float] = immune_signature.get("pdl1_ihc_tps")
        cd274_expression = immune_signature.get("cd274_expression", "unknown")
        immune_phenotype = immune_signature.get("immune_phenotype", "unknown")

        # Shared base - every recommendation carries the safety warning.
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

        # ---- Rule 1: MSI-H → pembrolizumab monotherapy ----
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

        # ---- Rule 2: PD-L1 IHC TPS ≥50% → pembrolizumab mono ----
        if pdl1_ihc_tps is not None and pdl1_ihc_tps >= 50:
            base["primary_drug"] = "pembrolizumab"
            base["regimen"] = "Pembrolizumab monotherapy (PD-L1 ≥50%)"
            base["confidence"] = "high"
            base["supporting_evidence"] = [
                "KEYNOTE-024",
            ]
            base["confirmatory_tests_required"] = [
                "Confirm PD-L1 IHC with 22C3 or SP263 assay",
                "Driver mutation testing (EGFR, ALK, ROS1, BRAF)",
            ]
            return base

        # ---- Rule 3: PD-L1 IHC TPS ≥1% → pembrolizumab + chemo ----
        if pdl1_ihc_tps is not None and pdl1_ihc_tps >= 1:
            base["primary_drug"] = "pembrolizumab"
            base["regimen"] = "Pembrolizumab + platinum-based chemotherapy"
            base["confidence"] = "high"
            base["supporting_evidence"] = [
                "KEYNOTE-189",
            ]
            base["confirmatory_tests_required"] = [
                "Confirm PD-L1 IHC with 22C3 or SP263 assay",
                "Driver mutation testing (EGFR, ALK, ROS1, BRAF)",
            ]
            return base

        # ---- Rule 4: CD274-high RNA proxy + inflamed → conditional ICI ----
        if cd274_expression == "high" and immune_phenotype == "inflamed":
            base["primary_drug"] = "anti-PD-1/PD-L1 agent"
            base["regimen"] = "Consider ICI-based regimen (CONDITIONAL)"
            base["confidence"] = "conditional"
            base["confirmatory_tests_required"] = [
                "PD-L1 IHC (22C3 or SP263) - REQUIRED before treatment",
                "Driver mutation testing (EGFR, ALK, ROS1, BRAF, etc.)",
            ]
            base["supporting_evidence"] = [
                "CD274 mRNA proxy correlates with IHC (r²=0.65-0.81, Kang et al. 2022)",
                "Pending PD-L1 IHC confirmation",
            ]
            base["alternatives"] = [
                "Chemo + ICI combination",
                "Clinical trial",
            ]
            return base

        # ---- Rule 5: Default → standard-of-care workup ----
        base["primary_drug"] = None
        base["regimen"] = "Standard-of-care workup recommended"
        base["confidence"] = "low"
        base["confirmatory_tests_required"] = [
            "PD-L1 IHC",
            "MSI/dMMR testing",
            "Driver mutation panel",
            "TMB if available",
        ]
        base["alternatives"] = [
            "Clinical trial enrollment",
        ]
        return base

    # ------------------------------------------------------------------
    # Clinical report generation
    # ------------------------------------------------------------------
    def generate_clinical_report(
        self,
        immune_signature: Dict[str, Any],
        recommendation: Dict[str, Any],
        cancer_type: str,
    ) -> str:
        """Produce a markdown-formatted clinical decision support report.

        Parameters
        ----------
        immune_signature:
            The same dict passed to ``get_recommendation``.
        recommendation:
            The dict returned by ``get_recommendation``.
        cancer_type:
            E.g. "NSCLC".

        Returns
        -------
        Markdown string suitable for display in Gradio / notebook output.
        """
        lines: List[str] = []

        lines.append("# ImmunoPath Clinical Decision Support Report")
        lines.append("")
        lines.append(f"> {SAFETY_DISCLAIMER}")
        lines.append("")

        # --- Patient context ---
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

        # --- Recommendation ---
        lines.append("## Treatment Recommendation")
        lines.append("")
        lines.append(f"- **Source:** {recommendation['source']}")
        lines.append(f"- **Primary drug:** {recommendation['primary_drug'] or 'None'}")
        lines.append(f"- **Regimen:** {recommendation['regimen']}")
        lines.append(f"- **Confidence:** {recommendation['confidence']}")
        lines.append("")

        # --- Evidence ---
        if recommendation["supporting_evidence"]:
            lines.append("## Supporting Evidence")
            lines.append("")
            for evidence in recommendation["supporting_evidence"]:
                lines.append(f"- {evidence}")
            lines.append("")

        # --- Confirmatory tests ---
        if recommendation["confirmatory_tests_required"]:
            lines.append("## Confirmatory Tests Required")
            lines.append("")
            for test in recommendation["confirmatory_tests_required"]:
                lines.append(f"- [ ] {test}")
            lines.append("")

        # --- Alternatives ---
        if recommendation["alternatives"]:
            lines.append("## Alternatives")
            lines.append("")
            for alt in recommendation["alternatives"]:
                lines.append(f"- {alt}")
            lines.append("")

        # --- Safety warnings ---
        if recommendation["safety_warnings"]:
            lines.append("## Safety Warnings")
            lines.append("")
            for warning in recommendation["safety_warnings"]:
                lines.append(f"- {warning}")
            lines.append("")

        # --- Drug reference (if a primary drug was recommended) ---
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


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _demo() -> None:
    """Demonstrate the rule-based guideline engine with sample signatures."""
    engine = ImmunotherapyGuidelines()

    cases: List[Dict[str, Any]] = [
        {
            "label": "Case 1: MSI-H tumor (high confidence)",
            "signature": {
                "msi_status": "MSI-H",
                "msi_probability": 0.92,
                "cd274_expression": "low",
                "immune_phenotype": "inflamed",
                "til_density": "high",
                "til_fraction": 0.55,
                "immune_score": 0.81,
            },
            "cancer_type": "NSCLC",
        },
        {
            "label": "Case 2: PD-L1 IHC ≥50%",
            "signature": {
                "msi_status": "MSS",
                "msi_probability": 0.05,
                "pdl1_ihc_tps": 65.0,
                "cd274_expression": "high",
                "immune_phenotype": "inflamed",
                "til_density": "moderate",
                "til_fraction": 0.30,
                "immune_score": 0.62,
            },
            "cancer_type": "NSCLC",
        },
        {
            "label": "Case 3: CD274-high + inflamed (conditional)",
            "signature": {
                "cd274_expression": "high",
                "msi_status": "MSS",
                "msi_probability": 0.12,
                "til_fraction": 0.42,
                "til_density": "high",
                "immune_phenotype": "inflamed",
                "cd8_infiltration": "high",
                "immune_score": 0.78,
            },
            "cancer_type": "NSCLC",
        },
        {
            "label": "Case 4: Immune desert (default workup)",
            "signature": {
                "msi_status": "MSS",
                "msi_probability": 0.03,
                "cd274_expression": "low",
                "immune_phenotype": "desert",
                "til_density": "low",
                "til_fraction": 0.05,
                "immune_score": 0.12,
            },
            "cancer_type": "NSCLC",
        },
    ]

    for case in cases:
        print("=" * 70)
        print(case["label"])
        print("=" * 70)
        rec = engine.get_recommendation(case["signature"], case["cancer_type"])
        print(json.dumps(rec, indent=2))
        print()
        report = engine.generate_clinical_report(
            case["signature"], rec, case["cancer_type"]
        )
        print(report)
        print()


if __name__ == "__main__":
    _demo()
