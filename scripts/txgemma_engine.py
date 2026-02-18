#!/usr/bin/env python3
"""TxGemma integration module for ImmunoPath.

TxGemma (google/txgemma-9b-chat) is a text-only causal LM trained on
Therapeutics Data Commons (TDC) — 66 drug-property prediction tasks.

It is used ONLY for drug explanations:
  - Mechanism of action (molecular level)
  - Toxicity profiles (ToxCast/ClinTox from TDC)
  - ADMET characteristics
  - General pharmacological considerations

It does NOT generate treatment recommendations.
Those come from the deterministic rule-based guideline_engine.py.

Usage:
    # Mock mode (default, no GPU required):
    explainer = TxGemmaExplainer(use_mock=True)
    result = explainer.get_drug_explanation("pembrolizumab", {"immune_phenotype": "inflamed"})

    # Real mode (requires GPU + ~18 GB VRAM for bfloat16 9B):
    explainer = TxGemmaExplainer(use_mock=False)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Mock drug knowledge base — curated from public pharmacology references
# ---------------------------------------------------------------------------

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
            "pharmacokinetics over the dose range of 0.1–10 mg/kg. "
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
            "~27 days. Steady state reached by cycle 6–9. IV administration "
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
            "Immune-mediated endocrinopathies (hypophysitis — characteristic of anti-CTLA-4)",
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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TxGemmaExplainer:
    """Explains drug properties using TxGemma (TDC-trained).

    TxGemma is used ONLY for drug explanations — mechanism of action,
    toxicity profiles, ADMET characteristics, and general pharmacological
    considerations.

    Treatment recommendations are NOT generated here; they come from the
    deterministic rule-based guideline_engine.py.

    Parameters
    ----------
    model_name : str
        Hugging Face model ID (default ``google/txgemma-9b-chat``).
    device : str
        Device map for model loading (``"auto"`` uses accelerate).
    use_mock : bool
        If ``True`` (default), return curated mock responses without loading
        the 9B model — suitable for development, demos, and CPU-only
        environments.
    """

    def __init__(
        self,
        model_name: str = "google/txgemma-9b-chat",
        device: str = "auto",
        use_mock: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_mock = use_mock

        self.tokenizer = None
        self.model = None

        if not use_mock:
            self._load_model()

    # ------------------------------------------------------------------
    # Model loading (only when use_mock=False)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load TxGemma-9B-Chat with bfloat16 precision."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Real mode requires `transformers` and `torch`. "
                "Install them or use use_mock=True."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def create_explanation_prompt(
        self,
        drug_name: str,
        immune_signature: Dict[str, Any],
    ) -> str:
        """Build the TxGemma prompt for a drug explanation.

        The prompt asks for TDC-trained knowledge only and explicitly
        forbids treatment recommendations or clinical-trial citations.
        """
        phenotype = immune_signature.get("immune_phenotype", "unknown")
        til_density = immune_signature.get("til_density", "unknown")

        prompt = (
            "You are a pharmaceutical scientist. Explain the drug properties "
            "and toxicity profile for the following immunotherapy drug.\n"
            "\n"
            "## Drug to Explain\n"
            f"Drug Name: {drug_name}\n"
            "\n"
            "## Context (for reference only — do NOT generate treatment "
            "recommendations)\n"
            "This drug was selected by a rule-based clinical guideline engine "
            "for a patient with:\n"
            f"- Immune phenotype: {phenotype}\n"
            f"- TIL density: {til_density}\n"
            "\n"
            "## Task (TDC-trained knowledge only)\n"
            "Based on your training on drug properties, explain:\n"
            "\n"
            "1. **Mechanism of Action**: How does this drug work at the "
            "molecular level?\n"
            "2. **Known Toxicity Profile**: Common adverse events (from "
            "TDC/clinical toxicity data)\n"
            "3. **ADMET Characteristics**: Absorption, distribution, "
            "metabolism, excretion, and toxicity properties\n"
            "4. **General Considerations**: Pharmacological factors to "
            "consider\n"
            "\n"
            "⚠️ WARNING: Do NOT provide treatment recommendations or cite "
            "specific clinical trials. Those are handled by the rule-based "
            "guideline engine.\n"
            "\n"
            "Format your response as JSON:\n"
            "{\n"
            f'    "drug_name": "{drug_name}",\n'
            '    "mechanism_of_action": "description",\n'
            '    "toxicity_profile": ["adverse_event_1", "adverse_event_2"],\n'
            '    "drug_properties": "ADMET summary if known",\n'
            '    "general_considerations": "pharmacological notes",\n'
            '    "disclaimer": "This is AI-generated drug context, not '
            'clinical guidance"\n'
            "}"
        )
        return prompt

    # ------------------------------------------------------------------
    # Drug explanation
    # ------------------------------------------------------------------

    def get_drug_explanation(
        self,
        drug_name: str,
        immune_signature: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a drug-context dict for *drug_name*.

        In mock mode the response is drawn from a curated knowledge base.
        In real mode the TxGemma-9B-Chat model is queried.

        Parameters
        ----------
        drug_name : str
            ICI drug name (e.g. ``"pembrolizumab"``).
        immune_signature : dict
            Immune-signature dict (used for prompt context).

        Returns
        -------
        dict
            Structured drug explanation with metadata fields.
        """
        drug_key = drug_name.lower().strip()

        if self.use_mock:
            explanation = self._mock_explanation(drug_key, immune_signature)
        else:
            explanation = self._real_explanation(drug_key, immune_signature)

        # Always stamp provenance and safety metadata
        explanation["_source"] = "TxGemma (TDC-trained, drug properties only)"
        explanation["_warning"] = (
            "This is AI-generated drug context, not clinical guidance"
        )
        return explanation

    # ------------------------------------------------------------------
    # Mock path
    # ------------------------------------------------------------------

    def _mock_explanation(
        self,
        drug_key: str,
        immune_signature: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a curated mock explanation for known ICI drugs."""
        if drug_key in _MOCK_EXPLANATIONS:
            # Return a copy so callers cannot mutate the template
            explanation = dict(_MOCK_EXPLANATIONS[drug_key])
            explanation["toxicity_profile"] = list(explanation["toxicity_profile"])
            explanation["disclaimer"] = (
                "This is AI-generated drug context, not clinical guidance"
            )
            return explanation

        # Unknown drug — return a generic placeholder
        return {
            "drug_name": drug_key,
            "mechanism_of_action": f"Mechanism of action for {drug_key} not available in mock mode.",
            "toxicity_profile": ["Data not available in mock mode"],
            "drug_properties": "Not available in mock mode.",
            "general_considerations": "Not available in mock mode.",
            "disclaimer": "This is AI-generated drug context, not clinical guidance",
        }

    # ------------------------------------------------------------------
    # Real model path
    # ------------------------------------------------------------------

    def _real_explanation(
        self,
        drug_key: str,
        immune_signature: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Query the loaded TxGemma model and parse its JSON response."""
        import torch

        prompt = self.create_explanation_prompt(drug_key, immune_signature)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return self._parse_json_response(response)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(response: str) -> Dict[str, Any]:
        """Extract the first ``{ ... }`` JSON block from *response*.

        Falls back to ``{"raw_response": response}`` if parsing fails.
        """
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(response[json_start:json_end])
            except json.JSONDecodeError:
                pass
        return {"raw_response": response}


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TxGemma Drug Explainer — Mock Mode Demo")
    print("=" * 70)

    explainer = TxGemmaExplainer(use_mock=True)

    sample_signature: Dict[str, Any] = {
        "immune_phenotype": "inflamed",
        "til_density": "high",
        "cd274_expression": "high",
        "msi_status": "MSS",
    }

    drugs = ["pembrolizumab", "nivolumab", "atezolizumab", "ipilimumab", "durvalumab"]

    for drug in drugs:
        print(f"\n{'—' * 60}")
        print(f"Drug: {drug}")
        print(f"{'—' * 60}")
        result = explainer.get_drug_explanation(drug, sample_signature)
        print(json.dumps(result, indent=2))

    # Unknown drug fallback
    print(f"\n{'—' * 60}")
    print("Drug: unknown_drug_xyz (fallback)")
    print(f"{'—' * 60}")
    result = explainer.get_drug_explanation("unknown_drug_xyz", sample_signature)
    print(json.dumps(result, indent=2))
