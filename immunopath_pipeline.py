#!/usr/bin/env python3
"""ImmunoPath Integration Pipeline - End-to-End H&E → Immunotherapy Recommendation.

This pipeline integrates 4 HAI-DEF models into a single clinical decision-support system:
  1. MedGemma (core)       - Fine-tuned VLM for immune profiling from H&E patches
  2. TxGemma (drug info)   - Drug explanation and pharmacology insights
  3. Path Foundation       - Patch-level visual feature embeddings
  4. MedSigLIP (zero-shot) - Zero-shot image-text confidence signal

All models run in mock mode by default (no GPU required). Toggle `use_mock=False` for real inference.

Usage:
    python immunopath_pipeline.py --test              # Run with mock data
    python immunopath_pipeline.py --image patch.jpg   # Run on a real image
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add scripts/ to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from guideline_engine import ImmunotherapyGuidelines
from txgemma_engine import TxGemmaExplainer


# ---------------------------------------------------------------------------
# Data classes for pipeline I/O
# ---------------------------------------------------------------------------

@dataclass
class ImmuneProfile:
    """Structured immune profile predicted by MedGemma."""
    cd274_expression: str = "low"          # "high" or "low"
    msi_status: str = "MSS"               # "MSI-H" or "MSS"
    tme_subtype: str = "D"                # "IE", "IE/F", "F", "D"
    til_fraction: float = 0.30
    til_density: str = "moderate"          # "low", "moderate", "high"
    immune_phenotype: str = "desert"       # "inflamed", "excluded", "desert"
    cd8_infiltration: str = "low"          # "low", "moderate", "high"
    immune_score: float = 0.35
    confidence: float = 0.80

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Complete pipeline output combining all 4 HAI-DEF models."""
    # Core immune profile (MedGemma)
    immune_profile: Dict[str, Any] = field(default_factory=dict)

    # Treatment recommendation (Guideline Engine)
    recommendation: Dict[str, Any] = field(default_factory=dict)
    clinical_report: str = ""

    # Drug explanation (TxGemma)
    drug_explanations: List[Dict[str, Any]] = field(default_factory=list)

    # Patch embeddings (Path Foundation)
    patch_embeddings: Optional[List[List[float]]] = None
    embedding_model: str = "google/path-foundation"

    # Zero-shot comparison (MedSigLIP)
    zero_shot_scores: Dict[str, float] = field(default_factory=dict)
    zero_shot_model: str = "google/medsiglip-448"

    # Metadata
    processing_time_s: float = 0.0
    models_used: List[str] = field(default_factory=list)
    mock_mode: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Mock models for demo / accessibility
# ---------------------------------------------------------------------------

class MockMedGemma:
    """Mock MedGemma for demo mode (no GPU required)."""

    MOCK_PROFILES = [
        # Index 0: inflamed + CD274-high → ICI recommended (3 imgs, 6 imgs)
        ImmuneProfile(
            cd274_expression="high", msi_status="MSS", tme_subtype="IE",
            til_fraction=0.72, til_density="high", immune_phenotype="inflamed",
            cd8_infiltration="high", immune_score=0.78, confidence=0.85,
        ),
        # Index 1: desert + CD274-low → standard workup (1 img, 4 imgs)
        ImmuneProfile(
            cd274_expression="low", msi_status="MSS", tme_subtype="D",
            til_fraction=0.15, til_density="low", immune_phenotype="desert",
            cd8_infiltration="low", immune_score=0.22, confidence=0.91,
        ),
        # Index 2: inflamed + CD274-high + MSI-H → pembrolizumab (2 imgs, 5 imgs) - best demo
        ImmuneProfile(
            cd274_expression="high", msi_status="MSI-H", tme_subtype="IE/F",
            til_fraction=0.45, til_density="moderate", immune_phenotype="inflamed",
            cd8_infiltration="moderate", immune_score=0.52, confidence=0.73,
        ),
    ]

    def __init__(self):
        self._call_count = 0

    def predict(self, image_paths: List[str]) -> ImmuneProfile:
        """Return a mock profile based on number of input images for variety."""
        idx = len(image_paths) % len(self.MOCK_PROFILES)
        return self.MOCK_PROFILES[idx]


class MockPathFoundation:
    """Mock Path Foundation for patch embedding generation."""

    def embed(self, image_paths: List[str]) -> List[List[float]]:
        import random
        random.seed(42)
        return [[random.gauss(0, 1) for _ in range(768)] for _ in image_paths]


class MockMedSigLIP:
    """Mock MedSigLIP for zero-shot image-text similarity scores."""

    PHENOTYPE_PROMPTS = {
        "inflamed": "H&E histopathology showing dense lymphocytic infiltration within tumor nests",
        "excluded": "H&E histopathology showing immune cells at tumor margins but not within tumor",
        "desert": "H&E histopathology showing minimal immune cell infiltration in tumor",
    }

    def score(self, image_paths: List[str]) -> Dict[str, float]:
        import random
        random.seed(len(image_paths))
        scores = {k: round(random.uniform(0.1, 0.9), 3) for k in self.PHENOTYPE_PROMPTS}
        # Normalize to sum to 1
        total = sum(scores.values())
        return {k: round(v / total, 3) for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class ImmunoPathPipeline:
    """End-to-end H&E → Immunotherapy Recommendation Pipeline.

    Integrates 4 HAI-DEF models:
      1. MedGemma      - immune profiling from H&E patches
      2. TxGemma       - drug pharmacology explanations
      3. Path Foundation - patch visual embeddings
      4. MedSigLIP     - zero-shot immune phenotype scoring

    Parameters
    ----------
    use_mock : bool
        If True (default), use mock models for all components.
        Set False for real GPU inference.
    cancer_type : str
        Cancer type for guideline lookup (default: "NSCLC").
    """

    def __init__(
        self,
        use_mock: bool = True,
        cancer_type: str = "NSCLC",
    ):
        self.use_mock = use_mock
        self.cancer_type = cancer_type

        # Initialize models
        self.medgemma = MockMedGemma() if use_mock else self._load_medgemma()
        self.guideline_engine = ImmunotherapyGuidelines()
        self.txgemma = TxGemmaExplainer(use_mock=use_mock)
        self.path_foundation = MockPathFoundation() if use_mock else self._load_path_foundation()
        self.medsiglip = MockMedSigLIP() if use_mock else self._load_medsiglip()

    def _load_medgemma(self):
        raise NotImplementedError(
            "Real MedGemma inference requires GPU + Colab. "
            "Use the Colab notebook for real inference, or set use_mock=True."
        )

    def _load_path_foundation(self):
        raise NotImplementedError(
            "Path Foundation requires GPU. Use use_mock=True for demo."
        )

    def _load_medsiglip(self):
        raise NotImplementedError(
            "MedSigLIP requires GPU. Use use_mock=True for demo."
        )

    def run(self, image_paths: List[str]) -> PipelineResult:
        """Run the full ImmunoPath pipeline on a set of H&E patch images.

        Parameters
        ----------
        image_paths : List[str]
            Paths to H&E patch images (512×512 px preferred).

        Returns
        -------
        PipelineResult
            Complete structured result from all 4 models.
        """
        t0 = time.time()
        models_used = []

        # ── Step 1: MedGemma - Immune Profiling ──────────────────────────
        profile = self.medgemma.predict(image_paths)
        immune_sig = profile.to_dict()
        models_used.append("MedGemma (google/medgemma-1.5-4b-it)")

        # ── Step 2: Guideline Engine - Treatment Recommendation ──────────
        recommendation = self.guideline_engine.get_recommendation(
            immune_sig, self.cancer_type
        )
        clinical_report = self.guideline_engine.generate_clinical_report(
            immune_sig, recommendation, self.cancer_type
        )
        models_used.append("Guideline Engine (rule-based, NCCN-aligned)")

        # ── Step 3: TxGemma - Drug Explanations ──────────────────────────
        drug_explanations = []
        primary_drug = recommendation.get("primary_drug")
        if primary_drug and primary_drug != "None":
            explanation = self.txgemma.get_drug_explanation(primary_drug, immune_sig)
            drug_explanations.append(explanation)
        # Also explain alternatives if present
        for alt in recommendation.get("alternatives", [])[:2]:
            alt_name = alt.lower().replace("chemo + ici combination", "pembrolizumab")
            if alt_name and alt_name != "clinical trial" and alt_name != "clinical trial enrollment":
                explanation = self.txgemma.get_drug_explanation(alt_name, immune_sig)
                drug_explanations.append(explanation)

        models_used.append("TxGemma (google/txgemma-9b-chat)")

        # ── Step 4: Path Foundation - Patch Embeddings ───────────────────
        embeddings = self.path_foundation.embed(image_paths)
        models_used.append("Path Foundation (google/path-foundation)")

        # ── Step 5: MedSigLIP - Zero-Shot Phenotype Scores ───────────────
        zs_scores = self.medsiglip.score(image_paths)
        models_used.append("MedSigLIP (google/medsiglip-448)")

        elapsed = time.time() - t0

        return PipelineResult(
            immune_profile=immune_sig,
            recommendation=recommendation,
            clinical_report=clinical_report,
            drug_explanations=drug_explanations,
            patch_embeddings=[e[:5] for e in embeddings],  # Truncate for display
            zero_shot_scores=zs_scores,
            processing_time_s=round(elapsed, 2),
            models_used=models_used,
            mock_mode=self.use_mock,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ImmunoPath Pipeline - H&E → Immunotherapy Recommendation"
    )
    parser.add_argument("--test", action="store_true", help="Run with mock data")
    parser.add_argument("--image", type=str, nargs="+", help="Path(s) to H&E patch images")
    parser.add_argument("--real", action="store_true", help="Use real models (requires GPU)")
    parser.add_argument("--output", type=str, help="Save output JSON to file")
    args = parser.parse_args()

    if not args.test and not args.image:
        parser.print_help()
        print("\n  Example: python immunopath_pipeline.py --test")
        return

    use_mock = not args.real
    pipeline = ImmunoPathPipeline(use_mock=use_mock)

    if args.test:
        image_paths = ["mock_patch_1.jpg", "mock_patch_2.jpg", "mock_patch_3.jpg", "mock_patch_4.jpg"]
    else:
        image_paths = args.image

    print("=" * 70)
    print("  ImmunoPath Pipeline")
    print(f"  Mode: {'MOCK (demo)' if use_mock else 'REAL (GPU)'}")
    print(f"  Images: {len(image_paths)}")
    print("=" * 70)

    result = pipeline.run(image_paths)

    # Display results
    print(f"\n{'─' * 70}")
    print("  IMMUNE PROFILE (MedGemma)")
    print(f"{'─' * 70}")
    for k, v in result.immune_profile.items():
        print(f"    {k:25s}: {v}")

    print(f"\n{'─' * 70}")
    print("  TREATMENT RECOMMENDATION (Guideline Engine)")
    print(f"{'─' * 70}")
    print(f"    Primary Drug:   {result.recommendation.get('primary_drug', 'N/A')}")
    print(f"    Regimen:        {result.recommendation.get('regimen', 'N/A')}")
    print(f"    Confidence:     {result.recommendation.get('confidence', 'N/A')}")
    evidence = result.recommendation.get("supporting_evidence", [])
    if evidence:
        print(f"    Evidence:       {', '.join(str(e) for e in evidence[:3])}")

    print(f"\n{'─' * 70}")
    print("  DRUG EXPLANATIONS (TxGemma)")
    print(f"{'─' * 70}")
    for exp in result.drug_explanations[:2]:
        name = exp.get("drug_name", "unknown")
        moa = exp.get("mechanism_of_action", "")[:120]
        print(f"    {name}: {moa}...")

    print(f"\n{'─' * 70}")
    print("  ZERO-SHOT PHENOTYPE SCORES (MedSigLIP)")
    print(f"{'─' * 70}")
    for phenotype, score in result.zero_shot_scores.items():
        bar = "█" * int(score * 30)
        print(f"    {phenotype:12s}: {score:.3f} {bar}")

    print(f"\n{'─' * 70}")
    print("  MODELS USED")
    print(f"{'─' * 70}")
    for i, m in enumerate(result.models_used, 1):
        print(f"    {i}. {m}")

    print(f"\n  Processing time: {result.processing_time_s:.2f}s")
    print(f"  Mock mode: {result.mock_mode}")
    print("=" * 70)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result.to_json())
        print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()
