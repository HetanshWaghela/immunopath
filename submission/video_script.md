# ImmunoPath - 3-Minute Video Script

## [0:00 – 0:20] Opening Hook

**[Screen: Title card with ImmunoPath logo]**

> "Every cancer patient gets an H&E slide. But only patients with access to expensive molecular testing get immunotherapy screening. What if we could bridge that gap with AI?"

**[Transition to problem statement slide]**

> "Immunotherapy selection requires PD-L1 IHC, MSI testing, and NGS panels - costing thousands of dollars and weeks of turnaround. ImmunoPath uses MedGemma to predict immune biomarkers directly from routine H&E histopathology slides."

---

## [0:20 – 1:00] Pipeline Demo

**[Screen: Architecture diagram]**

> "ImmunoPath integrates four HAI-DEF models. MedGemma is the core - fine-tuned with LoRA to predict 8 immune biomarkers from H&E patches. TxGemma provides drug pharmacology context. Path Foundation generates visual embeddings. MedSigLIP adds zero-shot confidence scoring."

**[Switch to Gradio demo - screen recording]**

> "Here's the demo. I upload H&E patches from a lung cancer biopsy. MedGemma produces a structured immune profile - CD274 expression, TME subtype, TIL density, immune phenotype. The guideline engine maps this to an NCCN-aligned recommendation. And TxGemma explains the drug's mechanism of action."

**[Show each tab briefly: Profile → Recommendation → Drug Info → Zero-Shot]**

---

## [1:00 – 1:40] Training & Results

**[Screen: Training pipeline diagram]**

> "We trained on 950 TCGA NSCLC patients - 60,000 patches extracted from diagnostic slides, diversity-selected with K-Means, and matched with RNA-seq immune signatures."

**[Screen: Results table]**

> "We iterated through three versions. Our TIL prediction meets its clinical target - an MAE of 0.16, below the 0.20 threshold. JSON compliance went from 16% in zero-shot to 100% in all fine-tuned versions - a 6x improvement showing production-ready structured output."

**[Screen: v1→v2→v3.1 iteration diagram]**

> "Each version applied systematic engineering fixes - response-only loss masking, PEFT weight tying corrections, and TIL normalization. This iterative approach is how real clinical AI gets built."

---

## [1:40 – 2:20] Impact & Feasibility

**[Screen: Health equity map]**

> "The real impact is access. H&E slides are already available everywhere - from major cancer centers to rural hospitals. If validated at scale, ImmunoPath could provide preliminary immune screening at the point of diagnosis, even where molecular testing isn't available."

**[Screen: Cost comparison]**

> "Molecular testing costs $2,000 to $5,000 per patient. ImmunoPath runs on a single GPU in under a minute. It's a triage tool - flagging patients for confirmatory testing, not replacing molecular assays."

---

## [2:20 – 2:50] Technical Highlights

**[Screen: 4 HAI-DEF models summary]**

> "To summarize: four HAI-DEF models in one pipeline. MedGemma fine-tuned for immune profiling. TxGemma for drug explanations. Path Foundation for visual features. MedSigLIP for zero-shot scoring. All backed by a deterministic guideline engine with safety disclaimers."

---

## [2:50 – 3:00] Closing

**[Screen: Title card]**

> "ImmunoPath - because every patient deserves immune profiling, not just those who can afford it."

**[End card: GitHub link, author name]**
