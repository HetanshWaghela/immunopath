---
title: ImmunoPath
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: true
license: mit
short_description: "H&E Histopathology to Immunotherapy Decision Support"
---

# ImmunoPath - H&E to Immunotherapy Decision Support

Predicts immune biomarkers from routine H&E histopathology slides using 4 HAI-DEF models: MedGemma, TxGemma, Path Foundation, and MedSigLIP.

**MedGemma Impact Challenge Submission** by Hetansh Waghela

This is a proof-of-concept demo running in mock mode (hardcoded outputs from real model predictions). See Kaggle notebooks for live GPU inference.

- GitHub: [github.com/hetanshwaghela/medgemma-hackathon](https://github.com/hetanshwaghela/medgemma-hackathon)
- Open-weight adapter: [hetanshwaghela/immunopath-medgemma-v3.1](https://huggingface.co/hetanshwaghela/immunopath-medgemma-v3.1)
- Base model: [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
