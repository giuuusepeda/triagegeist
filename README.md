# 🚨 TriageGeist

> **Predicting emergency department patient acuity with machine learning**  
> *A Kaggle competition project | MIMIC-IV ED dataset*

![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

**TriageGeist** is an end-to-end machine learning pipeline for predicting patient acuity at emergency department triage, developed as a submission for the [TriageGeist Kaggle Competition](https://www.kaggle.com/competitions/triagegeist/overview).

The project combines structured clinical data (vital signs, demographics, chief complaints) with NLP-based complaint encoding to train gradient boosting models capable of classifying acuity levels — ultimately supporting faster, more consistent triage decisions.

> ⚕️ **Data:** [MIMIC-IV Emergency Department (v2.2)](https://physionet.org/content/mimic-iv-ed/2.2/) — used under PhysioNet Credentialed Health Data License 1.5.0. Raw data is **not included** in this repository.

---

## Project Structure

```
triagegeist/
│
├── 01_Data/
│   └── mimic-iv-ed-demo-2.2/      # Demo subset (public, no PHI)
│
├── 02_EDA/                         # Exploratory Data Analysis
│   └── eda_mimic_ed.ipynb
│
├── 03_Baseline/                    # Baseline models & NLP pipeline
│   └── baseline_models.ipynb
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## What's Been Done

### ✅ Exploratory Data Analysis
- Distribution of acuity levels (ESI 1–5)
- Missing data audit across triage vitals
- Patterns in chief complaint text across acuity classes
- Feature correlation analysis

### ✅ NLP Pipeline — Chief Complaint Encoding
- Chief complaints are split on common delimiters to handle compound entries (e.g. *"chest pain / shortness of breath"*)
- Each complaint is encoded using **Bio_ClinicalBERT** (domain-adapted BERT for clinical text)
- Embeddings are clustered into **40 semantic groups** using KMeans
- Cluster assignments serve as a categorical feature in downstream models

### ✅ Baseline Models
- **Train/test split:** subject-wise `GroupShuffleSplit` to prevent patient-level data leakage
- **XGBoost** and **LightGBM** classifiers trained on combined structured + NLP features
- Evaluation metric: **Quadratic Weighted Kappa (QWK)**

| Model | QWK (validation) |
|-------|-----------------|
| XGBoost | ~0.53 |
| LightGBM | ~0.56 |
| Soft-voting Ensemble | ~0.53–0.56 |

> 📌 Preliminary results — ongoing optimization.

### 🔬 SHAP Analysis (preliminary)
- XGBoost relies more heavily on **vital sign features** (HR, SpO2, temperature)
- LightGBM places higher weight on **complaint cluster features**
- Ensemble leverages complementary signal from both

---

## In Progress

- [ ] Asymmetric cost matrix integration (clinical penalty weighting for under-triage)
- [ ] Hyperparameter tuning
- [ ] Neural baseline (tabular transformer / MLP)
- [ ] Final competition submission

---

## Setup

```bash
git clone https://github.com/giuuusepeda/triagegeist.git
cd triagegeist
pip install -r requirements.txt
```

> **Data access:** You must obtain credentialed access to MIMIC-IV ED via [PhysioNet](https://physionet.org/content/mimic-iv-ed/2.2/) and place the data in `01_Data/` before running any notebooks.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | `pandas`, `numpy` |
| NLP | `transformers` (Bio_ClinicalBERT), `scikit-learn` (KMeans) |
| Modeling | `xgboost`, `lightgbm`, `scikit-learn` |
| Explainability | `shap` |
| Environment | Python 3.10+, Jupyter Notebook |

---

## Data License

This project uses the **MIMIC-IV Emergency Department** dataset (v2.2), made available via [PhysioNet](https://physionet.org/) under the **PhysioNet Credentialed Health Data License 1.5.0**.

Access requires completion of the CITI Data or Specimens Only Research training and PhysioNet credentialing. This repository contains **no patient data**.

---

## Author

**Giulia Sepeda**  
Data Scientist · BSc Nursing · BSc Business Informatics  
[GitHub](https://github.com/giuuusepeda) · [LinkedIn](https://linkedin.com/in/giuliasepeda)

---

*Work in progress — last updated March 2026*
