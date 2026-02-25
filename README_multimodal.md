# 🧬 Multi-Modal Data Fusion — Cancer Risk Prediction

A Machine Learning project that combines three data modalities — genomic, microbiome, and clinical — using all three fusion strategies to demonstrate why integrated models outperform single-modality baselines. This is **Project 9 of 10** in my ML learning roadmap toward computational biology research.

---

## 📌 Project Overview

| Feature | Details |
|---|---|
| Data | Simulated multi-modal cancer risk dataset (600 patients) |
| Modality 1 | Genomic — 50 SNP/PRS features |
| Modality 2 | Microbiome — 30 bacterial taxa abundances |
| Modality 3 | Clinical — 8 variables (age, BMI, lifestyle, biomarkers) |
| Strategies | Early Fusion, Late Fusion, Intermediate Fusion (Stacking) |
| Libraries | `scikit-learn`, `xgboost`, `pandas`, `matplotlib` |

---

## 🧠 The Three Fusion Strategies

### Early Fusion (Feature Concatenation)
Combine all modalities into one feature vector before training a single model. Simple but can be overwhelmed by high-dimensional noisy modalities.

### Late Fusion (Decision Averaging)
Train separate models per modality, combine their output predictions. Handles missing modalities gracefully. Cannot learn cross-modal interactions.

### Intermediate Fusion (Stacking)
Train modality-specific models, use their cross-validated predictions as meta-features, train a meta-model on those. Best of both worlds — respects modality-specific properties while learning cross-modal interactions.

---

## 📊 Visualisations Generated

| Plot | What It Shows |
|---|---|
| ROC Curves | All 7 strategies on one plot (single modality + fusion) |
| AUC Comparison | Bar chart comparing all strategies with improvement over baseline |
| Ablation Study | AUC drop when each modality is removed |
| Modality Importance | Meta-model weights + ablation drops side by side |

---

## 🔍 Key Findings

- All fusion strategies outperform the best single modality — confirming the multi-modal hypothesis
- Intermediate fusion (stacking) achieves the highest AUC
- The ablation study reveals which modality contributes most to the integrated model
- Late fusion handles 20% missing microbiome data with minimal AUC degradation

---

## 📂 Project Structure

```
multimodal-fusion/
├── multimodal_fusion.py          # Main script
├── plot1_roc_curves.png
├── plot2_auc_comparison.png
├── plot3_ablation.png
├── plot4_modality_importance.png
└── README.md
```

---

## ⚙️ Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/Shaflovescoffee19/multimodal-fusion.git
cd multimodal-fusion
```

**2. Install dependencies**
```bash
pip3 install scikit-learn xgboost pandas matplotlib seaborn numpy
```

**3. Run the script**
```bash
python3 multimodal_fusion.py
```

---

## 🔬 Connection to Research Proposal

This project directly implements the core technical framework of **Aim 3** of a computational biology research proposal on CRC risk prediction in the Emirati population:

> *"The integrated risk score combines: (1) a polygenic risk score weighted by Emirati-specific allele frequencies, (2) a microbiome dysbiosis index based on CRC-associated taxa, and (3) clinical risk factors through a stacking ensemble"*

> *"The integrated model combining all three modalities is expected to achieve AUC 0.75–0.80, significantly outperforming single-modality models (AUC 0.60–0.70)"*

> *"Ablation studies will quantify each modality's independent and joint contribution"*

The stacking ensemble, modality importance analysis, and missing data handling implemented here are exactly the methods described in the proposal.

---

## 📚 What I Learned

- What **multi-modal learning** is and why it outperforms single-modality models
- The three fusion strategies — **early, late, and intermediate** — and when to use each
- How **stacking/meta-learning** learns to combine modality-specific predictions
- Why **cross-validated OOF predictions** are essential to prevent data leakage in stacking
- How to conduct an **ablation study** to quantify each modality's contribution
- How **late fusion handles missing modalities** gracefully where early fusion fails
- How this pipeline maps to real **multi-omics cancer research**

---

## 🗺️ Part of My ML Learning Roadmap

| # | Project | Status |
|---|---|---|
| 1 | Heart Disease EDA | ✅ Complete |
| 2 | Diabetes Data Cleaning | ✅ Complete |
| 3 | Cancer Risk Classification | ✅ Complete |
| 4 | Survival Analysis | ✅ Complete |
| 5 | Customer Segmentation | ✅ Complete |
| 6 | Gene Expression Clustering | ✅ Complete |
| 7 | Explainable AI with SHAP | ✅ Complete |
| 8 | Counterfactual Explanations | ✅ Complete |
| 9 | Multi-Modal Data Fusion | ✅ Complete |
| 10 | Transfer Learning | 🔜 Next |

---

## 🙋 Author

**Shaflovescoffee19** — building ML skills from scratch toward computational biology research.
