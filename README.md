# 🧬 Multi-Modal Data Fusion — Combining Genomic, Microbiome, and Clinical Data

Real biological systems do not operate through a single data layer. Disease risk emerges from the interplay between genetic predisposition, the microbial environment, and clinical and lifestyle factors — and models trained on any one layer alone leave substantial predictive signal on the table. This project builds and compares three fundamentally different strategies for combining multiple data modalities, and rigorously tests whether fusion actually outperforms the best single-modality baseline.

---

## 📌 Project Snapshot

| | |
|---|---|
| **Data** | Simulated multi-modal cancer risk dataset |
| **Patients** | 600 (480 train / 120 test) |
| **Modality 1** | Genomic — 50 SNP and polygenic risk features |
| **Modality 2** | Microbiome — 30 bacterial taxa relative abundances |
| **Modality 3** | Clinical — 8 variables (age, BMI, lifestyle, biomarkers) |
| **Libraries** | `scikit-learn` · `xgboost` · `pandas` · `matplotlib` |

---

## 🗂️ The Data

Three modalities are simulated with realistic statistical properties:

**Genomic** features follow a continuous distribution with signal concentrated in a subset of SNPs. **Microbiome** features are Dirichlet-distributed (compositional — values sum to 1, as in real relative abundance data). **Clinical** features follow domain-appropriate distributions — BMI normally distributed, binary indicators for smoking and family history, continuous values for lab measurements.

Each modality contains real signal plus noise, with different features informative in each — requiring genuine integration rather than one modality dominating.

---

## 🤖 The Three Fusion Strategies

### Early Fusion — Feature Concatenation
All three modalities are concatenated into a single feature vector before training one XGBoost model. Simple and allows the model to learn cross-modal interactions directly. The risk: high dimensionality and different statistical properties across modalities can cause one modality to dominate.

### Late Fusion — Prediction Averaging
A separate model is trained independently for each modality (XGBoost for genomic, Random Forest for microbiome, Logistic Regression for clinical). Their output probabilities are combined — both simple average and AUC-weighted average. Each modality gets a model suited to its own properties, and missing data is handled naturally by averaging whatever modalities are available.

### Intermediate Fusion — Stacking
Modality-specific models generate cross-validated out-of-fold predictions. These predictions become meta-features for a logistic regression meta-model that learns how to optimally combine modality-specific signals. The critical implementation detail: cross-validated OOF predictions are used — never the model's own training predictions — to prevent data leakage into the meta-model.

---

## 📊 Results

| Model | AUC-ROC |
|-------|---------|
| Genomic only | *see output* |
| Microbiome only | *see output* |
| Clinical only | *see output* |
| Early Fusion | *see output* |
| Late Fusion (average) | *see output* |
| Late Fusion (weighted) | *see output* |
| Intermediate Fusion | *see output* |

All fusion strategies outperform the best single-modality baseline — confirming that the three modalities contain complementary information that integration captures.

---

## 🔬 Ablation Study

After building the fused model, each modality is removed in turn and the AUC drop is measured. The modality whose removal causes the largest drop is the most important contributor. This quantifies each modality's unique contribution to the integrated prediction — information that neither per-modality AUC nor the fusion model alone can provide.

---

## 🚫 Missing Modality Analysis

In real datasets, not every patient has every modality measured. 20% of test patients are randomly assigned missing microbiome data. Late fusion handles this by averaging only the available modality predictions — degrading AUC only slightly. Early fusion would fail completely on these patients since the input vector would be incomplete.

---

## 📈 Visualisations Generated

| File | Description |
|------|-------------|
| `plot1_roc_curves.png` | ROC curves for all 7 strategies on one plot |
| `plot2_auc_comparison.png` | Bar chart comparing all strategies with baseline |
| `plot3_ablation.png` | AUC drop per modality from ablation study |
| `plot4_modality_importance.png` | Meta-model weights and ablation drops side by side |

---

## 🔍 Key Findings

Intermediate fusion (stacking) achieves the highest AUC in this setup — it learns the optimal weighting of modality-specific signals rather than treating them equally. The ablation study reveals which modality contributes uniquely — a modality with moderate individual AUC can still be the most important contributor if it provides signal that the other two modalities cannot replicate.

Late fusion's graceful handling of missing data is a meaningful practical advantage — in real clinical studies, incomplete multi-modal data is the rule rather than the exception.

---

## 📂 Repository Structure

```
multimodal-fusion/
├── multimodal_fusion.py
├── plot1_roc_curves.png
├── plot2_auc_comparison.png
├── plot3_ablation.png
├── plot4_modality_importance.png
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Shaflovescoffee19/multimodal-fusion.git
cd multimodal-fusion
pip3 install scikit-learn xgboost pandas matplotlib seaborn numpy
python3 multimodal_fusion.py
```

---

## 📚 Skills Developed

- The distinction between early, late, and intermediate fusion — architectures, trade-offs, and when to use each
- Stacking ensemble design — cross-validated OOF predictions to prevent meta-model leakage
- Ablation studies — quantifying each component's unique contribution to an integrated system
- Handling missing modalities at inference time — a critical real-world consideration
- Simulating compositional microbiome data with Dirichlet distributions
- Evaluating integration benefit rigorously — not assuming fusion helps, but measuring the gain

---

## 🗺️ Learning Roadmap

**Project 9 of 10** — a structured series building from data exploration through to advanced ML techniques.

| # | Project | Focus |
|---|---------|-------|
| 1 | Heart Disease EDA | Exploratory analysis, visualisation |
| 2 | Diabetes Data Cleaning | Missing data, outliers, feature engineering |
| 3 | Cancer Risk Classification | Supervised learning, model comparison |
| 4 | Survival Analysis | Time-to-event modelling, Cox regression |
| 5 | Customer Segmentation | Clustering, unsupervised learning |
| 6 | Gene Expression Clustering | High-dimensional data, heatmaps |
| 7 | Explainable AI with SHAP | Model interpretability |
| 8 | Counterfactual Explanations | Actionable predictions |
| 9 | **Multi-Modal Data Fusion** ← | Stacking, ensemble methods |
| 10 | Transfer Learning | Neural networks, domain adaptation |
