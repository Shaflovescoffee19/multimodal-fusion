# ============================================================
# PROJECT 9: Multi-Modal Data Fusion
# ============================================================
# WHAT THIS SCRIPT DOES:
#   1. Simulates 3 data modalities for cancer risk prediction:
#      - Genomic (SNP-based polygenic risk scores)
#      - Microbiome (bacterial taxa abundances)
#      - Clinical (demographics and lab values)
#   2. Trains single-modality baseline models
#   3. Implements all 3 fusion strategies:
#      - Early Fusion (feature concatenation)
#      - Late Fusion (prediction averaging)
#      - Intermediate Fusion (stacking/meta-learning)
#   4. Compares all strategies with AUC-ROC
#   5. Ablation study — which modality matters most?
#   6. Handles missing modality scenario
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
np.random.seed(42)

# ===========================================================
# STEP 1: SIMULATE MULTI-MODAL CANCER RISK DATA
# ===========================================================
# We simulate 3 modalities for 600 patients.
# Each modality contains real signal + noise to mimic
# the characteristics of actual clinical research data.

N_PATIENTS  = 600
N_GENOMIC   = 50    # SNP features / PRS components
N_MICROBIOME = 30   # Bacterial taxa abundances
N_CLINICAL  = 8     # Clinical variables

print("=" * 60)
print("STEP 1: SIMULATING MULTI-MODAL DATA")
print("=" * 60)
print(f"  Patients    : {N_PATIENTS}")
print(f"  Modality 1  : Genomic ({N_GENOMIC} SNP features)")
print(f"  Modality 2  : Microbiome ({N_MICROBIOME} taxa abundances)")
print(f"  Modality 3  : Clinical ({N_CLINICAL} variables)")
print()

# True underlying risk (hidden — what we want to predict)
true_risk = np.random.beta(2, 3, N_PATIENTS)
y = (true_risk > np.percentile(true_risk, 60)).astype(int)  # ~40% positive

# ── Modality 1: Genomic ───────────────────────────────────
# SNP data: binary (0/1/2 allele count) + continuous PRS
# Signal: first 15 SNPs are associated with outcome
genomic_noise = np.random.randn(N_PATIENTS, N_GENOMIC) * 0.8
genomic_signal = np.zeros((N_PATIENTS, N_GENOMIC))
genomic_signal[:, :15] = true_risk.reshape(-1, 1) * np.random.randn(1, 15) * 2
X_genomic = pd.DataFrame(
    genomic_noise + genomic_signal,
    columns=[f"SNP_{i:03d}" for i in range(N_GENOMIC)]
)

# ── Modality 2: Microbiome ────────────────────────────────
# Taxa abundances: Dirichlet-distributed (sum to 1)
# Signal: first 10 taxa associated with outcome
microbiome_base = np.random.dirichlet(np.ones(N_MICROBIOME) * 0.5, N_PATIENTS)
microbiome_signal = np.zeros((N_PATIENTS, N_MICROBIOME))
# CRC-associated taxa: higher in cases
microbiome_signal[:, :5] += true_risk.reshape(-1, 1) * 0.15
# Protective taxa: lower in cases
microbiome_signal[:, 5:10] -= true_risk.reshape(-1, 1) * 0.10
X_microbiome = pd.DataFrame(
    np.clip(microbiome_base + microbiome_signal, 0, None),
    columns=[f"Taxa_{i:03d}" for i in range(N_MICROBIOME)]
)
# Normalise to sum to 1 (relative abundance)
X_microbiome = X_microbiome.div(X_microbiome.sum(axis=1), axis=0)

# ── Modality 3: Clinical ──────────────────────────────────
clinical_data = {
    "Age":          np.random.normal(55, 12, N_PATIENTS),
    "BMI":          np.random.normal(27, 5, N_PATIENTS) + true_risk * 5,
    "FamilyHistory": (true_risk + np.random.randn(N_PATIENTS) * 0.3 > 0.7).astype(int),
    "SmokingStatus": (true_risk + np.random.randn(N_PATIENTS) * 0.4 > 0.75).astype(int),
    "AlcoholUse":   (np.random.rand(N_PATIENTS) > 0.7).astype(int),
    "PhysicalActivity": np.random.normal(3, 2, N_PATIENTS) - true_risk * 2,
    "FibreIntake":  np.random.normal(20, 8, N_PATIENTS) - true_risk * 10,
    "CEA_level":    np.random.exponential(2, N_PATIENTS) + true_risk * 8,
}
X_clinical = pd.DataFrame(clinical_data)
# Clip unrealistic values
X_clinical["Age"] = X_clinical["Age"].clip(20, 90)
X_clinical["BMI"] = X_clinical["BMI"].clip(15, 55)
X_clinical["PhysicalActivity"] = X_clinical["PhysicalActivity"].clip(0, 10)
X_clinical["FibreIntake"] = X_clinical["FibreIntake"].clip(0, 50)

print(f"  Cases (high risk)  : {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  Controls (low risk): {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
print()

# ===========================================================
# STEP 2: TRAIN / TEST SPLIT
# ===========================================================

idx_train, idx_test, y_train, y_test = train_test_split(
    np.arange(N_PATIENTS), y, test_size=0.2, random_state=42, stratify=y
)

# Split each modality
Xg_train = X_genomic.iloc[idx_train]
Xg_test  = X_genomic.iloc[idx_test]
Xm_train = X_microbiome.iloc[idx_train]
Xm_test  = X_microbiome.iloc[idx_test]
Xc_train = X_clinical.iloc[idx_train]
Xc_test  = X_clinical.iloc[idx_test]

print("=" * 60)
print("STEP 2: TRAIN / TEST SPLIT")
print("=" * 60)
print(f"  Training : {len(idx_train)} patients")
print(f"  Test     : {len(idx_test)} patients")
print()

# ===========================================================
# STEP 3: SINGLE-MODALITY BASELINE MODELS
# ===========================================================

print("=" * 60)
print("STEP 3: SINGLE-MODALITY BASELINE MODELS")
print("=" * 60)

def make_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

# Train one model per modality
genomic_model = make_pipeline(
    XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                  random_state=42, eval_metric="logloss", verbosity=0)
)
microbiome_model = make_pipeline(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
)
clinical_model = make_pipeline(
    LogisticRegression(C=1.0, max_iter=1000, random_state=42)
)

genomic_model.fit(Xg_train, y_train)
microbiome_model.fit(Xm_train, y_train)
clinical_model.fit(Xc_train, y_train)

# Predictions on test set
prob_genomic    = genomic_model.predict_proba(Xg_test)[:, 1]
prob_microbiome = microbiome_model.predict_proba(Xm_test)[:, 1]
prob_clinical   = clinical_model.predict_proba(Xc_test)[:, 1]

auc_genomic    = roc_auc_score(y_test, prob_genomic)
auc_microbiome = roc_auc_score(y_test, prob_microbiome)
auc_clinical   = roc_auc_score(y_test, prob_clinical)

print(f"  Genomic only    AUC : {auc_genomic:.4f}")
print(f"  Microbiome only AUC : {auc_microbiome:.4f}")
print(f"  Clinical only   AUC : {auc_clinical:.4f}")
print()

# ===========================================================
# STEP 4: EARLY FUSION — CONCATENATE ALL FEATURES
# ===========================================================

print("=" * 60)
print("STEP 4: EARLY FUSION (Feature Concatenation)")
print("=" * 60)

X_early_train = pd.concat([Xg_train.reset_index(drop=True),
                             Xm_train.reset_index(drop=True),
                             Xc_train.reset_index(drop=True)], axis=1)
X_early_test  = pd.concat([Xg_test.reset_index(drop=True),
                             Xm_test.reset_index(drop=True),
                             Xc_test.reset_index(drop=True)], axis=1)

early_model = make_pipeline(
    XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                  random_state=42, eval_metric="logloss", verbosity=0)
)
early_model.fit(X_early_train, y_train)
prob_early = early_model.predict_proba(X_early_test)[:, 1]
auc_early  = roc_auc_score(y_test, prob_early)

print(f"  Combined features  : {X_early_train.shape[1]}")
print(f"  Early Fusion AUC   : {auc_early:.4f}")
print()

# ===========================================================
# STEP 5: LATE FUSION — AVERAGE PREDICTIONS
# ===========================================================

print("=" * 60)
print("STEP 5: LATE FUSION (Prediction Averaging)")
print("=" * 60)

# Simple average
prob_late_avg = (prob_genomic + prob_microbiome + prob_clinical) / 3
auc_late_avg  = roc_auc_score(y_test, prob_late_avg)

# Weighted average — weight by individual AUC
weights = np.array([auc_genomic, auc_microbiome, auc_clinical])
weights = weights / weights.sum()
prob_late_weighted = (
    weights[0] * prob_genomic +
    weights[1] * prob_microbiome +
    weights[2] * prob_clinical
)
auc_late_weighted = roc_auc_score(y_test, prob_late_weighted)

print(f"  Simple average AUC    : {auc_late_avg:.4f}")
print(f"  Weighted average AUC  : {auc_late_weighted:.4f}")
print(f"  Weights used — Genomic: {weights[0]:.3f} | "
      f"Microbiome: {weights[1]:.3f} | Clinical: {weights[2]:.3f}")
print()

# ===========================================================
# STEP 6: INTERMEDIATE FUSION (STACKING)
# ===========================================================
# Train modality-specific models on training data.
# Use their CROSS-VALIDATED predictions as meta-features.
# Train a meta-model on those meta-features.
# Critical: use cross-val predictions to prevent leakage.

print("=" * 60)
print("STEP 6: INTERMEDIATE FUSION (Stacking)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Generate out-of-fold predictions for training set (no leakage)
oof_genomic    = np.zeros(len(idx_train))
oof_microbiome = np.zeros(len(idx_train))
oof_clinical   = np.zeros(len(idx_train))

for fold, (tr_idx, val_idx) in enumerate(cv.split(Xg_train, y_train)):
    # Genomic
    m = make_pipeline(XGBClassifier(n_estimators=100, max_depth=4,
                                     learning_rate=0.05, random_state=42,
                                     eval_metric="logloss", verbosity=0))
    m.fit(Xg_train.iloc[tr_idx], y_train[tr_idx])
    oof_genomic[val_idx] = m.predict_proba(Xg_train.iloc[val_idx])[:, 1]

    # Microbiome
    m = make_pipeline(RandomForestClassifier(n_estimators=100,
                                              random_state=42, n_jobs=-1))
    m.fit(Xm_train.iloc[tr_idx], y_train[tr_idx])
    oof_microbiome[val_idx] = m.predict_proba(Xm_train.iloc[val_idx])[:, 1]

    # Clinical
    m = make_pipeline(LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    m.fit(Xc_train.iloc[tr_idx], y_train[tr_idx])
    oof_clinical[val_idx] = m.predict_proba(Xc_train.iloc[val_idx])[:, 1]

# Stack OOF predictions as meta-features
X_meta_train = np.column_stack([oof_genomic, oof_microbiome, oof_clinical])

# Test meta-features = predictions from full models trained on all training data
X_meta_test = np.column_stack([prob_genomic, prob_microbiome, prob_clinical])

# Train meta-model (logistic regression — keeps it interpretable)
meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta_model.fit(X_meta_train, y_train)

prob_stacked = meta_model.predict_proba(X_meta_test)[:, 1]
auc_stacked  = roc_auc_score(y_test, prob_stacked)

print(f"  Stacking meta-features : 3 (one probability per modality)")
print(f"  Meta-model             : Logistic Regression")
print(f"  Intermediate Fusion AUC: {auc_stacked:.4f}")
print()
print("  Meta-model coefficients (modality weights):")
meta_coefs = meta_model.coef_[0]
for modality, coef in zip(["Genomic", "Microbiome", "Clinical"], meta_coefs):
    print(f"    {modality:<12s}: {coef:+.4f}")
print()

# ===========================================================
# STEP 7: ROC CURVES — ALL STRATEGIES
# ===========================================================

fig, ax = plt.subplots(figsize=(10, 8))
strategies = {
    "Genomic Only":        (prob_genomic,    "#9467BD", "--"),
    "Microbiome Only":     (prob_microbiome, "#8C564B", "--"),
    "Clinical Only":       (prob_clinical,   "#E377C2", "--"),
    "Early Fusion":        (prob_early,      "#4C72B0", "-"),
    "Late Fusion (avg)":   (prob_late_avg,   "#DD8452", "-"),
    "Late Fusion (wt)":    (prob_late_weighted, "#55A868", "-"),
    "Intermediate Fusion": (prob_stacked,    "#C44E52", "-"),
}

for name, (probs, color, ls) in strategies.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    lw = 2.5 if ls == "-" else 1.5
    ax.plot(fpr, tpr, color=color, linewidth=lw, linestyle=ls,
            label=f"{name} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Single Modality vs All Fusion Strategies",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot1_roc_curves.png")
plt.close()
print("Saved: plot1_roc_curves.png")

# ===========================================================
# STEP 8: AUC COMPARISON BAR CHART
# ===========================================================

all_aucs = {
    "Genomic":          auc_genomic,
    "Microbiome":       auc_microbiome,
    "Clinical":         auc_clinical,
    "Early Fusion":     auc_early,
    "Late (avg)":       auc_late_avg,
    "Late (weighted)":  auc_late_weighted,
    "Intermediate":     auc_stacked,
}

fig, ax = plt.subplots(figsize=(12, 6))
bar_colors = ["#9467BD", "#8C564B", "#E377C2",
              "#4C72B0", "#DD8452", "#55A868", "#C44E52"]

bars = ax.bar(list(all_aucs.keys()), list(all_aucs.values()),
              color=bar_colors, edgecolor="white", alpha=0.9)

# Baseline line = best single modality
best_single = max(auc_genomic, auc_microbiome, auc_clinical)
ax.axhline(y=best_single, color="red", linestyle="--", linewidth=1.5,
           label=f"Best single modality: {best_single:.3f}")

for bar, v in zip(bars, all_aucs.values()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_ylim(min(all_aucs.values()) - 0.05, 1.02)
ax.set_title("AUC Comparison — Single Modality vs Fusion Strategies",
             fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=20)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Add divider between single and fusion
ax.axvline(x=2.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(1.0, min(all_aucs.values()) - 0.04, "Single Modality",
        ha="center", fontsize=10, color="gray")
ax.text(5.0, min(all_aucs.values()) - 0.04, "Fusion Strategies",
        ha="center", fontsize=10, color="gray")

plt.tight_layout()
plt.savefig("plot2_auc_comparison.png")
plt.close()
print("Saved: plot2_auc_comparison.png")

# ===========================================================
# STEP 9: ABLATION STUDY
# ===========================================================
# Remove one modality at a time from the stacking model.
# The modality whose removal hurts AUC the most is
# the most important contributor.

print("=" * 60)
print("STEP 9: ABLATION STUDY")
print("=" * 60)

ablation_results = {}

# Remove genomic
X_meta_no_g = np.column_stack([oof_microbiome, oof_clinical])
X_test_no_g = np.column_stack([prob_microbiome, prob_clinical])
m_no_g = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
m_no_g.fit(X_meta_no_g, y_train)
auc_no_g = roc_auc_score(y_test, m_no_g.predict_proba(X_test_no_g)[:, 1])
ablation_results["Without Genomic"]    = auc_no_g

# Remove microbiome
X_meta_no_m = np.column_stack([oof_genomic, oof_clinical])
X_test_no_m = np.column_stack([prob_genomic, prob_clinical])
m_no_m = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
m_no_m.fit(X_meta_no_m, y_train)
auc_no_m = roc_auc_score(y_test, m_no_m.predict_proba(X_test_no_m)[:, 1])
ablation_results["Without Microbiome"] = auc_no_m

# Remove clinical
X_meta_no_c = np.column_stack([oof_genomic, oof_microbiome])
X_test_no_c = np.column_stack([prob_genomic, prob_microbiome])
m_no_c = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
m_no_c.fit(X_meta_no_c, y_train)
auc_no_c = roc_auc_score(y_test, m_no_c.predict_proba(X_test_no_c)[:, 1])
ablation_results["Without Clinical"]   = auc_no_c

print(f"  Full model AUC           : {auc_stacked:.4f}")
print()
for condition, auc_val in ablation_results.items():
    drop = auc_stacked - auc_val
    most_imp = " ← most important" if drop == max(
        auc_stacked - v for v in ablation_results.values()
    ) else ""
    print(f"  {condition:<22s}: {auc_val:.4f} (drop: {drop:+.4f}){most_imp}")
print()

fig, ax = plt.subplots(figsize=(9, 5))
ablation_labels = ["Full Model"] + list(ablation_results.keys())
ablation_aucs   = [auc_stacked] + list(ablation_results.values())
ablation_colors = ["#55A868"] + ["#C44E52", "#DD8452", "#4C72B0"]

bars = ax.bar(ablation_labels, ablation_aucs,
              color=ablation_colors, edgecolor="white", alpha=0.9)
ax.axhline(y=auc_stacked, color="green", linestyle="--",
           linewidth=1.5, alpha=0.7, label=f"Full model AUC: {auc_stacked:.3f}")

for bar, v in zip(bars, ablation_aucs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_ylim(min(ablation_aucs) - 0.05, 1.01)
ax.set_title("Ablation Study — Which Modality Contributes Most?\n"
             "(Largest AUC drop when removed = most important)",
             fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=15)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot3_ablation.png")
plt.close()
print("Saved: plot3_ablation.png")

# ===========================================================
# STEP 10: MISSING MODALITY SCENARIO
# ===========================================================
# Simulate 20% of test patients missing microbiome data.
# Compare late fusion with imputation vs ignoring.

print("=" * 60)
print("STEP 10: MISSING MODALITY SCENARIO")
print("=" * 60)

missing_pct = 0.20
n_missing = int(len(idx_test) * missing_pct)
missing_mask = np.zeros(len(idx_test), dtype=bool)
missing_mask[:n_missing] = True
np.random.shuffle(missing_mask)

print(f"  {missing_pct*100:.0f}% of test patients missing microbiome data ({n_missing} patients)")
print()

# Late fusion handles missing gracefully: use only available modalities
prob_late_missing = np.where(
    missing_mask,
    (prob_genomic + prob_clinical) / 2,    # Only 2 modalities
    (prob_genomic + prob_microbiome + prob_clinical) / 3  # All 3
)
auc_missing = roc_auc_score(y_test, prob_late_missing)

# Naive: replace missing microbiome with mean prediction
prob_microbiome_imputed = prob_microbiome.copy()
prob_microbiome_imputed[missing_mask] = prob_microbiome[~missing_mask].mean()
prob_late_imputed = (prob_genomic + prob_microbiome_imputed + prob_clinical) / 3
auc_imputed = roc_auc_score(y_test, prob_late_imputed)

print(f"  Full data AUC (late fusion)     : {auc_late_avg:.4f}")
print(f"  Missing 20% microbiome (skip)   : {auc_missing:.4f}")
print(f"  Missing 20% microbiome (impute) : {auc_imputed:.4f}")
print()

# ===========================================================
# STEP 11: MODALITY CONTRIBUTION VISUALISATION
# ===========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Meta-model weights
modality_names = ["Genomic", "Microbiome", "Clinical"]
modality_colors = ["#9467BD", "#8C564B", "#E377C2"]
meta_weights = np.abs(meta_coefs) / np.abs(meta_coefs).sum() * 100

axes[0].bar(modality_names, meta_weights,
            color=modality_colors, edgecolor="white", alpha=0.9)
axes[0].set_ylabel("Relative Weight (%)", fontsize=11)
axes[0].set_title("Meta-model Modality Weights\n(Intermediate Fusion)",
                  fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)
for i, v in enumerate(meta_weights):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

# Ablation drops
drops = [auc_stacked - v for v in ablation_results.values()]
drop_labels = ["Remove\nGenomic", "Remove\nMicrobiome", "Remove\nClinical"]
colors_drops = ["#9467BD", "#8C564B", "#E377C2"]
axes[1].bar(drop_labels, drops, color=colors_drops, edgecolor="white", alpha=0.9)
axes[1].set_ylabel("AUC Drop When Removed", fontsize=11)
axes[1].set_title("Ablation Study — AUC Drop\n(Larger drop = more important modality)",
                  fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)
for i, v in enumerate(drops):
    axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontweight="bold")

fig.suptitle("Modality Importance Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot4_modality_importance.png")
plt.close()
print("Saved: plot4_modality_importance.png")

# ===========================================================
# FINAL SUMMARY
# ===========================================================

best_fusion = max(
    {"Early": auc_early, "Late (avg)": auc_late_avg,
     "Late (weighted)": auc_late_weighted, "Intermediate": auc_stacked}.items(),
    key=lambda x: x[1]
)
most_important = max(ablation_results, key=lambda k: auc_stacked - ablation_results[k])

print()
print("=" * 60)
print("PROJECT 9 COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Dataset        : Simulated multi-modal cancer risk data")
print(f"  Patients       : {N_PATIENTS} (train={len(idx_train)}, test={len(idx_test)})")
print()
print("  Single-modality AUC:")
print(f"    Genomic      : {auc_genomic:.4f}")
print(f"    Microbiome   : {auc_microbiome:.4f}")
print(f"    Clinical     : {auc_clinical:.4f}")
print()
print("  Fusion AUC:")
print(f"    Early        : {auc_early:.4f}")
print(f"    Late (avg)   : {auc_late_avg:.4f}")
print(f"    Late (wt)    : {auc_late_weighted:.4f}")
print(f"    Intermediate : {auc_stacked:.4f}")
print()
print(f"  Best fusion strategy : {best_fusion[0]} (AUC={best_fusion[1]:.4f})")
print(f"  Most important modal : {most_important.replace('Without ', '')}")
print()
print("  Fusion vs best single modality:")
improvement = best_fusion[1] - best_single
print(f"    AUC improvement    : +{improvement:.4f}")
print()
print("  4 plots saved.")
print("  Ready to push to GitHub!")
print("=" * 60)
