#!/usr/bin/env python3
"""
Combined MAE Feature Regression/Classification Analyses
=======================================================

This script runs cognitive phenotype prediction using MAE features from BOTH:
- MoCo v2 features:
- /data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2/partial_fit_5std
- /data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2_T2_5std
- ViT features:
  - /data484_4/txia2/gwas_practice/individual_phenos/vit_t1_fixed
  - /data484_4/txia2/gwas_practice/individual_phenos/vit_t2_fixed

Phenotypes (all 8 outcomes from tian_cognitive_scores.csv, all treated as regression):
- participant.p21004_i2
- participant.p23324_i2
- participant.p6373_i2
- participant.p6348_i2
- participant.p6350_i2
- participant.p20016_i2
- participant.p4282_i2
- participant.p20023_i2

Results are stored under /data484_4/txia2/mocov2/combined_regression/<phenotype>/

Author: AI Assistant
Date: 2025-12-10
"""

import os
import glob
import time
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, SVR

import matplotlib.pyplot as plt
import seaborn as sns

# Optional xgboost
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ---------------------------------------------------------------------
# Paths / Feature sets
# ---------------------------------------------------------------------
MOCOV2_FEATURE_DIRS = [
    "/data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2/partial_fit_5std",
    "/data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2_T2_5std",
]

VIT_FEATURE_DIRS = [
    "/data484_4/txia2/gwas_practice/individual_phenos/vit_t1_fixed",
    "/data484_4/txia2/gwas_practice/individual_phenos/vit_t2_fixed",
]

IDP_FEATURES_PATH = "/data484_4/txia2/mocov2/IDP_PhenoWAS/merged_IDP_result_filtered.csv"

FEATURE_SETS = {
    "mocov2": MOCOV2_FEATURE_DIRS,
    "vit": VIT_FEATURE_DIRS,
    "idp": IDP_FEATURES_PATH,
}

# All 8 outcomes from tian_cognitive_scores.csv
PHENOTYPE_CSV = "/data484_4/txia2/mocov2/combined_regression/tian_cognitive_scores.csv"
OUTCOME_COLUMNS = [
    "participant.p21004_i2",
    "participant.p23324_i2",
    "participant.p6373_i2",
    "participant.p6348_i2",
    "participant.p6350_i2",
    "participant.p20016_i2",
    "participant.p4282_i2",
    "participant.p20023_i2",
]

OUTPUT_ROOT = "/data484_4/txia2/mocov2/combined_regression"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# UKB data path for F/G labels
UKB_PATH = "/data5/Ziqian/UKBB/UKB_data/UKB_all.csv"

# F and G label definitions (ICD-10 codes)
ICD10_LABELS = {
    "F": ["F"],  # Mental/behavioural disorders
    "G": ["G"],  # Nervous system disorders
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_feature_dir(feature_dir: str, prefix: str) -> pd.DataFrame:
    """Load all Feature_*.csv from a directory and return DataFrame indexed by IID."""
    feature_files = glob.glob(os.path.join(feature_dir, "Feature_*"))
    if not feature_files:
        raise FileNotFoundError(f"No Feature_*.csv files found in {feature_dir}")
    feature_files.sort(key=lambda x: int(Path(x).stem.split('_')[-1]))

    first = pd.read_csv(feature_files[0], sep=r"\s+", engine="python")
    n_samples = len(first)
    n_features = len(feature_files)

    mat = np.zeros((n_samples, n_features))
    iids = first["IID"].values

    for i, fpath in enumerate(feature_files):
        df = pd.read_csv(fpath, sep=r"\s+", engine="python")
        if len(df) != n_samples:
            raise ValueError(f"Feature file {fpath} has {len(df)} rows, expected {n_samples}")
        # assume value column is third col
        mat[:, i] = df.iloc[:, 2].values

    cols = [f"{prefix}_Feature_{i}" for i in range(n_features)]
    return pd.DataFrame(mat, index=iids, columns=cols)


def load_features_combined(feature_dirs: List[str]) -> pd.DataFrame:
    """Load and combine features from multiple directories (columns concatenated)."""
    dfs = []
    for idx, fdir in enumerate(feature_dirs):
        prefix = f"T{idx+1}"
        dfs.append(load_feature_dir(fdir, prefix))
    # inner join on index
    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.join(df, how="inner")
    return combined


def load_idp_features(idp_path: str) -> pd.DataFrame:
    """Load IDP features from CSV file. Returns DataFrame indexed by eid."""
    df = pd.read_csv(idp_path)
    if "eid" not in df.columns:
        raise ValueError(f"eid column not found in {idp_path}")
    
    # Set eid as index
    df = df.set_index("eid")
    # Rename columns to have IDP prefix
    df.columns = [f"IDP_{col}" for col in df.columns]
    return df


def load_phenotype(phenotype_col: str, csv_path: str, is_categorical: bool = False) -> pd.DataFrame:
    """Load phenotype from CSV file. phenotype_col should be the full column name like 'participant.p20016_i2'."""
    df = pd.read_csv(csv_path)
    if "participant.eid" not in df.columns:
        raise ValueError(f"participant.eid column not found in {csv_path}")
    if phenotype_col not in df.columns:
        raise ValueError(f"Column {phenotype_col} not found in {csv_path}")
    df = df.rename(columns={"participant.eid": "eid", phenotype_col: "label"})
    df = df.dropna(subset=["label"])
    if is_categorical:
        # convert to category codes
        df["label"] = df["label"].astype("category")
    return df[["eid", "label"]]


def split_data(features: pd.DataFrame, pheno: pd.DataFrame):
    # Reset index to get the ID column (could be 'index' or 'eid' depending on feature set)
    features_reset = features.reset_index()
    # The index column name might be 'index' (for MoCoV2/ViT) or 'eid' (for IDP)
    id_col = features_reset.columns[0]  # First column is the ID
    merged = features_reset.merge(pheno, left_on=id_col, right_on="eid", how="inner")
    merged = merged.drop(columns=["eid"])
    X = merged.drop(columns=["label"]).values
    y = merged["label"].values
    return merged, X, y


# ---------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------
def pearson_r(y_true, y_pred):
    """Compute Pearson correlation, returning 0 when variance is zero."""
    if len(y_true) < 2:
        return 0.0
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def run_regression(X, y, feature_names, out_dir, pheno_name, feature_set_name=None):
    os.makedirs(out_dir, exist_ok=True)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Only run SVR for IDP, otherwise run both models
    if feature_set_name == "idp":
        models = {
            "Support Vector Regression": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        }
    else:
        models = {
            "Lasso Regression": Lasso(alpha=0.1, random_state=42),
            "Support Vector Regression": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        }

    results = {}
    for name, model in models.items():
        t0 = time.time()
        # Both Lasso and SVR use scaled features
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r = pearson_r(y_test, y_pred)

        # CV
        cv_folds = min(5, max(2, len(X_train) // 20))
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scorer = make_scorer(pearson_r)
        # Both Lasso and SVR use scaled features
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring=scorer)

        results[name] = {
            "model": model,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r": r,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_test": y_test,
            "y_pred": y_pred,
            "time": time.time() - t0,
        }

    # Save summary
    summary = []
    for name, res in results.items():
        summary.append({
            "Model": name,
            "RMSE": res["rmse"],
            "MAE": res["mae"],
            "PearsonR": res["r"],
            "CV_PearsonR_Mean": res["cv_mean"],
            "CV_PearsonR_Std": res["cv_std"],
            "Time_s": res["time"],
        })
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, f"{pheno_name}_regression_summary.csv"), index=False)

    # Plot comparison
    models_order = list(results.keys())
    r_vals = [results[m]["r"] for m in models_order]
    cv_vals = [results[m]["cv_mean"] for m in models_order]

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(models_order))
    ax.bar(x - 0.2, r_vals, width=0.4, label="Test Pearson r")
    ax.bar(x + 0.2, cv_vals, width=0.4, label="CV Pearson r")
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=45, ha="right", fontsize=30)
    ax.set_ylabel("Pearson r", fontsize=30, fontweight='bold')
    ax.set_title(f"{pheno_name} Regression Performance", fontsize=30, fontweight='bold')
    ax.legend(fontsize=30, loc='upper right')
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{pheno_name}_regression_comparison.png"), dpi=300)
    plt.close()

    # Save best model
    best = max(results.items(), key=lambda kv: kv[1]["r"])
    with open(os.path.join(out_dir, f"{pheno_name}_best_model.pkl"), "wb") as f:
        pickle.dump(best[1]["model"], f)

    return results


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------
def run_classification(X, y, feature_names, out_dir, pheno_name):
    os.makedirs(out_dir, exist_ok=True)

    # Impute missing values
    imputer = SimpleImputer(strategy="most_frequent")
    X = imputer.fit_transform(X)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class="auto"),
        "Ridge Classifier": RidgeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            random_state=42, n_estimators=400, learning_rate=0.05,
            max_depth=4, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss",
        )

    results = {}
    for name, model in models.items():
        t0 = time.time()
        if name in ["Logistic Regression", "Ridge Classifier", "SVM"]:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_proba = model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        roc = None
        if y_proba is not None and len(np.unique(y)) == 2:
            roc = roc_auc_score(y_test, y_proba[:, 1])

        # CV
        cv_folds = min(5, max(2, len(X_train) // 20))
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        if name in ["Logistic Regression", "Ridge Classifier", "SVM"]:
            cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1_macro": f1,
            "roc_auc": roc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_test": y_test,
            "y_pred": y_pred,
            "time": time.time() - t0,
        }

    # Save summary
    summary = []
    for name, res in results.items():
        summary.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "F1_macro": res["f1_macro"],
            "ROC_AUC": res["roc_auc"],
            "CV_Acc_Mean": res["cv_mean"],
            "CV_Acc_Std": res["cv_std"],
            "Time_s": res["time"],
        })
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, f"{pheno_name}_classification_summary.csv"), index=False)

    # Plot comparison
    models_order = list(results.keys())
    acc_vals = [results[m]["accuracy"] for m in models_order]
    cv_vals = [results[m]["cv_mean"] for m in models_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models_order))
    ax.bar(x - 0.2, acc_vals, width=0.4, label="Test Accuracy")
    ax.bar(x + 0.2, cv_vals, width=0.4, label="CV Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{pheno_name} Classification Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{pheno_name}_classification_comparison.png"), dpi=300)
    plt.close()

    # Confusion matrix for best model
    best = max(results.items(), key=lambda kv: kv[1]["accuracy"])
    y_test = best[1]["y_test"]
    y_pred = best[1]["y_pred"]
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{pheno_name} Confusion Matrix - {best[0]}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{pheno_name}_confusion_matrix.png"), dpi=300)
    plt.close()

    # Save best model
    with open(os.path.join(out_dir, f"{pheno_name}_best_model.pkl"), "wb") as f:
        pickle.dump(best[1]["model"], f)

    return results


# ---------------------------------------------------------------------
# Comparison plotting
# ---------------------------------------------------------------------
def plot_vit_vs_mocov2_comparison(out_root: str, metric: str = "PearsonR"):
    """
    Create bar plots comparing IDP, ViT, and MoCoV2 for each model across all 8 outcomes.
    Order: IDP first, then ViT, then MoCoV2.
    
    Parameters:
    -----------
    out_root : str
        Root output directory containing mocov2/, vit/, and idp/ subdirectories
    metric : str
        Metric to compare (default: "PearsonR", options: "PearsonR", "CV_PearsonR_Mean", "RMSE", "MAE")
    """
    feature_sets = ["idp", "vit", "mocov2"]  # Order: IDP first, then ViT, then MoCoV2
    models = ["Support Vector Regression"]  # Only SVR for IDP, but we'll handle this in the loop
    
    # Extract phenotype names from OUTCOME_COLUMNS, remove "_i2" and "p" prefix for display
    phenotype_names = [col.replace("participant.", "").replace("_i2", "").replace("p", "") for col in OUTCOME_COLUMNS]
    # Keep original names with "_i2" for directory paths
    phenotype_names_with_i2 = [col.replace("participant.", "") for col in OUTCOME_COLUMNS]
    
    # Load results for each feature set and model
    results = {}
    results_std = {}  # Store standard deviations for error bars
    for feature_set in feature_sets:
        results[feature_set] = {}
        results_std[feature_set] = {}
        # For IDP, only SVR; for others, try both models
        if feature_set == "idp":
            model_list = ["Support Vector Regression"]
        else:
            model_list = ["Lasso Regression", "Support Vector Regression"]
        
        for model_name in model_list:
            results[feature_set][model_name] = {}
            results_std[feature_set][model_name] = {}
            for idx, pheno_name in enumerate(phenotype_names):
                # Use original name with "_i2" for directory path
                pheno_name_with_i2 = phenotype_names_with_i2[idx]
                summary_path = os.path.join(out_root, feature_set, pheno_name_with_i2, f"{pheno_name_with_i2}_regression_summary.csv")
                if os.path.exists(summary_path):
                    df = pd.read_csv(summary_path)
                    model_row = df[df["Model"] == model_name]
                    if not model_row.empty:
                        results[feature_set][model_name][pheno_name] = model_row[metric].values[0]
                        # Load std if metric is CV_PearsonR_Mean
                        if metric == "CV_PearsonR_Mean":
                            results_std[feature_set][model_name][pheno_name] = model_row["CV_PearsonR_Std"].values[0]
                        else:
                            results_std[feature_set][model_name][pheno_name] = 0
                    else:
                        print(f"Warning: Model '{model_name}' not found in {summary_path}")
                        results[feature_set][model_name][pheno_name] = None
                        results_std[feature_set][model_name][pheno_name] = 0
                else:
                    print(f"Warning: Summary file not found: {summary_path}")
                    results[feature_set][model_name][pheno_name] = None
                    results_std[feature_set][model_name][pheno_name] = 0
    
    # Create a plot for SVR model (common to all feature sets)
    model_name = "Support Vector Regression"
    fig, ax = plt.subplots(figsize=(18, 10))
    
    x = np.arange(len(phenotype_names))
    width = 0.25  # Three groups: IDP, ViT, MoCoV2
    
    idp_vals = [results["idp"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
    vit_vals = [results["vit"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
    mocov2_vals = [results["mocov2"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
    
    # Get std values for error bars (only for CV_PearsonR_Mean)
    if metric == "CV_PearsonR_Mean":
        idp_stds = [results_std["idp"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
        vit_stds = [results_std["vit"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
        mocov2_stds = [results_std["mocov2"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
        bars1 = ax.bar(x - width, idp_vals, width, label="IDP", alpha=0.8, color="#1e8449",
                       yerr=idp_stds, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
        bars2 = ax.bar(x, vit_vals, width, label="ViT", alpha=0.8, color="#e74c3c",
                       yerr=vit_stds, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
        bars3 = ax.bar(x + width, mocov2_vals, width, label="MoCoV2", alpha=0.8, color="#3498db",
                       yerr=mocov2_stds, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    else:
        bars1 = ax.bar(x - width, idp_vals, width, label="IDP", alpha=0.8, color="#1e8449")
        bars2 = ax.bar(x, vit_vals, width, label="ViT", alpha=0.8, color="#e74c3c")
        bars3 = ax.bar(x + width, mocov2_vals, width, label="MoCoV2", alpha=0.8, color="#3498db")
    
    ax.set_xlabel("Cognitive Scores", fontsize=30, fontweight='bold')
    ax.set_ylabel(metric.replace("_", " "), fontsize=34, fontweight='bold')
    ax.set_title("Cognitive Score prediction - Support Vector Regression", fontsize=34, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(phenotype_names, rotation=45, ha="right", fontsize=30)
    ax.legend(fontsize=26, loc='upper right')
    ax.tick_params(axis='y', labelsize=34)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    
    # Save plot
    metric_safe = metric.replace("_", "").lower()
    output_path = os.path.join(out_root, f"idp_vit_mocov2_{model_name.lower().replace(' ', '_')}_{metric_safe}_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    plt.close()
    
    # Also create a combined plot with both models side by side (for ViT and MoCoV2)
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    all_models = ["Lasso Regression", "Support Vector Regression"]
    for idx, model_name in enumerate(all_models):
        ax = axes[idx]
        x = np.arange(len(phenotype_names))
        width = 0.25
        
        # For Lasso, only ViT and MoCoV2 (no IDP)
        if model_name == "Lasso Regression":
            vit_vals = [results["vit"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
            mocov2_vals = [results["mocov2"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
            if metric == "CV_PearsonR_Mean":
                vit_stds = [results_std["vit"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
                mocov2_stds = [results_std["mocov2"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
                bars2 = ax.bar(x - width/2, vit_vals, width, label="ViT", alpha=0.8, color="#e74c3c",
                               yerr=vit_stds, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
                bars3 = ax.bar(x + width/2, mocov2_vals, width, label="MoCoV2", alpha=0.8, color="#3498db",
                               yerr=mocov2_stds, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
            else:
                bars2 = ax.bar(x - width/2, vit_vals, width, label="ViT", alpha=0.8, color="#e74c3c")
                bars3 = ax.bar(x + width/2, mocov2_vals, width, label="MoCoV2", alpha=0.8, color="#3498db")
            bars_list = [bars2, bars3]
        else:
            # For SVR, all three
            idp_vals = [results["idp"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
            vit_vals = [results["vit"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
            mocov2_vals = [results["mocov2"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
            if metric == "CV_PearsonR_Mean":
                idp_stds = [results_std["idp"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
                vit_stds = [results_std["vit"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
                mocov2_stds = [results_std["mocov2"][model_name].get(pheno, 0) or 0 for pheno in phenotype_names]
                bars1 = ax.bar(x - width, idp_vals, width, label="IDP", alpha=0.8, color="#1e8449",
                               yerr=idp_stds, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
                bars2 = ax.bar(x, vit_vals, width, label="ViT", alpha=0.8, color="#e74c3c",
                               yerr=vit_stds, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
                bars3 = ax.bar(x + width, mocov2_vals, width, label="MoCoV2", alpha=0.8, color="#3498db",
                               yerr=mocov2_stds, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
            else:
                bars1 = ax.bar(x - width, idp_vals, width, label="IDP", alpha=0.8, color="#1e8449")
                bars2 = ax.bar(x, vit_vals, width, label="ViT", alpha=0.8, color="#e74c3c")
                bars3 = ax.bar(x + width, mocov2_vals, width, label="MoCoV2", alpha=0.8, color="#3498db")
            bars_list = [bars1, bars2, bars3]
        
        ax.set_xlabel("Cognitive Scores", fontsize=30, fontweight='bold')
        ax.set_ylabel(metric.replace("_", " "), fontsize=34, fontweight='bold')
        if model_name == "Support Vector Regression":
            ax.set_title("Cognitive Score prediction - Support Vector Regression", fontsize=34, fontweight='bold', pad=20)
        else:
            ax.set_title(f"{model_name}", fontsize=34, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(phenotype_names, rotation=45, ha="right", fontsize=30)
        ax.legend(fontsize=26, loc='upper right')
        ax.tick_params(axis='y', labelsize=34)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.8)
    
    fig.suptitle(f"IDP vs ViT vs MoCoV2 Comparison Across All Outcomes ({metric.replace('_', ' ')})", 
                 fontsize=30, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save combined plot
    metric_safe = metric.replace("_", "").lower()
    output_path = os.path.join(out_root, f"idp_vit_mocov2_all_models_{metric_safe}_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined comparison plot: {output_path}")
    plt.close()


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------
def run_phenotype(phenotype_col: str, csv_path: str, out_root: str, feature_set_name: str, feature_dirs):
    # Extract phenotype name from column (e.g., "participant.p20016_i2" -> "p20016_i2")
    phenotype = phenotype_col.replace("participant.", "")
    # All outcomes are treated as regression (continuous) unless specified otherwise
    is_categorical = False
    out_dir = os.path.join(out_root, feature_set_name, phenotype)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Running {phenotype} ({'classification' if is_categorical else 'regression'}) ===")
    print(f"Output: {out_dir}")

    # Load features for this feature set (MoCo v2, ViT, or IDP)
    if feature_set_name == "idp":
        # IDP is a single CSV file
        features = load_idp_features(feature_dirs)
    else:
        # MoCoV2 or ViT: list of directories
        features = load_features_combined(feature_dirs)
    print(f"Combined features shape: {features.shape}")

    # Load phenotype
    pheno_df = load_phenotype(phenotype_col, csv_path, is_categorical=is_categorical)
    print(f"Phenotype shape: {pheno_df.shape}")

    # Merge
    merged, X, y = split_data(features, pheno_df)
    print(f"Merged samples: {len(merged)}")

    # Run analysis (all outcomes are regression)
    results = run_regression(X, y, merged.columns[:-1], out_dir, phenotype, feature_set_name=feature_set_name)

    # Save meta info
    meta = {
        "phenotype": phenotype,
        "feature_set": feature_set_name,
        "phenotype_column": phenotype_col,
        "is_categorical": is_categorical,
        "n_samples": len(merged),
        "n_features": X.shape[1],
        "feature_dirs": feature_dirs,
        "phenotype_file": csv_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------
# F/G Label Overlap Statistics
# ---------------------------------------------------------------------
def print_fg_overlap_stats():
    """Calculate and print overlap statistics for F/G labels with MoCoV2 features."""
    print("\n" + "="*70)
    print("F/G Label Overlap Statistics with MoCoV2 Features")
    print("="*70)
    
    # Load MoCoV2 feature IDs
    features = load_features_combined(MOCOV2_FEATURE_DIRS)
    mocov2_ids = set(features.index.astype(int))
    print(f"\nTotal unique MoCoV2 IIDs: {len(mocov2_ids)}")
    
    # Load UKB data and extract F/G labels
    f_ids = set()
    g_ids = set()
    
    chunk_size = 10000
    chunk_count = 0
    for chunk in pd.read_csv(UKB_PATH, chunksize=chunk_size):
        chunk_count += 1
        if "eid" not in chunk.columns:
            continue
        
        icd10_cols = [c for c in chunk.columns if c.startswith("41270-")]
        if not icd10_cols:
            continue
        
        # Filter to MoCoV2 IDs only
        chunk_filtered = chunk[chunk["eid"].isin(mocov2_ids)]
        if len(chunk_filtered) == 0:
            continue
        
        icd = chunk_filtered[icd10_cols + ["eid"]].copy()
        codes = icd[icd10_cols].astype(str)
        codes = codes.where(codes.notna(), '')
        
        # Vectorized check for F and G codes
        has_F = codes.apply(lambda row: any(str(v).startswith('F') for v in row if v and str(v).strip() != ''), axis=1)
        has_G = codes.apply(lambda row: any(str(v).startswith('G') for v in row if v and str(v).strip() != ''), axis=1)
        
        f_ids.update(icd.loc[has_F, "eid"].astype(int).tolist())
        g_ids.update(icd.loc[has_G, "eid"].astype(int).tolist())
        
        if chunk_count % 100 == 0:
            print(f"Processed {chunk_count} chunks...", end='\r')
    
    print(f"\nProcessed {chunk_count} chunks from UKB data")
    
    # Calculate overlaps
    f_overlap = f_ids & mocov2_ids
    g_overlap = g_ids & mocov2_ids
    
    print("\n" + "-"*70)
    print("F Label (Mental/behavioural disorders - ICD-10 F*):")
    print(f"  Total F-positive samples in UKB: {len(f_ids)}")
    print(f"  F-positive samples with MoCoV2 features: {len(f_overlap)}")
    print(f"  Percentage of F-positive with MoCoV2: {len(f_overlap)/len(f_ids)*100:.2f}%" if f_ids else "  N/A")
    print(f"  Percentage of MoCoV2 samples that are F-positive: {len(f_overlap)/len(mocov2_ids)*100:.2f}%")
    
    print("\n" + "-"*70)
    print("G Label (Nervous system disorders - ICD-10 G*):")
    print(f"  Total G-positive samples in UKB: {len(g_ids)}")
    print(f"  G-positive samples with MoCoV2 features: {len(g_overlap)}")
    print(f"  Percentage of G-positive with MoCoV2: {len(g_overlap)/len(g_ids)*100:.2f}%" if g_ids else "  N/A")
    print(f"  Percentage of MoCoV2 samples that are G-positive: {len(g_overlap)/len(mocov2_ids)*100:.2f}%")
    
    print("\n" + "-"*70)
    print("Combined:")
    fg_union = f_ids | g_ids
    fg_overlap = (f_ids | g_ids) & mocov2_ids
    print(f"  Total F or G-positive samples in UKB: {len(fg_union)}")
    print(f"  F or G-positive samples with MoCoV2 features: {len(fg_overlap)}")
    print(f"  Percentage of F/G-positive with MoCoV2: {len(fg_overlap)/len(fg_union)*100:.2f}%" if fg_union else "  N/A")
    print(f"  Percentage of MoCoV2 samples that are F or G-positive: {len(fg_overlap)/len(mocov2_ids)*100:.2f}%")
    
    print("="*70 + "\n")


def main():
    # Print F/G overlap statistics with MoCoV2 features
    print_fg_overlap_stats()
    
    # Run regression for cognitive outcomes - only IDP (ViT and MoCoV2 already finished)
    print("\n##### Running feature set: idp (ViT and MoCoV2 already completed) #####")
    for phenotype_col in OUTCOME_COLUMNS:
        run_phenotype(phenotype_col, PHENOTYPE_CSV, OUTPUT_ROOT, "idp", FEATURE_SETS["idp"])
    
    # Generate comparison plots after all analyses are complete
    print("\n##### Generating IDP vs ViT vs MoCoV2 comparison plots #####")
    plot_vit_vs_mocov2_comparison(OUTPUT_ROOT, metric="PearsonR")
    plot_vit_vs_mocov2_comparison(OUTPUT_ROOT, metric="CV_PearsonR_Mean")


if __name__ == "__main__":
    main()

