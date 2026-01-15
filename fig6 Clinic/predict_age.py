#!/usr/bin/env python3
"""
Age Prediction using IDP, ViT, and MoCoV2 Features
===================================================

This script predicts age using features from:
- IDP features
- ViT features
- MoCoV2 features

Age is loaded from UKB_all.csv, field code 21003.

Results report CV Pearson r and test Pearson r.

Author: AI Assistant
Date: 2025-12-18
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

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    make_scorer,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Optional xgboost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

import matplotlib.pyplot as plt

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

# UKB data path for age
UKB_PATH = "/data5/Ziqian/UKBB/UKB_data/UKB_all.csv"
AGE_FIELD_CODE = "21003"  # Age field code

OUTPUT_ROOT = "/data484_4/txia2/mocov2/combined_regression/age_prediction"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


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


def load_age_from_ukb(ukb_path: str, age_field_code: str) -> pd.DataFrame:
    """Load age from UKB_all.csv. Returns DataFrame with eid and age columns."""
    print(f"Loading age from UKB data (field {age_field_code})...")
    
    # Find the age column (field code 21003)
    # The column name format is typically "21003-0.0" or similar
    age_cols = []
    chunk_size = 10000
    first_chunk = True
    
    for chunk in pd.read_csv(ukb_path, chunksize=chunk_size):
        if first_chunk:
            # Find age column(s) - could be multiple instances
            age_cols = [c for c in chunk.columns if c.startswith(f"{age_field_code}-")]
            if not age_cols:
                raise ValueError(f"No columns found starting with '{age_field_code}-' in UKB data")
            # Use the first instance (typically "21003-0.0")
            age_col = age_cols[0]
            print(f"Found age column: {age_col}")
            first_chunk = False
            break
    
    # Now load the full age data
    age_df = pd.read_csv(ukb_path, usecols=["eid", age_col])
    age_df = age_df.rename(columns={age_col: "age"})
    age_df = age_df.dropna(subset=["age"])
    age_df["eid"] = age_df["eid"].astype(int)
    
    print(f"Loaded {len(age_df)} samples with age data")
    return age_df[["eid", "age"]]


def split_data(features: pd.DataFrame, pheno: pd.DataFrame):
    """Merge features and phenotype, return merged DataFrame and X, y arrays."""
    # Reset index to get the ID column (could be 'index' or 'eid' depending on feature set)
    features_reset = features.reset_index()
    # The index column name might be 'index' (for MoCoV2/ViT) or 'eid' (for IDP)
    id_col = features_reset.columns[0]  # First column is the ID
    
    # Convert ID to int for matching
    features_reset[id_col] = features_reset[id_col].astype(int)
    pheno["eid"] = pheno["eid"].astype(int)
    
    merged = features_reset.merge(pheno, left_on=id_col, right_on="eid", how="inner")
    merged = merged.drop(columns=["eid"])
    X = merged.drop(columns=["age"]).values
    y = merged["age"].values
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


def run_age_regression(X, y, feature_names, out_dir, feature_set_name):
    """Run age prediction regression for a feature set."""
    os.makedirs(out_dir, exist_ok=True)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Define all regression models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Support Vector Regression": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    }
    
    # Add XGBoost if available
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            random_state=42, n_estimators=400, learning_rate=0.05,
            max_depth=4, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1
        )

    results = {}
    for name, model in models.items():
        t0 = time.time()
        
        # Determine if model needs scaled features
        # Linear models and SVR use scaled features
        # Tree-based models (RF, GB, XGB) use raw features
        needs_scaling = name in [
            "Linear Regression", "Ridge Regression", "Lasso Regression", 
            "ElasticNet", "Support Vector Regression"
        ]
        
        if needs_scaling:
            X_train_model = X_train_s
            X_test_model = X_test_s
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        model.fit(X_train_model, y_train)
        y_pred = model.predict(X_test_model)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r = pearson_r(y_test, y_pred)

        # CV
        cv_folds = min(5, max(2, len(X_train) // 20))
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scorer = make_scorer(pearson_r)
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=cv, scoring=scorer)

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
            "MSE": res["mse"],
            "RMSE": res["rmse"],
            "MAE": res["mae"],
            "Test_PearsonR": res["r"],
            "CV_PearsonR_Mean": res["cv_mean"],
            "CV_PearsonR_Std": res["cv_std"],
            "Time_s": res["time"],
        })
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, f"age_regression_summary.csv"), index=False)

    # Print results
    print(f"\n=== Results for {feature_set_name.upper()} ===")
    for name, res in results.items():
        print(f"{name}:")
        print(f"  Test Pearson r: {res['r']:.4f}")
        print(f"  CV Pearson r (mean ± std): {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
        print(f"  RMSE: {res['rmse']:.4f}")
        print(f"  MAE: {res['mae']:.4f}")

    # Plot comparison
    models_order = list(results.keys())
    r_vals = [results[m]["r"] for m in models_order]
    cv_vals = [results[m]["cv_mean"] for m in models_order]

    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(models_order))
    ax.bar(x - 0.2, r_vals, width=0.4, label="Test Pearson r")
    ax.bar(x + 0.2, cv_vals, width=0.4, label="CV Pearson r")
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=45, ha="right", fontsize=24)
    ax.set_ylabel("Pearson r", fontsize=30, fontweight='bold')
    ax.set_title(f"Age Prediction - {feature_set_name.upper()}", fontsize=30, fontweight='bold')
    ax.legend(fontsize=30, loc='upper right')
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"age_regression_comparison.png"), dpi=300)
    plt.close()

    # Save best model
    best = max(results.items(), key=lambda kv: kv[1]["r"])
    with open(os.path.join(out_dir, f"age_best_model.pkl"), "wb") as f:
        pickle.dump(best[1]["model"], f)

    return results


def run_phenotype(feature_set_name: str, feature_dirs):
    """Run age prediction for a feature set."""
    out_dir = os.path.join(OUTPUT_ROOT, feature_set_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Running Age Prediction for {feature_set_name.upper()} ===")
    print(f"Output: {out_dir}")

    # Load features for this feature set (MoCo v2, ViT, or IDP)
    if feature_set_name == "idp":
        # IDP is a single CSV file
        features = load_idp_features(feature_dirs)
    else:
        # MoCoV2 or ViT: list of directories
        features = load_features_combined(feature_dirs)
    print(f"Combined features shape: {features.shape}")

    # Load age
    age_df = load_age_from_ukb(UKB_PATH, AGE_FIELD_CODE)
    print(f"Age data shape: {age_df.shape}")

    # Merge
    merged, X, y = split_data(features, age_df)
    print(f"Merged samples: {len(merged)}")

    # Run analysis
    results = run_age_regression(X, y, merged.columns[:-1], out_dir, feature_set_name)

    # Save meta info
    meta = {
        "phenotype": "age",
        "feature_set": feature_set_name,
        "age_field_code": AGE_FIELD_CODE,
        "n_samples": len(merged),
        "n_features": X.shape[1],
        "feature_dirs": feature_dirs if isinstance(feature_dirs, list) else [feature_dirs],
        "ukb_path": UKB_PATH,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    """Main function to run age prediction for all feature sets."""
    print("="*70)
    print("Age Prediction using IDP, ViT, and MoCoV2 Features")
    print("="*70)
    
    # Run for each feature set
    for feature_set_name, feature_dirs in FEATURE_SETS.items():
        run_phenotype(feature_set_name, feature_dirs)
    
    print("\n" + "="*70)
    print("Age Prediction Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

