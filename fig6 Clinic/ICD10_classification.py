#!/usr/bin/env python3
"""
Nested CV comparison for ICD-10 condition prediction using ViT and MoCoV2
features (T1 and T2) with class-weighted models. Uses balanced accuracy as the primary
metric in the inner/outer folds and saves results/plots under
`/data484_4/txia2/mocov2/classification`.
"""

import os
import glob
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


class MultiFeatureSetPredictor:
    """Compare feature sets with nested CV using balanced accuracy."""

    def __init__(
        self,
        ukb_path: str,
        vit_t1_features_path: str,
        vit_t2_features_path: str,
        mocov2_t1_features_path: str,
        mocov2_t2_features_path: str,
        idp_features_path: str,
        output_dir: str,
    ):
        self.ukb_path = ukb_path
        self.vit_t1_features_path = vit_t1_features_path
        self.vit_t2_features_path = vit_t2_features_path
        self.mocov2_t1_features_path = mocov2_t1_features_path
        self.mocov2_t2_features_path = mocov2_t2_features_path
        self.idp_features_path = idp_features_path
        self.output_dir = output_dir
        self.vit_features: Optional[pd.DataFrame] = None
        self.mocov2_features: Optional[pd.DataFrame] = None
        self.idp_features: Optional[pd.DataFrame] = None
        self.ukb_data: Optional[pd.DataFrame] = None

        # Classify for entire categories: F (mental/behavioural disorders) and G (nervous system disorders)
        self.conditions = {
            "mental_behavioural_disorder_diagnosis_F": ["F"],  # All F codes
            "nerve_system_diagnosis_G": ["G"],  # All G codes
        }

        os.makedirs(output_dir, exist_ok=True)

    # ---------------------- Feature loading helpers ---------------------- #
    def _load_features_generic(
        self, feature_files: List[str], sep: str, feature_name_parser
    ) -> pd.DataFrame:
        if len(feature_files) == 0:
            raise ValueError("No feature files found.")

        first_feature = pd.read_csv(
            feature_files[0], sep=sep, header=0, names=["FID", "IID", "value"]
        )
        n_samples = len(first_feature)
        n_features = len(feature_files)

        feature_matrix = np.zeros((n_samples, n_features))
        sample_ids = first_feature["IID"].values

        for i, feature_file in enumerate(feature_files):
            feature_data = pd.read_csv(
                feature_file, sep=sep, header=0, names=["FID", "IID", "value"]
            )
            if len(feature_data) != n_samples:
                raise ValueError(
                    f"Feature {feature_file} has {len(feature_data)} samples "
                    f"(expected {n_samples})."
                )
            feature_matrix[:, i] = feature_data["value"].values

        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        columns = [feature_name_parser(i) for i in range(n_features)]
        return pd.DataFrame(feature_matrix, index=sample_ids, columns=columns)

    def load_vit_features(self) -> pd.DataFrame:
        # Load T1 features
        t1_feature_files = glob.glob(os.path.join(self.vit_t1_features_path, "Feature_*"))
        t1_feature_files = [f for f in t1_feature_files if not os.path.splitext(f)[1]]
        t1_feature_files.sort(key=lambda x: int(x.split("Feature_")[-1]))
        print(f"Found {len(t1_feature_files)} ViT T1 feature files.")
        
        vit_t1 = self._load_features_generic(
            t1_feature_files, sep=r"\s+", feature_name_parser=lambda i: f"ViT_T1_Feature_{i}"
        )
        print(f"ViT T1 features loaded: {vit_t1.shape}")
        
        # Load T2 features
        t2_feature_files = glob.glob(os.path.join(self.vit_t2_features_path, "Feature_*"))
        t2_feature_files = [f for f in t2_feature_files if not os.path.splitext(f)[1]]
        t2_feature_files.sort(key=lambda x: int(x.split("Feature_")[-1]))
        print(f"Found {len(t2_feature_files)} ViT T2 feature files.")
        
        vit_t2 = self._load_features_generic(
            t2_feature_files, sep=r"\s+", feature_name_parser=lambda i: f"ViT_T2_Feature_{i}"
        )
        print(f"ViT T2 features loaded: {vit_t2.shape}")
        
        # Concatenate T1 and T2 features
        self.vit_features = pd.concat([vit_t1, vit_t2], axis=1, join="inner")
        print(f"ViT features (T1+T2) loaded: {self.vit_features.shape}")
        return self.vit_features

    def load_mocov2_features(self) -> pd.DataFrame:
        # Load T1 features
        t1_feature_files = glob.glob(os.path.join(self.mocov2_t1_features_path, "Feature_*.csv"))
        t1_feature_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        print(f"Found {len(t1_feature_files)} MoCoV2 T1 feature files.")
        if len(t1_feature_files) == 0:
            raise ValueError("No MoCoV2 T1 feature files found.")

        first_t1_feature = pd.read_csv(t1_feature_files[0], sep="\t", header=0)
        feature_col = [c for c in first_t1_feature.columns if c not in ["FID", "IID"]][0]
        n_samples_t1 = len(first_t1_feature)
        n_features_t1 = len(t1_feature_files)

        t1_feature_matrix = np.zeros((n_samples_t1, n_features_t1))
        sample_ids_t1 = first_t1_feature["IID"].values

        for i, feature_file in enumerate(t1_feature_files):
            feature_data = pd.read_csv(feature_file, sep="\t", header=0)
            if len(feature_data) != n_samples_t1:
                raise ValueError(
                    f"Feature {feature_file} has {len(feature_data)} samples "
                    f"(expected {n_samples_t1})."
                )
            feature_col = [c for c in feature_data.columns if c not in ["FID", "IID"]][0]
            t1_feature_matrix[:, i] = feature_data[feature_col].values

        t1_feature_matrix = np.nan_to_num(t1_feature_matrix, nan=0.0)
        t1_columns = [f"MoCoV2_T1_Feature_{i}" for i in range(n_features_t1)]
        mocov2_t1 = pd.DataFrame(t1_feature_matrix, index=sample_ids_t1, columns=t1_columns)
        print(f"MoCoV2 T1 features loaded: {mocov2_t1.shape}")
        
        # Load T2 features
        t2_feature_files = glob.glob(os.path.join(self.mocov2_t2_features_path, "Feature_*.csv"))
        t2_feature_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        print(f"Found {len(t2_feature_files)} MoCoV2 T2 feature files.")
        if len(t2_feature_files) == 0:
            raise ValueError("No MoCoV2 T2 feature files found.")

        first_t2_feature = pd.read_csv(t2_feature_files[0], sep="\t", header=0)
        feature_col = [c for c in first_t2_feature.columns if c not in ["FID", "IID"]][0]
        n_samples_t2 = len(first_t2_feature)
        n_features_t2 = len(t2_feature_files)

        t2_feature_matrix = np.zeros((n_samples_t2, n_features_t2))
        sample_ids_t2 = first_t2_feature["IID"].values

        for i, feature_file in enumerate(t2_feature_files):
            feature_data = pd.read_csv(feature_file, sep="\t", header=0)
            if len(feature_data) != n_samples_t2:
                raise ValueError(
                    f"Feature {feature_file} has {len(feature_data)} samples "
                    f"(expected {n_samples_t2})."
                )
            feature_col = [c for c in feature_data.columns if c not in ["FID", "IID"]][0]
            t2_feature_matrix[:, i] = feature_data[feature_col].values

        t2_feature_matrix = np.nan_to_num(t2_feature_matrix, nan=0.0)
        t2_columns = [f"MoCoV2_T2_Feature_{i}" for i in range(n_features_t2)]
        mocov2_t2 = pd.DataFrame(t2_feature_matrix, index=sample_ids_t2, columns=t2_columns)
        print(f"MoCoV2 T2 features loaded: {mocov2_t2.shape}")
        
        # Concatenate T1 and T2 features
        self.mocov2_features = pd.concat([mocov2_t1, mocov2_t2], axis=1, join="inner")
        print(f"MoCoV2 features (T1+T2) loaded: {self.mocov2_features.shape}")
        return self.mocov2_features

    def load_idp_features(self) -> pd.DataFrame:
        """Load baseline IDP features from a merged CSV (one row per eid)."""
        idp_df = pd.read_csv(self.idp_features_path)
        if "eid" not in idp_df.columns:
            raise ValueError("IDP features file must contain an 'eid' column.")

        idp_df = idp_df.set_index("eid")
        # Clean up any problematic values
        idp_df = idp_df.replace([np.inf, -np.inf], np.nan)
        idp_df = idp_df.dropna(axis=1, how="all")
        idp_df = idp_df.fillna(0.0)

        self.idp_features = idp_df
        print(f"IDP features loaded: {self.idp_features.shape}")
        return self.idp_features

    # ---------------------- Data and labeling ---------------------- #
    def load_ukb_data(self) -> pd.DataFrame:
        if all(f is None for f in [self.vit_features, self.mocov2_features, self.idp_features]):
            raise RuntimeError("Load at least one feature set before loading UKB data.")

        feature_id_sets = []
        if self.vit_features is not None:
            feature_id_sets.append(set(self.vit_features.index))
        if self.mocov2_features is not None:
            feature_id_sets.append(set(self.mocov2_features.index))
        if self.idp_features is not None:
            feature_id_sets.append(set(self.idp_features.index))

        all_patient_ids = set.union(*feature_id_sets)
        chunk_size = 10000
        filtered_chunks = []
        for chunk in pd.read_csv(self.ukb_path, chunksize=chunk_size):
            chunk_filtered = chunk[chunk["eid"].isin(all_patient_ids)]
            if len(chunk_filtered) > 0:
                filtered_chunks.append(chunk_filtered)

        if len(filtered_chunks) == 0:
            raise ValueError("No matching patients found in UKB data.")

        self.ukb_data = pd.concat(filtered_chunks, ignore_index=True)
        print(f"UKB data loaded: {self.ukb_data.shape}")
        return self.ukb_data

    def get_condition_labels(self, condition_codes: List[str]) -> pd.DataFrame:
        icd10_columns = [c for c in self.ukb_data.columns if c.startswith("41270-")]
        icd10_data = self.ukb_data[["eid"] + icd10_columns].copy()

        labels = []
        for _, row in icd10_data.iterrows():
            has_condition = False
            for col in icd10_columns:
                code = row[col]
                if pd.notna(code) and str(code).strip():
                    code_str = str(code).strip()
                    # Check if code starts with any of the condition codes
                    # For F and G categories, check if code starts with F or G
                    if any(code_str.startswith(c) for c in condition_codes):
                        has_condition = True
                        break
            labels.append(has_condition)
        icd10_data["has_condition"] = labels
        return icd10_data

    # ---------------------- Modeling ---------------------- #
    def _build_base_model(self, model_name: str, y_train: np.ndarray):
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

        if model_name == "Logistic Regression":
            return LogisticRegression(
                max_iter=2000, class_weight="balanced", solver="lbfgs"
            )
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
        if model_name == "SVM":
            return SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            )
        if model_name == "XGBoost":
            return xgb.XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
            )
        raise ValueError(f"Unknown model {model_name}")

    def _get_param_grid(self, model_name: str) -> Dict[str, List]:
        if model_name == "Logistic Regression":
            return {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__penalty": ["l2", "l1", "none"],
            }
        if model_name == "Random Forest":
            return {
                "clf__max_depth": [10, None],
                "clf__min_samples_leaf": [5, 10],
                "clf__max_features": ["sqrt"],
            }
        if model_name == "SVM":
            return {
                "clf__C": [0.1, 1.0],
                "clf__gamma": ["scale"],
            }
        if model_name == "XGBoost":
            return {
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05, 0.1],
                "clf__n_estimators": [300],
                "clf__min_child_weight": [5, 10],
            }
            
        return {}

    def _nested_cv(
        self, X: np.ndarray, y: np.ndarray, model_name: str
    ) -> Tuple[float, float, float, float, List[dict]]:
        outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        bal_acc_scores: List[float] = []
        auc_scores: List[float] = []
        best_params_per_fold: List[dict] = []

        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            base_model = self._build_base_model(model_name, y_train)
            pipeline = Pipeline([("scaler", StandardScaler()), ("clf", base_model)])
            param_grid = self._get_param_grid(model_name)

            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring="balanced_accuracy",
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            y_proba_test = best_model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba_test >= 0.5).astype(int)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba_test)

            bal_acc_scores.append(bal_acc)
            auc_scores.append(auc)
            best_params_per_fold.append(search.best_params_)

        return (
            float(np.mean(bal_acc_scores)),
            float(np.std(bal_acc_scores)),
            float(np.mean(auc_scores)),
            float(np.std(auc_scores)),
            best_params_per_fold,
        )

    def _prepare_xy(
        self, features_df: pd.DataFrame, condition_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        merged = condition_data.merge(
            features_df.reset_index().rename(columns={"index": "eid"}), on="eid", how="inner"
        )
        merged = merged.dropna(subset=["has_condition"])

        # Use all feature columns from the feature dataframe while excluding eid,
        # the label column, and raw ICD-10 code columns.
        feature_cols = [
            c
            for c in merged.columns
            if c != "eid"
            and c != "has_condition"
            and not c.startswith("41270-")
        ]
        X = merged[feature_cols].values
        y = merged["has_condition"].astype(int).values

        valid_rows = ~np.isnan(X).all(axis=1)
        X = np.nan_to_num(X[valid_rows], nan=0.0)
        y = y[valid_rows]
        return X, y

    def run(self) -> pd.DataFrame:
        self.load_vit_features()
        self.load_mocov2_features()
        # Baseline using imaging-derived phenotypes (IDPs)
        self.load_idp_features()
        self.load_ukb_data()

        models = ["Logistic Regression"]
        results = []

        for condition_name, codes in self.conditions.items():
            print(f"\nProcessing {condition_name}...")
            condition_data = self.get_condition_labels(codes)

            for feature_set_name, features_df in [
                ("VIT", self.vit_features),
                ("MoCoV2", self.mocov2_features),
                ("IDP", self.idp_features),
            ]:
                X, y = self._prepare_xy(features_df, condition_data)
                if y.sum() < 5 or len(y) < 30:
                    print(
                        f"Skipping {condition_name}-{feature_set_name}: "
                        f"insufficient samples (n={len(y)}, positives={y.sum()})."
                    )
                    continue

                for model_name in models:
                    try:
                        (
                            bal_mean,
                            bal_std,
                            auc_mean,
                            auc_std,
                            best_params,
                        ) = self._nested_cv(X, y, model_name)
                        results.append(
                            {
                                "condition": condition_name,
                                "feature_set": feature_set_name,
                                "model": model_name,
                                "balanced_acc_mean": bal_mean,
                                "balanced_acc_std": bal_std,
                                "roc_auc_mean": auc_mean,
                                "roc_auc_std": auc_std,
                                "n_samples": len(y),
                                "n_positive": int(y.sum()),
                                "n_negative": int(len(y) - y.sum()),
                                "best_params_per_fold": best_params,
                            }
                        )
                        print(
                            f"{condition_name}-{feature_set_name}-{model_name} | "
                            f"balanced_acc: {bal_mean:.3f}±{bal_std:.3f}, "
                            f"roc_auc: {auc_mean:.3f}±{auc_std:.3f}"
                        )
                    except Exception as exc:
                        print(
                            f"Error for {condition_name}-{feature_set_name}-{model_name}: {exc}"
                        )

        results_df = pd.DataFrame(results)
        results_csv = os.path.join(self.output_dir, "model_comparison_balanced.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"\nSaved results to {results_csv}")

        if not results_df.empty:
            self._create_plots(results_df)

        return results_df

    # ---------------------- Plotting ---------------------- #
    def _create_plots(self, results_df: pd.DataFrame) -> None:
        def _bar_plot(metric: str, title: str, fname: str):
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot = results_df.pivot_table(
                values=metric, index=["feature_set", "model"], columns="condition"
            )
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="RdYlBu_r",
                vmin=0,
                vmax=1,
                cbar_kws={"label": metric},
                ax=ax,
            )
            ax.set_title(title, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, fname), bbox_inches="tight")
            plt.close()

        _bar_plot(
            "balanced_acc_mean",
            "Nested CV Balanced Accuracy",
            "balanced_accuracy_heatmap.png",
        )
        _bar_plot("roc_auc_mean", "Nested CV ROC-AUC", "roc_auc_heatmap.png")


def main():
    ukb_path = "/data5/Ziqian/UKBB/UKB_data/UKB_all.csv"
    vit_t1_features_path = "/data484_4/txia2/gwas_practice/individual_phenos/vit_t1_fixed"
    vit_t2_features_path = "/data484_4/txia2/gwas_practice/individual_phenos/vit_t2_fixed"
    mocov2_t1_features_path = (
        "/data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2/partial_fit_5std"
    )
    mocov2_t2_features_path = (
        "/data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2_T2_5std"
    )
    idp_features_path = "/data484_4/txia2/mocov2/IDP_PhenoWAS/merged_IDP_result_filtered.csv"
    output_dir = "/data484_4/txia2/mocov2/classification/IDP"

    predictor = MultiFeatureSetPredictor(
        ukb_path,
        vit_t1_features_path,
        vit_t2_features_path,
        mocov2_t1_features_path,
        mocov2_t2_features_path,
        idp_features_path,
        output_dir,
    )
    predictor.run()


if __name__ == "__main__":
    main()

