#!/usr/bin/env python3
"""
CCA IDP Univariate Association Study
=====================================
Uses pre-computed CCA variates to test associations with IDPs.

For each of 128 CCA components:
- For each IDP: 
  - IDP ~ T1[:, j] + covariates (separate regression)
  - IDP ~ T2[:, j] + covariates (separate regression)
- Track statistics:
  - max_t1_neg_log10_p: Maximum -log10(p-value) for T1 across all IDPs
  - percentile_95_t1: 95th percentile of -log10(p-values) for T1
  - n_sig_t1: Number of significant associations (Bonferroni corrected) for T1
  - Same for T2
- Plot max -log10(p-values), 95th percentile, and number of significant associations
- Performs analysis for both MoCoV2 and ViT CCA variates
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
idp_path = Path("/data484_4/txia2/mocov2/IDP_PhenoWAS/merged_IDP_result_filtered.csv")
ukb_all_path = Path("/data5/Ziqian/UKBB/UKB_data/UKB_all.csv")

# MoCoV2 CCA variates
cca_t1_variates_mocov2_path = Path("/data484_4/txia2/mocov2/CCA/cca_t1_variates.npy")
cca_t2_variates_mocov2_path = Path("/data484_4/txia2/mocov2/CCA/cca_t2_variates.npy")

# ViT CCA variates
cca_t1_variates_vit_path = Path("/data484_4/txia2/mocov2/CCA/cca_t1_variates_vit.npy")
cca_t2_variates_vit_path = Path("/data484_4/txia2/mocov2/CCA/cca_t2_variates_vit.npy")

# Need to get IID mapping from the original CCA data
t1_dir = "/data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2/partial_fit_5std"
output_dir = Path("/data484_4/txia2/mocov2/CCA/cca_idp_univariate")
output_dir.mkdir(parents=True, exist_ok=True)

min_idp_samples = 10000  # Filter to IDP fields with > 10000 samples

# --- Helper utilities ------------------------------------------------------
def _pick_field(field_id: str, available_cols: list[str], preferred_instances: tuple[str, ...]) -> str | None:
    """Return the first column that matches the UKB field id and preferred instance."""
    matches = [c for c in available_cols if c.startswith(f"{field_id}-")]
    if not matches:
        return None
    for suffix in preferred_instances:
        candidate = f"{field_id}-{suffix}"
        if candidate in matches:
            return candidate
    return matches[0]

# Field IDs for covariates
brain_field_specs = {
    "age": ("21003", ("2.0", "0.0", "1.0", "3.0")),
    "sex": ("31", ("0.0", "2.0", "1.0", "3.0")),
    "scanner_lateral": ("25756", ("2.0", "3.0", "0.0", "1.0")),
    "scanner_transverse": ("25757", ("2.0", "3.0", "0.0", "1.0")),
    "scanner_longitudinal": ("25758", ("2.0", "3.0", "0.0", "1.0")),
    "head_motion": ("25741", ("2.0", "3.0", "0.0", "1.0")),
    "intracranial_volume": ("25000", ("2.0", "3.0", "0.0", "1.0")),
    "body_weight": ("21002", ("2.0", "0.0", "1.0", "3.0")),
    "height": ("50", ("2.0", "0.0", "1.0", "3.0")),
    "waist_circumference": ("48", ("2.0", "0.0", "1.0", "3.0")),
    "bmi": ("23104", ("2.0", "0.0", "1.0", "3.0")),
    "assessment_centre": ("54", ("2.0", "0.0", "1.0", "3.0")),
}

# Load UKB covariates
print("Loading covariates...")
ukb_columns = list(pd.read_csv(ukb_all_path, nrows=0).columns)
field_to_column: dict[str, str] = {}
missing_covars: list[str] = []
ukb_usecols = {"eid"}
for covar, (field_id, preferred_instances) in brain_field_specs.items():
    column = _pick_field(field_id, ukb_columns, preferred_instances)
    if column is None:
        missing_covars.append(covar)
        continue
    field_to_column[covar] = column
    ukb_usecols.add(column)

pc_candidates = sorted([c for c in ukb_columns if c.startswith("22009-0.")], key=lambda x: int(x.split(".")[-1]))
pc_columns = pc_candidates[:40]
for idx, column in enumerate(pc_columns, start=1):
    name = f"pc{idx}"
    field_to_column[name] = column
    ukb_usecols.add(column)

if missing_covars:
    print(f"Missing covariate columns in UKB_all.csv: {', '.join(missing_covars)}")

covars = pd.read_csv(ukb_all_path, usecols=sorted(ukb_usecols))
covars["eid"] = covars["eid"].astype(int)
covars = covars.rename(columns={v: k for k, v in field_to_column.items()})

categorical_covars = [c for c in ["sex", "assessment_centre"] if c in covars.columns]
continuous_covars = [c for c in [
    "age",
    "scanner_lateral",
    "scanner_transverse",
    "scanner_longitudinal",
    "head_motion",
    "intracranial_volume",
    "body_weight",
    "height",
    "waist_circumference",
    "bmi",
] if c in covars.columns]
continuous_covars.extend([f"pc{i}" for i in range(1, len(pc_columns) + 1)])

# Load IDP data
print("Loading IDP data...")
idp_data = pd.read_csv(idp_path)
idp_data["eid"] = idp_data["eid"].astype(int)
meta_cols = ["eid"]
idp_columns = [c for c in idp_data.columns if c not in meta_cols]

# Filter IDP fields
print(f"Filtering IDP fields to those with > {min_idp_samples} samples...")
idp_sample_counts = idp_data[idp_columns].notna().sum()
valid_idp_fields = idp_sample_counts[idp_sample_counts > min_idp_samples].index.tolist()
print(f"Found {len(valid_idp_fields)} IDP fields with > {min_idp_samples} samples (out of {len(idp_columns)} total)")
idp_columns = valid_idp_fields

# Load CCA variates and get IID mapping
print("Loading CCA variates...")
# Load T1 and T2 data the same way as the CCA script to get correct IID order
import os
t2_dir = "/data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2_T2_5std"
n_features = 128

print("Loading T1 features to get IID mapping...")
t1_data = None
for i in tqdm(range(n_features), desc="Loading T1 features"):
    file_path = os.path.join(t1_dir, f"Feature_{i}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep="\t")
        if i == 0:
            t1_data = df[['IID', f'pca_{i}']].copy()
        else:
            t1_data = t1_data.merge(df[['IID', f'pca_{i}']], on='IID', how='outer')

print("Loading T2 features...")
t2_data = None
for i in tqdm(range(n_features), desc="Loading T2 features"):
    file_path = os.path.join(t2_dir, f"Feature_{i}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep="\t")
        if i == 0:
            t2_data = df[['IID', f'pca_{i}']].copy()
        else:
            t2_data = t2_data.merge(df[['IID', f'pca_{i}']], on='IID', how='outer')

# Merge T1 and T2 data by IID (same as CCA script)
merged_cca_data = t1_data.merge(t2_data, on='IID', how='inner', suffixes=('_T1', '_T2'))
# Drop NA values (same as CCA script)
merged_cca_data = merged_cca_data.dropna()

# Get IID mapping (this is the order used in CCA variates)
iid_mapping = merged_cca_data[['IID']].copy()
iid_mapping['IID'] = iid_mapping['IID'].astype(int)
iid_mapping['eid'] = iid_mapping['IID']  # Assuming IID == eid

print(f"IID mapping shape: {iid_mapping.shape}")

# Load MoCoV2 CCA variates
print("\nLoading MoCoV2 CCA variates...")
cca_t1_variates_mocov2 = np.load(cca_t1_variates_mocov2_path)
cca_t2_variates_mocov2 = np.load(cca_t2_variates_mocov2_path)

print(f"MoCoV2 T1 variates shape: {cca_t1_variates_mocov2.shape}")
print(f"MoCoV2 T2 variates shape: {cca_t2_variates_mocov2.shape}")

# Load ViT CCA variates
print("\nLoading ViT CCA variates...")
cca_t1_variates_vit = np.load(cca_t1_variates_vit_path)
cca_t2_variates_vit = np.load(cca_t2_variates_vit_path)

print(f"ViT T1 variates shape: {cca_t1_variates_vit.shape}")
print(f"ViT T2 variates shape: {cca_t2_variates_vit.shape}")

# Check if variates match IID length
for name, variates in [("MoCoV2 T1", cca_t1_variates_mocov2), 
                       ("MoCoV2 T2", cca_t2_variates_mocov2),
                       ("ViT T1", cca_t1_variates_vit),
                       ("ViT T2", cca_t2_variates_vit)]:
    if len(iid_mapping) != variates.shape[0]:
        print(f"Warning: IID mapping length ({len(iid_mapping)}) doesn't match {name} variates shape ({variates.shape[0]})")
        print("Using variates length for alignment...")
        if len(iid_mapping) > variates.shape[0]:
            iid_mapping = iid_mapping.iloc[:variates.shape[0]]
        else:
            print(f"Error: {name} variates have more samples than IID mapping!")

# Determine number of components
n_components = min(
    cca_t1_variates_mocov2.shape[1], cca_t2_variates_mocov2.shape[1],
    cca_t1_variates_vit.shape[1], cca_t2_variates_vit.shape[1]
)
print(f"\nNumber of CCA components: {n_components}")

# Create IID mapping dataframe
iid_df = iid_mapping.copy()
iid_df['eid'] = iid_df['IID']  # Assuming IID == eid

# Merge IDP, covariates, and CCA variates
print("Merging IDP, covariates, and CCA variates...")
# First merge IDP with covariates
merged = idp_data.merge(covars, on="eid", how="inner")

# Then merge with IID mapping to align with CCA variates
merged = merged.merge(iid_df, on="eid", how="inner")

print(f"Total merged samples: {len(merged)}")

# Align CCA variates with merged data
# The IID order in merged should match the order in iid_df (which matches Z)
# Create a mapping from IID to row index in Z
iid_to_idx = {iid: idx for idx, iid in enumerate(iid_df['IID'].values)}
merged_iids = merged['IID'].values
valid_indices = np.array([iid_to_idx.get(iid, -1) for iid in merged_iids])
valid_mask = valid_indices >= 0

if not valid_mask.all():
    print(f"Warning: {np.sum(~valid_mask)} samples could not be aligned with CCA variates")
    merged = merged.iloc[valid_mask]
    valid_indices = valid_indices[valid_mask]

# Align all variate sets
T1_aligned_mocov2 = cca_t1_variates_mocov2[valid_indices, :]
T2_aligned_mocov2 = cca_t2_variates_mocov2[valid_indices, :]
T1_aligned_vit = cca_t1_variates_vit[valid_indices, :]
T2_aligned_vit = cca_t2_variates_vit[valid_indices, :]

print(f"Samples after alignment: {len(merged)}")
print(f"Aligned MoCoV2 T1 variates shape: {T1_aligned_mocov2.shape}")
print(f"Aligned MoCoV2 T2 variates shape: {T2_aligned_mocov2.shape}")
print(f"Aligned ViT T1 variates shape: {T1_aligned_vit.shape}")
print(f"Aligned ViT T2 variates shape: {T2_aligned_vit.shape}")

# Prepare covariates matrix
print("Preparing covariates...")
local_cont = [c for c in continuous_covars if c in merged.columns]
local_cat = [c for c in categorical_covars if c in merged.columns]

# Handle continuous covariates: fill NA with median
numeric_covars = merged[local_cont].apply(pd.to_numeric, errors="coerce")
for col in numeric_covars.columns:
    if numeric_covars[col].isna().any():
        median_val = numeric_covars[col].median()
        if pd.notna(median_val):
            numeric_covars[col] = numeric_covars[col].fillna(median_val)
        else:
            numeric_covars = numeric_covars.drop(columns=[col])
            if col in local_cont:
                local_cont.remove(col)

# Handle categorical covariates: one-hot encode
cat_frames = []
for col in local_cat:
    cat_series = merged[col].fillna("missing")
    cat_frames.append(pd.get_dummies(cat_series, prefix=col, drop_first=True))

if cat_frames:
    covar_matrix = pd.concat([numeric_covars] + cat_frames, axis=1)
else:
    covar_matrix = numeric_covars.copy()

# Ensure all columns are numeric
for col in covar_matrix.columns:
    if covar_matrix[col].dtype == 'object':
        covar_matrix[col] = pd.to_numeric(covar_matrix[col], errors='coerce')

# Drop rows with any remaining NA in covariates
covar_matrix = covar_matrix.dropna()
valid_covar_indices = covar_matrix.index
merged = merged.loc[valid_covar_indices]
T1_aligned_mocov2 = T1_aligned_mocov2[merged.index.get_indexer(valid_covar_indices), :]
T2_aligned_mocov2 = T2_aligned_mocov2[merged.index.get_indexer(valid_covar_indices), :]
T1_aligned_vit = T1_aligned_vit[merged.index.get_indexer(valid_covar_indices), :]
T2_aligned_vit = T2_aligned_vit[merged.index.get_indexer(valid_covar_indices), :]

print(f"Samples after covariate filtering: {len(merged)}")
print(f"Number of covariate features: {covar_matrix.shape[1]}")

# Function to perform univariate regression for a given set of variates
def run_cca_analysis(T1_aligned, T2_aligned, model_name):
    """Run univariate regression analysis for CCA variates.
    Performs separate regressions: IDP ~ T1 + covariates and IDP ~ T2 + covariates.
    """
    print(f"\n" + "="*60)
    print(f"PERFORMING UNIVARIATE REGRESSIONS - {model_name}")
    print("="*60)
    
    all_results = []
    component_summary = []
    n_idps = len(idp_columns)
    bonferroni_threshold = -np.log10(0.05 / n_idps)
    
    for comp_idx in tqdm(range(n_components), desc=f"Processing {model_name} CCA components"):
        # Get T1 and T2 components
        t1_component = T1_aligned[:, comp_idx]
        t2_component = T2_aligned[:, comp_idx]
        
        # Store results for this component across all IDPs
        t1_stats = []
        t2_stats = []
        t1_logp = []
        t2_logp = []
        
        for idp_field in tqdm(idp_columns, desc=f"{model_name} Component {comp_idx+1}/{n_components}", leave=False):
            # Get IDP values
            idp_values = pd.to_numeric(merged[idp_field], errors="coerce")
            
            # T1 regression: IDP ~ T1 + covariates
            valid_mask_t1 = (idp_values.notna() & np.isfinite(t1_component))
            if valid_mask_t1.sum() < 100:  # Minimum samples
                t1_tstat = np.nan
                t1_p = np.nan
                t1_logp_val = np.nan
            else:
                y_t1 = idp_values[valid_mask_t1].values.astype(float)
                t1_subset = t1_component[valid_mask_t1]
                covar_subset_t1 = covar_matrix.loc[valid_mask_t1].values.astype(float)
            
            # Check for valid data
                if (len(y_t1) < 100 or 
                    np.any(~np.isfinite(y_t1)) or 
                np.any(~np.isfinite(t1_subset)) or 
                    np.any(~np.isfinite(covar_subset_t1))):
                    t1_tstat = np.nan
                    t1_p = np.nan
                    t1_logp_val = np.nan
                else:
            try:
                        # Build design matrix: [constant, t1_component, covariates]
                        X_t1 = np.hstack([
                            np.ones((len(y_t1), 1)),  # constant
                    t1_subset.reshape(-1, 1),  # T1 CCA component
                            covar_subset_t1  # covariates
                ])
                
                # Fit OLS model
                        model_t1 = sm.OLS(y_t1, X_t1).fit()
                
                        # Get t-statistics and p-values for T1 (index 1)
                        if len(model_t1.pvalues) > 1:
                            t1_tstat = model_t1.tvalues[1]  # t-statistic for T1
                            t1_p = model_t1.pvalues[1]
                            
                            if (pd.notna(t1_p) and np.isfinite(t1_p) and t1_p > 0):
                                t1_logp_val = -np.log10(t1_p)
                            else:
                                t1_logp_val = np.nan
                        else:
                            t1_tstat = np.nan
                            t1_p = np.nan
                            t1_logp_val = np.nan
                    except Exception as e:
                        t1_tstat = np.nan
                        t1_p = np.nan
                        t1_logp_val = np.nan
            
            # T2 regression: IDP ~ T2 + covariates
            valid_mask_t2 = (idp_values.notna() & np.isfinite(t2_component))
            if valid_mask_t2.sum() < 100:  # Minimum samples
                t2_tstat = np.nan
                t2_p = np.nan
                t2_logp_val = np.nan
            else:
                y_t2 = idp_values[valid_mask_t2].values.astype(float)
                t2_subset = t2_component[valid_mask_t2]
                covar_subset_t2 = covar_matrix.loc[valid_mask_t2].values.astype(float)
                
                # Check for valid data
                if (len(y_t2) < 100 or 
                    np.any(~np.isfinite(y_t2)) or 
                    np.any(~np.isfinite(t2_subset)) or 
                    np.any(~np.isfinite(covar_subset_t2))):
                    t2_tstat = np.nan
                    t2_p = np.nan
                    t2_logp_val = np.nan
                else:
                    try:
                        # Build design matrix: [constant, t2_component, covariates]
                        X_t2 = np.hstack([
                            np.ones((len(y_t2), 1)),  # constant
                            t2_subset.reshape(-1, 1),  # T2 CCA component
                            covar_subset_t2  # covariates
                        ])
                        
                        # Fit OLS model
                        model_t2 = sm.OLS(y_t2, X_t2).fit()
                        
                        # Get t-statistics and p-values for T2 (index 1)
                        if len(model_t2.pvalues) > 1:
                            t2_tstat = model_t2.tvalues[1]  # t-statistic for T2
                            t2_p = model_t2.pvalues[1]
                            
                            if (pd.notna(t2_p) and np.isfinite(t2_p) and t2_p > 0):
                                t2_logp_val = -np.log10(t2_p)
                            else:
                                t2_logp_val = np.nan
                        else:
                            t2_tstat = np.nan
                            t2_p = np.nan
                            t2_logp_val = np.nan
                    except Exception as e:
                        t2_tstat = np.nan
                        t2_p = np.nan
                        t2_logp_val = np.nan
            
            # Store results if valid
            if pd.notna(t1_logp_val) and np.isfinite(t1_logp_val):
                        t1_stats.append(t1_tstat)
                t1_logp.append(t1_logp_val)
            
            if pd.notna(t2_logp_val) and np.isfinite(t2_logp_val):
                        t2_stats.append(t2_tstat)
                        t2_logp.append(t2_logp_val)
                        
            # Store individual results
            if pd.notna(t1_logp_val) or pd.notna(t2_logp_val):
                        all_results.append({
                            "cca_component": comp_idx + 1,
                            "idp_field": idp_field,
                    "t1_tstat": t1_tstat if pd.notna(t1_tstat) else None,
                    "t2_tstat": t2_tstat if pd.notna(t2_tstat) else None,
                    "t1_p_value": t1_p if pd.notna(t1_p) else None,
                    "t2_p_value": t2_p if pd.notna(t2_p) else None,
                    "t1_neg_log10_p": t1_logp_val if pd.notna(t1_logp_val) else None,
                    "t2_neg_log10_p": t2_logp_val if pd.notna(t2_logp_val) else None,
                    "n_samples_t1": int(valid_mask_t1.sum()) if valid_mask_t1.sum() >= 100 else None,
                    "n_samples_t2": int(valid_mask_t2.sum()) if valid_mask_t2.sum() >= 100 else None,
                })
        
        # Calculate statistics for this component
        if len(t1_logp) > 0:
            t1_logp_array = np.array(t1_logp)
            max_t1_neg_log10_p = np.max(t1_logp_array)
            percentile_95_t1 = np.percentile(t1_logp_array, 95)
            n_sig_t1 = np.sum(t1_logp_array > bonferroni_threshold)
        else:
            max_t1_neg_log10_p = 0.0
            percentile_95_t1 = 0.0
            n_sig_t1 = 0
        
        if len(t2_logp) > 0:
            t2_logp_array = np.array(t2_logp)
            max_t2_neg_log10_p = np.max(t2_logp_array)
            percentile_95_t2 = np.percentile(t2_logp_array, 95)
            n_sig_t2 = np.sum(t2_logp_array > bonferroni_threshold)
        else:
            max_t2_neg_log10_p = 0.0
            percentile_95_t2 = 0.0
            n_sig_t2 = 0
        
            component_summary.append({
                "cca_component": comp_idx + 1,
            "max_t1_neg_log10_p": max_t1_neg_log10_p,
            "max_t2_neg_log10_p": max_t2_neg_log10_p,
            "percentile_95_t1": percentile_95_t1,
            "percentile_95_t2": percentile_95_t2,
            "n_sig_t1": n_sig_t1,
            "n_sig_t2": n_sig_t2,
            "n_idps": len(t1_logp),
            })
    
    return all_results, component_summary

# Run analysis for MoCoV2
mocov2_results, mocov2_summary = run_cca_analysis(T1_aligned_mocov2, T2_aligned_mocov2, "MoCoV2")

# Run analysis for ViT
vit_results, vit_summary = run_cca_analysis(T1_aligned_vit, T2_aligned_vit, "ViT")

# Process and save results for both models
def process_and_save_results(all_results, component_summary, model_name):
    """Process, save, and plot results for a given model."""
    if not all_results:
        print(f"\nNo results generated for {model_name}. Check data availability.")
        return
    
    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(component_summary)
    
    print(f"\n" + "="*60)
    print(f"SUMMARY - {model_name}")
    print("="*60)
    print(f"Total IDP-component pairs analyzed: {len(results_df)}")
    print(f"Number of unique IDPs: {results_df['idp_field'].nunique()}")
    print(f"Number of CCA components: {n_components}")
    
    # Summary statistics
    print(f"\nT1 -log10(p-value) statistics by component:")
    print(f"  Max: {summary_df['max_t1_neg_log10_p'].max():.4f}")
    print(f"  Mean of max: {summary_df['max_t1_neg_log10_p'].mean():.4f}")
    print(f"  95th percentile: {summary_df['percentile_95_t1'].max():.4f}")
    print(f"  Mean of 95th percentile: {summary_df['percentile_95_t1'].mean():.4f}")
    print(f"  Total significant associations (Bonferroni): {summary_df['n_sig_t1'].sum()}")
    
    print(f"\nT2 -log10(p-value) statistics by component:")
    print(f"  Max: {summary_df['max_t2_neg_log10_p'].max():.4f}")
    print(f"  Mean of max: {summary_df['max_t2_neg_log10_p'].mean():.4f}")
    print(f"  95th percentile: {summary_df['percentile_95_t2'].max():.4f}")
    print(f"  Mean of 95th percentile: {summary_df['percentile_95_t2'].mean():.4f}")
    print(f"  Total significant associations (Bonferroni): {summary_df['n_sig_t2'].sum()}")
    
    # Save results
    output_file = output_dir / f"cca_idp_univariate_results_{model_name.lower()}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    summary_file = output_dir / f"cca_component_summary_{model_name.lower()}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Component summary saved to: {summary_file}")
    
    # Top components
    print(f"\nTop 10 CCA Components by Max T1 -log10(p-value) ({model_name}):")
    top_t1 = summary_df.nlargest(10, 'max_t1_neg_log10_p')
    print(top_t1[['cca_component', 'max_t1_neg_log10_p', 'percentile_95_t1', 'n_sig_t1']].to_string(index=False))
    
    print(f"\nTop 10 CCA Components by Max T2 -log10(p-value) ({model_name}):")
    top_t2 = summary_df.nlargest(10, 'max_t2_neg_log10_p')
    print(top_t2[['cca_component', 'max_t2_neg_log10_p', 'percentile_95_t2', 'n_sig_t2']].to_string(index=False))
    

# Process results for both models
process_and_save_results(mocov2_results, mocov2_summary, "MoCoV2")
process_and_save_results(vit_results, vit_summary, "ViT")

print("\n=== Analysis Complete ===")
print("Note: To create the combined plot, run: plot_cca_percentile95_combined.py")
