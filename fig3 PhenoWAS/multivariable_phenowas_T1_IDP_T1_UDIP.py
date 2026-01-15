#!/usr/bin/env python3
"""
Omnibus (partial F-test) per IDP:
IDP ~ all traits + covariates
vs
IDP ~ covariates

T1 IDPs with T1 UDIP traits (MoCoV2 and ViT)
Outputs ONE p-value per IDP, comparing two trait datasets.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable

# ---------------- Paths ----------------
idp_path = Path("/data484_4/txia2/mocov2/IDP_PhenoWAS/merged_IDP_result_filtered.csv")
trait_dir1 = Path("/data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2/partial_fit_5std")
trait_dir2 = Path("/data484_4/txia2/gwas_practice/individual_phenos/vit_t1_fixed")
ukb_all_path = Path("/data5/Ziqian/UKBB/UKB_data/UKB_all.csv")
output_dir = Path("/data484_4/txia2/mocov2/IDP_PhenoWAS/idp_omnibus_ftest_T1_IDP_T1_UDIP")
output_dir.mkdir(parents=True, exist_ok=True)

MIN_N = 10000
N_PCS = 40

print("=== Omnibus Partial F-test: T1 IDPs with T1 UDIP Traits ===\n")
print(f"Trait dataset 1 (MoCoV2 T1 UDIP): {trait_dir1}")
print(f"Trait dataset 2 (ViT T1 UDIP): {trait_dir2}\n")

# ---------------- Load T1-related IDP IDs ----------------
print("Loading T1-related IDP IDs...")
idp_id_xlsx = Path("/data484_4/txia2/mocov2/IDP_PhenoWAS/IDP_ID.xlsx")
df_idp_ids = pd.read_excel(idp_id_xlsx)
t1_mask_desc = df_idp_ids['IDP description'].str.contains('T1', case=False, na=False)
t1_mask_short = df_idp_ids['IDP short name'].str.contains('T1', case=False, na=False)
t1_combined = t1_mask_desc | t1_mask_short
t1_ukb_ids = [str(int(x)) for x in df_idp_ids[t1_combined]['UKB ID'].tolist() if pd.notna(x)]
print(f"Found {len(t1_ukb_ids)} T1-related UKB IDs: {t1_ukb_ids}")

# ---------------- Load IDPs ----------------
idp_df = pd.read_csv(idp_path)
idp_df["eid"] = idp_df["eid"].astype(int)
all_idp_cols = [c for c in idp_df.columns if c != "eid"]

# Filter to T1-related IDPs
t1_idp_cols = [col for col in all_idp_cols if any(ukb_id in col for ukb_id in t1_ukb_ids)]
print(f"Found {len(t1_idp_cols)} T1-related IDP columns in merged data")

# Filter IDP fields to those with > MIN_N samples
print(f"Filtering T1 IDP fields to those with > {MIN_N} samples...")
idp_sample_counts = idp_df[t1_idp_cols].notna().sum()
idp_cols = idp_sample_counts[idp_sample_counts > MIN_N].index.tolist()
print(f"Found {len(idp_cols)} T1 IDP fields with > {MIN_N} samples")

# ---------------- Load covariates ----------------
print("\nLoading covariates...")
ukb_columns = list(pd.read_csv(ukb_all_path, nrows=0).columns)

covar_cols = []

# age, sex
covar_cols += [c for c in ukb_columns if c.startswith("21003-")][:1]  # Take first match
covar_cols += [c for c in ukb_columns if c.startswith("31-")][:1]

# scanner + ICV + motion
for prefix in ["25756-", "25757-", "25758-", "25741-", "25000-"]:
    matches = [c for c in ukb_columns if c.startswith(prefix)]
    if matches:
        covar_cols.append(matches[0])

# body
for prefix in ["21002-", "50-", "48-", "23104-"]:
    matches = [c for c in ukb_columns if c.startswith(prefix)]
    if matches:
        covar_cols.append(matches[0])

# PCs
pc_cols = sorted([c for c in ukb_columns if c.startswith("22009-0.")])[:N_PCS]
covar_cols += pc_cols

# Remove duplicates while preserving order
covar_cols = list(dict.fromkeys(covar_cols))

print(f"Loading {len(covar_cols)} covariate columns...")
covars = pd.read_csv(ukb_all_path, usecols=["eid"] + covar_cols)
covars["eid"] = covars["eid"].astype(int)

# numeric + median impute
covars = covars.apply(pd.to_numeric, errors="coerce")
for c in covar_cols:
    if c in covars.columns:
        covars[c] = covars[c].fillna(covars[c].median())

print(f"Covariates loaded: {covars.shape}")

# ---------------- Load traits function ----------------
def load_traits(trait_dir):
    """Load all traits from a directory into a single DataFrame."""
    trait_frames = []

    for f in tqdm(sorted(trait_dir.glob("Feature_*")), desc=f"Loading traits from {trait_dir.name}"):
        df = pd.read_csv(f, sep=r"\s+")
        trait_col = [c for c in df.columns if c.upper() not in {"FID", "IID"}]
        if not trait_col:
            continue
        trait_col = trait_col[0]
        df = df.rename(columns={"IID": "eid"})[["eid", trait_col]]
        df = df.dropna()
        df = df.rename(columns={trait_col: f.stem})
        trait_frames.append(df)

    if not trait_frames:
        return pd.DataFrame()

    traits = trait_frames[0]
    for df in trait_frames[1:]:
        traits = traits.merge(df, on="eid", how="inner")

    traits["eid"] = traits["eid"].astype(int)
    return traits

# ---------------- Omnibus F-test function ----------------
def run_omnibus_ftest(traits_df, idp_df, covars, idp_cols, covar_cols, dataset_name):
    """Run omnibus partial F-test: IDP ~ all_traits + covariates vs IDP ~ covariates"""

    trait_cols = [c for c in traits_df.columns if c != "eid"]
    print(f"\n=== Running Omnibus F-test for {dataset_name} ===")
    print(f"Number of traits: {len(trait_cols)}")
    print(f"Number of IDPs: {len(idp_cols)}")

    # Merge everything
    merged = (
        idp_df
        .merge(traits_df, on="eid", how="inner")
        .merge(covars, on="eid", how="inner")
    )
    print(f"Merged sample size: {merged.shape[0]}")

    results = []

    # Prepare trait matrix with constant
    X_traits = merged[trait_cols].apply(pd.to_numeric, errors="coerce")
    X_traits = X_traits.fillna(X_traits.median())
    X_traits = sm.add_constant(X_traits, has_constant="add")

    # Prepare covariate matrix with constant
    valid_covar_cols = [c for c in covar_cols if c in merged.columns]
    X_cov = merged[valid_covar_cols].apply(pd.to_numeric, errors="coerce")
    X_cov = X_cov.fillna(X_cov.median())
    X_cov = sm.add_constant(X_cov, has_constant="add")

    for idp in tqdm(idp_cols, desc=f"Omnibus F-test ({dataset_name})"):
        y = pd.to_numeric(merged[idp], errors="coerce")
        mask = y.notna()

        if mask.sum() < MIN_N:
            continue

        y_clean = y[mask]

        # Full model: IDP ~ constant + all_traits + covariates
        X_full = pd.concat([X_traits.loc[mask], X_cov.loc[mask].drop(columns="const")], axis=1)

        # Reduced model: IDP ~ constant + covariates only
        X_reduced = X_cov.loc[mask]

        try:
            full_model = sm.OLS(y_clean, X_full).fit()
            reduced_model = sm.OLS(y_clean, X_reduced).fit()

            F, p, df_diff = full_model.compare_f_test(reduced_model)

            results.append({
                "idp_field": idp,
                "dataset": dataset_name,
                "n_samples": int(mask.sum()),
                "n_traits": len(trait_cols),
                "n_covariates": len(valid_covar_cols),
                "F_stat": F,
                "p_omnibus": p,
                "df_traits": df_diff,
                "r2_full": full_model.rsquared,
                "r2_reduced": reduced_model.rsquared,
                "r2_traits_only": full_model.rsquared - reduced_model.rsquared,
            })

        except np.linalg.LinAlgError:
            print(f"  Warning: LinAlgError for IDP {idp}")
            continue

    return results

# ---------------- Run analysis ----------------
# Load traits from both directories
print("\n" + "="*60)
traits1 = load_traits(trait_dir1)
print(f"Dataset 1 (MoCoV2 T1 UDIP) traits loaded: {traits1.shape}")

traits2 = load_traits(trait_dir2)
print(f"Dataset 2 (ViT T1 UDIP) traits loaded: {traits2.shape}")

# Run omnibus F-test for both datasets
results1 = run_omnibus_ftest(traits1, idp_df, covars, idp_cols, covar_cols, "mocov2_T1_UDIP")
results2 = run_omnibus_ftest(traits2, idp_df, covars, idp_cols, covar_cols, "vit_T1_UDIP")

# Combine results
all_results = results1 + results2
res_df = pd.DataFrame(all_results)

if len(res_df) > 0:
    # Calculate FDR correction per dataset
    for dataset in res_df['dataset'].unique():
        dataset_mask = res_df['dataset'] == dataset
        pvals = res_df.loc[dataset_mask, 'p_omnibus'].values
        if len(pvals) > 0:
            _, fdr_pvals, _, _ = sm.stats.multipletests(pvals, method='fdr_bh')
            res_df.loc[dataset_mask, 'p_fdr'] = fdr_pvals

    # Sort by p-value
    res_df = res_df.sort_values('p_omnibus')

    # Save results
    output_path = output_dir / "idp_omnibus_ftest_T1_IDP_T1_UDIP.csv"
    res_df.to_csv(output_path, index=False)
    print(f"\n=== Results saved to: {output_path} ===")

    # Summary statistics
    print("\n=== Summary Statistics ===")
    for dataset in res_df['dataset'].unique():
        dataset_df = res_df[res_df['dataset'] == dataset]
        print(f"\n{dataset}:")
        print(f"  Number of IDPs tested: {len(dataset_df)}")
        print(f"  Mean F-statistic: {dataset_df['F_stat'].mean():.4f}")
        print(f"  Median F-statistic: {dataset_df['F_stat'].median():.4f}")
        print(f"  Mean Incremental R²: {dataset_df['r2_traits_only'].mean():.4f}")
        sig_raw = (dataset_df['p_omnibus'] < 0.05).sum()
        sig_fdr = (dataset_df['p_fdr'] < 0.05).sum()
        print(f"  Significant (p < 0.05): {sig_raw} / {len(dataset_df)} ({100*sig_raw/len(dataset_df):.1f}%)")
        print(f"  Significant (FDR < 0.05): {sig_fdr} / {len(dataset_df)} ({100*sig_fdr/len(dataset_df):.1f}%)")

    # Create separate figures with larger fonts
    print("\n=== Creating Comparison Visualizations ===")

    # Color scheme: MoCoV2 = blue, ViT = red
    color_map = {'mocov2_T1_UDIP': '#3498db', 'vit_T1_UDIP': '#e74c3c'}

    # Clean IDP field names (remove -2.0 suffix)
    res_df['idp_field'] = res_df['idp_field'].str.replace('-2.0', '', regex=False)

    # Prepare pivots
    pivot_f = res_df.pivot(index='idp_field', columns='dataset', values='F_stat')
    res_df['neg_log10_p'] = -np.log10(res_df['p_omnibus'] + 1e-300)
    pivot_p = res_df.pivot(index='idp_field', columns='dataset', values='neg_log10_p')
    pivot_r2 = res_df.pivot(index='idp_field', columns='dataset', values='r2_traits_only')

    # Reorder columns so ViT comes first (red), then MoCoV2 (blue)
    col_order = ['vit_T1_UDIP', 'mocov2_T1_UDIP']
    col_order = [c for c in col_order if c in pivot_f.columns]
    colors = [color_map[c] for c in col_order]

    # Figure 1: F-statistics
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    pivot_f[col_order].plot(kind='bar', ax=ax1, width=0.8, color=colors)
    ax1.set_xlabel('T1 IDP Field', fontsize=20, fontweight='bold')
    ax1.set_ylabel('F-statistic', fontsize=20, fontweight='bold')
    ax1.set_title('Omnibus F-statistic by T1 IDP (T1 UDIP Traits)', fontsize=20, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.legend(['ViT_T1_UDIP', 'MoCoV2_T1_UDIP'], title='Dataset', fontsize=20, title_fontsize=20)
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig1.savefig(output_dir / "omnibus_fstat_by_idp.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'omnibus_fstat_by_idp.png'}")
    plt.close(fig1)

    # Figure 2: -log10(p-value)
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    pivot_p[col_order].plot(kind='bar', ax=ax2, width=0.8, color=colors)
    ax2.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=2, label='p=0.05')
    ax2.set_xlabel('T1 IDP Field', fontsize=20, fontweight='bold')
    ax2.set_ylabel('-log10(P-value)', fontsize=20, fontweight='bold')
    ax2.set_title('Omnibus P-value by T1 IDP (T1 UDIP Traits)', fontsize=20, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.legend(['ViT_T1_UDIP', 'MoCoV2_T1_UDIP', 'p=0.05'], title='Dataset', fontsize=20, title_fontsize=20)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig2.savefig(output_dir / "omnibus_pvalue_by_idp.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'omnibus_pvalue_by_idp.png'}")
    plt.close(fig2)

    # Figure 3: Incremental R² (traits contribution after covariates)
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    pivot_r2[col_order].plot(kind='bar', ax=ax3, width=0.8, color=colors)
    ax3.set_xlabel('T1 IDP', fontsize=20, fontweight='bold')
    ax3.set_ylabel('Incremental R²', fontsize=20, fontweight='bold')
    ax3.set_title('Incremental R² (T1 UDIP Traits Contribution After Covariates)', fontsize=20, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45, labelsize=20)
    ax3.tick_params(axis='y', labelsize=20)
    ax3.legend(['ViT_T1_UDIP', 'MoCoV2_T1_UDIP'], fontsize=18, loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig3.savefig(output_dir / "omnibus_incremental_r2_by_idp.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'omnibus_incremental_r2_by_idp.png'}")
    plt.close(fig3)

    # Figure 4: Scatter plot comparing F-statistics
    fig4, ax4 = plt.subplots(figsize=(10, 10))
    if 'mocov2_T1_UDIP' in pivot_f.columns and 'vit_T1_UDIP' in pivot_f.columns:
        valid_mask = pivot_f.notna().all(axis=1)
        if valid_mask.sum() > 0:
            x = pivot_f.loc[valid_mask, 'mocov2_T1_UDIP']
            y = pivot_f.loc[valid_mask, 'vit_T1_UDIP']
            ax4.scatter(x, y, s=150, alpha=0.7, c='purple', edgecolors='black', linewidths=1)

            # Add IDP labels
            for idx in pivot_f.loc[valid_mask].index:
                ax4.annotate(idx.split('-')[0], (x[idx], y[idx]), fontsize=14, alpha=0.8)

            max_val = max(x.max(), y.max()) * 1.1
            ax4.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='y=x')
            ax4.set_xlabel('F-statistic: MoCoV2 T1 UDIP', fontsize=20, fontweight='bold')
            ax4.set_ylabel('F-statistic: ViT T1 UDIP', fontsize=20, fontweight='bold')
            ax4.set_title('F-statistic Comparison (T1 IDP, T1 UDIP Traits)', fontsize=20, fontweight='bold')
            ax4.tick_params(axis='both', labelsize=20)
            ax4.legend(fontsize=20)
            ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    fig4.savefig(output_dir / "omnibus_fstat_scatter_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'omnibus_fstat_scatter_comparison.png'}")
    plt.close(fig4)

    print("\n=== Analysis Complete ===")
else:
    print("No results generated. Check data availability.")


# ---------------- Plotting from saved CSV (line plot) ----------------
def plot_line_from_saved_csv(csv_path=None):
    """Plot line figures from saved omnibus F-test results CSV."""
    if csv_path is None:
        csv_path = output_dir / "idp_omnibus_ftest_T1_IDP_T1_UDIP.csv"

    print(f"\n=== Plotting from saved CSV: {csv_path} ===")
    res_df = pd.read_csv(csv_path)

    # Color scheme: MoCoV2 = blue, ViT = red
    color_map = {'mocov2_T1_UDIP': '#3498db', 'vit_T1_UDIP': '#e74c3c'}

    # Clean IDP field names (remove -2.0 suffix)
    res_df['idp_field'] = res_df['idp_field'].str.replace('-2.0', '', regex=False)

    # Prepare pivots
    pivot_f = res_df.pivot(index='idp_field', columns='dataset', values='F_stat')
    res_df['neg_log10_p'] = -np.log10(res_df['p_omnibus'] + 1e-300)
    pivot_p = res_df.pivot(index='idp_field', columns='dataset', values='neg_log10_p')
    pivot_r2 = res_df.pivot(index='idp_field', columns='dataset', values='r2_traits_only')

    # Reorder columns so ViT comes first (red), then MoCoV2 (blue)
    col_order = ['vit_T1_UDIP', 'mocov2_T1_UDIP']
    col_order = [c for c in col_order if c in pivot_f.columns]
    colors = [color_map[c] for c in col_order]

    # Sort by one dataset for consistent ordering
    if col_order:
        pivot_f = pivot_f.sort_values(col_order[0], ascending=False)
        pivot_p = pivot_p.loc[pivot_f.index]
        pivot_r2 = pivot_r2.loc[pivot_f.index]

    x_indices = np.arange(len(pivot_f))

    # Figure 1: F-statistics (line plot)
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    for col, color in zip(col_order, colors):
        label = 'ViT_T1' if col == 'vit_T1_UDIP' else 'MoCoV2_T1'
        ax1.plot(x_indices, pivot_f[col].values, marker='o', color=color, label=label, linewidth=2, markersize=4)
    ax1.set_xlabel('IDP Index', fontsize=16)
    ax1.set_ylabel('F-statistic', fontsize=16)
    ax1.set_title('Omnibus F-statistic by T1 IDP (T1 UDIP Traits)', fontsize=20, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_xticks([])  # Hide x-axis labels since there are too many IDPs
    ax1.legend(loc='upper left', fontsize=18)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(output_dir / "omnibus_fstat_by_idp_line.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'omnibus_fstat_by_idp_line.png'}")
    plt.close(fig1)

    # Figure 2: -log10(p-value) (line plot)
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    for col, color in zip(col_order, colors):
        label = 'ViT_T1' if col == 'vit_T1_UDIP' else 'MoCoV2_T1'
        ax2.plot(x_indices, pivot_p[col].values, marker='o', color=color, label=label, linewidth=2, markersize=4)
    ax2.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=2, label='p=0.05')
    ax2.set_xlabel('IDP Index', fontsize=16)
    ax2.set_ylabel('-log10(P-value)', fontsize=16)
    ax2.set_title('Omnibus P-value by T1 IDP (T1 UDIP Traits)', fontsize=20, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_xticks([])
    ax2.legend(loc='upper left', fontsize=18)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(output_dir / "omnibus_pvalue_by_idp_line.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'omnibus_pvalue_by_idp_line.png'}")
    plt.close(fig2)

    # Figure 3: Incremental R² (line plot)
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    for col, color in zip(col_order, colors):
        label = 'ViT_T1' if col == 'vit_T1_UDIP' else 'MoCoV2_T1'
        ax3.plot(x_indices, pivot_r2[col].values, marker='o', color=color, label=label, linewidth=4, markersize=8)
    ax3.set_xlabel('T1 IDP', fontsize=20, fontweight='bold')
    ax3.set_ylabel('Incremental R²', fontsize=20, fontweight='bold')
    ax3.set_title('Incremental R² (T1 UDIP Traits Contribution After Covariates)', fontsize=20, fontweight='bold')
    ax3.tick_params(axis='x', labelsize=16)
    ax3.tick_params(axis='y', labelsize=16)
    ax3.set_xticks([])
    ax3.legend(loc='upper right', fontsize=18)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(output_dir / "omnibus_incremental_r2_by_idp_line.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'omnibus_incremental_r2_by_idp_line.png'}")
    plt.close(fig3)

    # Figure 4: Scatter plot comparing F-statistics
    fig4, ax4 = plt.subplots(figsize=(10, 10))
    if 'mocov2_T1_UDIP' in pivot_f.columns and 'vit_T1_UDIP' in pivot_f.columns:
        valid_mask = pivot_f.notna().all(axis=1)
        if valid_mask.sum() > 0:
            x = pivot_f.loc[valid_mask, 'mocov2_T1_UDIP']
            y = pivot_f.loc[valid_mask, 'vit_T1_UDIP']
            ax4.scatter(x, y, s=150, alpha=0.7, c='purple', edgecolors='black', linewidths=1)
            max_val = max(x.max(), y.max()) * 1.1
            ax4.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='y=x')
            ax4.set_xlabel('F-statistic: MoCoV2 T1 UDIP', fontsize=16)
            ax4.set_ylabel('F-statistic: ViT T1 UDIP', fontsize=16)
            ax4.set_title('F-statistic Comparison (T1 IDP, T1 UDIP Traits)', fontsize=20, fontweight='bold')
            ax4.tick_params(axis='both', labelsize=16)
            ax4.legend(loc='upper left', fontsize=18)
            ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    fig4.savefig(output_dir / "omnibus_fstat_scatter_comparison_line.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'omnibus_fstat_scatter_comparison_line.png'}")
    plt.close(fig4)

    print("=== Plotting Complete ===")
