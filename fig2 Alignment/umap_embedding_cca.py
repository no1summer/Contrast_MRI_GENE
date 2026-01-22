#!/usr/bin/env python3
"""
UMAP Embedding Analysis for MoCoV2 and ViT CCA Variates
======================================================

Performs UMAP embedding on CCA variates for both MoCoV2 and ViT models.
Uses the same data loading logic as CCA_IDP_Univariate.py but applies UMAP to CCA variates.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Set paths - same as CCA_IDP_Univariate.py
# MoCoV2 CCA variates
cca_t1_variates_mocov2_path = Path("/data484_4/txia2/mocov2/CCA/cca_t1_variates.npy")
cca_t2_variates_mocov2_path = Path("/data484_4/txia2/mocov2/CCA/cca_t2_variates.npy")

# ViT CCA variates
cca_t1_variates_vit_path = Path("/data484_4/txia2/mocov2/CCA/cca_t1_variates_vit.npy")
cca_t2_variates_vit_path = Path("/data484_4/txia2/mocov2/CCA/cca_t2_variates_vit.npy")

# Need to get IID mapping from the original CCA data - same as CCA script
t1_dir = "/data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2/partial_fit_5std"
t2_dir = "/data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2_T2_5std"

output_dir = Path("/data484_4/txia2/mocov2/UMAP")
output_dir.mkdir(parents=True, exist_ok=True)

print("Starting UMAP embedding analysis for CCA variates of both MoCoV2 and ViT models...")

def load_cca_variates_and_iid_mapping():
    """
    Load CCA variates and get IID mapping - exactly like CCA_IDP_Univariate.py
    """
    print("Loading CCA variates and IID mapping...")

    # Load T1 and T2 data the same way as the CCA script to get correct IID order
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

    # Check if variates match IID length and align them
    def align_variates(variates, name):
        """Align variates with IID mapping by truncating to match lengths."""
        if len(iid_mapping) != variates.shape[0]:
            print(f"Warning: IID mapping length ({len(iid_mapping)}) doesn't match {name} variates shape ({variates.shape[0]})")
            min_samples = min(len(iid_mapping), variates.shape[0])
            print(f"Truncating to {min_samples} samples for alignment...")
            aligned_iid_mapping = iid_mapping.iloc[:min_samples]
            aligned_variates = variates[:min_samples]
            return aligned_variates, aligned_iid_mapping
        return variates, iid_mapping

    # Align all variates
    cca_t1_variates_mocov2, iid_mapping_mocov2 = align_variates(cca_t1_variates_mocov2, "MoCoV2 T1")
    cca_t2_variates_mocov2, _ = align_variates(cca_t2_variates_mocov2, "MoCoV2 T2")
    cca_t1_variates_vit, iid_mapping_vit = align_variates(cca_t1_variates_vit, "ViT T1")
    cca_t2_variates_vit, _ = align_variates(cca_t2_variates_vit, "ViT T2")

    # Determine number of components
    n_components = min(
        cca_t1_variates_mocov2.shape[1], cca_t2_variates_mocov2.shape[1],
        cca_t1_variates_vit.shape[1], cca_t2_variates_vit.shape[1]
    )
    print(f"\nNumber of CCA components: {n_components}")

    return (cca_t1_variates_mocov2, cca_t2_variates_mocov2,
            cca_t1_variates_vit, cca_t2_variates_vit,
            iid_mapping_mocov2, iid_mapping_vit, n_components)

def perform_umap_embedding(cca_t1, cca_t2, model_name, iid_mapping):
    """
    Perform UMAP embedding on CCA variates.
    Fit on T1 CCA variates, transform both T1 and T2 to same embedding space.
    """
    print(f"\nPerforming UMAP embedding for {model_name} CCA variates...")

    # Fit UMAP on T1 CCA variates, then transform both T1 and T2
    print(f"Fitting UMAP on {model_name} T1 CCA variates...")
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1, verbose=True)
    X_umap_t1 = umap_model.fit_transform(cca_t1)

    print(f"\nTransforming {model_name} T2 CCA variates using the fitted UMAP model...")
    X_umap_t2 = umap_model.transform(cca_t2)

    print(f"\n{model_name} UMAP results:")
    print(f"  T1 CCA embedding shape: {X_umap_t1.shape}")
    print(f"  T2 CCA embedding shape: {X_umap_t2.shape}")

    return X_umap_t1, X_umap_t2

def create_visualization(X_umap_t1, X_umap_t2, model_name, output_dir):
    """
    Create and save UMAP visualization for a model.
    """
    plt.figure(figsize=(14, 10))

    # Plot T1 samples in blue
    plt.scatter(X_umap_t1[:, 0], X_umap_t1[:, 1], alpha=0.5, s=15,
                c='steelblue', label='T1', edgecolors='darkblue', linewidths=0.1)

    # Plot T2 samples in red
    plt.scatter(X_umap_t2[:, 0], X_umap_t2[:, 1], alpha=0.5, s=15,
                c='coral', label='T2', edgecolors='darkred', linewidths=0.1)

    plt.xlabel('UMAP Component 1', fontsize=20)
    plt.ylabel('UMAP Component 2', fontsize=20)
    plt.title(f'UMAP Embedding: {model_name} T1 vs T2 CCA Variates', fontsize=20, pad=20)
    plt.legend(fontsize=20, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Save the plot
    output_path = output_dir / f'umap_embedding_{model_name.lower()}_cca_t1_t2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{model_name} plot saved to: {output_path}")
    plt.close()

def save_umap_coordinates(iid_mapping, X_umap_t1, X_umap_t2, model_name, output_dir):
    """
    Save UMAP coordinates to CSV file.
    """
    umap_df = pd.DataFrame({
        'IID': iid_mapping['IID'].values,
        'umap_t1_1': X_umap_t1[:, 0],
        'umap_t1_2': X_umap_t1[:, 1],
        'umap_t2_1': X_umap_t2[:, 0],
        'umap_t2_2': X_umap_t2[:, 1]
    })

    output_csv = output_dir / f'umap_embedding_coordinates_{model_name.lower()}_cca_t1_t2.csv'
    umap_df.to_csv(output_csv, index=False)
    print(f"{model_name} CCA UMAP coordinates saved to: {output_csv}")

    return umap_df

def main():
    """
    Main function to perform UMAP embedding for CCA variates of both models.
    """
    # Load CCA variates and IID mapping - same as CCA_IDP_Univariate.py
    try:
        (cca_t1_mocov2, cca_t2_mocov2,
         cca_t1_vit, cca_t2_vit,
         iid_mapping_mocov2, iid_mapping_vit, n_components) = load_cca_variates_and_iid_mapping()
    except Exception as e:
        print(f"Error loading CCA variates: {e}")
        return

    # Process MoCoV2 model
    try:
        print("\n" + "="*60)
        print("PROCESSING MoCoV2 CCA VARIATES")
        print("="*60)

        X_umap_t1_mocov2, X_umap_t2_mocov2 = perform_umap_embedding(
            cca_t1_mocov2, cca_t2_mocov2, "MoCoV2", iid_mapping_mocov2
        )

        create_visualization(X_umap_t1_mocov2, X_umap_t2_mocov2, "MoCoV2", output_dir)

        mocov2_df = save_umap_coordinates(
            iid_mapping_mocov2, X_umap_t1_mocov2, X_umap_t2_mocov2, "MoCoV2", output_dir
        )

        print(f"\n=== MoCoV2 CCA Summary ===")
        print(f"Total samples in embeddings: {len(iid_mapping_mocov2)}")
        print(f"CCA components: {n_components}")
        print(f"T1 CCA variates: {cca_t1_mocov2.shape[1]} dimensions")
        print(f"T2 CCA variates: {cca_t2_mocov2.shape[1]} dimensions")

    except Exception as e:
        print(f"Error processing MoCoV2 CCA variates: {e}")

    # Process ViT model
    try:
        print("\n" + "="*60)
        print("PROCESSING ViT CCA VARIATES")
        print("="*60)

        X_umap_t1_vit, X_umap_t2_vit = perform_umap_embedding(
            cca_t1_vit, cca_t2_vit, "ViT", iid_mapping_vit
        )

        create_visualization(X_umap_t1_vit, X_umap_t2_vit, "ViT", output_dir)

        vit_df = save_umap_coordinates(
            iid_mapping_vit, X_umap_t1_vit, X_umap_t2_vit, "ViT", output_dir
        )

        print(f"\n=== ViT CCA Summary ===")
        print(f"Total samples in embeddings: {len(iid_mapping_vit)}")
        print(f"CCA components: {n_components}")
        print(f"T1 CCA variates: {cca_t1_vit.shape[1]} dimensions")
        print(f"T2 CCA variates: {cca_t2_vit.shape[1]} dimensions")

    except Exception as e:
        print(f"Error processing ViT CCA variates: {e}")

    print(f"\nUMAP parameters used:")
    print(f"  - n_neighbors: 30")
    print(f"  - min_dist: 0.1")
    print(f"  - n_components: 2")
    print(f"  - Random state: 42")
    print(f"  - Fitted on T1 CCA variates, transformed on T2 CCA variates (same embedding space)")

    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
