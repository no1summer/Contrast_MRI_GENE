#!/usr/bin/env python
"""Apply PCA to pre-extracted features from contrastive learning model.

Usage:
    python apply_pca_to_features_T2.py \
        --features_dir /path/to/features/directory \
        --output_dir /path/to/output/ \
        [--n_components 128] [--chunk_size 1000]
"""

import os
import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import joblib


def apply_pca_to_features(features_dir, output_dir, n_components=128, chunk_size=1000):
    """
    Load pre-extracted features and apply PCA transformation.
    
    Args:
        features_dir (str): Directory containing features_before_pca.npy, eids.npy, and metadata.json
        output_dir (str): Path to save the PCA-reduced features and models
        n_components (int): Number of PCA components (default: 128)
        chunk_size (int): Chunk size for processing large arrays (default: 1000)
    """
    # Load features and metadata
    features_path = os.path.join(features_dir, 'features_before_pca.npy')
    eids_path = os.path.join(features_dir, 'eids.npy')
    metadata_path = os.path.join(features_dir, 'metadata.json')
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not os.path.exists(eids_path):
        raise FileNotFoundError(f"EIDs file not found: {eids_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    print("Loading features and metadata...")
    all_compute_pools = np.load(features_path)
    dataset_eids = np.load(eids_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded features with shape: {all_compute_pools.shape}")
    print(f"Loaded {len(dataset_eids)} EIDs")
    print(f"Metadata: {metadata}")
    
    # Verify sizes match
    if len(dataset_eids) != all_compute_pools.shape[0]:
        raise ValueError(f"EID count ({len(dataset_eids)}) doesn't match features count ({all_compute_pools.shape[0]})")
    
    # Apply IncrementalPCA to reduce to n_components features
    print(f"Applying IncrementalPCA to reduce features to {n_components}...")
    
    # Standardize the features before PCA - use chunking for memory efficiency with large arrays
    print("Standardizing features (this may take a while for large datasets)...")
    scaler = StandardScaler()
    
    # For very large arrays, compute mean and std incrementally to avoid memory issues
    n_samples = all_compute_pools.shape[0]
    n_features = all_compute_pools.shape[1]
    
    print(f"Computing mean in chunks of {chunk_size} samples...")
    # Compute mean incrementally (use float32 for memory efficiency)
    mean_sum = np.zeros(n_features, dtype=np.float32)
    for i in tqdm(range(0, n_samples, chunk_size), desc="Computing mean"):
        chunk = all_compute_pools[i:i+chunk_size].astype(np.float32)
        mean_sum += chunk.sum(axis=0).astype(np.float32)
    mean = (mean_sum / n_samples).astype(np.float32)
    
    print(f"Computing std in chunks of {chunk_size} samples...")
    # Compute std incrementally (use float32 for memory efficiency)
    var_sum = np.zeros(n_features, dtype=np.float32)
    for i in tqdm(range(0, n_samples, chunk_size), desc="Computing std"):
        chunk = all_compute_pools[i:i+chunk_size].astype(np.float32)
        var_sum += ((chunk - mean) ** 2).sum(axis=0).astype(np.float32)
    std = np.sqrt(var_sum / n_samples).astype(np.float32)
    std[std == 0] = 1.0  # Avoid division by zero
    
    # Set scaler parameters
    scaler.mean_ = mean.astype(np.float64)  # StandardScaler expects float64 internally
    scaler.scale_ = std.astype(np.float64)
    scaler.var_ = (std ** 2).astype(np.float64)
    scaler.n_samples_seen_ = n_samples
    
    # Apply IncrementalPCA using partial_fit in chunks (more memory efficient)
    print(f"Fitting IncrementalPCA using partial_fit in chunks (this may take a while)...")
    pca = IncrementalPCA(n_components=n_components, batch_size=min(chunk_size, 1000))
    
    # First pass: partial_fit on scaled chunks
    print("First pass: Fitting PCA incrementally...")
    for i in tqdm(range(0, n_samples, chunk_size), desc="Fitting PCA"):
        chunk = all_compute_pools[i:i+chunk_size].astype(np.float32)
        chunk_scaled = scaler.transform(chunk).astype(np.float32)  # Scale and convert to float32
        pca.partial_fit(chunk_scaled)
    
    print("Second pass: Transforming data with fitted PCA...")
    # Second pass: transform in chunks and collect results
    all_compute_pools_pca = []
    for i in tqdm(range(0, n_samples, chunk_size), desc="Transforming with PCA"):
        chunk = all_compute_pools[i:i+chunk_size].astype(np.float32)
        chunk_scaled = scaler.transform(chunk).astype(np.float32)  # Scale and convert to float32
        chunk_pca = pca.transform(chunk_scaled).astype(np.float32)  # Transform and keep as float32
        all_compute_pools_pca.append(chunk_pca)
    
    # Concatenate PCA results (float32 to save memory)
    all_compute_pools_pca = np.concatenate(all_compute_pools_pca, axis=0).astype(np.float32)
    print("IncrementalPCA completed.")
    
    print(f"PCA completed. Reduced features shape: {all_compute_pools_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Individual explained variance ratios (first 10): {pca.explained_variance_ratio_[:10]}")
    
    # Create a pandas DataFrame with PCA-reduced features
    df = pd.DataFrame(all_compute_pools_pca, columns=[f'pca_{i}' for i in range(n_components)])
    
    # Add EIDs from the dataset
    df['IID'] = dataset_eids
    df['FID'] = dataset_eids
    df = df.drop_duplicates(subset=['FID'], keep='first')
    
    # Filter to discovery set
    discovery_iid = pd.read_csv("/data484_4/txia2/DeepENDO/UDIP/output/test_again/UDIP_id_discovery.csv", sep=" ")
    df = df[df['IID'].isin(discovery_iid['eid'])]
    
    # Save discovery set CSV
    os.makedirs(output_dir, exist_ok=True)
    discovery_path = os.path.join(output_dir, "UDIP_id_discovery_T2.csv")
    df.to_csv(discovery_path, sep=" ", index=False)
    print(f"✅ Saved discovery set to: {discovery_path}")
    
    # Save PCA-reduced features
    num_features = all_compute_pools_pca.shape[1]
    print(f"Saving {num_features} PCA-reduced features from contrastive learning model")
    print(f"Original features: {metadata['num_patches']} patches * {metadata['embed_dim']} embed_dim = {metadata['feature_dim']}")
    print(f"PCA-reduced features: {num_features}")
    print(f"Will save {num_features} individual feature files (Feature_0.csv to Feature_{num_features-1}.csv)")
    
    # Save PCA model and scaler for future use
    print("Saving PCA model and scaler...")
    pca_path = os.path.join(output_dir, 'pca_model.pkl')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(pca, pca_path)
    print(f"✅ Saved PCA model to: {pca_path}")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved scaler to: {scaler_path}")
    
    # Save individual PCA features (n_components)
    print("Saving individual feature files...")
    for i in tqdm(range(0, n_components), desc="Saving feature files"):
        df_to_save = df[['FID', 'IID', f'pca_{i}']]
        file_name = f'Feature_{i}.csv'
        df_to_save.to_csv(os.path.join(output_dir, file_name), sep=' ', index=False)
    print("✅ All feature files saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply PCA to pre-extracted features from contrastive learning model.')
    parser.add_argument('--features_dir', type=str, required=True, 
                        help='Directory containing features_before_pca.npy, eids.npy, and metadata.json')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Path to save the PCA-reduced features and models')
    parser.add_argument('--n_components', type=int, default=128, 
                        help='Number of PCA components (default: 128)')
    parser.add_argument('--chunk_size', type=int, default=1000, 
                        help='Chunk size for processing large arrays (default: 1000)')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    apply_pca_to_features(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        n_components=args.n_components,
        chunk_size=args.chunk_size
    )



