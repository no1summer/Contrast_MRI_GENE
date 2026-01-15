#!/usr/bin/env python
"""Apply PCA to pre-extracted features from contrastive learning model.

Usage:
    python apply_pca_to_features_T1.py \
        --checkpoint_path /path/to/inference_checkpoint.pt \
        --output_dir /path/to/output/ \
        [--n_components 128] [--chunk_size 1000]
"""

import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import joblib
import torch


def apply_pca_to_features(checkpoint_path, output_dir, n_components=128, chunk_size=1000):
    """
    Load pre-extracted features and apply PCA transformation.
    
    Args:
        checkpoint_path (str): Path to the inference_checkpoint.pt file containing features, eids, and metadata
        output_dir (str): Path to save the PCA-reduced features and models
        n_components (int): Number of PCA components (default: 128)
        chunk_size (int): Chunk size for processing large arrays (default: 1000)
    """
    # Load checkpoint file
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract features and eids from checkpoint
    if 'features' not in checkpoint_data:
        raise KeyError("Checkpoint does not contain 'features' key")
    if 'eids' not in checkpoint_data:
        raise KeyError("Checkpoint does not contain 'eids' key")
    
    print("Extracting features and eids from checkpoint...")
    features_tensor = checkpoint_data['features']
    dataset_eids = checkpoint_data['eids']
    
    # Convert features to numpy array
    if isinstance(features_tensor, torch.Tensor):
        all_compute_pools = features_tensor.numpy()
    else:
        all_compute_pools = np.array(features_tensor)
    
    # Convert eids to numpy array if needed
    if isinstance(dataset_eids, torch.Tensor):
        dataset_eids = dataset_eids.numpy()
    elif not isinstance(dataset_eids, np.ndarray):
        dataset_eids = np.array(dataset_eids)
    
    # Extract metadata from checkpoint
    metadata = {
        'num_patches': checkpoint_data.get('num_patches', None),
        'embed_dim': checkpoint_data.get('embed_dim', None),
        'feature_dim': checkpoint_data.get('feature_dim', all_compute_pools.shape[1]),
        'total_samples': checkpoint_data.get('total_samples', len(dataset_eids))
    }
    
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
    chunk_size = 1000
    n_samples = all_compute_pools.shape[0]
    n_features = all_compute_pools.shape[1]
    
    print(f"Computing mean in chunks of {chunk_size} samples...")
    # Compute mean incrementally
    mean_sum = np.zeros(n_features)
    for i in tqdm(range(0, n_samples, chunk_size), desc="Computing mean"):
        chunk = all_compute_pools[i:i+chunk_size]
        mean_sum += chunk.sum(axis=0)
    mean = mean_sum / n_samples
    
    print(f"Computing std in chunks of {chunk_size} samples...")
    # Compute std incrementally
    var_sum = np.zeros(n_features)
    for i in tqdm(range(0, n_samples, chunk_size), desc="Computing std"):
        chunk = all_compute_pools[i:i+chunk_size]
        var_sum += ((chunk - mean) ** 2).sum(axis=0)
    std = np.sqrt(var_sum / n_samples)
    std[std == 0] = 1.0  # Avoid division by zero
    
    # Set scaler parameters
    scaler.mean_ = mean
    scaler.scale_ = std
    scaler.var_ = std ** 2
    scaler.n_samples_seen_ = n_samples
    
    print("Transforming features with StandardScaler in chunks (this may take a while)...")
    # Transform in chunks to avoid memory issues
    all_compute_pools_scaled = []
    for i in tqdm(range(0, n_samples, chunk_size), desc="Scaling features"):
        chunk = all_compute_pools[i:i+chunk_size]
        chunk_scaled = scaler.transform(chunk)
        all_compute_pools_scaled.append(chunk_scaled)
    
    all_compute_pools_scaled = np.concatenate(all_compute_pools_scaled, axis=0)
    print("StandardScaler completed.")
    # Apply IncrementalPCA using partial_fit in chunks (more memory efficient)
    print(f"Fitting IncrementalPCA using partial_fit in chunks (this may take a while)...")
    pca = IncrementalPCA(n_components=n_components, batch_size=min(chunk_size, 1000))
    
    n_samples = all_compute_pools_scaled.shape[0]
    
    # First pass: partial_fit on chunks (no standardization)
    print("First pass: Fitting PCA incrementally...")
    for i in tqdm(range(0, n_samples, chunk_size), desc="Fitting PCA"):
        chunk = all_compute_pools_scaled[i:i+chunk_size].astype(np.float32)
        pca.partial_fit(chunk)
    
    print("Second pass: Transforming data with fitted PCA...")
    # Second pass: transform in chunks and collect results
    all_compute_pools_pca = []
    for i in tqdm(range(0, n_samples, chunk_size), desc="Transforming with PCA"):
        chunk = all_compute_pools_scaled[i:i+chunk_size].astype(np.float32)
        chunk_pca = pca.transform(chunk).astype(np.float32)  # Transform and keep as float32
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
    discovery_path = os.path.join(output_dir, "UDIP_id_discovery_T1.csv")
    df.to_csv(discovery_path, sep=" ", index=False)
    print(f"✅ Saved discovery set to: {discovery_path}")
    
    # Save PCA-reduced features
    num_features = all_compute_pools_pca.shape[1]
    print(f"Saving {num_features} PCA-reduced features from contrastive learning model")
    print(f"Original features: {metadata['num_patches']} patches * {metadata['embed_dim']} embed_dim = {metadata['feature_dim']}")
    print(f"PCA-reduced features: {num_features}")
    print(f"Will save {num_features} individual feature files (Feature_0.csv to Feature_{num_features-1}.csv)")
    
    # Save PCA model for future use
    print("Saving PCA model...")
    pca_path = os.path.join(output_dir, 'pca_model.pkl')
    joblib.dump(pca, pca_path)
    print(f"✅ Saved PCA model to: {pca_path}")
    
    # Save individual PCA features (n_components)
    print("Saving individual feature files...")
    for i in tqdm(range(0, n_components), desc="Saving feature files"):
        df_to_save = df[['FID', 'IID', f'pca_{i}']]
        file_name = f'Feature_{i}.csv'
        df_to_save.to_csv(os.path.join(output_dir, file_name), sep=' ', index=False)
    print("✅ All feature files saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply PCA to pre-extracted features from contrastive learning model.')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the inference_checkpoint.pt file containing features, eids, and metadata')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Path to save the PCA-reduced features and models')
    parser.add_argument('--n_components', type=int, default=128, 
                        help='Number of PCA components (default: 128)')
    parser.add_argument('--chunk_size', type=int, default=1000, 
                        help='Chunk size for processing large arrays (default: 1000)')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    apply_pca_to_features(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        n_components=args.n_components,
        chunk_size=args.chunk_size
    )

