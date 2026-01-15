#!/usr/bin/env python
"""Extract compute_pool features from a trained contrastive learning ViT model.

Usage:
    python extract_compute_pool_contrast_learning.py \
        --checkpoint /path/to/contrast_learning/checkpoint.pth \
        --datafile /path/to/data.csv \
        --output_dir /path/to/output/ \
        [--batch_size 32] [--num_workers 4]
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys

sys.path.append("/data484_4/txia2/DeepENDO/training")

# Import contrast learning model
from engine_128_T1_T2_vit_contrast_learning import MoCoV2Dual
import nibabel as nib

# Custom dataset for T1 and T2 from separate CSV files
class aedataset_T1T2_separate(torch.utils.data.Dataset):
    def __init__(self, datafile_T1, datafile_T2, transforms=None):
        self.df_T1 = pd.read_csv(datafile_T1)
        self.df_T2 = pd.read_csv(datafile_T2)
        self.transforms = transforms
        
        print(f"Original T1 dataset size: {len(self.df_T1)}")
        print(f"Original T2 dataset size: {len(self.df_T2)}")
        
        # Check for duplicates in each dataset
        t1_duplicates = self.df_T1['EID'].duplicated().sum()
        t2_duplicates = self.df_T2['EID'].duplicated().sum()
        print(f"T1 dataset has {t1_duplicates} duplicate EIDs")
        print(f"T2 dataset has {t2_duplicates} duplicate EIDs")
        
        # Remove duplicates by keeping first occurrence
        if t1_duplicates > 0:
            self.df_T1 = self.df_T1.drop_duplicates(subset=['EID'], keep='first')
            print(f"After removing duplicates - T1 dataset size: {len(self.df_T1)}")
        
        if t2_duplicates > 0:
            self.df_T2 = self.df_T2.drop_duplicates(subset=['EID'], keep='first')
            print(f"After removing duplicates - T2 dataset size: {len(self.df_T2)}")
        
        # Ensure both datasets have the same EIDs
        common_eids = set(self.df_T1['EID']) & set(self.df_T2['EID'])
        print(f"Found {len(common_eids)} common subjects between T1 and T2 datasets")
        
        # Filter to common EIDs and sort by EID to ensure alignment
        self.df_T1 = self.df_T1[self.df_T1['EID'].isin(common_eids)].sort_values('EID').reset_index(drop=True)
        self.df_T2 = self.df_T2[self.df_T2['EID'].isin(common_eids)].sort_values('EID').reset_index(drop=True)
        
        print(f"After filtering - T1 dataset size: {len(self.df_T1)}")
        print(f"After filtering - T2 dataset size: {len(self.df_T2)}")
        
        # Verify alignment
        if len(self.df_T1) != len(self.df_T2):
            raise ValueError(f"Dataset size mismatch: T1={len(self.df_T1)}, T2={len(self.df_T2)}")
        
        # Verify EID alignment
        if not (self.df_T1['EID'] == self.df_T2['EID']).all():
            raise ValueError("EID alignment failed - EIDs don't match between T1 and T2 datasets")
        
        print("Dataset alignment verified successfully")

    def __len__(self):
        return len(self.df_T1)

    def __getitem__(self, idx):
        # Safety check to prevent IndexError
        if idx >= len(self.df_T1) or idx >= len(self.df_T2):
            raise IndexError(f"Index {idx} out of bounds. T1 length: {len(self.df_T1)}, T2 length: {len(self.df_T2)}")
        
        row_T1 = self.df_T1.iloc[idx]
        row_T2 = self.df_T2.iloc[idx]
        
        # Verify EID match for this specific index
        if row_T1['EID'] != row_T2['EID']:
            raise ValueError(f"EID mismatch at index {idx}: T1 EID={row_T1['EID']}, T2 EID={row_T2['EID']}")
        
        x_T1 = self.load_sample(row_T1, "mri_names", self.transforms)
        x_T2 = self.load_sample(row_T2, "mri_names", self.transforms)
        # Use T1 mask for loss
        mask_T1 = self.load_mask(row_T1, "mri_names")
        mask_T2 = self.load_mask(row_T2, "mri_names")
        return x_T1, x_T2, mask_T1, mask_T2
    
    def load_sample(self, row, modality_col, transforms):
        img_path = row[modality_col]
        img = nib.load(img_path)
        img = img.get_fdata()
        img = torch.from_numpy(img)
        img = torch.nn.functional.pad(img, (0,0,3,3,0,0))  # padding image from 182x218x182 to 182x224x182
        # padding needs to be done before normalization
        mask = img != 0
        img = (img - img[mask].mean()) / img[mask].std()
        if transforms:
            img = transforms(img)
        img = img.type(torch.float)
        return img

    def load_mask(self, row, modality_col):
        img_path = row[modality_col]
        img = nib.load(img_path)
        img = img.get_fdata()
        img = torch.from_numpy(img)
        img = torch.nn.functional.pad(img, (0,0,3,3,0,0))  # padding image from 182x218x182 to 182x224x182
        mask = img != 0
        mask = torch.tensor(mask)
        return mask

# A wrapper to extract latent representations from the encoder
class InferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_T1, x_T2):
        """
        Extract latent representations from encoder_q.
        Returns the latent before pooling/projection, which serves as compute_pool.
        """
        # Use encoder_q to encode T1 images
        y_T2 = x_T2
        latent, batch_mask, non_zero_patch = self.model.encoder_q(x_T2, y_T2)
        
        # Return latent representation (B, num_patches, embed_dim)
        # This is the equivalent of compute_pool for this architecture
        return latent, batch_mask, non_zero_patch

def extract_compute_pool(checkpoint_path, data_file_T1, data_file_T2, output_path, batch_size=32, num_workers=4):
    """
    Loads a contrastive learning model from a checkpoint, processes data, extracts the latent
    representations from encoder_q, and saves them as PCA-reduced features.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        data_file_T1 (str): Path to the T1 csv file for the dataset.
        data_file_T2 (str): Path to the T2 csv file for the dataset.
        output_path (str): Path to save the output CSV files.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for the DataLoader.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the contrastive learning model
    model = MoCoV2Dual(
        lr=0.0001,
        img_size=182,
        patch_size=14,
        num_frames=224,
        tubelet_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        proj_dim=256,
        K=65536,
        m=0.999,
        T=0.07,
        symmetric=True,
        non_zero_patch_opt=False  # Set to False for consistent output shapes
    )
    print("Contrastive learning model instantiated.")

    # Load the checkpoint to CPU first, before wrapping the model
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # The model might be saved with a 'module.' prefix if trained with DDP
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    print("Checkpoint loaded successfully into the contrastive learning model.")

    # Wrap the model for inference to extract latent representations
    inference_model = InferenceWrapper(model)

    # Note: Using single GPU for inference to avoid shape mismatch errors
    print(f"Using single GPU for inference (device: {device})")

    inference_model = inference_model.to(device)
    inference_model.eval()

    # Create dataset and dataloader
    # Uses separate T1 and T2 CSV files with "mri_names" column
    dataset = aedataset_T1T2_separate(
        datafile_T1=data_file_T1,
        datafile_T2=data_file_T2,
        transforms=None
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print("DataLoader created.")

    # Store compute pools (latent representations)
    all_compute_pools = []
    num_patches = None
    embed_dim = None

    with torch.no_grad():
        for i, (x_T1, x_T2, mask_T1, mask_T2) in enumerate(tqdm(dataloader, desc="Extracting Features")):
            x_T1 = x_T1.to(device)
            x_T2 = x_T2.to(device)  # Not used for extraction, but needed for dataset
            
            # Extract latent representations from encoder_q
            latent, batch_mask, non_zero_patch = inference_model(x_T1, x_T2)
            
            # latent has shape (B, num_patches, embed_dim)
            # With non_zero_patch_opt=False, num_patches = grid_size * grid_depth * grid_size
            # grid_size = 182 // 14 = 13, grid_depth = 224 // 16 = 14
            # num_patches = 13 * 14 * 13 = 2366
            # embed_dim = 384
            B, num_patches, embed_dim = latent.shape
            if i == 0:
                print(f"Batch {i}: latent shape = {latent.shape}")
            
            # Flatten the patches dimension to get (B, num_patches * embed_dim)
            # This gives us a fixed-size feature vector per sample
            # Use float32 to reduce memory usage (half the size of float64)
            compute_pool_flat = latent.view(B, -1).cpu().numpy().astype(np.float32)
            all_compute_pools.append(compute_pool_flat)

    # Concatenate all batches (keep as float32 for memory efficiency)
    print("Concatenating batches...")
    all_compute_pools = np.concatenate(all_compute_pools, axis=0).astype(np.float32)
    print(f"Extracted compute pools with shape: {all_compute_pools.shape} (dtype: {all_compute_pools.dtype})")

    # Get the EIDs from the processed dataset
    dataset_eids = dataset.df_T1['EID'].values
    
    # Save features before PCA
    print("Saving features before PCA...")
    os.makedirs(output_path, exist_ok=True)
    
    # Save features as numpy array
    features_path = os.path.join(output_path, 'features_before_pca.npy')
    np.save(features_path, all_compute_pools)
    print(f"✅ Saved features to: {features_path}")
    print(f"   Features shape: {all_compute_pools.shape}")
    print(f"   Features dtype: {all_compute_pools.dtype}")
    
    # Save EIDs
    eids_path = os.path.join(output_path, 'eids.npy')
    np.save(eids_path, dataset_eids)
    print(f"✅ Saved EIDs to: {eids_path}")
    print(f"   EIDs count: {len(dataset_eids)}")
    
    # Save metadata
    metadata = {
        'num_patches': num_patches,
        'embed_dim': embed_dim,
        'feature_dim': num_patches * embed_dim,
        'n_samples': all_compute_pools.shape[0],
        'n_features': all_compute_pools.shape[1]
    }
    import json
    metadata_path = os.path.join(output_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to: {metadata_path}")
    
    print("\n✅ Feature extraction completed. Features saved before PCA.")
    print(f"   Run the PCA script separately to apply PCA transformation.")
    print(f"   Features saved in: {output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract compute_pool from a trained contrastive learning UDIP-ViT model.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--datafile_T1', type=str, default="/data4012/kpatel38/brain_imaging/T1_128_gwas.csv", help='Path to the T1 data CSV file. Must contain columns: EID and mri_names.')
    parser.add_argument('--datafile_T2', type=str, default="/data4012/kpatel38/brain_imaging/T2_128_gwas.csv", help='Path to the T2 data CSV file. Must contain columns: EID and mri_names.')
    parser.add_argument('--output_dir', type=str, default="compute_pool_features_contrast_learning", help='Path to save the output CSV files.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the DataLoader.')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    extract_compute_pool(
        checkpoint_path=args.checkpoint,
        data_file_T1=args.datafile_T1,
        data_file_T2=args.datafile_T2,
        output_path=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

