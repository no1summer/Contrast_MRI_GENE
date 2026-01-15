# MoCoV2 Contrastive Learning for Neuroimaging GWAS Pipeline

This repository contains the complete pipeline for training a contrastive learning model on T1 and T2 MRI images, extracting features for UK Biobank discovery cohort, performing PCA dimension reduction, and conducting GWAS analysis using FastGWA.

## Pipeline Overview

The pipeline consists of four main steps as illustrated in Figure 1 (`fig1.png`):

1. **Model Training**: Train a contrastive learning Vision Transformer (ViT) model on T1 and T2 MRI images
2. **Feature Extraction**: Extract features from the trained model for UK Biobank discovery cohort
3. **PCA Dimension Reduction**: Apply PCA to reduce feature dimensions
4. **GWAS Analysis**: Perform genome-wide association studies using FastGWA

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Model Training

Train the contrastive learning model using T1 and T2 MRI images:

```bash
python engine_128_T1_T2_mocov2_contrast_learning.py [arguments]
```

This script trains a MoCoV2-based dual encoder model that learns representations from paired T1 and T2 MRI scans using contrastive learning.

**Key Features:**
- Vision Transformer (ViT) architecture
- Distributed training support
- Contrastive learning with momentum updates
- Saves model checkpoints during training

### Step 2: Feature Extraction for UK Biobank Discovery Cohort

Extract features from the trained model for the UK Biobank discovery cohort:

```bash
python extract_compute_pool_contrast_learning_saveT1.py \
    --checkpoint /path/to/trained/model/checkpoint.pth \
    --datafile /path/to/ukbiobank/data.csv \
    --output_dir /path/to/output/ \
    [--batch_size 32] \
    [--num_workers 4]
```

**Parameters:**
- `--checkpoint`: Path to the trained model checkpoint
- `--datafile`: Path to CSV file containing UK Biobank discovery cohort data
- `--output_dir`: Directory to save extracted features
- `--batch_size`: Batch size for feature extraction (default: 32)
- `--num_workers`: Number of worker processes (default: 4)

This script extracts compute pool features from the trained model and saves them for downstream analysis.
The pretained checkpoint could be found in the following link.
https://drive.google.com/file/d/1VvmKhfLDk-JVpbYgHMu_djLvdA7SItMr/view?usp=sharing

### Step 3: PCA Dimension Reduction

Apply PCA to reduce the dimensionality of extracted features:

```bash
python apply_pca_to_features_T1.py \
    --checkpoint_path /path/to/inference_checkpoint.pt \
    --output_dir /path/to/output/ \
    [--n_components 128] \
    [--chunk_size 1000]
```

**Parameters:**
- `--checkpoint_path`: Path to the inference checkpoint containing extracted features
- `--output_dir`: Directory to save PCA-reduced features and models
- `--n_components`: Number of PCA components (default: 128)
- `--chunk_size`: Chunk size for processing large arrays (default: 1000)

This script performs incremental PCA on the extracted features to reduce dimensionality while preserving the most important variance.

### Step 4: GWAS Analysis with FastGWA

Perform genome-wide association studies on the PCA-reduced features using FastGWA:

```bash
python fastGWAS_csv.py [arguments]
```

This script runs FastGWA (fast genome-wide association analysis) on each PCA component as a phenotype, identifying genetic associations with the learned neuroimaging features.

**Output:**
- FastGWA results for each PCA component
- Association statistics and p-values
- Results can be used for downstream genetic analysis

## Pipeline Workflow

Refer to `fig1.png` for a visual representation of the complete pipeline workflow, showing the data flow from raw MRI images through model training, feature extraction, dimension reduction, and GWAS analysis.

## File Structure

- `engine_128_T1_T2_mocov2_contrast_learning.py`: Main training script for contrastive learning model
- `extract_compute_pool_contrast_learning_saveT1.py`: Feature extraction script for T1 images
- `apply_pca_to_features_T1.py`: PCA dimension reduction script
- `fastGWAS_csv.py`: FastGWA analysis script
- `fig1.png`: Pipeline workflow diagram

## Notes

- The pipeline is designed for distributed training and can utilize multiple GPUs
- Ensure sufficient disk space for storing extracted features and intermediate results
- The PCA step uses incremental PCA to handle large datasets efficiently
- FastGWA requires appropriate genetic data files (BGEN format) and sample files

## Citation

If you use this pipeline, please cite the relevant papers for:
- MoCoV2 contrastive learning framework
- Vision Transformer architecture
- FastGWA method

