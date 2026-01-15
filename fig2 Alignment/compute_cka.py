import argparse
import glob
import os
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def load_embedding_matrix(directory: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load per-feature CSVs into a subject-aligned matrix."""
    #pattern = os.path.join(directory, "Feature_*.csv")
    pattern = os.path.join(directory, "Feature_*")
    feature_files = sorted(
        glob.glob(pattern),
        key=lambda path: int(os.path.splitext(os.path.basename(path))[0].split("_")[1]),
    )
    if not feature_files:
        raise FileNotFoundError(f"No feature files found under {directory}")

    iids = None
    feature_columns = []

    for feature_path in feature_files:
        df = pd.read_csv(feature_path, sep=r"\s+")
        value_col = df.columns[-1]
        df_subset = df[["IID", value_col]]
        current_ids = df_subset["IID"].to_numpy()
        if iids is None:
            iids = current_ids
        elif not np.array_equal(iids, current_ids):
            raise ValueError(f"IID order mismatch detected in {feature_path}")
        # Convert to numpy array, handling NA values
        feature_values = df_subset[value_col].to_numpy(dtype=np.float32)
        # Check for NA values and warn if found
        if pd.isna(feature_values).any():
            na_count = pd.isna(feature_values).sum()
            print(f"Warning: Found {na_count} NA values in {os.path.basename(feature_path)}")
        feature_columns.append(feature_values)

    matrix = np.column_stack(feature_columns).astype(np.float32)
    return iids, matrix


def align_embeddings(
    ids_a: np.ndarray, emb_a: np.ndarray, ids_b: np.ndarray, emb_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align two embedding matrices by intersecting subject IDs and remove rows with NA values."""
    common_ids, idx_a, idx_b = np.intersect1d(ids_a, ids_b, return_indices=True)
    if common_ids.size == 0:
        raise ValueError("No overlapping IID entries between the two embeddings")
    
    emb_a_aligned = emb_a[idx_a]
    emb_b_aligned = emb_b[idx_b]
    
    # Remove rows with any NA values in either embedding
    # Check for NaN (which is what numpy uses for missing values)
    na_mask_a = np.isnan(emb_a_aligned).any(axis=1)
    na_mask_b = np.isnan(emb_b_aligned).any(axis=1)
    na_mask = na_mask_a | na_mask_b
    
    if na_mask.any():
        n_removed = na_mask.sum()
        print(f"Removing {n_removed} samples with NA values (out of {common_ids.size} common samples)")
        emb_a_aligned = emb_a_aligned[~na_mask]
        emb_b_aligned = emb_b_aligned[~na_mask]
        common_ids = common_ids[~na_mask]
    
    if common_ids.size == 0:
        raise ValueError("No samples remaining after removing NA values")
    
    return emb_a_aligned, emb_b_aligned, common_ids


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the linear CKA similarity between two centered embeddings."""
    # Check for NA values (should not happen if align_embeddings was used, but safety check)
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("NA values detected in embeddings passed to linear_cka")
    
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)

    cross_cov = x_centered.T @ y_centered
    numerator = np.linalg.norm(cross_cov, "fro") ** 2
    denom = np.linalg.norm(x_centered.T @ x_centered, "fro") * np.linalg.norm(
        y_centered.T @ y_centered, "fro"
    )
    if denom == 0:
        raise ZeroDivisionError("Encountered zero variance in embeddings while computing CKA")
    return numerator / denom


def sample_positive_pairs(
    emb_a: np.ndarray, emb_b: np.ndarray, num_samples: int, seed: int
) -> np.ndarray:
    """Compute positive-pair CKA by bootstrap sampling subjects (aligned embeddings)."""
    rng = np.random.default_rng(seed)
    n_samples = emb_a.shape[0]
    if n_samples != emb_b.shape[0]:
        raise ValueError("Aligned embedding matrices must share the same number of rows")

    scores = []
    for _ in range(num_samples):
        # Bootstrap sample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        scores.append(linear_cka(emb_a[indices], emb_b[indices]))
    return np.asarray(scores, dtype=np.float32)


def sample_negative_pairs(
    emb_a: np.ndarray, emb_b: np.ndarray, num_shuffles: int, seed: int
) -> np.ndarray:
    """Shuffle the second embedding matrix to estimate negative-pair CKA distribution."""
    rng = np.random.default_rng(seed)
    n_samples = emb_a.shape[0]
    if n_samples != emb_b.shape[0]:
        raise ValueError("Aligned embedding matrices must share the same number of rows")

    scores = []
    base_indices = np.arange(n_samples)
    for _ in range(num_shuffles):
        permuted = rng.permutation(n_samples)
        # Ensure we do not accidentally keep subjects aligned.
        while np.any(permuted == base_indices):
            permuted = rng.permutation(n_samples)
        scores.append(linear_cka(emb_a, emb_b[permuted]))
    return np.asarray(scores, dtype=np.float32)


def plot_positive_distribution(values: np.ndarray, output_path: str) -> None:
    """Plot and save a histogram of the positive-pair CKA distribution."""
    plt.figure(figsize=(14, 6))
    plt.hist(values, bins=30, color="#c44e52", alpha=0.85, edgecolor="black", linewidth=0.5)
    plt.xlabel("Linear CKA (positive pairs)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Positive-pair CKA Distribution", fontsize=14)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_distributions(
    pos_values: np.ndarray, neg_values: np.ndarray, output_path: str,
    model_name: str = None, density: bool = False
) -> None:
    """Plot and save histograms of both positive and negative CKA distributions with broken x-axis."""
    # Create figure with two subplots that share y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    
    # Determine x-axis ranges
    neg_min, neg_max = neg_values.min(), neg_values.max()
    pos_min, pos_max = pos_values.min(), pos_values.max()
    
    # Add some padding to ranges
    neg_range = neg_max - neg_min
    pos_range = pos_max - pos_min
    neg_min -= neg_range * 0.05
    neg_max += neg_range * 0.05
    pos_min -= pos_range * 0.05
    pos_max += pos_range * 0.05
    
    # Plot negative pairs on the left
    ax1.hist(
        neg_values, bins=20, color="#4c72b0", alpha=0.7, edgecolor="black", linewidth=0.5, density=density
    )
    ax1.set_xlim(neg_min, neg_max)
    ax1.set_xlabel("Linear CKA", fontsize=20)
    ax1.set_ylabel("Density" if density else "Count", fontsize=20)
    ax1.set_title("Negative Pairs - "+model_name, fontsize=20, fontweight="bold")
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=2))
    ax1.tick_params(labelsize=20)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    
    # Plot positive pairs on the right
    ax2.hist(
        pos_values, bins=20, color="#c44e52", alpha=0.7, edgecolor="black", linewidth=0.5, density=density
    )
    ax2.set_xlim(pos_min, pos_max)
    ax2.set_xlabel("Linear CKA", fontsize=20)
    ax2.tick_params(labelsize=20)
    ax2.set_ylabel("")  # Remove duplicate y-label
    ax2.set_title("Positive Pairs - "+model_name, fontsize=20, fontweight="bold")
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    
    # Add broken axis indicators (// marks)
    # Draw diagonal lines on the right side of ax1 and left side of ax2
    d = 0.015  # Size of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.5)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multiple_models(
    model_data: list, output_path: str, density: bool = False
) -> None:
    """
    Plot multiple models (e.g., ViT and MoCoV2) with separate subplots for each positive pair.
    
    Args:
        model_data: List of tuples (model_name, pos_values, neg_values, neg_color, pos_color)
        output_path: Path to save the plot
        density: Whether to use density normalization (False shows actual counts)
    """
    # Create figure with 3 subplots: negative pairs (shared), then separate positive pairs for each model
    num_models = len(model_data)
    # Make negative pair slightly wider and positive pairs closer together
    width_ratios = [4] + [3] * num_models  # Negative pair is 4/3 = 1.33x wider (2/3 of previous 2x)
    fig, axes = plt.subplots(1, 1 + num_models, figsize=(6 + 9*num_models, 6), 
                             sharey=True, gridspec_kw={'wspace': 0.15, 'width_ratios': width_ratios})
    
    ax_neg = axes[0]
    ax_pos_list = axes[1:]
    
    # Determine overall x-axis ranges for negative pairs
    all_neg_values = np.concatenate([neg for _, _, neg, _, _ in model_data])
    neg_min, neg_max = all_neg_values.min(), all_neg_values.max()
    neg_range = neg_max - neg_min
    neg_min -= neg_range * 0.05
    neg_max += neg_range * 0.05
    
    # Plot negative pairs on the left (all models overlaid)
    for model_name, pos_values, neg_values, neg_color, pos_color in model_data:
        ax_neg.hist(
            neg_values, bins=30, color=neg_color, alpha=0.6, edgecolor="black", 
            linewidth=0.5, density=density, label=model_name
        )
    
    # Set labels and titles for negative pairs
    ax_neg.set_xlim(neg_min, neg_max)
    ax_neg.set_xlabel("Linear CKA (Negative pairs)", fontsize=20)
    ax_neg.set_ylabel("Density" if density else "Count", fontsize=20)
    ax_neg.set_title("Negative Pairs", fontsize=20, fontweight="bold")
    ax_neg.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_neg.tick_params(labelsize=20)
    ax_neg.grid(axis="y", linestyle="--", alpha=0.4)
    ax_neg.legend(fontsize=20, loc='upper center')
    
    # Plot each model's positive pairs in separate subplots
    for idx, (model_name, pos_values, neg_values, neg_color, pos_color) in enumerate(model_data):
        ax_pos = ax_pos_list[idx]
        
        # Determine x-axis range for this model's positive pairs
        pos_min, pos_max = pos_values.min(), pos_values.max()
        pos_range = pos_max - pos_min
        pos_min -= pos_range * 0.1
        pos_max += pos_range * 0.1
        
        # Plot positive pairs with more bins to spread out
        ax_pos.hist(
            pos_values, bins=50, color=pos_color, alpha=0.7, edgecolor="black", 
            linewidth=0.5, density=density, label=model_name
        )
        
        # Set labels and titles for this positive pair subplot
        ax_pos.set_xlim(pos_min, pos_max)
        ax_pos.set_xlabel("Linear CKA (Positive pairs)", fontsize=20)
        ax_pos.set_ylabel("")  # Remove duplicate y-label
        ax_pos.set_title("Positive Pairs", fontsize=20, fontweight="bold")
        ax_pos.tick_params(labelsize=20)
        # Limit the number of x-axis ticks to reduce clutter
        ax_pos.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_pos.grid(axis="y", linestyle="--", alpha=0.4)
        ax_pos.legend(fontsize=20)
    
    # Add broken axis indicators (// marks) between negative and first positive subplot
    d = 0.015
    kwargs = dict(transform=ax_neg.transAxes, color='k', clip_on=False, linewidth=1.5)
    ax_neg.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax_neg.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs.update(transform=ax_pos_list[0].transAxes)
    ax_pos_list[0].plot((-d, +d), (-d, +d), **kwargs)
    ax_pos_list[0].plot((-d, +d), (1-d, 1+d), **kwargs)
    
    # Add broken axis indicators between positive pair subplots if more than one model
    for i in range(1, len(ax_pos_list)):
        kwargs.update(transform=ax_pos_list[i-1].transAxes)
        ax_pos_list[i-1].plot((1-d, 1+d), (-d, +d), **kwargs)
        ax_pos_list[i-1].plot((1-d, 1+d), (1-d, 1+d), **kwargs)
        
        kwargs.update(transform=ax_pos_list[i].transAxes)
        ax_pos_list[i].plot((-d, +d), (-d, +d), **kwargs)
        ax_pos_list[i].plot((-d, +d), (1-d, 1+d), **kwargs)
    
    # Add overall title
    fig.suptitle("CKA Distribution: Positive vs Negative Pairs (Multiple Models)", 
                 fontsize=20, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute positive and negative-pair CKA distributions."
    )
    parser.add_argument(
        "--first_dir",
        default="/data484_4/txia2/gwas_practice/individual_phenos/vit_t1_fixed",
        help="Directory containing Feature_*.csv embeddings for the first model.",
    )
    parser.add_argument(
        "--second_dir",
        default="/data484_4/txia2/gwas_practice/individual_phenos/vit_t2_fixed",
        help="Directory containing Feature_*.csv embeddings for the second model.",
    )
    parser.add_argument(
        "--mocov2_first_dir",
        default=None,
        #default='/data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2/first_5std',
        help="Directory containing Feature_*.csv embeddings for MoCoV2 T1 (optional, for multi-model plotting).",
    )
    parser.add_argument(
        "--mocov2_second_dir",
        default=None,
        #default='/data484_4/txia2/gwas_practice/individual_phenos/contrast_learning_mocov2_T2_5std',
        help="Directory containing Feature_*.csv embeddings for MoCoV2 T2 (optional, for multi-model plotting).",
    )
    parser.add_argument(
        "--model_name",
        default='ViT',
        help="Name of the model for single-model plots (e.g., 'ViT' or 'MoCoV2').",
    )
    parser.add_argument(
        "--num_shuffles",
        type=int,
        default=200,
        help="Number of random negative pairs to sample (by shuffling subjects).",
    )
    parser.add_argument(
        "--num_bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap samples for positive pairs.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling.")
    parser.add_argument(
        "--output_plot",
        default="/data484_4/txia2/mocov2/CKA/cka_distributions_vit.png",
        help="File path to save the histogram plot.",
    )
    parser.add_argument(
        "--positive_only",
        action="store_true",
        help="Plot only positive pairs distribution (without negative pairs).",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Use density normalization instead of counts (default: False, shows actual counts).",
    )

    args = parser.parse_args()

    # Check if we should plot multiple models
    plot_multiple = args.mocov2_first_dir is not None and args.mocov2_second_dir is not None

    # Process first model (ViT by default)
    ids_first, emb_first = load_embedding_matrix(args.first_dir)
    ids_second, emb_second = load_embedding_matrix(args.second_dir)
    emb_first, emb_second, common_ids = align_embeddings(
        ids_first, emb_first, ids_second, emb_second
    )

    n_features_first = emb_first.shape[1]
    n_features_second = emb_second.shape[1]
    n_subjects = common_ids.size

    model_name_1 = args.model_name if args.model_name else "ViT"
    if not plot_multiple:
        print(f"Aligned {n_subjects} shared subjects across the two embeddings.")
        print(f"First embedding: {n_features_first} features")
        print(f"Second embedding: {n_features_second} features")
        print(f"\nComputing CKA between full embedding matrices (all features combined):")
        print(f"  - Positive pairs: {args.num_bootstrap} bootstrap samples (aligned subjects)")
        print(f"  - Negative pairs: {args.num_shuffles} random shuffles (misaligned subjects)")

    # Compute positive pairs (aligned embeddings with bootstrap)
    pos_scores_1 = sample_positive_pairs(emb_first, emb_second, args.num_bootstrap, args.seed)

    if not plot_multiple:
        print(f"\n{model_name_1} - Positive-pair CKA statistics:")
        print(f"  mean = {pos_scores_1.mean():.6f}")
        print(f"  std  = {pos_scores_1.std(ddof=1):.6f}")
        print(f"  min  = {pos_scores_1.min():.6f}")
        print(f"  max  = {pos_scores_1.max():.6f}")

    # Always compute negative pairs (shuffled embeddings)
    neg_scores_1 = sample_negative_pairs(emb_first, emb_second, args.num_shuffles, args.seed)

    if not plot_multiple:
        print(f"\n{model_name_1} - Negative-pair CKA statistics:")
        print(f"  mean = {neg_scores_1.mean():.6f}")
        print(f"  std  = {neg_scores_1.std(ddof=1):.6f}")
        print(f"  min  = {neg_scores_1.min():.6f}")
        print(f"  max  = {neg_scores_1.max():.6f}")
        print(f"\nNote: Both positive and negative pairs have {args.num_bootstrap} samples each.")
        print(f"      The histogram shows {'density' if args.density else 'count'} on the y-axis.")

    # Process second model (MoCoV2) if provided
    if plot_multiple:
        print(f"\nProcessing {model_name_1}...")
        print(f"  Aligned {n_subjects} shared subjects")
        print(f"  Positive pairs: {len(pos_scores_1)} samples")
        print(f"  Negative pairs: {len(neg_scores_1)} samples")
        
        ids_mocov2_first, emb_mocov2_first = load_embedding_matrix(args.mocov2_first_dir)
        ids_mocov2_second, emb_mocov2_second = load_embedding_matrix(args.mocov2_second_dir)
        emb_mocov2_first, emb_mocov2_second, common_ids_mocov2 = align_embeddings(
            ids_mocov2_first, emb_mocov2_first, ids_mocov2_second, emb_mocov2_second
        )
        
        print(f"\nProcessing MoCoV2...")
        print(f"  Aligned {common_ids_mocov2.size} shared subjects")
        
        pos_scores_2 = sample_positive_pairs(emb_mocov2_first, emb_mocov2_second, args.num_bootstrap, args.seed)
        neg_scores_2 = sample_negative_pairs(emb_mocov2_first, emb_mocov2_second, args.num_shuffles, args.seed)
        
        print(f"  Positive pairs: {len(pos_scores_2)} samples")
        print(f"  Negative pairs: {len(neg_scores_2)} samples")
        print(f"\nNote: Both models have {args.num_bootstrap} samples each for positive and negative pairs.")
        print(f"      The histogram shows {'density' if args.density else 'count'} on the y-axis.")
        
        # Plot both models together
        # Format: (model_name, pos_values, neg_values, neg_color, pos_color)
        model_data = [
            (model_name_1, pos_scores_1, neg_scores_1, "#1f77b4", "#c44e52"),  # Dark blue (neg), Dark red (pos) for ViT
            ("MoCoV2", pos_scores_2, neg_scores_2, "#87ceeb", "#ff6b6b"),  # Light blue (neg), Light red (pos) for MoCoV2
        ]
        plot_multiple_models(model_data, args.output_plot, density=args.density)
        print(f"\nSaved multi-model histogram to {args.output_plot}")
    elif args.positive_only:
        plot_positive_distribution(pos_scores_1, args.output_plot)
        print(f"\nSaved positive-pair histogram to {args.output_plot}")
    else:
        plot_distributions(pos_scores_1, neg_scores_1, args.output_plot, 
                          model_name=model_name_1, density=args.density)
        print(f"\nSaved histogram to {args.output_plot}")


if __name__ == "__main__":
    main()

