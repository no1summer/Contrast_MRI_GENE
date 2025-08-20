# imports

# PyTorch
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Custom imports
# Note: Remove MONAI dependency - we'll handle transforms manually

class PatchEmbedding3D(nn.Module):
    """3D patch embedding for Vision Transformer"""
    def __init__(self, img_size=(96, 112, 96), patch_size=(16, 16, 16), in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, n_patches_d, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    """Cross attention between T1 and T2 modalities"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        B, N_q, C = query.shape
        B, N_kv, C = key_value.shape
        
        q = self.q(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """Standard Transformer block"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformerEncoder(nn.Module):
    """Vision Transformer Encoder"""
    def __init__(self, img_size=(96, 112, 96), patch_size=(16, 16, 16), in_chans=1, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding3D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]  # Return CLS token

class VisionTransformerDecoder(nn.Module):
    """Vision Transformer Decoder for reconstruction"""
    def __init__(self, embed_dim=768, img_size=(96, 112, 96), patch_size=(16, 16, 16), 
                 out_chans=1, depth=4, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_chans = out_chans
        
        # Calculate number of patches
        self.n_patches_d = img_size[0] // patch_size[0]
        self.n_patches_h = img_size[1] // patch_size[1]
        self.n_patches_w = img_size[2] // patch_size[2]
        self.n_patches = self.n_patches_d * self.n_patches_h * self.n_patches_w
        
        # Project from feature dimension to patch dimension
        patch_dim = patch_size[0] * patch_size[1] * patch_size[2] * out_chans
        
        # Learnable queries for decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_dim)
        
        # Initialize weights
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Project encoder output
        x = self.decoder_embed(x)  # (B, embed_dim)
        x = x.unsqueeze(1)  # (B, 1, embed_dim)
        
        # Create mask tokens for all patches
        mask_tokens = self.mask_token.repeat(B, self.n_patches, 1)
        
        # Add positional embeddings
        x_with_pos = mask_tokens + self.decoder_pos_embed
        
        # Combine with encoded feature (broadcast to all positions)
        x_expanded = x.repeat(1, self.n_patches, 1)
        decoder_input = x_with_pos + x_expanded
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            decoder_input = blk(decoder_input)
            
        decoder_input = self.decoder_norm(decoder_input)
        
        # Predict patches
        x = self.decoder_pred(decoder_input)  # (B, n_patches, patch_dim)
        
        # Reshape to image
        x = self.unpatchify(x)
        return x
    
    def unpatchify(self, x):
        """Convert patches back to image"""
        B = x.shape[0]
        
        # Reshape from (B, n_patches, patch_dim) to (B, n_patches_d, n_patches_h, n_patches_w, patch_d, patch_h, patch_w, out_chans)
        x = x.reshape(B, self.n_patches_d, self.n_patches_h, self.n_patches_w, 
                     self.patch_size[0], self.patch_size[1], self.patch_size[2], self.out_chans)
        
        # Permute and reshape to final image shape
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # (B, out_chans, n_patches_d, patch_d, n_patches_h, patch_h, n_patches_w, patch_w)
        x = x.reshape(B, self.out_chans, self.img_size[0], self.img_size[1], self.img_size[2])
        
        return x

class engine_AE(nn.Module):
    def __init__(self, lr, concat_modalities=True, use_modality='T1', 
                 img_size=(96, 112, 96), patch_size=(16, 16, 16), embed_dim=768,
                 encoder_depth=6, decoder_depth=4, num_heads=12):
        super().__init__()
        self.lr = lr
        self.concat_modalities = concat_modalities
        self.use_modality = use_modality
        
        self.hidden_dim = embed_dim  # Use embed_dim directly (512)
        
        # Vision Transformer Encoders for T1 and T2
        self.encoder_T1 = VisionTransformerEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=1,
            embed_dim=embed_dim, depth=encoder_depth, num_heads=num_heads
        )
        
        self.encoder_T2 = VisionTransformerEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=1,
            embed_dim=embed_dim, depth=encoder_depth, num_heads=num_heads
        )
        
        # Cross attention for modality interaction (at embed_dim level)
        self.cross_attn_T1_to_T2 = CrossAttention(embed_dim, num_heads)
        self.cross_attn_T2_to_T1 = CrossAttention(embed_dim, num_heads)
        
        # Projection layer (from embed_dim back to embed_dim for decoders)
        self.projection_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Vision Transformer Decoders
        self.decoder_T1 = VisionTransformerDecoder(
            embed_dim=embed_dim, img_size=img_size, patch_size=patch_size,
            out_chans=1, depth=decoder_depth, num_heads=num_heads
        )
        
        self.decoder_T2 = VisionTransformerDecoder(
            embed_dim=embed_dim, img_size=img_size, patch_size=patch_size,
            out_chans=1, depth=decoder_depth, num_heads=num_heads
        )
        
        # Loss functions
        self.train_loss_function1 = torch.nn.MSELoss(reduction="none")
        self.valid_loss_function = torch.nn.MSELoss(reduction="none")

    def forward(self, x_T1, x_T2):
        if not self.concat_modalities:
            # Single modality mode
            if self.use_modality == 'T1':
                encoded_T1 = self.encoder_T1(x_T1)  # (B, embed_dim)
                recon = self.decoder_T1(encoded_T1)  # (B, 1, D, H, W)
                return recon, encoded_T1  # Return embed_dim features directly
            else:
                encoded_T2 = self.encoder_T2(x_T2)  # (B, embed_dim)
                recon = self.decoder_T2(encoded_T2)  # (B, 1, D, H, W)
                return recon, encoded_T2  # Return embed_dim features directly
        else:
            # Multi-modal mode with cross-attention (NO concatenation, NO feature projection)
            # Encode both modalities
            encoded_T1 = self.encoder_T1(x_T1)  # (B, embed_dim)
            encoded_T2 = self.encoder_T2(x_T2)  # (B, embed_dim)
            
            # Add sequence dimension for cross-attention at embed_dim level
            encoded_T1_seq = encoded_T1.unsqueeze(1)  # (B, 1, embed_dim)
            encoded_T2_seq = encoded_T2.unsqueeze(1)  # (B, 1, embed_dim)
            
            # Cross-attention between modalities at embed_dim level
            T1_attended = self.cross_attn_T1_to_T2(encoded_T1_seq, encoded_T2_seq)  # (B, 1, embed_dim)
            T2_attended = self.cross_attn_T2_to_T1(encoded_T2_seq, encoded_T1_seq)  # (B, 1, embed_dim)
            
            # Remove sequence dimension
            T1_attended = T1_attended.squeeze(1)  # (B, embed_dim)
            T2_attended = T2_attended.squeeze(1)  # (B, embed_dim)
            
            # Optional projection (identity for now, can be removed entirely)
            T1_embedding = self.projection_layer(T1_attended)  # (B, embed_dim)
            T2_embedding = self.projection_layer(T2_attended)  # (B, embed_dim)
            
            # Decode each modality from its own cross-attended embedding
            recon_T1 = self.decoder_T1(T1_embedding)  # (B, 1, D, H, W)
            recon_T2 = self.decoder_T2(T2_embedding)  # (B, 1, D, H, W)
            
            # Return original encoder features (before cross-attention) for endophenotype
            return (recon_T1, recon_T2), torch.cat([encoded_T1, encoded_T2], dim=1)

# Validation function for the new architecture
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x_T1, x_T2, mask = batch
            x_T1 = x_T1.to(device)
            x_T2 = x_T2.to(device)
            mask = mask.to(device)
            
            (recon_T1, recon_T2), _ = model(x_T1, x_T2)
            
            # Handle both DDP and non-DDP cases for accessing the loss function
            loss_fn = model.module.valid_loss_function if hasattr(model, 'module') else model.valid_loss_function
            
            # Calculate loss for both T1 and T2
            loss_T1 = loss_fn(x_T1, recon_T1)
            loss_T1 = loss_T1.squeeze(1) * mask
            loss_T1 = loss_T1.sum() / mask.sum()
            
            loss_T2 = loss_fn(x_T2, recon_T2)
            loss_T2 = loss_T2.squeeze(1) * mask
            loss_T2 = loss_T2.sum() / mask.sum()
            
            # Combined loss
            loss = (loss_T1 + loss_T2) / 2.0
            total_loss += loss.item() * x_T1.size(0)
            
            # PSNR/SSIM (per sample for both T1 and T2)
            for i in range(x_T1.shape[0]):
                # Process T1
                gt_T1 = x_T1[i].cpu().numpy()
                pred_T1 = recon_T1[i].detach().cpu().numpy()
                msk = mask[i].cpu().numpy().astype(bool)
                
                # Reshape mask to match the input dimensions
                msk = msk[None, ...]
                
                # Apply mask while preserving dimensions
                gt_T1_masked = gt_T1[msk].reshape(-1)
                pred_T1_masked = pred_T1[msk].reshape(-1)
                
                # Process T2
                gt_T2 = x_T2[i].cpu().numpy()
                pred_T2 = recon_T2[i].detach().cpu().numpy()
                gt_T2_masked = gt_T2[msk].reshape(-1)
                pred_T2_masked = pred_T2[msk].reshape(-1)
                
                if len(gt_T1_masked) > 0:
                    # PSNR for T1
                    data_range_T1 = gt_T1_masked.max() - gt_T1_masked.min()
                    if data_range_T1 > 0:
                        psnr_T1 = compare_psnr(gt_T1_masked, pred_T1_masked, data_range=data_range_T1)
                    else:
                        psnr_T1 = 0.0
                    
                    # PSNR for T2
                    data_range_T2 = gt_T2_masked.max() - gt_T2_masked.min()
                    if data_range_T2 > 0:
                        psnr_T2 = compare_psnr(gt_T2_masked, pred_T2_masked, data_range=data_range_T2)
                    else:
                        psnr_T2 = 0.0
                    
                    # SSIM for T1
                    try:
                        ssim_T1 = compare_ssim(gt_T1_masked, pred_T1_masked, data_range=data_range_T1)
                    except Exception:
                        ssim_T1 = 0.0
                    
                    # SSIM for T2
                    try:
                        ssim_T2 = compare_ssim(gt_T2_masked, pred_T2_masked, data_range=data_range_T2)
                    except Exception:
                        ssim_T2 = 0.0
                else:
                    psnr_T1 = psnr_T2 = 0.0
                    ssim_T1 = ssim_T2 = 0.0
                
                # Average metrics across modalities
                total_psnr += (psnr_T1 + psnr_T2) / 2
                total_ssim += (ssim_T1 + ssim_T2) / 2
                n_samples += 1
                
    avg_loss = total_loss / len(dataloader.dataset)
    avg_psnr = total_psnr / n_samples if n_samples > 0 else 0.0
    avg_ssim = total_ssim / n_samples if n_samples > 0 else 0.0
    return avg_loss, avg_psnr, avg_ssim

# Custom dataset for T1 and T2 modalities
import pandas as pd
import nibabel as nib
import torch
import numpy as np

def load_sample(row, modality_col, transforms=None):
    img_path = row[modality_col]
    img = nib.load(img_path)
    img = img.get_fdata()
    mask = img != 0
    img = (img - img[mask].mean()) / img[mask].std()
    
    # Convert to tensor and ensure channel first (C, D, H, W)
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    if img.ndim == 3:  # Add channel dimension if missing
        img = img.unsqueeze(0)
    
    return img

def load_mask(row, modality_col):
    img_path = row[modality_col]
    img = nib.load(img_path)
    img = img.get_fdata()
    mask = img != 0
    mask = torch.tensor(mask)
    return mask

class aedataset_T1T2(torch.utils.data.Dataset):
    def __init__(self, datafile, modality_T1, modality_T2, transforms=None):
        self.df = pd.read_csv(datafile)
        self.modality_T1 = modality_T1
        self.modality_T2 = modality_T2
        # transforms parameter kept for compatibility but not used

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_T1 = load_sample(row, self.modality_T1)
        x_T2 = load_sample(row, self.modality_T2)
        mask = load_mask(row, self.modality_T1)
        return x_T1, x_T2, mask

# Define datasets
train_dataset = aedataset_T1T2(
    datafile="/data4012/kpatel38/backups/autoencoder_ethnicity/train_mixed_ethnicity.csv",
    modality_T1="T1_unbiased_linear",
    modality_T2="T2_unbiased_linear",
)

val_dataset = aedataset_T1T2(
    datafile="/data4012/kpatel38/backups/autoencoder_ethnicity/val_mixed_ethnicity.csv",
    modality_T1="T1_unbiased_linear",
    modality_T2="T2_unbiased_linear",
)

# Directory name to save checkpoints and metrics
dir_name = "T1_128/output/vit_crossattention/"

# Main training loop using standard PyTorch
if __name__ == "__main__":
    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data.distributed import DistributedSampler
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DeepENDO ViT model')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch (default: 0)')
    args = parser.parse_args()

    # Set PyTorch memory allocator configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    
    # CUDA settings for better memory management
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.max_split_size_mb = 512
    
    # DDP setup
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    print(f"Environment variables: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    
    if world_size > 1:
        print(f"Initializing process group with rank={rank}, world_size={world_size}")
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        is_main_process = (rank == 0)
    else:
        print("Not running in DDP mode - environment variables not found")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    print(f'Rank: {rank}, Using CUDA device: {torch.cuda.current_device()}')
    
    # Create ViT model with smaller parameters for memory efficiency
    AE_model = engine_AE(
        lr=0.0005248074602497723, 
        concat_modalities=True,
        img_size=(96, 112, 96),
        patch_size=(16, 16, 16),
        embed_dim=512,  # Reduced from 768
        encoder_depth=4,  # Reduced from 6
        decoder_depth=2,  # Reduced from 4
        num_heads=8  # Reduced from 12
    )
    AE_model = AE_model.to(device)
    
    if dist.is_available() and dist.is_initialized():
        AE_model = DDP(AE_model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(AE_model.parameters(), lr=AE_model.module.lr if hasattr(AE_model, 'module') else AE_model.lr, weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=4, min_lr=(AE_model.module.lr if hasattr(AE_model, 'module') else AE_model.lr)/1000, factor=0.5)

    # Resume from checkpoint if specified
    start_epoch = args.start_epoch
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            
            # Load model state
            if hasattr(AE_model, 'module'):
                AE_model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                AE_model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded checkpoint. Resuming from epoch {start_epoch}")

    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(dir_name, exist_ok=True)

    # TensorBoard writer (only on main process)
    if is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(dir_name, "tb_logs"))
    else:
        writer = None

    # Hyperparameters optimized for ViT memory efficiency
    batch_size = 2  # Very small for ViT
    accumulation_steps = 24  # Large accumulation to maintain effective batch size
    num_workers = 1
    persistent_workers = False
    prefetch_factor = 1
    num_epochs = 100
    
    if world_size > 1:
        print(f"Using {num_workers} workers per GPU, total workers: {num_workers * world_size}")
        print(f"Effective batch size per GPU: {batch_size * accumulation_steps}")
        print(f"Total effective batch size: {batch_size * accumulation_steps * world_size}")
    
    # Distributed samplers for DDP
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, shuffle=False, sampler=train_sampler,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, shuffle=False, sampler=val_sampler,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, pin_memory=True, 
            num_workers=num_workers, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, pin_memory=True, 
            num_workers=num_workers, shuffle=False
        )

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        if dist.is_available() and dist.is_initialized():
            train_sampler.set_epoch(epoch)
        
        AE_model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        
        for i, batch in enumerate(train_loader):
            x_T1 = batch[0].to(device, non_blocking=True)
            x_T2 = batch[1].to(device, non_blocking=True)
            mask = batch[2].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                (recon_T1, recon_T2), _ = AE_model(x_T1, x_T2)
                
                # Calculate losses
                loss_T1 = AE_model.module.train_loss_function1(x_T1, recon_T1) if hasattr(AE_model, 'module') else AE_model.train_loss_function1(x_T1, recon_T1)
                loss_T1 = (loss_T1.squeeze(1) * mask).sum() / mask.sum()
                
                loss_T2 = AE_model.module.train_loss_function1(x_T2, recon_T2) if hasattr(AE_model, 'module') else AE_model.train_loss_function1(x_T2, recon_T2)
                loss_T2 = (loss_T2.squeeze(1) * mask).sum() / mask.sum()
                
                loss = (loss_T1 + loss_T2) / 2.0
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(AE_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            
            total_loss += loss.item() * accumulation_steps * x_T1.size(0)
            
            # Clean up
            del loss, loss_T1, loss_T2, recon_T1, recon_T2, x_T1, x_T2, mask
            torch.cuda.empty_cache()
            
            if i % (accumulation_steps * 2) == 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        
        train_loss = total_loss / len(train_loader.dataset)
        
        # Validation
        val_loss, val_psnr, val_ssim = validate_one_epoch(AE_model, val_loader, device)
        scheduler.step(val_loss)
        
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.3f} | Val SSIM: {val_ssim:.3f} | LR: {current_lr:.6f}")
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/psnr", val_psnr, epoch)
            writer.add_scalar("val/ssim", val_ssim, epoch)
            writer.add_scalar("lr", current_lr, epoch)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': (AE_model.module if hasattr(AE_model, 'module') else AE_model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim
            }
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(dir_name, "best_model.pth"))
            
            torch.save(checkpoint, os.path.join(dir_name, "last_model.pth"))
            if (epoch + 1) % 10 == 0:
                torch.save(checkpoint, os.path.join(dir_name, f"checkpoint_epoch_{epoch+1}.pth"))
                
    if is_main_process and writer is not None:
        writer.close()
