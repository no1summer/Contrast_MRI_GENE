# ====================================
# PyTorch and Core Libraries
# ====================================
import os
import math
import argparse
from functools import partial

# ====================================
# Data Handling
# ====================================
import numpy as np
import pandas as pd
import nibabel as nib

# ====================================
# Deep Learning
# ====================================
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

# ====================================
# Metrics and Progress
# ====================================
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# ====================================
# UTILITIES from udip_vit_merged.py
# ====================================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization helper function."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# ====================================
# POSITIONAL EMBEDDINGS from udip_vit_merged.py
# ====================================

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sinusoidal positional embeddings from grid positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """Generate 3D sinusoidal positional embeddings."""
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim/6)*2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

# ====================================
# PATCH EMBEDDING from udip_vit_merged.py
# ====================================

class PatchEmbed3D(nn.Module):
    """3D Volume to Patch Embedding."""
    def __init__(self, patch_size=14, tubelet_size=16, in_chans=1, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # Use (tubelet, patch, patch) ordering so temporal/depth dimension comes first
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(patch_size, tubelet_size,patch_size),
            stride=(patch_size,tubelet_size,  patch_size),
        )

    def forward(self, x, **kwargs):
        # The UDIP model expects (B, C, D, H, W)
        # Our data loader provides (B, D, H, W), so we add the channel dim
        x = x.unsqueeze(1)  # Add channel dimension if needed
        B, C, T, H, W = x.shape
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# ====================================
# ATTENTION AND MLP MODULES from udip_vit_merged.py
# ====================================

class MLP(nn.Module):
    """Multi-Layer Perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        # Try to use SDPA, but fall back if it's not available (e.g., older PyTorch)
        self.use_sdpa = use_sdpa and hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            # scaled_dot_product_attention does not return attention weights
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 grid_size=None, grid_depth=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, mask=None):
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x

# ====================================
# VISION TRANSFORMER ENCODER from udip_vit_merged.py
# ====================================

class VisionTransformer(nn.Module):
    """Vision Transformer encoder with optional non-zero patch optimization."""
    def __init__(self, img_size=182, patch_size=14, num_frames=224, tubelet_size=16,
                 in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 norm_layer=nn.LayerNorm, init_std=0.02, out_layers=None,
                 uniform_power=False, non_zero_patch_opt=True, **kwargs):
        super().__init__()
        self.non_zero_patch_opt = non_zero_patch_opt
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers

        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        self.grid_size = self.input_size // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, tubelet_size=tubelet_size,
            in_chans=in_chans, embed_dim=embed_dim)
        
        self.num_patches = self.grid_size * self.grid_depth * self.grid_size

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                act_layer=nn.GELU, grid_size=self.grid_size, grid_depth=self.grid_depth,
                attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        self._init_pos_embed(self.pos_embed.data)
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        """Initialize positional embeddings with sine-cosine."""
        embed_dim = pos_embed.size(-1)
        sincos = get_3d_sincos_pos_embed(embed_dim, self.grid_size, self.grid_depth, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        """Initialize module weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        """Rescale transformer block weights for better initialization."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def non_zero_patch(self, x, y):
        """Filter out patches that are all zeros across the batch (match reference)."""
        non_zero_mask = y.view(
            -1, self.grid_size, self.patch_size, self.grid_depth, self.tubelet_size,
            self.grid_size, self.patch_size).sum((2,4,6)) != 0
        non_zero_mask = non_zero_mask.view(-1, self.grid_size * self.grid_depth * self.grid_size)
        batch_mask = non_zero_mask.max(0)[0]
        x = x[:, batch_mask, :]
        return x, batch_mask, non_zero_mask

    def forward(self, x, y):
        """Forward pass through vision transformer (match reference)."""
        x = self.patch_embed(x)
        x += self.pos_embed
        if self.non_zero_patch_opt:
            x, batch_mask, non_zero_patch = self.non_zero_patch(x, y)
        else:
            batch_mask = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
            non_zero_patch = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        for blk in self.blocks:
            x = blk(x)
        if self.norm is not None:
            x = self.norm(x)
        return x, batch_mask, non_zero_patch


# ====================================
# DECODER from udip_vit_merged.py
# ====================================

class Decoder(nn.Module):
    """UDIP-style decoder with positional token concatenation."""
    def __init__(self, img_size=182, patch_size=14, num_frames=224, tubelet_size=16,
                 embed_dim=384, decoder_embed_dim=192, depth=4, num_heads=6,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, norm_layer=nn.LayerNorm, init_std=0.02):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        
        self.num_patches = (self.input_size // self.patch_size) ** 2 * (self.num_frames // self.tubelet_size)
        self.grid_size = self.input_size // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size
        
        # Positional embedding for decoder
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.num_patches, self.decoder_embed_dim), requires_grad=False)
        self._init_pos_embed(self.pos_emb.data)
        
        # Decoder layers
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                act_layer=nn.GELU, grid_size=self.grid_size, grid_depth=self.grid_depth,
                attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(self.decoder_embed_dim)
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)
        
        # Output projection
        patch_numel = self.patch_size ** 2 * self.tubelet_size
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, patch_numel, bias=True)
        
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        """Rescale block weights."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_pos_embed(self, pos_embed):
        """Initialize positional embeddings."""
        embed_dim = pos_embed.size(-1)
        sincos = get_3d_sincos_pos_embed(embed_dim, self.grid_size, self.grid_depth, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def unpatchify(self, x, batch_mask):
        """Reconstruct image from patches."""
        B = x.shape[0]
        # Create a full tensor of zeros for all patches
        full_patches = torch.zeros(B, self.num_patches, x.shape[-1], device=x.device)
        # Place the predicted patches into the correct positions
        full_patches[:, batch_mask, :] = x
        
        # Reshape to image dimensions
        x = full_patches.view(
            B, self.grid_depth, self.grid_size, self.grid_size, 
            self.tubelet_size, self.patch_size, self.patch_size
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.num_frames, self.input_size, self.input_size)
        return x

    def forward(self, x, batch_mask):
        """Forward pass through decoder."""
        # Get positional embeddings for active patches
        pos_emb = self.pos_emb[:, batch_mask, :].repeat(x.shape[0], 1, 1)
        num_latent = x.shape[1]
        
        # Project encoder output
        x = self.decoder_embed(x)
        
        # Concatenate memory tokens with positional tokens (UDIP strategy)
        x = torch.cat([x, pos_emb], dim=1)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        # Predict from positional tokens only
        x = self.decoder_pred(x[:, num_latent:, :])
        return x

# ====================================
# MAIN MODEL WRAPPER (replaces engine_AE)
# ====================================

class MoCoV2Dual(nn.Module):
    """MoCo v2 style contrastive learning treating T1 and T2 as two views (positive pair).
    Uses a VisionTransformer backbone and momentum encoder. Supports symmetric loss (both directions)."""
    def __init__(self, lr, img_size=182, patch_size=14, num_frames=224, tubelet_size=16,
                 in_chans=1, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, m=0.999, T=0.07,
                 K=65536, proj_dim=256, symmetric=True, non_zero_patch_opt=False):
        super().__init__()
        self.lr = lr
        self.m = m
        self.T = T
        self.K = K
        self.symmetric = symmetric
        # Query encoder
        self.encoder_q = VisionTransformer(
            img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim,
            depth, num_heads, mlp_ratio, qkv_bias, None, drop_rate, attn_drop_rate,
            partial(nn.LayerNorm, eps=1e-6), 0.02, None, False, non_zero_patch_opt)
        # Key encoder
        self.encoder_k = VisionTransformer(
            img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim,
            depth, num_heads, mlp_ratio, qkv_bias, None, drop_rate, attn_drop_rate,
            partial(nn.LayerNorm, eps=1e-6), 0.02, None, False, non_zero_patch_opt)
        # Projection heads (MoCo v2 MLP with BN and ReLU)
        self.projector_q = self._build_mlp(embed_dim, proj_dim)
        self.projector_k = self._build_mlp(embed_dim, proj_dim)
        # Initialize key encoder to query encoder weights
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # Create the queue
        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _build_mlp(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # keys: (B, C)
        keys = concat_all_gather(keys) if dist.is_available() and dist.is_initialized() else keys
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        K = self.K
        if batch_size > K:
            keys = keys[:K]
            batch_size = K
        if ptr + batch_size <= K:
            self.queue[:, ptr:ptr+batch_size] = keys.T
        else:
            first = K - ptr
            self.queue[:, ptr:] = keys[:first].T
            remaining = batch_size - first
            if remaining > 0:
                self.queue[:, :remaining] = keys[first:first+remaining].T
        ptr = (ptr + batch_size) % K
        self.queue_ptr[0] = ptr

    def _encode(self, encoder, projector, x, y):
        latent, _, _ = encoder(x, y)
        pooled = latent.mean(1)
        if projector is not None:
            z = projector(pooled)  # (B, proj_dim)
        else:
            z = pooled  # (B, embed_dim) - use pooled features directly
        z = F.normalize(z, dim=1)
        return z

    def forward(self, x_T1, x_T2):
        # x_T1 as query view, x_T2 as key view
        y_T1 = x_T1
        y_T2 = x_T2
        q1 = self._encode(self.encoder_q, self.projector_q, x_T1, y_T1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k2 = self._encode(self.encoder_k, self.projector_k, x_T2, y_T2)
        # logits for direction T1->T2
        l_pos_12 = torch.einsum('nc,nc->n', [q1, k2]).unsqueeze(-1)  # (B,1)
        l_neg_12 = torch.einsum('nc,ck->nk', [q1, self.queue.clone().detach()])  # (B,K)
        logits_12 = torch.cat([l_pos_12, l_neg_12], dim=1) / self.T
        labels = torch.zeros(logits_12.size(0), dtype=torch.long, device=logits_12.device)
        loss_12 = F.cross_entropy(logits_12, labels)

        if self.symmetric:
            # Second direction T2->T1
            q2 = self._encode(self.encoder_q, self.projector_q, x_T2, y_T2)
            with torch.no_grad():
                k1 = self._encode(self.encoder_k, self.projector_k, x_T1, y_T1)
            l_pos_21 = torch.einsum('nc,nc->n', [q2, k1]).unsqueeze(-1)
            l_neg_21 = torch.einsum('nc,ck->nk', [q2, self.queue.clone().detach()])
            logits_21 = torch.cat([l_pos_21, l_neg_21], dim=1) / self.T
            loss_21 = F.cross_entropy(logits_21, labels)
            loss = 0.5 * (loss_12 + loss_21)
        else:
            loss = loss_12

        # Update queue with keys from current minibatch (use k2 only)
        self._dequeue_and_enqueue(k2.detach())
        return loss, {
            'loss_12': loss_12.detach() if self.symmetric else loss.detach(),
            'queue_ptr': int(self.queue_ptr.item())
        }
    def forward_val(self, x_T1, x_T2):
        # Encode only using encoder (projector optional)
        q = self._encode(self.encoder_q, None, x_T1,x_T1)  # omit projector if you want
        k = self._encode(self.encoder_q, None, x_T2,x_T2)  # note: use encoder_q only

        # Normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        # Cosine similarity matrix: (B, B)
        sim_matrix = torch.matmul(q, k.T) / self.T

        # Positive labels on diagonal
        labels = torch.arange(q.size(0), device=sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
    
    def get_encoder_features(self, x_T1, x_T2):
        """Extract encoder features (before projection) for evaluation metrics."""
        # Get latent representations from encoders (before pooling and projection)
        latent_T1, _, _ = self.encoder_q(x_T1, x_T1)
        latent_T2, _, _ = self.encoder_q(x_T2, x_T2)
        
        # Pool to get global features (mean pooling over patches)
        feat_T1 = latent_T1.mean(1)  # (B, embed_dim)
        feat_T2 = latent_T2.mean(1)  # (B, embed_dim)
        
        return feat_T1, feat_T2

@torch.no_grad()
def concat_all_gather(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

# ====================================
# EVALUATION METRICS
# ====================================

def compute_cross_modal_retrieval_metrics(feat_T1, feat_T2, k_values=[1, 5, 10]):
    """
    Compute cross-modal retrieval accuracy metrics.
    
    For each T1, find if the nearest neighbor T2 is its real partner.
    
    Args:
        feat_T1: (N, D) tensor of T1 encoder features
        feat_T2: (N, D) tensor of T2 encoder features (paired with T1)
        k_values: List of k values for recall@k
    
    Returns:
        dict with metrics: top1_acc, recall_at_k, median_rank
    """
    # Normalize features
    feat_T1 = F.normalize(feat_T1, dim=1)
    feat_T2 = F.normalize(feat_T2, dim=1)
    
    # Compute similarity matrix: (N, N)
    # sim[i, j] = similarity between T1[i] and T2[j]
    sim_matrix = torch.matmul(feat_T1, feat_T2.T)  # (N, N)
    
    # For each T1, find the rank of its true T2 partner
    # True partner is on the diagonal (i, i)
    N = sim_matrix.size(0)
    true_labels = torch.arange(N, device=sim_matrix.device)
    
    # Get sorted indices (descending order of similarity)
    _, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)
    
    # Find rank of true partner for each T1
    ranks = []
    for i in range(N):
        rank = (sorted_indices[i] == true_labels[i]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    ranks = torch.tensor(ranks, device=sim_matrix.device, dtype=torch.float32)
    
    # Top-1 accuracy: percentage where rank == 1
    top1_acc = (ranks == 1).float().mean().item()
    
    # Recall@k: percentage where rank <= k
    recall_at_k = {}
    for k in k_values:
        recall_at_k[f'recall@{k}'] = (ranks <= k).float().mean().item()
    
    # Median rank
    median_rank = ranks.median().item()
    
    return {
        'top1_accuracy': top1_acc,
        'recall_at_k': recall_at_k,
        'median_rank': median_rank,
        'mean_rank': ranks.mean().item()
    }


def compute_cka(feat_T1, feat_T2, kernel='linear', center=True):
    """
    Compute Centered Kernel Alignment (CKA) between T1 and T2 encoder outputs.
    
    CKA measures the similarity between two representations.
    Higher CKA means better cross-modal alignment.
    
    Args:
        feat_T1: (N, D1) tensor of T1 encoder features
        feat_T2: (N, D2) tensor of T2 encoder features
        kernel: 'linear' or 'rbf' kernel type
        center: whether to center the kernel matrices
    
    Returns:
        CKA value (scalar)
    """
    # Convert to numpy for easier computation
    if isinstance(feat_T1, torch.Tensor):
        feat_T1 = feat_T1.cpu().numpy()
    if isinstance(feat_T2, torch.Tensor):
        feat_T2 = feat_T2.cpu().numpy()
    
    # Compute kernel matrices
    if kernel == 'linear':
        K = np.dot(feat_T1, feat_T1.T)
        L = np.dot(feat_T2, feat_T2.T)
    elif kernel == 'rbf':
        # RBF kernel with median distance as bandwidth
        try:
            from sklearn.metrics.pairwise import rbf_kernel
            from scipy.spatial.distance import pdist
            gamma_T1 = 1.0 / (2.0 * np.median(pdist(feat_T1)) ** 2) if feat_T1.shape[0] > 1 else 1.0
            gamma_T2 = 1.0 / (2.0 * np.median(pdist(feat_T2)) ** 2) if feat_T2.shape[0] > 1 else 1.0
            K = rbf_kernel(feat_T1, gamma=gamma_T1)
            L = rbf_kernel(feat_T2, gamma=gamma_T2)
        except ImportError:
            # Fallback: use linear kernel if sklearn/scipy not available
            print("Warning: sklearn/scipy not available, using linear kernel for CKA")
            K = np.dot(feat_T1, feat_T1.T)
            L = np.dot(feat_T2, feat_T2.T)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")
    
    # Center the kernel matrices
    if center:
        N = K.shape[0]
        H = np.eye(N) - np.ones((N, N)) / N
        K = H @ K @ H
        L = H @ L @ H
    
    # Compute CKA
    # CKA = ||K^T L||_F^2 / (||K^T K||_F ||L^T L||_F)
    numerator = np.trace(K @ L)
    denominator = np.sqrt(np.trace(K @ K) * np.trace(L @ L))
    
    if denominator == 0:
        return 0.0
    
    cka = numerator / denominator
    return float(cka)

      

# Validation function for the new architecture
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from tqdm import tqdm


# ====================================
# DATASET
# ====================================
import pandas as pd
import nibabel as nib



class aedataset(torch.utils.data.Dataset):
    def __init__(self, datafile, modality):
        """
        Args:
            datafile (type: csv or list): the datafile mentioning the location of images or a list of file locations.
            modality (type: string): column containing location of modality of interest in the datafile.
            transforms (type: pytorch specific transforms): to add channel to the image and convert to tensor.
        Returns:
            img [torch tensor]: img file normalized 
            mask [torch tensor]: mask excluding background
            img_name [string]: name of the image
        """
        self.datafile = pd.read_csv(datafile)
        self.unbiased_brain = self.datafile[modality]

    def __len__(self):
        return len(self.unbiased_brain)

    def __getitem__(self, idxx=int):
        img_name = self.unbiased_brain[idxx]
        img = nib.load(img_name)
        img = img.get_fdata()
        img = torch.from_numpy(img)
        img = torch.nn.functional.pad(img, (0,0,3,3,0,0)) # padding image from 182x218x182 to 182x224x182
        # padding needs to be done before normalization
        mask = img != 0
        img = (img - img[img != 0].mean()) / img[img != 0].std()
        img = img.type(torch.float)
        #mask = mask.int()
        return img, mask

class aedataset_T1T2(torch.utils.data.Dataset):
    """Dual modality dataset returning (T1, T2, mask_T1, mask_T2)."""
    def __init__(self, datafile, modality_T1, modality_T2):
        self.df = pd.read_csv(datafile)
        self.modality_T1 = modality_T1
        self.modality_T2 = modality_T2

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        img = nib.load(path).get_fdata()
        img = torch.from_numpy(img)
        img = torch.nn.functional.pad(img, (0,0,3,3,0,0))
        mask = img != 0
        img = (img - img[mask].mean()) / img[mask].std()
        return img.float(), mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        t1, m1 = self._load(row[self.modality_T1])
        t2, m2 = self._load(row[self.modality_T2])
        return t1, t2, m1, m2

    
# ====================================
# MAIN TRAINING SCRIPT
# ====================================
if __name__ == "__main__":
    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data.distributed import DistributedSampler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
        is_main_process = True

    print(f'Rank: {rank}, Using CUDA device: {torch.cuda.current_device() if torch.cuda.is_available() else "CPU"}')
    
    # Create MoCo contrastive model
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
        symmetric=True
    ).to(device)
    print(f"MoCoV2Dual model loaded on {device}")
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=model.module.lr if hasattr(model,'module') else model.lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Resume from checkpoint if specified
    start_epoch = args.start_epoch
    best_val_loss = float('inf')  # track contrastive val loss
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded checkpoint. Resuming from epoch {start_epoch}")

    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    dir_name = "/data484_4/txia2/DeepENDO/training/T1_128/output/mocov2_replicate2_fixed"
    os.makedirs(dir_name, exist_ok=True)

    if is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(dir_name, "tb_logs"))
    else:
        writer = None

    # DataLoaders
    train_dataset = aedataset_T1T2(
        datafile="/data4012/kpatel38/backups/autoencoder_ethnicity/train_mixed_ethnicity.csv",
        modality_T1="T1_unbiased_linear",
        modality_T2="T2_unbiased_linear"
    )
    val_dataset = aedataset_T1T2(
        datafile="/data4012/kpatel38/backups/autoencoder_ethnicity/val_mixed_ethnicity.csv",
        modality_T1="T1_unbiased_linear",
        modality_T2="T2_unbiased_linear"
    )
    
    batch_size = 4
    num_workers = 4
    
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, sampler=train_sampler, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, sampler=val_sampler, drop_last=False
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, pin_memory=True, 
            num_workers=num_workers, shuffle=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, shuffle=False, drop_last=False
        )

    # Training Loop
    num_epochs = 300
    for epoch in range(start_epoch, num_epochs):
        if dist.is_available() and dist.is_initialized():
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for i, batch in enumerate(pbar):
            x_T1, x_T2, _, _ = batch
            x_T1 = x_T1.to(device)
            x_T2 = x_T2.to(device)
            with torch.cuda.amp.autocast():
                loss, extra = model(x_T1, x_T2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            running_loss += loss.item()
            pbar.set_postfix({'CLoss': f'{loss.item():.6f}'})
            if writer and is_main_process:
                writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + i)

        avg_train_loss = running_loss / len(train_loader)
        if writer and is_main_process:
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('LR/train', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step()

        # validation cosine similarity and evaluation metrics
        model.eval()
        val_loss_sum = 0.0
        
        # Get the actual model (handle DDP wrapper)
        model_for_eval = model.module if hasattr(model, 'module') else model
        
        # Collect features for evaluation metrics
        all_feat_T1 = []
        all_feat_T2 = []
        
        with torch.no_grad():
            for batch in val_loader:
                x_T1, x_T2, _, _ = batch
                x_T1, x_T2 = x_T1.to(device), x_T2.to(device)
                
                # Compute validation loss
                loss = model_for_eval.forward_val(x_T1, x_T2)
                val_loss_sum += loss.item()
                
                # Extract encoder features for evaluation metrics
                feat_T1, feat_T2 = model_for_eval.get_encoder_features(x_T1, x_T2)
                all_feat_T1.append(feat_T1)
                all_feat_T2.append(feat_T2)
        
        avg_val_loss = val_loss_sum / len(val_loader)
        
        # Concatenate features from this process
        if all_feat_T1:  # Only if we collected any features
            all_feat_T1 = torch.cat(all_feat_T1, dim=0)  # (N_local, embed_dim)
            all_feat_T2 = torch.cat(all_feat_T2, dim=0)  # (N_local, embed_dim)
            embed_dim = all_feat_T1.shape[1]
        else:
            # Create empty tensors if no features collected (need to know embed_dim)
            # Get embed_dim from model (already have model_for_eval)
            embed_dim = model_for_eval.encoder_q.embed_dim
            all_feat_T1 = torch.empty(0, embed_dim, device=device)
            all_feat_T2 = torch.empty(0, embed_dim, device=device)
        
        # In DDP mode, gather features from all processes (all processes must participate)
        if dist.is_available() and dist.is_initialized():
            all_feat_T1 = concat_all_gather(all_feat_T1)
            all_feat_T2 = concat_all_gather(all_feat_T2)
        
        # Compute evaluation metrics (only on main process to avoid duplication)
        if is_main_process and len(all_feat_T1) > 0:
            
            # Compute cross-modal retrieval metrics
            retrieval_metrics = compute_cross_modal_retrieval_metrics(
                all_feat_T1, all_feat_T2, k_values=[1, 5, 10, 50]
            )
            
            # Compute CKA
            cka_linear = compute_cka(all_feat_T1, all_feat_T2, kernel='linear', center=True)
            cka_rbf = compute_cka(all_feat_T1, all_feat_T2, kernel='rbf', center=True)
            
            # Log metrics to tensorboard
            if writer:
                writer.add_scalar('Metrics/retrieval_top1_accuracy', retrieval_metrics['top1_accuracy'], epoch)
                writer.add_scalar('Metrics/retrieval_median_rank', retrieval_metrics['median_rank'], epoch)
                writer.add_scalar('Metrics/retrieval_mean_rank', retrieval_metrics['mean_rank'], epoch)
                for k, recall in retrieval_metrics['recall_at_k'].items():
                    writer.add_scalar(f'Metrics/retrieval_{k}', recall, epoch)
                writer.add_scalar('Metrics/CKA_linear', cka_linear, epoch)
                writer.add_scalar('Metrics/CKA_rbf', cka_rbf, epoch)
            
            # Print metrics
            print(f"  Retrieval - Top-1 Acc: {retrieval_metrics['top1_accuracy']:.4f}, "
                  f"Median Rank: {retrieval_metrics['median_rank']:.2f}, "
                  f"Mean Rank: {retrieval_metrics['mean_rank']:.2f}")
            print(f"  Recall@1: {retrieval_metrics['recall_at_k']['recall@1']:.4f}, "
                  f"Recall@5: {retrieval_metrics['recall_at_k']['recall@5']:.4f}, "
                  f"Recall@10: {retrieval_metrics['recall_at_k']['recall@10']:.4f}")
            print(f"  CKA (linear): {cka_linear:.4f}, CKA (RBF): {cka_rbf:.4f}")


        if writer and is_main_process:
            writer.add_scalar('Loss/val_contrastive', avg_val_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs} - Train CLoss: {avg_train_loss:.6f} | Val CLoss: {avg_val_loss:.6f}")

        # Save checkpoint
        if is_main_process:
            is_best = avg_val_loss < best_val_loss
            best_val_loss = min(avg_val_loss, best_val_loss)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }
            
            save_path = os.path.join(dir_name, 'latest_checkpoint.pth')
            torch.save(checkpoint, save_path)
            if is_best:
                best_path = os.path.join(dir_name, 'best_model.pth')
                torch.save(checkpoint, best_path)
            if (epoch + 1) % 20 == 0:
                epoch_save_path = os.path.join(dir_name, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(checkpoint, epoch_save_path)

    if writer and is_main_process:
        writer.close()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    print("Training finished.")
