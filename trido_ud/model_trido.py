"""
TriDo-JiT: Triple-Domain Joint-in-Time Transformer for PET Denoising
======================================================================
Three-domain architecture:
  1. Sinogram Domain — sino encoder with body-part conditioning (Cross-Domain Recon)
  2. Image Domain     — JiT transformer-based refinement (Flow Matching v-prediction)
  3. Frequency Domain — GFP high-frequency enhancement (DCT-based)

The three domains operate in a cascade:
  Input image → Radon → Sinogram → SinoEncoder → FBP → JiT Transformer → GFP → Output

This file defines the core TriDoJiT nn.Module that combines all sub-modules.
It reuses the proven JiT backbone from model_jit_ud.py, extending it with
sinogram and frequency domain processing.

References:
  - SiT: https://github.com/willisma/SiT
  - Lightning-DiT: https://github.com/hustvl/LightningDiT
  - Cross-Domain Reconstruction (sinogram domain)
  - Prior Knowledge-Guided Triple-Domain Transformer-GAN
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Reuse proven components from _ud version
import sys
sys.path.insert(0, '/Users/Zhuanz/projects/JiT-main')
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm

# TriDo-specific modules (support both package and direct invocation)
try:
    from trido_ud.sino_processor import SinogramEncoder
    from trido_ud.radon_transform import SinoImageBridge
    from trido_ud.gfp_module import GFPModule
except ImportError:
    from sino_processor import SinogramEncoder
    from radon_transform import SinoImageBridge
    from gfp_module import GFPModule


# ==============================================================================
# Reused JiT Building Blocks (from model_jit_ud.py)
# ==============================================================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=2, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * 1000.0, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def scaled_dot_product_attention(query, key, value, dropout_p=0.0):
    return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = rope(q)
        k = rope(k)
        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ==============================================================================
# TriDo-JiT: Three-Domain Architecture
# ==============================================================================

class TriDoJiT(nn.Module):
    """
    Triple-Domain Joint-in-Time Transformer for PET Denoising.

    Three domains cascaded:
      ┌─────────────────────────────────────────────────────────────┐
      │  ① Sinogram Domain   ② Image Domain     ③ Frequency Domain  │
      │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
      │  │ Radon → Sino │→  │ FBP → JiT    │→  │ GFP Enhance  │→   │
      │  │   Encoder    │   │  Transformer │   │   Module     │    │
      │  └──────────────┘   └──────────────┘   └──────────────┘    │
      └─────────────────────────────────────────────────────────────┘

    Key features:
      - Body part embedding injected at sinogram AND image domain
      - FBP provides differentiable sinogram→image bridge
      - JiT backbone with v-prediction flow matching
      - GFP enhances high-frequencies lost in denoising

    Args:
        input_size: Input image size (square)
        patch_size: ViT patch size
        in_channels: Input channels (condition + noisy = 2)
        out_channels: Output channels (denoised image = 1)
        hidden_size: Transformer hidden dim
        depth: Number of JiT blocks
        num_heads: Attention heads
        mlp_ratio: MLP expansion ratio
        attn_drop: Attention dropout
        proj_drop: Projection dropout
        bottleneck_dim: Patch embedding bottleneck
        n_views: Sinogram projection views
        sino_base_ch: Sinogram encoder base channels
        use_sino_domain: Enable sinogram domain processing
        use_freq_domain: Enable frequency domain enhancement
    """

    def __init__(self, input_size=256, patch_size=16, in_channels=2, out_channels=1,
                 hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, bottleneck_dim=128,
                 n_views=256, sino_base_ch=32,
                 use_sino_domain=True, use_freq_domain=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_sino_domain = use_sino_domain
        self.use_freq_domain = use_freq_domain

        # --- Time embedding ---
        self.t_embedder = TimestepEmbedder(hidden_size)

        # --- Body part embedding (3 classes: 0=brain, 1=chest, 2=abdomen) ---
        self.body_part_embedder = nn.Embedding(3, hidden_size)

        # --- Domain ①: Sinogram ---
        if use_sino_domain:
            self.sino_bridge = SinoImageBridge(
                n_views=n_views, img_size=input_size,
                det_size=input_size
            )
            self.sino_encoder = SinogramEncoder(
                n_views=n_views, n_bins=input_size,
                base_channels=sino_base_ch, cond_dim=128
            )
        else:
            self.sino_bridge = None
            self.sino_encoder = None

        # --- Domain ②: Image (JiT Transformer) ---
        # After FBP, we have a 1-channel image. Concatenate with condition.
        # But the JiT expects in_channels at input. If using sino domain,
        # the image-domain JiT receives the FBP output + condition.
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0
        )

        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop, proj_drop=proj_drop)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # --- Domain ③: Frequency (GFP) ---
        if use_freq_domain:
            self.gfp = GFPModule(img_size=input_size, body_part_cond=True)
        else:
            self.gfp = None

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.body_part_embedder.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, model_input, t, body_part):
        """
        Full three-domain forward pass.

        Args:
            model_input: (B, 2, H, W) — concat(z, condition)
            t: (B,) — flow matching timestep
            body_part: (B,) LongTensor — body part category (0=brain, 1=chest, 2=abdomen)

        Returns:
            output: (B, 1, H, W) — predicted clean image
        """
        B, C, H, W = model_input.shape

        # --- Body part conditioning ---
        body_part_emb = self.body_part_embedder(body_part)   # (B, hidden_size)

        # --- Timestep conditioning ---
        t_emb = self.t_embedder(t)

        # [核心融合]：时间上下文与身体部位上下文物理叠加
        c = t_emb + body_part_emb

        # --- Extract z (noisy) and condition separately ---
        z = model_input[:, 0:1, :, :]       # (B, 1, H, W) — noisy image
        cond = model_input[:, 1:2, :, :]    # (B, 1, H, W) — condition image

        # ================================================================
        # Domain ①: Sinogram Domain Processing
        # ================================================================
        if self.use_sino_domain:
            # Forward project condition to sinogram domain
            sino_cond = self.sino_bridge.forward_project(cond)  # (B, 1, n_views, det_size)

            # Sino encoder denoising with body part conditioning
            sino_denoised = self.sino_encoder(sino_cond, body_part)  # (B, 1, n_views, det_size)

            # FBP reconstruction back to image domain
            image_from_sino = self.sino_bridge.reconstruct(sino_denoised)  # (B, 1, H, W)

            # Combine FBP output with original z for the image domain
            # This creates a 2-channel input for the JiT: z + sino_recon
            image_input = torch.cat([z, image_from_sino], dim=1)  # (B, 2, H, W)
        else:
            # Without sino domain, use z + cond directly
            image_input = torch.cat([z, cond], dim=1)

        # ================================================================
        # Domain ②: Image Domain (JiT Transformer)
        # ================================================================
        x = self.x_embedder(image_input)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x, c, self.feat_rope)

        x = self.final_layer(x, c)
        image_output = self.unpatchify(x, self.patch_size)  # (B, 1, H, W)

        # ================================================================
        # Domain ③: Frequency Domain (GFP Enhancement)
        # ================================================================
        if self.use_freq_domain:
            output = self.gfp(image_output, body_part)
        else:
            output = image_output

        return output


def TriDoJiT_Large(**kwargs):
    """TriDo-JiT Large: 24-layer, 1024-dim transformer."""
    patch_size = kwargs.pop('patch_size', 16)
    return TriDoJiT(
        depth=24, hidden_size=1024, num_heads=16,
        bottleneck_dim=128, patch_size=patch_size, **kwargs
    )


def TriDoJiT_Base(**kwargs):
    """TriDo-JiT Base: 12-layer, 768-dim transformer (faster training)."""
    patch_size = kwargs.pop('patch_size', 16)
    return TriDoJiT(
        depth=12, hidden_size=768, num_heads=12,
        bottleneck_dim=128, patch_size=patch_size, **kwargs
    )


def TriDoJiT_Small(**kwargs):
    """TriDo-JiT Small: 6-layer, 512-dim transformer (quick experiments)."""
    patch_size = kwargs.pop('patch_size', 16)
    return TriDoJiT(
        depth=6, hidden_size=512, num_heads=8,
        bottleneck_dim=128, patch_size=patch_size, **kwargs
    )


TriDoJiT_models = {
    'TriDoJiT-Large': TriDoJiT_Large,
    'TriDoJiT-Base': TriDoJiT_Base,
    'TriDoJiT-Small': TriDoJiT_Small,
}
