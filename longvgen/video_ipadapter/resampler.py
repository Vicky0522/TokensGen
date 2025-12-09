# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import FeedForward

from longvgen.models.embeddings import get_3d_rotary_pos_embed, apply_rotary_emb

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    start_temporal_idx: int = 0,
    grid_crops_coords = None 
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)

    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
        start_temporal_idx=start_temporal_idx
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

## FFN
#def FeedForward(dim, mult=4):
#    inner_dim = int(dim * mult)
#    return nn.Sequential(
#        nn.LayerNorm(dim),
#        nn.Linear(dim, inner_dim, bias=False),
#        nn.GELU(),
#        nn.Linear(inner_dim, dim, bias=False),
#    )

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, qk_norm=True):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.qk_norm = qk_norm

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        if qk_norm:
            self.norm_q = nn.LayerNorm(dim_head, eps=1e-6)
            self.norm_k = nn.LayerNorm(dim_head, eps=1e-6)

    def forward(self, x, latents, image_rotary_emb=None, sampling_rotary_emb=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)

        if image_rotary_emb is not None:
            k[:,:,:-l] = apply_rotary_emb(k[:,:,:-l], image_rotary_emb)

        if sampling_rotary_emb is not None:
            tmp1, tmp2 = sampling_rotary_emb
            q = apply_rotary_emb(q, sampling_rotary_emb)
            k[:,:,-l:] = apply_rotary_emb(k[:,:,-l:], sampling_rotary_emb)

        # attention
#        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
#        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
#        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
#        out = weight @ v
        scale = 1 / math.sqrt(self.dim_head)
        out = F.scaled_dot_product_attention(q, k, v, scale=scale)

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_height_queries=6,
        num_width_queries=6,
        num_temporal_queries=13,
        embedding_dim=1280,
        output_dim=1024,
        ff_mult=4,
        max_height_seq_len: int = 16,
        max_width_seq_len: int = 16,
        max_temporal_seq_len: int = 49,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        ff_inner_dim: Optional[int] = None,
        final_dropout: bool = True,
        ff_bias: bool = True,
#        bottleneck_dim=16,
        **kwargs
    ):
        super().__init__()

        self.num_height_queries = num_height_queries
        self.num_width_queries = num_width_queries
        self.num_temporal_queries = num_temporal_queries

        self.latents = nn.Parameter(torch.randn(1, num_height_queries*num_width_queries*num_temporal_queries, dim) / dim**0.5) 

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.pca = None

#        self.bottleneck = nn.ModuleList(
#            [
#                nn.Linear(output_dim, bottleneck_dim),
#                nn.Linear(bottleneck_dim, output_dim),
#            ]
#        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim, 
                            dim_head=dim_head, 
                            heads=heads, 
                            qk_norm=True
                        ),
                        FeedForward(
                            dim=dim, 
                            dropout=dropout,
                            activation_fn=activation_fn,
                            final_dropout=final_dropout,
                            inner_dim=ff_inner_dim,
                            bias=ff_bias,
                        ),
                    ]
                )
            )

    def set_pca(self, pca_path=None, device="cuda"):
        if pca_path is None:
            self.pca = None
        else:
            print(f"Successfully set pca: {pca_path}")
            self.pca = torch.load(pca_path).to(device)

    def forward(self, x, image_rotary_emb=None, sampling_rotary_emb=None):
        '''
        x shape (1+(chunk_size-1)*num_chunks, n, D)
        return shape (1, (num_chunks+1) * spatial_seq + (num_chunks+1) * temporal_seq, D) 
        '''

        latents = self.latents

        b = x.shape[0]
        x = rearrange(x, "b f n d -> (b f) n d")
        x = self.proj_in(x)
        x = rearrange(x, "(b f) n d -> b (f n) d", b=b)

        latents = repeat(latents, "1 ... -> b ...", b=b)

        for attn, ff in self.layers:
            latents = attn(x, latents, image_rotary_emb, sampling_rotary_emb) + latents
            latents = ff(latents) + latents

        latents = self.norm_out(self.proj_out(latents))

        if self.pca is not None:
            b = latents.shape[0]
            dtype = latents.dtype
            latents = rearrange(latents, "b n d -> (b n) d").to(self.pca.components_.dtype)
            latents = self.pca.transform(latents)
            latents[:,16:] = 0.0
            latents = self.pca.inverse_transform(latents)
            latents = rearrange(latents, "(b n) d -> b n d", b=b).to(dtype)

#        latents = self.bottleneck[0](latents)
#        latents = self.bottleneck[1](latents)

        latents = rearrange(latents, "b (f h w) d -> b f d h w", f=self.num_temporal_queries, h=self.num_height_queries)

        return latents

