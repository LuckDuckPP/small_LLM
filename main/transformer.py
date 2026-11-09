"""
transformere.py
===========================

A modern, decoder-only Transformer stack (LLaMA-style) in pure PyTorch.

This file contains ONLY the transformer architecture (the "body" of the model).
It does not contain the token embedding layer or the language-modelling head;
those live in `model.py`, which wraps this decoder.

Building blocks
---------------
* TransformerConfig     - one dataclass holding every hyper-parameter
* RMSNorm               - cheaper, more stable alternative to LayerNorm
* RotaryEmbedding       - rotary positional embeddings (RoPE), cache-aware
* KVCache               - pre-allocated key/value cache for fast generation
* CausalSelfAttention   - grouped-query attention (GQA) + RoPE + KV cache,
                          uses PyTorch's fused SDPA (FlashAttention) kernel
* SwiGLUFeedForward     - gated feed-forward network (SwiGLU)
* TransformerBlock      - pre-norm residual block (attention + FFN)
* TransformerDecoder    - the stacked blocks + final norm

Data flow of a single block (pre-norm):

    x ──► RMSNorm ──► Attention(RoPE, GQA, causal) ──┐
    │                                                 ▼
    └─────────────────────────────────────────────── (+) ──► h
    h ──► RMSNorm ──► SwiGLU FFN ───────────────────┐
    │                                                ▼
    └──────────────────────────────────────────── (+) ──► out
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TransformerConfig:
    """All hyper-parameters of the model in one place."""

    vocab_size: int = 50256
    max_seq_len: int = 2048          # maximum context length
    d_model: int = 512               # hidden size
    n_layers: int = 8                # number of transformer blocks
    n_heads: int = 8                 # number of query heads
    n_kv_heads: int = 4              # number of key/value heads (GQA); = n_heads -> MHA, 1 -> MQA
    ffn_multiple_of: int = 64        # FFN hidden size is rounded up to a multiple of this
    ffn_hidden_mult: float = 8 / 3   # SwiGLU uses ~8/3 * d_model to match a 4x GELU FFN in params
    dropout: float = 0.0             # dropout for attention / residual paths
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0     # RoPE base frequency
    bias: bool = False               # bias in linear layers (LLaMA-style models use none)
    tie_word_embeddings: bool = True # share input embedding and output projection weights
    use_flash_attention: bool = True # use torch.nn.functional.scaled_dot_product_attention
    gradient_checkpointing: bool = False  # trade compute for memory during training
    init_std: float = 0.02           # std of the normal weight initialisation

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        hidden = int(self.ffn_hidden_mult * self.d_model)
        m = self.ffn_multiple_of
        return m * ((hidden + m - 1) // m)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        return cls(**d)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer norm: x / rms(x) * weight (no mean-centering, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()  # compute statistics in fp32 for numerical stability
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


# --------------------------------------------------------------------------- #
# Rotary positional embeddings
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k. Shapes: q (B,H,T,D), k (B,Hkv,T,D), cos/sin (1,1,T,D)."""
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


class RotaryEmbedding(nn.Module):
    """
    Pre-computes the cos/sin tables for RoPE.

    RoPE encodes *relative* position by rotating query/key pairs by an angle that
    is proportional to their absolute position, so that q_m . k_n depends only on (m - n).
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)          # (T, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)           # (T, D)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(
        self, seq_len: int, start_pos: int = 0, dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        end = start_pos + seq_len
        if end > self.max_seq_len:
            raise ValueError(
                f"Sequence position {end} exceeds max_seq_len={self.max_seq_len}"
            )
        cos = self.cos_cached[start_pos:end][None, None]  # (1,1,T,D)
        sin = self.sin_cached[start_pos:end][None, None]
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin


# --------------------------------------------------------------------------- #
# KV cache
# --------------------------------------------------------------------------- #
class KVCache:
    """
    Pre-allocated key/value cache.

    During generation, keys/values of already-processed tokens never change, so we
    store them and only compute attention inputs for the newest token(s).
    `length` is advanced ONCE per forward pass (by the decoder) after all layers
    have written their new keys/values.
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        max_len: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (n_layers, batch_size, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.length = 0

    def update(
        self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = k_new.size(2)
        start, end = self.length, self.length + t
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > {self.max_len}")
        self.k[layer_idx, :, :, start:end] = k_new
        self.v[layer_idx, :, :, start:end] = v_new
        return self.k[layer_idx, :, :, :end], self.v[layer_idx, :, :, :end]

    def advance(self, n_tokens: int) -> None:
        self.length += n_tokens

    def reset(self) -> None:
        self.length = 0


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, Hkv, S, D) -> (B, Hkv * n_rep, S, D): share each KV head across n_rep query heads."""
    if n_rep == 1:
        return x
    b, h_kv, s, d = x.shape
    return x[:, :, None].expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with
      * grouped-query attention (n_kv_heads <= n_heads) -> smaller KV cache
      * rotary positional embeddings
      * optional KV cache for incremental decoding
      * fused FlashAttention-style kernel through PyTorch SDPA
    """

    def __init__(self, cfg: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.use_flash = cfg.use_flash_attention

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        bsz, t, _ = x.shape

        q = self.q_proj(x).view(bsz, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)
            k, v = k.to(q.dtype), v.to(q.dtype)

        s = k.size(2)  # total key length (cached + new)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Which keys may each query attend to?
        #   t == s : plain causal mask (training / prefill without cache)
        #   t == 1 : the new token may see everything cached -> no mask needed
        #   else   : causal mask shifted by the number of cached tokens
        is_causal = (t == s) and t > 1
        attn_mask = None
        if not is_causal and t > 1:
            attn_mask = torch.ones(t, s, dtype=torch.bool, device=x.device).tril(diagonal=s - t)

        if self.use_flash:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:  # explicit (slow, but transparent) implementation
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                attn_mask = torch.ones(t, s, dtype=torch.bool, device=x.device).tril()
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            probs = F.softmax(scores.float(), dim=-1).type_as(q)
            probs = F.dropout(probs, p=self.dropout, training=self.training)
            out = probs @ v

        out = out.transpose(1, 2).contiguous().view(bsz, t, self.n_heads * self.head_dim)
        return self.resid_dropout(self.o_proj(out))


# --------------------------------------------------------------------------- #
# Feed-forward network
# --------------------------------------------------------------------------- #
class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN:  down( silu(gate(x)) * up(x) )."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        hidden = cfg.ffn_hidden_dim
        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.w_up = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.w_down = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# --------------------------------------------------------------------------- #
# Transformer block + decoder stack
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """One pre-norm decoder block: x + Attn(Norm(x)), then h + FFN(Norm(h))."""

    def __init__(self, cfg: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLUFeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerDecoder(nn.Module):
    """
    Stack of `n_layers` decoder blocks followed by a final RMSNorm.

    Input : embedded tokens         (batch, seq_len, d_model)
    Output: contextualised states   (batch, seq_len, d_model)
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.layers = nn.ModuleList(TransformerBlock(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        t = x.size(1)
        start_pos = kv_cache.length if kv_cache is not None else 0
        cos, sin = self.rope(t, start_pos, dtype=x.dtype)

        use_ckpt = self.cfg.gradient_checkpointing and self.training and kv_cache is None
        for layer in self.layers:
            if use_ckpt:
                x = checkpoint(layer, x, cos, sin, None, use_reentrant=False)
            else:
                x = layer(x, cos, sin, kv_cache)

        if kv_cache is not None:
            kv_cache.advance(t)
        return self.final_norm(x)