"""
model.py
=================

Decoder-only language model for text generation.

It wires together:

    token ids ──► Embedding ──► TransformerDecoder ──► LM head ──► logits over vocab
                  (this file)    (transformer.py)                   (this file)
 
and uses the logits to do next-token prediction:

  * training  : cross-entropy between logits[t] and the token that follows position t
  * inference : autoregressive sampling (temperature, top-k, top-p, repetition
                penalty, EOS handling) accelerated by a KV cache

Run `python model.py` for a tiny end-to-end demo (train + generate).
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import KVCache, TransformerConfig, TransformerDecoder


class LMOutput(NamedTuple):
    logits: torch.Tensor                  # (batch, seq_len, vocab)  or (batch, 1, vocab)
    loss: Optional[torch.Tensor] = None   # scalar, only when targets are given


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #
def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Down-weight tokens that already appeared (CTRL-style). penalty > 1 discourages repeats."""
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, generated)
    score = torch.where(score > 0, score / penalty, score * penalty)
    return logits.scatter(1, generated, score)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.size(-1))
    kth_best = torch.topk(logits, k, dim=-1).values[:, -1, None]
    return logits.masked_fill(logits < kth_best, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose probability mass >= p."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    remove = (cumulative - probs) > p          # always keeps the most likely token
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class DecoderOnlyLM(nn.Module):
    """Embedding layer + decoder-only transformer + LM head."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1) Embedding layer: token id -> d_model vector
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # 2) Transformer architecture (positions are injected inside attention via RoPE,
        #    so no learned positional embedding table is needed)
        self.decoder = TransformerDecoder(cfg)

        # 3) LM head: d_model vector -> logits over the vocabulary
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight  # weight tying

        self.apply(self._init_weights)
        # GPT-2 style: scale the residual-branch output projections by 1/sqrt(2 * n_layers)
        residual_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    # ----------------------------- init & utils ----------------------------- #
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights are counted once
        if exclude_embeddings:
            n -= self.tok_embeddings.weight.numel()
        return n

    def configure_optimizers(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple = (0.9, 0.95),
    ) -> torch.optim.AdamW:
        """AdamW with weight decay on matrices only (not on norms / biases)."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused or None)

    def new_kv_cache(self, batch_size: int) -> KVCache:
        p = self.tok_embeddings.weight
        return KVCache(
            n_layers=self.cfg.n_layers,
            batch_size=batch_size,
            n_kv_heads=self.cfg.n_kv_heads,
            max_len=self.cfg.max_seq_len,
            head_dim=self.cfg.head_dim,
            device=p.device,
            dtype=p.dtype,
        )

    # -------------------------------- forward -------------------------------- #
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        last_token_only: bool = False,
    ) -> LMOutput:
        """
        input_ids : (batch, seq_len) integer token ids
        targets   : (batch, seq_len) where targets[:, t] is the token that FOLLOWS
                    input_ids[:, t]. Use -100 to ignore a position in the loss.
        kv_cache  : pass a KVCache for incremental decoding
        last_token_only : compute logits just for the final position (saves memory
                    and compute at inference time)
        """
        if targets is not None and last_token_only:
            raise ValueError("last_token_only cannot be combined with targets")

        x = self.emb_dropout(self.tok_embeddings(input_ids))   # (B, T, d_model)
        h = self.decoder(x, kv_cache=kv_cache)                  # (B, T, d_model)
        if last_token_only:
            h = h[:, -1:, :]
        logits = self.lm_head(h)                                # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return LMOutput(logits=logits, loss=loss)

    # ------------------------------- generation ------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation (next-token prediction in a loop).

        1. Prefill  : run the whole prompt once, filling the KV cache.
        2. Decode   : repeatedly pick the next token from the last position's
                      logits, append it, and feed ONLY that token back in.

        temperature == 0 -> greedy decoding.
        Returns the prompt followed by the generated tokens: (batch, T + new).
        """
        was_training = self.training
        self.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids[None]
        max_len = self.cfg.max_seq_len
        if max_new_tokens >= max_len:
            raise ValueError("max_new_tokens must be smaller than max_seq_len")
        if input_ids.size(1) + max_new_tokens > max_len:   # keep the most recent context
            input_ids = input_ids[:, -(max_len - max_new_tokens):]

        bsz = input_ids.size(0)
        cache = self.new_kv_cache(bsz)
        generated = input_ids
        finished = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)

        logits = self(input_ids, kv_cache=cache, last_token_only=True).logits[:, -1]  # prefill

        for step in range(max_new_tokens):
            logits = apply_repetition_penalty(logits.float(), generated, repetition_penalty)

            if temperature == 0:
                next_token = logits.argmax(dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    logits = top_k_filter(logits, top_k)
                if top_p is not None and top_p < 1.0:
                    logits = top_p_filter(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished, torch.full_like(next_token, eos_token_id), next_token
                )
                finished |= next_token == eos_token_id

            generated = torch.cat([generated, next_token[:, None]], dim=1)
            if finished.all() or step == max_new_tokens - 1:
                break
            logits = self(next_token[:, None], kv_cache=cache, last_token_only=True).logits[:, -1]

        self.train(was_training)
        return generated

    # ------------------------------ save / load ------------------------------ #
    def save_pretrained(self, path: str) -> None:
        torch.save({"config": self.cfg.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> "DecoderOnlyLM":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(TransformerConfig.from_dict(ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model


# --------------------------------------------------------------------------- #
# Demo: train a tiny character-level model, then generate text with it
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "a transformer predicts the next token from all previous tokens. "
        "attention lets every token look at the tokens that came before it. "
    ) * 40

    # --- minimal character-level tokenizer ---
    chars = sorted(set(corpus))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    data = torch.tensor(encode(corpus), dtype=torch.long)

    cfg = TransformerConfig(
        vocab_size=len(chars), max_seq_len=128, d_model=128, n_layers=4,
        n_heads=4, n_kv_heads=2, dropout=0.0,
    )
    model = DecoderOnlyLM(cfg).to(device)
    print(f"parameters: {model.num_parameters() / 1e6:.2f}M  |  device: {device}")

    optimizer = model.configure_optimizers(lr=3e-3, weight_decay=0.01)
    block, batch = 64, 32
    steps = 300
    model.train()
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(data) - block - 1, (batch,))
        x = torch.stack([data[i : i + block] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + block + 1] for i in ix]).to(device)  # shifted by one

        lr = 3e-3 * 0.5 * (1 + math.cos(math.pi * step / steps))  # cosine decay
        for g in optimizer.param_groups:
            g["lr"] = lr
        loss = model(x, targets=y).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    prompt = "attention lets"
    ids = torch.tensor([encode(prompt)], device=device)
    out = model.generate(ids, max_new_tokens=100, temperature=0.7, top_k=10, top_p=0.9)
    print("\n--- generated ---")
    print(decode(out[0].tolist()))