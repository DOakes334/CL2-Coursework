"""Encoder-decoder Transformer for character-level G2P."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Core modules

class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        inner_dim = int(hidden_dim * 2 / 3)
        self.gate_proj   = nn.Linear(input_dim, inner_dim, bias=False)
        self.value_proj  = nn.Linear(input_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, input_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.silu(self.gate_proj(X)) * self.value_proj(X))


class LayerNorm(nn.Module):
    """Layer normalisation."""
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias   = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = torch.var(x, dim=-1, correction=0, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class LexicalEmbedding(nn.Module):
    """Token embedding."""
    def __init__(self, vocab_size: int, input_dim: int, padding_idx: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, input_dim, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x.long())


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute position embedding."""
    def __init__(self, d_model: int, max_len: int = 514, padding_idx: int = 1):
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.position_embeddings = nn.Embedding(max_len + 1, d_model, padding_idx=0)
        with torch.no_grad():
            self.position_embeddings.weight[0].zero_()

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != self.padding_idx).long()
        position_ids = torch.cumsum(mask, dim=1) * mask
        return x + self.position_embeddings(position_ids)


# RoPE

class RotaryEmbedding(nn.Module):
    """Rotary position embedding."""

    def __init__(self, d_head: int, max_len: int = 512, base: int = 10000):
        super().__init__()
        self.d_head = d_head
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int) -> None:
        t = torch.arange(max_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :])

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        if seq_len > self.cos_cache.shape[2]:
            self._build_cache(seq_len * 2)

        cos = self.cos_cache[:, :, :seq_len, :]
        sin = self.sin_cache[:, :, :seq_len, :]

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# GQA

class GroupedQueryAttention(nn.Module):
    """Grouped-query attention with RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        max_len: int = 512,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_groups   = n_heads // self.n_kv_heads
        self.d_head     = d_model // n_heads

        assert n_heads % self.n_kv_heads == 0, \
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

        kv_dim = self.n_kv_heads * self.d_head
        self.q_proj  = nn.Linear(d_model, d_model,  bias=False)
        self.k_proj  = nn.Linear(d_model, kv_dim,   bias=False)
        self.v_proj  = nn.Linear(d_model, kv_dim,   bias=False)
        self.out_proj = nn.Linear(d_model, d_model,  bias=False)

        self.rope    = RotaryEmbedding(self.d_head, max_len)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        Q = self.q_proj(query).unflatten(-1, (self.n_heads,    self.d_head)).transpose(1, 2)
        K = self.k_proj(key  ).unflatten(-1, (self.n_kv_heads, self.d_head)).transpose(1, 2)
        V = self.v_proj(value).unflatten(-1, (self.n_kv_heads, self.d_head)).transpose(1, 2)

        Q, K = self.rope(Q, K)

        # Grouped KV heads
        if self.n_groups > 1:
            K = K.repeat_interleave(self.n_groups, dim=1)
            V = V.repeat_interleave(self.n_groups, dim=1)

        # Masks
        bias = None
        if key_padding_mask is not None:
            bias = torch.zeros(B, 1, T_q, T_k, device=query.device, dtype=query.dtype)
            bias = bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        if attn_mask is not None:
            bias = attn_mask if bias is None else bias + attn_mask

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=bias, dropout_p=dropout_p)

        out = out.transpose(1, 2).flatten(start_dim=-2)
        return self.out_proj(out)


# MHA

class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.query_projection  = nn.Linear(d_model, d_model, bias=False)
        self.key_projection    = nn.Linear(d_model, d_model, bias=False)
        self.value_projection  = nn.Linear(d_model, d_model, bias=False)
        self.final_projections = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        causal_mask: bool = False,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        def split_heads(t: torch.Tensor, T: int) -> torch.Tensor:
            return t.unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2)

        Q = split_heads(self.query_projection(query),  T_q)
        K = split_heads(self.key_projection(key),      T_k)
        V = split_heads(self.value_projection(value),  T_k)

        # Masks
        bias = None
        if causal_mask:
            bias = torch.triu(
                torch.full((T_q, T_k), float("-inf"), device=query.device, dtype=query.dtype),
                diagonal=1,
            )
        if key_padding_mask is not None:
            pad_bias = torch.zeros(B, 1, T_q, T_k, device=query.device, dtype=query.dtype)
            pad_bias = pad_bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            bias = pad_bias if bias is None else bias + pad_bias

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=bias, dropout_p=dropout_p)

        out = out.transpose(1, 2).flatten(start_dim=-2)
        return self.final_projections(out)


# Encoder layer

class EncoderLayer(nn.Module):
    """Pre-LN encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_drop: float = 0.0,
    ):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        self.ff        = SwiGLU(d_model, d_ff)
        self.norm1     = LayerNorm(d_model)
        self.norm2     = LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)
        self.layer_drop = layer_drop

    def forward(self, x: torch.Tensor, src_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Stochastic depth
        if self.training and self.layer_drop > 0.0:
            if torch.rand(1).item() < self.layer_drop:
                return x

        # Pre-LN residuals
        normed = self.norm1(x)
        x = x + self.drop(self.self_attn(normed, normed, normed, key_padding_mask=src_pad_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# Decoder layer

class DecoderLayer(nn.Module):
    """Pre-LN decoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_drop: float = 0.0,
    ):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff         = SwiGLU(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
        self.layer_drop = layer_drop

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        mem_pad:  torch.Tensor | None = None,
        tgt_pad:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training and self.layer_drop > 0.0:
            if torch.rand(1).item() < self.layer_drop:
                return x

        # Self-attention
        normed = self.norm1(x)
        x = x + self.drop(
            self.self_attn(normed, normed, normed, causal_mask=True, key_padding_mask=tgt_pad)
        )
        # Cross-attention
        normed2 = self.norm2(x)
        x = x + self.drop(
            self.cross_attn(normed2, memory, memory, causal_mask=False, key_padding_mask=mem_pad)
        )
        # Feed-forward
        x = x + self.drop(self.ff(self.norm3(x)))
        return x


# Full model

class G2PTransformer(nn.Module):
    """Encoder-decoder Transformer."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:    int = 384,
        n_heads:    int = 6,
        n_kv_heads: int | None = None,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        d_ff:       int = 1024,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
        dropout:    float = 0.15,
        layer_drop: float = 0.1,
        pad_idx:    int = 0,
    ):
        super().__init__()
        self.d_model       = d_model
        self.pad_idx       = pad_idx
        self.tgt_vocab_size = tgt_vocab_size
        n_kv_heads = n_kv_heads or max(1, n_heads // 3)

        # Embeddings
        self.src_embed = LexicalEmbedding(src_vocab_size, d_model, pad_idx)
        self.tgt_embed = LexicalEmbedding(tgt_vocab_size, d_model, pad_idx)
        self.tgt_pos = LearnedPositionalEmbedding(d_model, max_tgt_len, padding_idx=pad_idx)

        self.embed_drop = nn.Dropout(dropout)

        # Encoder
        enc_drops = [layer_drop * i / max(n_enc_layers - 1, 1) for i in range(n_enc_layers)]
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout, enc_drops[i])
            for i in range(n_enc_layers)
        ])
        self.encoder_norm = LayerNorm(d_model)

        # Decoder
        dec_drops = [layer_drop * i / max(n_dec_layers - 1, 1) for i in range(n_dec_layers)]
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, dec_drops[i])
            for i in range(n_dec_layers)
        ])
        self.decoder_norm = LayerNorm(d_model)

        # Weight tying
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.output_proj.weight = self.tgt_embed.emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1 and "position_embeddings" not in name:
                nn.init.xavier_uniform_(p)
            elif "emb" in name or "position" in name:
                nn.init.normal_(p, std=self.d_model ** -0.5)

    def encode(
        self,
        src: torch.Tensor,
        src_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_drop(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_pad_mask=src_pad_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        tgt:     torch.Tensor,
        memory:  torch.Tensor,
        mem_pad: torch.Tensor | None = None,
        tgt_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.embed_drop(self.tgt_pos(x, tgt))
        for layer in self.decoder_layers:
            x = layer(x, memory, mem_pad=mem_pad, tgt_pad=tgt_pad)
        x = self.decoder_norm(x)
        return self.output_proj(x)

    def forward(
        self,
        src:     torch.Tensor,
        tgt:     torch.Tensor,
        src_pad: torch.Tensor | None = None,
        tgt_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_pad)
        return self.decode(tgt, memory, mem_pad=src_pad, tgt_pad=tgt_pad)

    @torch.no_grad()
    def greedy_decode(
        self,
        src:     torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 128,
    ) -> list[list[int]]:
        """Greedy decoding."""
        self.eval()
        B, device = src.size(0), src.device
        src_pad = (src == self.pad_idx)
        memory  = self.encode(src, src_pad)

        ys       = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results  = [[] for _ in range(B)]

        for _ in range(max_len):
            logits   = self.decode(ys, memory, mem_pad=src_pad)
            next_tok = logits[:, -1, :].argmax(-1)
            for i in range(B):
                if not finished[i]:
                    if next_tok[i].item() == eos_idx:
                        finished[i] = True
                    else:
                        results[i].append(next_tok[i].item())
            if finished.all():
                break
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)

        return results

    @torch.no_grad()
    def beam_decode(
        self,
        src:           torch.Tensor,
        bos_idx:       int,
        eos_idx:       int,
        beam_size:     int = 4,
        max_len:       int = 128,
        length_penalty: float = 0.6,
    ) -> list[list[int]]:
        """Beam search with length normalisation."""
        assert src.size(0) == 1, "beam_decode takes one example at a time"
        self.eval()
        device  = src.device
        src_pad = (src == self.pad_idx)
        memory  = self.encode(src, src_pad)

        candidates: list[tuple[float, list[int]]] = [(0.0, [bos_idx])]
        completed:  list[tuple[float, list[int]]] = []

        for _ in range(max_len):
            if not candidates:
                break
            expanded = []
            for log_prob, tokens in candidates:
                ys    = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                logits = self.decode(ys, memory, mem_pad=src_pad)
                lp    = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                top_lp, top_idx = lp.topk(beam_size)
                for lp_val, tok in zip(top_lp.tolist(), top_idx.tolist()):
                    new_score  = log_prob + lp_val
                    new_tokens = tokens + [tok]
                    if tok == eos_idx:
                        norm = ((5 + len(new_tokens)) / 6) ** length_penalty
                        completed.append((new_score / norm, new_tokens[1:]))  # strip BOS
                    else:
                        expanded.append((new_score, new_tokens))

            expanded.sort(key=lambda t: t[0], reverse=True)
            candidates = expanded[:beam_size]

        if not completed:
            candidates.sort(key=lambda t: t[0], reverse=True)
            return [candidates[0][1][1:]] if candidates else [[]]

        completed.sort(key=lambda t: t[0], reverse=True)
        return [completed[0][1]]


# Checkpoint averaging

def average_checkpoints(checkpoint_paths: list, model: G2PTransformer) -> G2PTransformer:
    """Average checkpoint weights."""
    avg_state = None
    n = len(checkpoint_paths)
    for path in checkpoint_paths:
        state = torch.load(path, map_location="cpu")["model_state_dict"]
        if avg_state is None:
            avg_state = {k: v.float().clone() / n for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float() / n

    orig = torch.load(checkpoint_paths[0], map_location="cpu")["model_state_dict"]
    for k in avg_state:
        avg_state[k] = avg_state[k].to(orig[k].dtype)
    model.load_state_dict(avg_state)
    print(f"[Avg] Averaged {n} checkpoints.")
    return model
