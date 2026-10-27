# Pure PyTorch attention over the paged KV cache, used where FlashAttention
# and Triton are unavailable (Apple Silicon / MPS, CPU).
import torch
import torch.nn.functional as F


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    slots = slot_mapping.long()
    k_cache.view(-1, *k_cache.shape[2:])[slots] = key
    v_cache.view(-1, *v_cache.shape[2:])[slots] = value


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float):
    # q: (B, Lq, H, D), k/v: (B, Lk, Hkv, D), mask: broadcastable to (B, H, Lq, Lk)
    num_groups = q.size(2) // k.size(2)
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=2)
        v = v.repeat_interleave(num_groups, dim=2)
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, scale=scale)
    return o.transpose(1, 2)


def prefill_attention(q, k, v, k_cache, v_cache, context, scale):
    cu_seqlens_q = context.cu_seqlens_q.tolist()
    cu_seqlens_k = context.cu_seqlens_k.tolist()
    o = torch.empty_like(q)
    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        seqlen_q = q_end - q_start
        seqlen_k = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
        if context.block_tables is None:
            k_i, v_i = k[q_start:q_end], v[q_start:q_end]
        else:    # prefix cache / chunked prefill: earlier tokens live in the cache
            block_size = k_cache.size(1)
            block_ids = context.block_tables[i, :(seqlen_k + block_size - 1) // block_size].long()
            k_i = k_cache[block_ids].flatten(0, 1)[:seqlen_k]
            v_i = v_cache[block_ids].flatten(0, 1)[:seqlen_k]
        # queries are the last seqlen_q positions of the sequence
        mask = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device).tril(seqlen_k - seqlen_q)
        o[q_start:q_end] = sdpa(q[q_start:q_end].unsqueeze(0), k_i.unsqueeze(0), v_i.unsqueeze(0), mask, scale)[0]
    return o


def decode_attention(q, k_cache, v_cache, context, scale):
    block_ids = context.block_tables.long().clamp_min(0)    # padding (-1) is masked out below
    k = k_cache[block_ids].flatten(1, 2)
    v = v_cache[block_ids].flatten(1, 2)
    positions = torch.arange(k.size(1), device=q.device)
    mask = (positions[None, :] < context.context_lens[:, None])[:, None, None, :]
    return sdpa(q.unsqueeze(1), k, v, mask, scale).squeeze(1)
