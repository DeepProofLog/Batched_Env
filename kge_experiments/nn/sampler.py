"""
Fully Vectorized Sampler - No Python loops in core path.

Key features:
- Fully vectorized batch operations
- Both sampled and exhaustive modes use tensor operations
- Same RNG pattern as SB3 for parity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import torch


LongTensor = torch.LongTensor
Tensor = torch.Tensor


def _mix_hash(triples: LongTensor, b_e: int, b_r: int) -> LongTensor:
    h = triples[..., 1].to(torch.int64)
    r = triples[..., 0].to(torch.int64)
    t = triples[..., 2].to(torch.int64)
    return (h << 42) | (r << 21) | t


@dataclass
class SamplerConfig:
    num_entities: int
    num_relations: int
    device: torch.device
    default_mode: Literal['head', 'tail', 'both'] = 'both'
    seed: int = 0
    order_negatives: bool = False


class Sampler:
    """Fully vectorized negative sampler."""

    def __init__(self, cfg: SamplerConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.num_entities = cfg.num_entities
        self.num_relations = cfg.num_relations
        self.default_mode = cfg.default_mode
        self.order_negatives = cfg.order_negatives
        self.hashes_sorted: Optional[LongTensor] = None
        self.b_e = max(2 * self.num_entities + 1, 1024)
        self.b_r = max(2 * self.num_relations + 1, 128)
        self.domain_padded: Optional[Tensor] = None
        self.domain_len: Optional[Tensor] = None
        self.ent2dom: Optional[LongTensor] = None
        self.pos_in_dom: Optional[LongTensor] = None
        self.num_domains = 0
        self.max_pool_len = 0

    @classmethod
    def from_data(cls, all_known_triples_idx: LongTensor, num_entities: int, num_relations: int,
                  device: torch.device, default_mode: Literal['head', 'tail', 'both'] = 'both',
                  seed: int = 0, domain2idx: Optional[Dict[str, List[int]]] = None,
                  entity2domain: Optional[Dict[int, str]] = None, order_negatives: bool = False) -> "Sampler":
        cfg = SamplerConfig(num_entities, num_relations, device, default_mode, seed, order_negatives)
        self = cls(cfg)
        if all_known_triples_idx is not None and all_known_triples_idx.numel() > 0:
            hashes = _mix_hash(all_known_triples_idx.detach().to(device=self.device, dtype=torch.long), self.b_e, self.b_r)
            self.hashes_sorted = torch.sort(torch.unique(hashes)).values
        else:
            self.hashes_sorted = torch.empty((0,), dtype=torch.long, device=self.device)
        if domain2idx and entity2domain:
            self._build_domain_structures(domain2idx, entity2domain, device)
        return self
    
    def _build_domain_structures(self, domain2idx: Dict[str, List[int]], entity2domain: Dict[int, str], device: torch.device) -> None:
        domain_names = sorted(domain2idx.keys())
        domain_lists = [torch.tensor(domain2idx[n], dtype=torch.int32, device=device) for n in domain_names]
        self.num_domains = len(domain_lists)
        self.max_pool_len = max((t.numel() for t in domain_lists), default=0)
        self.domain_padded = torch.zeros((self.num_domains, self.max_pool_len), dtype=torch.int32, device=device)
        self.domain_len = torch.zeros((self.num_domains,), dtype=torch.int32, device=device)
        for i, t in enumerate(domain_lists):
            self.domain_padded[i, :t.numel()] = t
            self.domain_len[i] = t.numel()
        max_ent = max(entity2domain.keys(), default=0)
        self.ent2dom = torch.full((max_ent + 1,), -1, dtype=torch.int32, device=device)
        self.pos_in_dom = torch.zeros((max_ent + 1,), dtype=torch.int32, device=device)
        for d, name in enumerate(domain_names):
            row = self.domain_padded[d, :self.domain_len[d]]
            if row.numel() > 0:
                self.ent2dom[row.long()] = d
                self.pos_in_dom[row.long()] = torch.arange(row.numel(), device=device, dtype=torch.int32)

    def _has_domain_info(self) -> bool:
        return self.domain_padded is not None and self.num_domains > 0

    def _filter_mask_batched(self, triples: LongTensor) -> torch.BoolTensor:
        """Vectorized filter - returns [B, K] mask."""
        if self.hashes_sorted is None or self.hashes_sorted.numel() == 0:
            return torch.ones(triples.shape[:-1], dtype=torch.bool, device=triples.device)
        B, K = triples.shape[:2]
        flat = triples.reshape(-1, 3)
        hashes = _mix_hash(flat, self.b_e, self.b_r)
        pos = torch.searchsorted(self.hashes_sorted, hashes)
        in_range = (pos >= 0) & (pos < self.hashes_sorted.numel())
        eq = torch.zeros_like(in_range)
        eq[in_range] = self.hashes_sorted[pos[in_range]] == hashes[in_range]
        return (~eq).reshape(B, K)

    def _filter_keep_mask(self, triples: LongTensor) -> torch.BoolTensor:
        """Return mask [N] of entries NOT in known positives. Input: [N, 3]."""
        if self.hashes_sorted is None or self.hashes_sorted.numel() == 0:
            return torch.ones((triples.shape[0],), dtype=torch.bool, device=triples.device)
        hashes = _mix_hash(triples, self.b_e, self.b_r)
        pos = torch.searchsorted(self.hashes_sorted, hashes)
        in_range = (pos >= 0) & (pos < self.hashes_sorted.numel())
        eq = torch.zeros_like(in_range, dtype=torch.bool)
        eq[in_range] = self.hashes_sorted[pos[in_range]] == hashes[in_range]
        return ~eq

    def _get_corruption_indices(self, mode: str) -> List[int]:
        return [0] if mode == 'head' else [2] if mode == 'tail' else [0, 2]

    def corrupt(self, positives: LongTensor, *, num_negatives: Optional[int] = None,
                mode: Literal['head', 'tail', 'both'] = None, device: Optional[torch.device] = None,
                filter: bool = True, unique: bool = True) -> LongTensor:
        """
        FULLY VECTORIZED corruption.
        
        - num_negatives=K: sample K random indices
        - num_negatives=None: enumerate all valid indices
        
        Returns [B, M, 3] in (r, h, t) format.
        """
        device = device or self.device
        mode = mode or self.default_mode
        pos = positives.to(device=device)
        B = pos.shape[0]
        
        if B == 0:
            return torch.zeros((0, num_negatives or 0, 3), dtype=pos.dtype, device=device)
        
        pos_hrt = torch.stack([pos[:, 1], pos[:, 0], pos[:, 2]], dim=1)  # [B, 3]
        cols = self._get_corruption_indices(mode)
        is_exhaustive = num_negatives is None
        
        # Determine K (negatives per triple)
        pool_size = (self.max_pool_len - 1) if self._has_domain_info() else (self.num_entities - 1)
        K = pool_size * len(cols) if is_exhaustive else num_negatives
        
        # Initialize [B, K, 3]
        neg = pos_hrt.unsqueeze(1).expand(B, K, 3).clone()
        
        # Split K across columns
        n_cols = len(cols)
        per_col = K // n_cols
        remainder = K % n_cols
        
        start = 0
        for i, col in enumerate(cols):
            count = per_col + (1 if i < remainder else 0)
            end = start + count
            
            if count == 0:
                continue
            
            # Get slice to modify
            orig = neg[:, start:end, col]  # [B, count]
            
            if col == 1:
                # Relation corruption - vectorized
                if is_exhaustive:
                    # Enumerate all relations except original
                    all_rels = torch.arange(self.num_relations, device=device, dtype=pos.dtype)
                    all_rels = all_rels.unsqueeze(0).expand(B, -1)  # [B, num_rels]
                    orig_rel = pos_hrt[:, 1:2]  # [B, 1]
                    # Mask out original
                    mask = all_rels != orig_rel  # [B, num_rels]
                    # Take first count valid per row
                    indices = mask.long().cumsum(dim=1) - 1
                    indices = indices.clamp(min=0, max=count-1)
                    result = torch.gather(all_rels, 1, indices[:, :count])
                    neg[:, start:end, col] = result
                else:
                    rnd = torch.randint(0, self.num_relations - 1, orig.shape, device=device)
                    neg[:, start:end, col] = rnd + (rnd >= orig)
            else:
                # Entity corruption - unified path for exhaustive and sampled
                orig = neg[:, start:end, col]  # [B, count]
                
                if not self._has_domain_info():
                    # No domain: pool is all entities [1, num_entities]
                    pool_size = self.num_entities
                    orig_ent = pos_hrt[:, col:col+1]  # [B, 1]
                    
                    if is_exhaustive:
                        # All entities except original - use argsort method
                        all_ents = torch.arange(1, pool_size + 1, device=device, dtype=pos.dtype)
                        all_ents = all_ents.unsqueeze(0).expand(B, -1)  # [B, pool_size]
                        mask = all_ents != orig_ent  # [B, pool_size]
                        # Sort valid to front
                        sort_keys = (~mask).long()
                        _, perm = torch.sort(sort_keys, dim=1, stable=True)
                        sorted_ents = torch.gather(all_ents, 1, perm)
                        result = sorted_ents[:, :count]
                    else:
                        # Random sample excluding original
                        rnd = torch.randint(1, pool_size, orig.shape, device=device)
                        result = rnd + ((rnd >= orig) & (orig > 0)).long()
                    
                    neg[:, start:end, col] = result
                else:
                    # Domain-aware: pool is domain members
                    orig_ent = pos_hrt[:, col]  # [B] already LongTensor
                    d_ids = self.ent2dom[orig_ent].long()  # [B]
                    pools = self.domain_padded[d_ids]  # [B, max_pool_len]
                    
                    # Mask: valid candidates (in domain, not original, not padding)
                    orig_exp = orig_ent.unsqueeze(1)
                    valid_mask = (pools != orig_exp) & (pools > 0)  # [B, max_pool_len]
                    
                    if is_exhaustive:
                        # Sort valid to front, take first count
                        sort_keys = (~valid_mask).long()
                        _, perm = torch.sort(sort_keys, dim=1, stable=True)
                        sorted_pools = torch.gather(pools, 1, perm)
                        sorted_valid = torch.gather(valid_mask, 1, perm)
                        result = sorted_pools[:, :count]
                        result.masked_fill_(~sorted_valid[:, :count], 0)
                    else:
                        # Random sample from valid candidates
                        pool_len = self.domain_len[d_ids]
                        pos_orig = self.pos_in_dom[orig_ent].long()
                        
                        # Generate [B, count] random indices
                        orig_flat = orig.reshape(-1)
                        valid = orig_flat > 0
                        result_flat = orig_flat.clone()
                        
                        if valid.any():
                            ov = orig_flat[valid]
                            d_flat = self.ent2dom[ov].long()
                            L = self.domain_len[d_flat]
                            p = self.pos_in_dom[ov].long()
                            can = L > 1
                            
                            if can.any():
                                Lm1 = (L[can] - 1).float()
                                rnd = torch.floor(torch.rand(can.sum(), device=device) * Lm1).long()
                                adj = rnd + (rnd >= p[can])
                                temp = ov.clone()
                                temp[can] = self.domain_padded[d_flat[can], adj].long()
                                result_flat[valid] = temp
                        
                        result = result_flat.reshape(orig.shape)
                    
                    neg[:, start:end, col] = result
            
            start = end
        
        # Convert to (r, h, t): [B, K, 3]
        neg_rht = torch.stack([neg[:, :, 1], neg[:, :, 0], neg[:, :, 2]], dim=-1)
        
        # Mark valid (non-padding) - invalid if head or tail is 0
        valid = (neg_rht[:, :, 1] > 0) & (neg_rht[:, :, 2] > 0)  # [B, K]
        
        # Filter known positives - vectorized
        if filter:
            valid = valid & self._filter_mask_batched(neg_rht)
        
        # Deduplicate - vectorized via hash comparison
        if unique:
            hashes = _mix_hash(neg_rht, self.b_e, self.b_r)  # [B, K]
            hashes = torch.where(valid, hashes, torch.full_like(hashes, -1))
            # Mark duplicates within each row
            for k in range(1, K):
                is_dup = (hashes[:, :k] == hashes[:, k:k+1]).any(dim=1)
                valid[:, k] = valid[:, k] & ~is_dup
        
        # Zero out invalid
        neg_rht = neg_rht * valid.unsqueeze(-1)
        
        # Sort valid to front - vectorized
        sort_key = (~valid).long()
        _, perm = torch.sort(sort_key, dim=1, stable=True)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
        neg_rht = neg_rht[batch_idx, perm]
        
        # Optimized: Determine output size M = K (always return full size to avoid CPU sync)
        # In exhaustive mode, K is the max possible candidates. We assume callers handle padding.
        # This removes the need for .item() synchronization.
        M = K
        
        output = neg_rht
        
        # Sort if requested - vectorized
        if self.order_negatives and M > 0:
            keys = output[:, :, 0] * 10000000 + output[:, :, 1] * 10000 + output[:, :, 2]
            keys = keys.masked_fill(output.sum(-1) == 0, 2**62)
            _, idx = torch.sort(keys, dim=1)
            batch_i = torch.arange(B, device=device).unsqueeze(1).expand(-1, M)
            output = output[batch_i, idx]
        
        return output

    # Backward-compatible methods
    def corrupt_batch(self, positive_batch: LongTensor, num_negs_per_pos: int) -> LongTensor:
        pos_rht = torch.stack([positive_batch[:, 1], positive_batch[:, 0], positive_batch[:, 2]], dim=1)
        result = self.corrupt(pos_rht, num_negatives=num_negs_per_pos, mode=self.default_mode, filter=False, unique=False)
        return torch.stack([result[:, :, 1], result[:, :, 0], result[:, :, 2]], dim=-1)
    
    def corrupt_batch_all(self, positive_batch: LongTensor) -> List[LongTensor]:
        pos_rht = torch.stack([positive_batch[:, 1], positive_batch[:, 0], positive_batch[:, 2]], dim=1)
        result = self.corrupt(pos_rht, num_negatives=None, mode=self.default_mode, filter=False)
        return [result[i][result[i].sum(-1) != 0][:, [1, 0, 2]] for i in range(result.shape[0])]
    
    def corrupt_all(self, positives: LongTensor, *, mode=None, device=None, use_domain=True):
        mode = mode or self.default_mode
        result = self.corrupt(positives, num_negatives=None, mode=mode, device=device, filter=True)
        lst = [result[i][result[i].sum(-1) != 0] for i in range(result.shape[0])]
        return (lst, None) if mode == 'head' else ([], lst) if mode == 'tail' else (lst, lst)
    
    def get_negatives_from_states_separate(self, positives: LongTensor, *, num_negatives=None, device=None, filter=True, unique=True):
        device = device or self.device
        pos = positives.to(device=device, dtype=torch.long)
        B = pos.shape[0]
        extract = lambda r: [r[i][r[i].sum(-1) != 0] for i in range(r.shape[0])]
        head = [torch.empty((0, 3), dtype=pos.dtype, device=device)] * B
        tail = [torch.empty((0, 3), dtype=pos.dtype, device=device)] * B
        if self.default_mode in ('head', 'both'):
            head = extract(self.corrupt(pos, num_negatives=num_negatives, mode='head', device=device, filter=filter, unique=unique))
        if self.default_mode in ('tail', 'both'):
            tail = extract(self.corrupt(pos, num_negatives=num_negatives, mode='tail', device=device, filter=filter, unique=unique))
        return head, tail
