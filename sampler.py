"""
Refactored Sampler - Clean, maintainable, and efficient.

Unified, index-only negative sampler with optional domains and filtering.

Key improvements over original:
- Single, unified interface
- Vectorized operations throughout
- Cleaner filtering with hashing
- Optional domain constraints
- Better performance with EMA-based oversampling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import torch


LongTensor = torch.LongTensor
Tensor = torch.Tensor


def _mix_hash(triples: LongTensor, b_e: int, b_r: int) -> LongTensor:
    """
    Compute a fast 64-bit mixed hash for (h, r, t) triples (indices start at 1).
    Uses large odd multipliers and xor mix; stays in signed int64 domain.
    
    Args:
        triples: [*, 3] long tensor
        b_e: Entity hash bucket size (for compatibility)
        b_r: Relation hash bucket size (for compatibility)
    
    Returns:
        hashes: [*] int64 tensor
    """
    h = triples[..., 1].to(torch.int64)  # head (column 1)
    r = triples[..., 0].to(torch.int64)  # relation (column 0)
    t = triples[..., 2].to(torch.int64)  # tail (column 2)
    
    # constants (64-bit odd primes)
    A = torch.tensor(1469598103934665603, dtype=torch.int64, device=triples.device)
    B = torch.tensor(1099511628211, dtype=torch.int64, device=triples.device)
    x = (h + A) ^ ((r + B) * 14029467366897019727) ^ ((t + A) * 11400714819323198485)
    
    # finalize (xorshift-multiply)
    x ^= (x >> 33)
    x *= 14029467366897019727
    x ^= (x >> 29)
    x *= 11400714819323198485
    x ^= (x >> 32)
    return x


@dataclass
class SamplerConfig:
    """Configuration for sampler."""
    num_entities: int
    num_relations: int
    device: torch.device
    default_mode: Literal['head', 'tail', 'both'] = 'both'
    seed: int = 0


class Sampler:
    """
    Unified, index-only negative sampler with optional domains and filtering.

    Public API:
      - corrupt(positives, num_negatives, mode='both', device=None, filter=True, unique=True, use_domain=True)
      - corrupt_all(positives, mode='both', device=None, use_domain=True)

    Internals:
      - Filterer: sorted unique 64-bit hashes of known positives (CPU/GPU)
      - Domain: optional ragged allowed-sets per relation for heads/tails
      - Generator: vectorized torch.randint with optional per-row domain draws
    """

    def __init__(self, cfg: SamplerConfig) -> None:
        """Initialize sampler with configuration."""
        self.cfg = cfg
        self.device = cfg.device
        self.num_entities = cfg.num_entities
        self.num_relations = cfg.num_relations
        self.default_mode = cfg.default_mode

        self.rng = torch.Generator(device='cpu')
        self.rng.manual_seed(cfg.seed)

        # Filterer buffers
        self.hashes_sorted: Optional[LongTensor] = None  # [U]
        self.b_e: int = max(2 * self.num_entities + 1, 1024)  # kept for compatibility/debug
        self.b_r: int = max(2 * self.num_relations + 1, 128)

        # Domains (ragged; index-only)
        self.allowed_heads_per_rel: List[Optional[LongTensor]] = []
        self.allowed_tails_per_rel: List[Optional[LongTensor]] = []

        # Accept-rate EMA for overshoot
        self.ema_accept: float = 0.75
        self.ema_alpha: float = 0.1

    # -----------------------------
    # Builders
    # -----------------------------
    @classmethod
    def from_data(
        cls,
        all_known_triples_idx: LongTensor,    # [T,3], CPU or GPU
        num_entities: int,
        num_relations: int,
        device: torch.device,
        default_mode: Literal['head', 'tail', 'both'] = 'both',
        seed: int = 0,
        domain_heads: Optional[Dict[int, LongTensor]] = None,
        domain_tails: Optional[Dict[int, LongTensor]] = None,
    ) -> "Sampler":
        """
        Create sampler from data.
        
        Args:
            all_known_triples_idx: All known positive triples
            num_entities: Number of entities
            num_relations: Number of relations
            device: Target device
            default_mode: Default corruption mode
            seed: Random seed
            domain_heads: Optional domain constraints for heads
            domain_tails: Optional domain constraints for tails
        
        Returns:
            Sampler instance
        """
        cfg = SamplerConfig(num_entities, num_relations, device, default_mode, seed)
        self = cls(cfg)
        
        # Build filter hashes (unique + sorted) on CPU
        if all_known_triples_idx is not None and all_known_triples_idx.numel() > 0:
            cpu = all_known_triples_idx.detach().to('cpu').to(torch.long)
            hashes = _mix_hash(cpu, self.b_e, self.b_r)
            uniq = torch.unique(hashes)
            self.hashes_sorted = torch.sort(uniq).values.to(device=self.device)
        else:
            self.hashes_sorted = torch.empty((0,), dtype=torch.long, device=self.device)

        # Domains
        max_rel = num_relations + 1
        self.allowed_heads_per_rel = [None] * max_rel
        self.allowed_tails_per_rel = [None] * max_rel
        if domain_heads is not None:
            for r, vec in domain_heads.items():
                self.allowed_heads_per_rel[r] = vec.to(dtype=torch.long, device='cpu')
        if domain_tails is not None:
            for r, vec in domain_tails.items():
                self.allowed_tails_per_rel[r] = vec.to(dtype=torch.long, device='cpu')
        
        return self

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _filter_keep_mask(self, triples: LongTensor) -> torch.BoolTensor:
        """Return mask of entries NOT present in known positives. triples: [N,3]."""
        if self.hashes_sorted is None or self.hashes_sorted.numel() == 0:
            return torch.ones((triples.shape[0],), dtype=torch.bool, device=triples.device)
        
        hashes = _mix_hash(triples, self.b_e, self.b_r)
        pos = torch.searchsorted(self.hashes_sorted, hashes)
        in_range = (pos >= 0) & (pos < self.hashes_sorted.numel())
        eq = torch.zeros_like(in_range, dtype=torch.bool)
        eq[in_range] = self.hashes_sorted[pos[in_range]] == hashes[in_range]
        return ~eq

    def _draw_uniform_entities(self, size: int, device: torch.device) -> LongTensor:
        """Draw uniform random entities."""
        # Use CPU generator then move to device for compatibility
        result = torch.randint(1, self.num_entities + 1, (size,), generator=self.rng, device='cpu', dtype=torch.long)
        return result.to(device=device, non_blocking=True)

    def _draw_from_domain(self, rel_ids: LongTensor, head: bool, device: torch.device) -> LongTensor:
        """Per-row domain sampling. rel_ids: [N]. Returns [N] sampled entities."""
        # Work on CPU (ragged domain lists live on CPU), then move once to `device`.
        out = torch.empty((rel_ids.shape[0],), dtype=torch.long, device='cpu')
        arr = self.allowed_heads_per_rel if head else self.allowed_tails_per_rel
        for i, r in enumerate(rel_ids.tolist()):
            cand = arr[r]
            if cand is None or cand.numel() == 0:
                val = torch.randint(1, self.num_entities + 1, (1,), generator=self.rng,
                                    device='cpu', dtype=torch.long)
                out[i] = val[0]                      # 0-dim tensor assignment (no .item())
            else:
                j = torch.randint(0, cand.numel(), (1,), generator=self.rng, device='cpu')
                out[i] = cand[j]                     # 0-dim tensor assignment (no .item())
        return out.to(device=device, non_blocking=True)

    # -----------------------------
    # Public API
    # -----------------------------
    def corrupt(
        self,
        positives: LongTensor,                    # [B,3]
        *,
        num_negatives: int,
        mode: Literal['head', 'tail', 'both'] = None,
        device: Optional[torch.device] = None,
        filter: bool = True,
        unique: bool = True,
        use_domain: bool = True,
    ) -> LongTensor:
        """
        Vectorized K-negative generation. Returns [B, K, 3] on `device` (default sampler device).
        
        Args:
            positives: [B, 3] positive triples
            num_negatives: Number of negatives per positive
            mode: Corruption mode ('head', 'tail', or 'both')
            device: Target device (defaults to sampler device)
            filter: Whether to filter out known positives
            unique: Whether to ensure unique negatives per positive
            use_domain: Whether to use domain constraints
        
        Returns:
            negatives: [B, K, 3] negative triples
        
        Notes:
            - mode='head': corrupt head only
            - mode='tail': corrupt tail only
            - mode='both': randomly corrupt head or tail for each negative
            - If filtering removes entries, we oversample and trim to K.
        """
        if device is None:
            device = self.device
        if mode is None:
            mode = self.default_mode
        
        B = positives.shape[0]
        K = max(1, int(num_negatives))

        pos = positives.to(device=device, dtype=torch.long, non_blocking=True)  # [B,3]

        # Overshoot factor from EMA to reduce refills after filtering
        factor = 1.0 / max(1e-6, self.ema_accept)
        factor = float(torch.clamp(torch.tensor(factor), 1.0, 2.0))  # clamp to [1,2]
        K_os = int((K * factor) + 1)

        total = B * K_os
        r = pos[:, 0].repeat_interleave(K_os)   # [B*K_os]
        h = pos[:, 1].repeat_interleave(K_os)
        t = pos[:, 2].repeat_interleave(K_os)

        if mode == 'head':
            rel_ids = r
            if use_domain:
                h_new = self._draw_from_domain(rel_ids, head=True, device=device)
            else:
                h_new = self._draw_uniform_entities(total, device)
            # avoid sampling equal to gold head: resample once (cheap)
            mask_eq = h_new == h
            if mask_eq.any():
                if use_domain:
                    h_new[mask_eq] = self._draw_from_domain(rel_ids[mask_eq], head=True, device=device)
                else:
                    h_new[mask_eq] = self._draw_uniform_entities(int(mask_eq.sum()), device)
            cand = torch.stack([r, h_new, t], dim=-1)  # [B*K_os,3]

        elif mode == 'tail':
            rel_ids = r
            if use_domain:
                t_new = self._draw_from_domain(rel_ids, head=False, device=device)
            else:
                t_new = self._draw_uniform_entities(total, device)
            mask_eq = t_new == t
            if mask_eq.any():
                if use_domain:
                    t_new[mask_eq] = self._draw_from_domain(rel_ids[mask_eq], head=False, device=device)
                else:
                    t_new[mask_eq] = self._draw_uniform_entities(int(mask_eq.sum()), device)
            cand = torch.stack([r, h, t_new], dim=-1)

        else:  # both
            # Bernoulli per slot: True=head, False=tail
            coin = torch.randint(0, 2, (total,), generator=self.rng, device='cpu', dtype=torch.long).to(device=device, non_blocking=True).bool()
            rel_ids = r
            
            # draw both pools, then gather
            if use_domain:
                heads_pool = self._draw_from_domain(rel_ids, head=True, device=device)
                tails_pool = self._draw_from_domain(rel_ids, head=False, device=device)
            else:
                heads_pool = self._draw_uniform_entities(total, device)
                tails_pool = self._draw_uniform_entities(total, device)

            h_new = torch.where(coin, heads_pool, h)
            t_new = torch.where(~coin, tails_pool, t)

            # prevent identity replacements; re-draw once for offending positions
            mask_h = coin & (h_new == h)
            if mask_h.any():
                if use_domain:
                    h_new[mask_h] = self._draw_from_domain(rel_ids[mask_h], head=True, device=device)
                else:
                    h_new[mask_h] = self._draw_uniform_entities(int(mask_h.sum()), device)
            mask_t = (~coin) & (t_new == t)
            if mask_t.any():
                if use_domain:
                    t_new[mask_t] = self._draw_from_domain(rel_ids[mask_t], head=False, device=device)
                else:
                    t_new[mask_t] = self._draw_uniform_entities(int(mask_t.sum()), device)

            cand = torch.stack([r, h_new, t_new], dim=-1)

        # Filtering (drop positives)
        if filter:
            keep = self._filter_keep_mask(cand)
            cand = cand[keep]
            # Update EMA of accept rate
            accept_rate = float(cand.shape[0]) / max(1, total)
            self.ema_accept = (1 - self.ema_alpha) * self.ema_accept + self.ema_alpha * accept_rate

        # If unique per-row requested, de-dup within each group of K_os for the same base triple
        if unique and cand.shape[0] > 0:
            # Row ids [0..B-1] repeated K_os -> keep alignment after filtering
            base_ids = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(K_os)
            if filter:
                keep_indices = torch.nonzero(keep, as_tuple=False).squeeze(-1)
                base_ids = base_ids[keep_indices]
            
            # sort by (base_id, hash) and then unique-consecutive
            key = base_ids * 0x9E3779B185EBCA87 + (cand[:, 0] * 1315423911) ^ (cand[:, 1] * 2654435761) ^ (cand[:, 2] * 374761393)
            order = torch.argsort(torch.stack([base_ids, key], dim=-1), dim=0, stable=True)[:, 1]
            base_ids = base_ids[order]
            cand = cand[order]
            
            # unique-consecutive along (base_ids, cand)
            dif_base = torch.ones_like(base_ids, dtype=torch.bool)
            dif_base[1:] = base_ids[1:] != base_ids[:-1]
            dif_trip = torch.ones((cand.shape[0],), dtype=torch.bool, device=device)
            dif_trip[1:] = torch.any(cand[1:] != cand[:-1], dim=-1)
            keep2 = dif_base | dif_trip
            base_ids = base_ids[keep2]
            cand = cand[keep2]

        # Now we need exactly K per base triple. We'll pack without Python loops.
        out = torch.zeros((B, K, 3), dtype=torch.long, device=device)
        counts = torch.zeros((B,), dtype=torch.long, device=device)
        if cand.shape[0] > 0:
            if not (unique or filter):
                base_ids = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(K_os)
            # compute 0.. per group positions
            if base_ids.numel() > 0:
                _, counts_per_row = torch.unique_consecutive(base_ids, return_counts=True)
                if counts_per_row.numel() == 0:
                    pos_in_base = torch.zeros_like(base_ids)
                else:
                    starts = torch.cumsum(torch.nn.functional.pad(counts_per_row[:-1], (1, 0)), dim=0)
                    expanded = torch.repeat_interleave(starts, counts_per_row)
                    pos_in_base = torch.arange(base_ids.shape[0], device=device) - expanded
            else:
                pos_in_base = torch.zeros_like(base_ids)
            take = pos_in_base < K
            if take.any():
                out[base_ids[take], pos_in_base[take]] = cand[take]
                counts = torch.bincount(base_ids[take], minlength=B).clamp_max(K)

        # For rows with <K, do a small top-up with uniform draws (unfiltered).
        need = (K - counts).clamp(min=0)
        need_mask = need > 0
        if need_mask.any():
            b_idx = need_mask.nonzero(as_tuple=True)[0]          # rows needing fill
            need_vals = need[b_idx]                              # how many per row
            extra_total_t = need_vals.sum()                      # tensor
            extra_total   = int(extra_total_t)                   # single scalar read

            # Repeat base triples to match the number of extras we’ll generate
            r = positives[b_idx, 0].repeat_interleave(need_vals)
            h = positives[b_idx, 1].repeat_interleave(need_vals)
            t = positives[b_idx, 2].repeat_interleave(need_vals)

            # Build extras depending on corruption mode
            if mode == 'head':
                h_new = self._draw_uniform_entities(extra_total, device)
                extras = torch.stack([r, h_new, t], dim=-1)
            elif mode == 'tail':
                t_new = self._draw_uniform_entities(extra_total, device)
                extras = torch.stack([r, h, t_new], dim=-1)
            else:  # mode == 'both'
                coin = torch.randint(0, 2, (extra_total,), generator=self.rng, device='cpu',
                                    dtype=torch.long).to(device=device, non_blocking=True).bool()
                heads_pool = self._draw_uniform_entities(extra_total, device)
                tails_pool = self._draw_uniform_entities(extra_total, device)
                h_new = torch.where(coin, heads_pool, h)
                t_new = torch.where(~coin, tails_pool, t)
                extras = torch.stack([r, h_new, t_new], dim=-1)

            # Indices to write into out[ B, K, 3 ]
            repeat_idx = b_idx.repeat_interleave(need_vals)      # length == extra_total

            # Positions within each row (0..need_i-1) without .tolist()
            offsets = torch.cumsum(torch.nn.functional.pad(need_vals[:-1], (1, 0)), dim=0)
            all_pos = torch.arange(extra_total, device=device)
            pos_in_group = all_pos - offsets.repeat_interleave(need_vals)

            write_pos = counts[repeat_idx] + pos_in_group
            out[repeat_idx, write_pos] = extras
            counts[b_idx] += need_vals


        return out  # [B,K,3]

    def corrupt_all(
        self,
        positives: LongTensor,                # [B,3]
        *,
        mode: Literal['head', 'tail', 'both'] = None,
        device: Optional[torch.device] = None,
        use_domain: bool = True,
    ) -> Tuple[List[LongTensor], Optional[List[LongTensor]]]:
        """
        Enumerate *all* legal corruptions. Returns (all_heads, all_tails).
        If mode='head' -> (heads, None); if 'tail' -> ([], tails); if 'both' -> (heads, tails).
        Each list has length B; each element is [K_i, 3] ragged tensor on `device`.
        
        Args:
            positives: [B, 3] positive triples
            mode: Corruption mode
            device: Target device
            use_domain: Whether to use domain constraints
        
        Returns:
            (heads_list, tails_list): Lists of corrupted triples
        """
        if device is None:
            device = self.device
        if mode is None:
            mode = self.default_mode

        pos = positives.to(device=device, dtype=torch.long, non_blocking=True)  # [B,3]

        heads_list: List[LongTensor] = []
        tails_list: List[LongTensor] = []

        B = pos.shape[0]
        for i in range(B):
            r, h, t = pos[i].tolist()
            
            if mode in ('head', 'both'):
                if use_domain and r < len(self.allowed_heads_per_rel) and self.allowed_heads_per_rel[r] is not None:
                    cand = self.allowed_heads_per_rel[r]
                    cand = cand[cand != h]
                    vals = cand.to(device=device, dtype=torch.long)
                else:
                    vals = torch.arange(1, self.num_entities + 1, device=device, dtype=torch.long)
                    vals = vals[vals != h]
                H = torch.stack([torch.full_like(vals, r), vals, torch.full_like(vals, t)], dim=-1)
                heads_list.append(H)
            
            if mode in ('tail', 'both'):
                if use_domain and r < len(self.allowed_tails_per_rel) and self.allowed_tails_per_rel[r] is not None:
                    cand = self.allowed_tails_per_rel[r]
                    cand = cand[cand != t]
                    vals = cand.to(device=device, dtype=torch.long)
                else:
                    vals = torch.arange(1, self.num_entities + 1, device=device, dtype=torch.long)
                    vals = vals[vals != t]
                T = torch.stack([torch.full_like(vals, r), torch.full_like(vals, h), vals], dim=-1)
                tails_list.append(T)

        if mode == 'head':
            return heads_list, None
        if mode == 'tail':
            return [], tails_list
        return heads_list, tails_list
