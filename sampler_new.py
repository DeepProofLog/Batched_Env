
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import torch


LongTensor = torch.LongTensor
Tensor = torch.Tensor


def _mix_hash(triples: LongTensor, b_e: int, b_r: int) -> LongTensor:
    """Compute a fast 64-bit mixed hash for (h, r, t) triples (indices start at 1).
    Uses large odd multipliers and xor mix; stays in signed int64 domain.
    triples: [*, 3] long.
    """
    h = triples[..., 0].to(torch.int64)
    r = triples[..., 1].to(torch.int64)
    t = triples[..., 2].to(torch.int64)
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
    num_entities: int
    num_relations: int
    device: torch.device
    default_mode: Literal['head','tail','both'] = 'both'
    seed: int = 0


class Sampler:
    """Unified, index-only negative sampler with optional domains and filtering.

    Public API:
      - corrupt(positives, num_negatives, mode='both', device=None, filter=True, unique=True, use_domain=True)
      - corrupt_all(positives, mode='both', device=None, use_domain=True)

    Internals:
      - Filterer: sorted unique 64-bit hashes of known positives (CPU/GPU)
      - Domain: optional ragged allowed-sets per relation for heads/tails
      - Generator: vectorized torch.randint with optional per-row domain draws
    """

    def __init__(self, cfg: SamplerConfig) -> None:
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
        default_mode: Literal['head','tail','both'] = 'both',
        seed: int = 0,
        domain_heads: Optional[Dict[int, LongTensor]] = None,
        domain_tails: Optional[Dict[int, LongTensor]] = None,
    ) -> "Sampler":
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
        self.allowed_heads_per_rel = [None] * (max_rel)
        self.allowed_tails_per_rel = [None] * (max_rel)
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
        return torch.randint(1, self.num_entities + 1, (size,), generator=self.rng, device=device, dtype=torch.long)

    def _draw_from_domain(self, rel_ids: LongTensor, head: bool, device: torch.device) -> LongTensor:
        """Per-row domain sampling. rel_ids: [N]. Returns [N] sampled entities."""
        out = torch.empty((rel_ids.shape[0],), dtype=torch.long, device=device)
        arr = self.allowed_heads_per_rel if head else self.allowed_tails_per_rel
        # CPU sampling per-row to handle ragged domains; small overhead amortized by vectorized calls
        for i, r in enumerate(rel_ids.tolist()):
            cand = arr[r]
            if cand is None or cand.numel() == 0:
                out[i] = torch.randint(1, self.num_entities + 1, (1,), generator=self.rng).item()
            else:
                j = torch.randint(0, cand.numel(), (1,), generator=self.rng).item()
                out[i] = cand[j].item()
        return out

    # -----------------------------
    # Public API
    # -----------------------------
    def corrupt(
        self,
        positives: LongTensor,                    # [B,3]
        *,
        num_negatives: int,
        mode: Literal['head','tail','both'] = None,
        device: Optional[torch.device] = None,
        filter: bool = True,
        unique: bool = True,
        use_domain: bool = True,
    ) -> LongTensor:
        """
        Vectorized K-negative generation. Returns [B, K, 3] on `device` (default sampler device).
        - mode='head'|'tail'|'both' (in 'both', each slot decides independently which side to corrupt).
        - If filtering removes entries, we oversample mildly and trim to K.
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
            target = 'head'
            rel_ids = r
            if use_domain:
                h_new = self._draw_from_domain(rel_ids, head=True, device=device)
            else:
                h_new = self._draw_uniform_entities(total, device)
            # avoid sampling equal to gold head: resample once (cheap)
            mask_eq = h_new == h
            if mask_eq.any():
                h_new[mask_eq] = self._draw_uniform_entities(int(mask_eq.sum()), device)
            cand = torch.stack([r, h_new, t], dim=-1)  # [B*K_os,3]

        elif mode == 'tail':
            target = 'tail'
            rel_ids = r
            if use_domain:
                t_new = self._draw_from_domain(rel_ids, head=False, device=device)
            else:
                t_new = self._draw_uniform_entities(total, device)
            mask_eq = t_new == t
            if mask_eq.any():
                t_new[mask_eq] = self._draw_uniform_entities(int(mask_eq.sum()), device)
            cand = torch.stack([r, h, t_new], dim=-1)

        else:  # both
            target = 'both'
            # Bernoulli per slot: True=head, False=tail
            coin = torch.randint(0, 2, (total,), generator=self.rng, device=device, dtype=torch.long).bool()
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
                h_new[mask_h] = self._draw_uniform_entities(int(mask_h.sum()), device)
            mask_t = (~coin) & (t_new == t)
            if mask_t.any():
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
                base_ids = base_ids[keep]
            # sort by (base_id, hash) and then unique-consecutive
            key = base_ids * 0x9E3779B185EBCA87 + (cand[:, 0] * 1315423911) ^ (cand[:, 1] * 2654435761) ^ (cand[:, 2] * 374761393)
            order = torch.argsort(torch.stack([base_ids, key], dim=-1), dim=0)[:, 1]
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

        # Now we need exactly K per base triple. We'll pack row-by-row.
        out = torch.zeros((B, K, 3), dtype=torch.long, device=device)
        counts = torch.zeros((B,), dtype=torch.long, device=device)
        # scatter-fill
        if cand.shape[0] > 0:
            # recompute base_ids if we didn't compute earlier
            if not (unique or filter):
                base_ids = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(K_os)
            # clamp to available slots
            for i in range(cand.shape[0]):
                b = int(base_ids[i].item())
                c = int(counts[b].item())
                if c < K:
                    out[b, c] = cand[i]
                    counts[b] += 1

        # For rows with <K, do a small top-up with uniform draws (unfiltered to avoid long loops).
        need = (K - counts).clamp(min=0)
        if int(need.sum().item()) > 0:
            b_idx = need.nonzero(as_tuple=False).squeeze(-1)
            extra_total = int(need.sum().item())
            # draw extras for tails by default (balanced choice is okay)
            r = positives[b_idx, 0].repeat_interleave(need[b_idx])
            h = positives[b_idx, 1].repeat_interleave(need[b_idx])
            t = positives[b_idx, 2].repeat_interleave(need[b_idx])
            if mode == 'head':
                h_new = self._draw_uniform_entities(extra_total, device)
                extras = torch.stack([r, h_new, t], dim=-1)
            elif mode == 'tail':
                t_new = self._draw_uniform_entities(extra_total, device)
                extras = torch.stack([r, h, t_new], dim=-1)
            else:
                coin = torch.randint(0, 2, (extra_total,), generator=self.rng, device=device, dtype=torch.long).bool()
                heads_pool = self._draw_uniform_entities(extra_total, device)
                tails_pool = self._draw_uniform_entities(extra_total, device)
                h_new = torch.where(coin, heads_pool, h)
                t_new = torch.where(~coin, tails_pool, t)
                extras = torch.stack([r, h_new, t_new], dim=-1)
            # fill remaining slots
            ptrs = torch.zeros_like(need)
            for i, b in enumerate(b_idx.tolist()):
                kneed = int(need[b].item())
                start = int(ptrs[b].item())
                out[b, counts[b]:counts[b] + kneed] = extras[start:start + kneed]
                counts[b] += kneed
                ptrs[b] += kneed

        return out  # [B,K,3]

    def corrupt_all(
        self,
        positives: LongTensor,                # [B,3]
        *,
        mode: Literal['head','tail','both'] = None,
        device: Optional[torch.device] = None,
        use_domain: bool = True,
    ) -> Tuple[List[LongTensor], Optional[List[LongTensor]]]:
        """
        Enumerate *all* legal corruptions. Returns (all_heads, all_tails).
        If mode='head' -> (heads, None); if 'tail' -> ([], tails); if 'both' -> (heads, tails).
        Each list has length B; each element is [K_i, 3] ragged tensor on `device`.
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
