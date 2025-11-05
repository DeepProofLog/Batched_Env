
import math
import logging
import types
from typing import Collection, Optional, Union, Literal, Dict, List, Tuple

import numpy as np
import torch

from utils import Term
from dataset import DataHandler
from index_manager import IndexManager

# ---------------------------------------------------------------------
# Minimal, PyKEEN-free negative sampling utilities
# ---------------------------------------------------------------------

Target = Literal["head", "relation", "tail"]
COLUMN_HEAD, COLUMN_RELATION, COLUMN_TAIL = 0, 1, 2
LABEL_HEAD, LABEL_RELATION, LABEL_TAIL = "head", "relation", "tail"
TARGET_TO_INDEX = {LABEL_HEAD: COLUMN_HEAD, LABEL_RELATION: COLUMN_RELATION, LABEL_TAIL: COLUMN_TAIL}


# ------------------------------------------------------------------- #
# 1.  GPU filter with O(log N) search instead of torch.isin           #
# ------------------------------------------------------------------- #
class SortedHashTripleFilter(torch.nn.Module):
    """
    Hash-encode triples and test membership via `torch.searchsorted`
    on a pre-sorted GPU tensor – noticeably faster than `torch.isin`
    for |candidates| ≪ |training triples|.
    """

    def __init__(self, true_triples: torch.Tensor):
        super().__init__()
        if true_triples.numel() == 0:
            hashes = torch.empty((0,), dtype=torch.int64)
        else:
            hashes = (
                (true_triples[:, 0].to(torch.int64) << 42)
                | (true_triples[:, 1].to(torch.int64) << 21)
                |  true_triples[:, 2].to(torch.int64)
            )
        self.register_buffer("_hashes_sorted", torch.sort(hashes.unique())[0])
        # legacy alias used elsewhere
        self._hashes = self._hashes_sorted

    def forward(self, triples: torch.Tensor) -> torch.BoolTensor:
        """
        triples: (..., 3) → bool mask with the same leading shape (True = keep).
        Safe against out-of-range positions returned by searchsorted.
        """
        if triples.numel() == 0:
            return torch.ones((*triples.shape[:-1],), dtype=torch.bool, device=triples.device)

        flat = triples.view(-1, 3)
        h    = (flat[:, 0].to(torch.int64) << 42) | (flat[:, 1].to(torch.int64) << 21) | flat[:, 2].to(torch.int64)

        pos  = torch.searchsorted(self._hashes_sorted, h)          # 1-D indices
        L    = self._hashes_sorted.numel()

        in_set = torch.zeros_like(h, dtype=torch.bool)             # default: not found
        if L > 0:
            valid  = pos < L                                       # only safe positions
            if valid.any():
                in_set[valid] = self._hashes_sorted[pos[valid]] == h[valid]

        return (~in_set).view(*triples.shape[:-1])                 # True = keep candidate


# ------------------------------------------------------------------- #
# 2.  Minimal base sampler (no PyKEEN)                                #
# ------------------------------------------------------------------- #
class _BaseSampler(torch.nn.Module):
    """Lightweight base holding sizes & (optional) filterer."""

    def __init__(
        self,
        mapped_triples: torch.Tensor,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        filtered: bool = True,
    ):
        super().__init__()
        self.mapped_triples = mapped_triples
        if num_entities is None:
            self.num_entities = int(mapped_triples[:, [0, 2]].max().item()) + 1 if mapped_triples.numel() > 0 else 0
        else:
            self.num_entities = int(num_entities)

        if num_relations is None:
            self.num_relations = int(mapped_triples[:, 1].max().item()) + 1 if mapped_triples.numel() > 0 else 0
        else:
            self.num_relations = int(num_relations)

        self.filterer: Optional[SortedHashTripleFilter] = (
            SortedHashTripleFilter(mapped_triples) if filtered else None
        )


# ------------------------------------------------------------------- #
# 3.  Simple "random replacement" helper                               #
# ------------------------------------------------------------------- #
def _random_replacement_(
    batch: torch.Tensor, index: int, selection: slice, size: int, max_index: int, pad_idx: int = 0
) -> None:
    """
    Replace a column of a batch of indices by random indices, excluding
    the original value and the padding index.
    """
    if max_index <= 1:
        return

    orig = batch[selection, index].to(torch.int64)
    if pad_idx == 0:
        # sample from [1, max_index-1], then shift past 'orig' where needed
        rng = torch.randint(1, max_index, (size,), device=batch.device, dtype=torch.int64)
        shift = (rng >= orig) & (orig > 0)
        replacement = (rng + shift.to(rng.dtype)).to(batch.dtype)
        batch[selection, index] = replacement
    else:
        cand = torch.randint(0, max_index, (size,), device=batch.device, dtype=torch.int64)
        bad  = (cand == orig) | (cand == pad_idx)
        while bad.any():
            cand[bad] = torch.randint(0, max_index, (int(bad.sum().item()),), device=batch.device, dtype=torch.int64)
            bad = (cand == orig) | (cand == pad_idx)
        batch[selection, index] = cand.to(batch.dtype)


# ------------------------------------------------------------------- #
# 4.  Domain-aware & uniform samplers                                  #
# ------------------------------------------------------------------- #
class BasicNegativeSamplerDomain(_BaseSampler):
    """
    Domain-aware negative sampler.

    All domain lookups are precomputed once and stored as device tensors
    so that batch corruption is fully vectorized.
    """

    def __init__(
        self,
        mapped_triples: torch.Tensor,
        domain2idx: Dict[str, List[int]],
        entity2domain: Dict[int, str],
        filtered: bool = True,
        corruption_scheme: List[str] = ('tail',),
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(mapped_triples=mapped_triples, filtered=filtered)
        self.domain2idx = domain2idx
        self.entity2domain = entity2domain
        self.device = device

        # ---- Build stable integer IDs for domains ----
        self.domain_names = sorted(self.domain2idx.keys())
        self.domain_str2int = {name: i for i, name in enumerate(self.domain_names)}
        self.domain_int2str = {i: name for i, name in enumerate(self.domain_names)}

        # ---- Build per-domain pools as a single padded 2-D tensor ----
        storage_dtype = torch.int32
        domain_lists: List[torch.Tensor] = []
        for name in self.domain_names:
            ents = torch.tensor(self.domain2idx[name], dtype=storage_dtype, device=device)
            domain_lists.append(ents)

        self.num_domains = len(domain_lists)
        self.max_pool_len = max((t.numel() for t in domain_lists), default=0)
        # (D, Lmax) padded with 0 (padding id); entities start at 1 so 0 is safe
        self.domain_padded = torch.zeros((self.num_domains, self.max_pool_len), dtype=storage_dtype, device=device)
        self.domain_len = torch.zeros((self.num_domains,), dtype=storage_dtype, device=device)
        for i, t in enumerate(domain_lists):
            self.domain_padded[i, :t.numel()] = t
            self.domain_len[i] = t.numel()

        # ---- Fast maps: entity -> domain_id   and   entity -> position within its domain ----
        max_ent_id = max(entity2domain.keys(), default=0)
        ent2dom = torch.full((max_ent_id + 1,), -1, dtype=storage_dtype, device=device)
        pos_in_dom = torch.zeros((max_ent_id + 1,), dtype=storage_dtype, device=device)
        # We iterate once over our padded pools to fill both maps
        for d, name in enumerate(self.domain_names):
            row = self.domain_padded[d, : int(self.domain_len[d].item())]
            # guard: row can be empty for corner cases
            if row.numel() == 0:
                continue
            row_long = row.to(torch.int64)
            ent2dom[row_long] = d
            # position of each entity in its domain
            pos_in_dom[row_long] = torch.arange(row.numel(), device=device, dtype=storage_dtype)

        self.ent2dom = ent2dom
        self.pos_in_dom = pos_in_dom

        # Validate corruption indices (0=head, 1=rel, 2=tail)
        self.corruption_scheme: List[str] = list(corruption_scheme) or [LABEL_HEAD, LABEL_TAIL]
        self._corruption_indices = [TARGET_TO_INDEX[c] for c in self.corruption_scheme]
        if any(idx not in (0, 1, 2) for idx in self._corruption_indices):
            raise ValueError(f"Invalid corruption index in scheme: {self._corruption_indices}")

        # Infer counts from mapped triples
        if self.mapped_triples.numel() > 0:
            self.num_relations = int(self.mapped_triples[:, 1].max().item()) + 1

    def _replace_relation_uniform_(self, batch: torch.Tensor, sel: slice) -> None:
        """Uniformly replace relations in batch[sel, 1] excluding the original id."""
        if self.num_relations <= 1:
            return
        orig = batch[sel, 1]
        high = torch.full_like(orig, self.num_relations - 1)
        rnd = torch.floor(torch.rand_like(orig, dtype=torch.float32) * high.to(torch.float32)).to(torch.int64)
        batch[sel, 1] = (rnd + (rnd >= orig)).to(batch.dtype)

    def corrupt_batch(self, positive_batch: torch.Tensor, num_negs_per_pos: int) -> torch.Tensor:
        """
        Vectorized corruption for head/tail within the same domain and (optional) relation.
        """
        batch_shape = positive_batch.shape[:-1]
        neg = positive_batch.view(-1, 3).repeat_interleave(num_negs_per_pos, dim=0)
        total = neg.size(0)
        if total == 0:
            return neg.view(*batch_shape, num_negs_per_pos, 3)

        # Split work roughly equally across the selected indices (0/1/2)
        step = math.ceil(total / max(1, len(self._corruption_indices)))
        for col, start in zip(self._corruption_indices, range(0, total, step)):
            stop = min(start + step, total)
            sel = slice(start, stop)
            if col == 1:
                self._replace_relation_uniform_(neg, sel)
                continue

            # ---- entity corruption within the same domain ----
            orig = neg[sel, col]
            valid = orig > 0  # ignore padding rows defensively
            if not valid.any():
                continue
            orig_valid = orig[valid]
            orig_valid_long = orig_valid.to(torch.int64)
            d_ids = self.ent2dom[orig_valid_long]                 # (N,)
            d_ids_long = d_ids.to(torch.int64)
            pool_len = self.domain_len[d_ids_long]                # (N,)
            pos = self.pos_in_dom[orig_valid_long].to(torch.int64)  # (N,)

            # rows where the domain only has one entity cannot be corrupted
            can = pool_len > 1
            if not can.any():
                continue

            # Draw per-row index in [0, pool_len-2] then add +1 for rows where idx >= pos
            Lm1 = (pool_len[can] - 1).to(torch.float32)
            rnd = torch.floor(torch.rand(Lm1.shape, device=neg.device) * Lm1).to(torch.int64)
            adj = rnd + (rnd >= pos[can].to(torch.int64))
            repl = self.domain_padded[d_ids_long[can], adj].to(orig.dtype)

            # write back only for rows that can be corrupted
            if can.any():
                orig_valid = orig_valid.clone()
                orig_valid[can] = repl
                orig[valid] = orig_valid
                neg[sel, col] = orig

        return neg.view(*batch_shape, num_negs_per_pos, 3)

    def corrupt_batch_all(self, positive_batch: torch.Tensor) -> List[torch.Tensor]:
        """
        Enumerate all legal domain-respecting corruptions for each triple.
        Uses vectorized repeat+assign instead of Python loops per candidate.
        """
        out: List[torch.Tensor] = []
        for triple in positive_batch:
            parts = []
            for col in self._corruption_indices:
                if col == 1:
                    # enumerate all relations except the current one
                    if self.num_relations <= 1:
                        continue
                    rels = torch.arange(self.num_relations, device=triple.device, dtype=triple.dtype)
                    rels = rels[rels != triple[1]]
                    if rels.numel() == 0:
                        continue
                    t = triple.repeat(rels.numel(), 1)
                    t[:, 1] = rels.to(t.dtype)
                    parts.append(t)
                else:
                    e = triple[col].item()
                    if e <= 0:
                        continue
                    d = int(self.ent2dom[e].item())
                    if d < 0:
                        continue
                    L = int(self.domain_len[d].item())
                    if L <= 1:
                        continue
                    pool = self.domain_padded[d, :L]
                    cand = pool[pool != e]
                    if cand.numel() == 0:
                        continue
                    t = triple.repeat(cand.numel(), 1)
                    t[:, col] = cand
                    parts.append(t)
            out.append(torch.cat(parts, dim=0) if parts else triple.new_empty((0, 3)))
        return out


class BasicNegativeSamplerCustom(_BaseSampler):
    """
    Uniform negative sampler (head/tail by default). Efficient random replacement,
    excludes padding and original value.
    """

    def __init__(
        self,
        mapped_triples: torch.Tensor,
        num_entities: int,
        num_relations: int,
        filtered: bool = True,
        corruption_scheme: Optional[Collection[Target]] = None,
        padding_idx: Optional[int] = 0,
    ):
        super().__init__(
            mapped_triples=mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            filtered=filtered,
        )
        self.pad_idx: int = int(padding_idx)
        if self.pad_idx != 0:
            logging.warning(
                "Efficient replacement assumes pad_idx = 0; "
                "fallback to slower rejection-sampling."
            )
        # Determine corruption scheme and indices
        self.corruption_scheme = list(corruption_scheme) if corruption_scheme else [LABEL_HEAD, LABEL_TAIL]
        self._corruption_indices = [TARGET_TO_INDEX[side] for side in self.corruption_scheme]
        if any(idx not in {0, 1, 2} for idx in self._corruption_indices):
            raise ValueError(f"Invalid corruption index found in scheme: {self._corruption_indices}.")

    def corrupt_batch(self, positive_batch: torch.Tensor, num_negs_per_pos: int) -> torch.Tensor:
        """
        Corrupts a batch of positive triples using the specified scheme,
        efficiently excluding the padding index (assumed 0) and the original triple value.
        """
        batch_shape = positive_batch.shape[:-1]
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(num_negs_per_pos, dim=0)
        total_num_negatives = negative_batch.shape[0]
        if total_num_negatives == 0:
            return negative_batch.view(*batch_shape, num_negs_per_pos, 3)

        num_corruption_indices = len(self._corruption_indices)
        split_idx = math.ceil(total_num_negatives / max(1, num_corruption_indices))

        current_start = 0
        for index in self._corruption_indices:
            stop = min(current_start + split_idx, total_num_negatives)
            if stop <= current_start:
                continue
            selection = slice(current_start, stop)
            size = stop - current_start
            current_max_index = self.num_relations if index == 1 else self.num_entities
            _random_replacement_(
                batch=negative_batch,
                index=index,
                selection=selection,
                size=size,
                max_index=current_max_index,
                pad_idx=self.pad_idx,
            )
            current_start = stop

        return negative_batch.view(*batch_shape, num_negs_per_pos, 3)

    def corrupt_batch_all(self, positive_batch: torch.Tensor) -> List[torch.Tensor]:
        """
        Exhaustively enumerate every legal head/relation/tail corruption
        defined by ``self._corruption_indices`` for each triple.
        """
        device = positive_batch.device
        negatives: List[torch.Tensor] = []

        pool_dtype = positive_batch.dtype
        ent_pool = torch.arange(1, self.num_entities, device=device, dtype=pool_dtype)
        if self.pad_idx is not None:
            ent_pool = ent_pool[ent_pool != self.pad_idx]

        rel_pool = torch.arange(self.num_relations, device=device, dtype=pool_dtype)

        for triple in positive_batch:              # loop over batch (B is usually small)
            triple_negs = []
            for col in self._corruption_indices:
                pool = rel_pool if col == 1 else ent_pool
                cand = pool[pool != triple[col]]
                if cand.numel() == 0:
                    continue
                reps = triple.repeat(cand.numel(), 1)
                reps[:, col] = cand.to(reps.dtype)
                triple_negs.append(reps)
            if triple_negs:
                negatives.append(torch.cat(triple_negs, dim=0))
            else:
                negatives.append(triple.new_empty((0, 3)))
        return negatives


# ------------------------------------------------------------------- #
# 5.  High-level helpers used by training / evaluation                #
# ------------------------------------------------------------------- #
def get_negatives(
    self,
    sub_indices: torch.Tensor,
    padding_atoms: int,
    max_arity: int,
    device: torch.device,
    num_negs: Optional[int] = None,        # None ⇒ enumerate *all* corruptions
    debug: bool = False,
) -> torch.Tensor:
    """
    Generate negative samples for a batch of query states.

    Returns
    -------
    neg_subs : torch.Tensor
        Shape (B, M, padding_atoms, max_arity+1) where M is either
        `num_negs` (sampled mode) or the per-batch maximum when enumerating
        all corruptions.  Unused slots are padded with `self.index_manager.padding_idx`.
    """
    expected_dtype = getattr(getattr(self, "index_manager", None), "idx_dtype", torch.int32)
    if sub_indices.dtype != expected_dtype:
        raise TypeError(
            f"sub_indices dtype {sub_indices.dtype} does not match expected {expected_dtype}. "
            "Ensure environment/index manager emit int32 tensors prior to negative sampling."
        )

    # ensure filterer device
    if self.filterer is not None and self.filterer._hashes_sorted.device != sub_indices.device:
        self.filterer = self.filterer.to(sub_indices.device)

    # ensure domain tensors are on the same device
    if hasattr(self, 'ent2dom') and self.ent2dom.device != sub_indices.device:
        self.ent2dom = self.ent2dom.to(sub_indices.device)
        self.pos_in_dom = self.pos_in_dom.to(sub_indices.device)
        self.domain_len = self.domain_len.to(sub_indices.device)
        self.domain_padded = self.domain_padded.to(sub_indices.device)

    B = sub_indices.size(0)

    # Extract (r,h,t) from first atom slot
    rels  = sub_indices[:, 0, 0]
    heads = sub_indices[:, 0, 1]
    tails = sub_indices[:, 0, 2]
    pos_batch = torch.stack([heads, rels, tails], dim=1)  # (B, 3)

    # Enumerate *all* corruptions
    if num_negs is None:
        neg_batches = self.corrupt_batch_all(pos_batch)    # list length B
        lengths     = [nb.size(0) for nb in neg_batches]
        total_rows  = sum(lengths)

        if total_rows == 0:                                # nothing to pad
            neg_subs = torch.full(
                (B, 0, padding_atoms, max_arity + 1),
                fill_value=self.index_manager.padding_idx,
                dtype=expected_dtype,
                device=device,
            )
            return neg_subs

        # filter once on flattened tensor
        flat  = torch.cat(neg_batches, dim=0).to(expected_dtype)  # (total_rows, 3)
        mask  = self.filterer(flat) if self.filterer is not None else torch.ones(flat.size(0), dtype=torch.bool, device=flat.device)

        # slice the mask back
        filtered_batches: List[torch.Tensor] = []
        cursor = 0
        for L in lengths:
            if L == 0:
                filtered_batches.append(flat.new_empty((0, 3)))
                continue
            seg_mask = mask[cursor: cursor + L]
            seg      = flat[cursor: cursor + L][seg_mask]
            filtered_batches.append(seg)
            cursor += L

        # pad to equal length and write into output tensor
        max_M = max(nb.size(0) for nb in filtered_batches) if filtered_batches else 0
        neg_subs = torch.full(
            (B, max_M, padding_atoms, max_arity + 1),
            fill_value=self.index_manager.padding_idx,
            dtype=expected_dtype,
            device=device,
        )
        for i, nb in enumerate(filtered_batches):
            if nb.numel() == 0:
                continue
            m = nb.size(0)
            neg_subs[i, :m, 0, 1] = nb[:, 0]   # head
            neg_subs[i, :m, 0, 0] = nb[:, 1]   # relation
            neg_subs[i, :m, 0, 2] = nb[:, 2]   # tail
        return neg_subs

    # Sampled negatives (oversample then filter/unique)
    overshoot = 3
    cand = self.corrupt_batch(
        pos_batch,
        num_negs_per_pos=overshoot * num_negs
    ).view(-1, 3).to(expected_dtype)           # (B·overshoot·num_negs, 3)

    mask = self.filterer(cand) if self.filterer is not None else torch.ones(cand.size(0), dtype=torch.bool, device=cand.device)
    cand = cand[mask]

    chosen = cand.unique(dim=0, return_inverse=False)
    chosen = chosen[: B * num_negs]            # simple truncation

    actual_num_candidates = chosen.size(0)
    expected_num_candidates = B * num_negs

    neg_subs = torch.full(
        (B, num_negs, padding_atoms, max_arity + 1),
        fill_value=self.index_manager.padding_idx,
        dtype=expected_dtype,
        device=device,
    )

    if actual_num_candidates < expected_num_candidates:
        if debug:
            print(
                f"Negative sampling produced fewer candidates than requested: "
                f"got {actual_num_candidates}, expected {expected_num_candidates} ",
                RuntimeWarning
            )
        chosen = chosen.view(B, -1, 3)  # (B, actual_per_batch, 3)
        actual_per_batch = chosen.size(1) if chosen.numel() > 0 else 0
        if actual_per_batch > 0:
            neg_subs[:, :actual_per_batch, 0, 1] = chosen[:, :, 0]
            neg_subs[:, :actual_per_batch, 0, 0] = chosen[:, :, 1]
            neg_subs[:, :actual_per_batch, 0, 2] = chosen[:, :, 2]
    else:
        chosen = chosen.view(B, num_negs, 3)  # (B, num_negs, 3)
        neg_subs[:, :, 0, 1] = chosen[:, :, 0]
        neg_subs[:, :, 0, 0] = chosen[:, :, 1]
        neg_subs[:, :, 0, 2] = chosen[:, :, 2]

    return neg_subs

def get_negatives_from_states_separate(
    self: Union[BasicNegativeSamplerCustom, BasicNegativeSamplerDomain],
    states: List[List[Term]],
    device: torch.device,
    num_negs: Optional[int] = None,
    return_states: bool = True,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[List[List[Term]], List[List[Term]]]]:
    """
    Convert a list of Term-lists to sub-indices, generate negatives separately for head and tail, 
    and return separate tensors/lists.

    Args:
        states: list of B query states, each a list of Term
        num_negs: number of negatives per corruption type, or None for all
        return_states: whether to return Term lists or tensor sub-indices
        
    Returns:
        If return_states=True: (head_neg_terms, tail_neg_terms) - both List[List[Term]]
        If return_states=False: (head_neg_subs, tail_neg_subs) - both torch.Tensor
    """
    # Handle input format
    if isinstance(states, Term):
        states = [[states]]
    elif isinstance(states, list) and states and isinstance(states[0], Term):
        states = [states]
    
    # Build sub-indices for each state
    subs = [self.index_manager.get_atom_sub_index(state) for state in states]
    target_device = self.filterer._hashes_sorted.device if self.filterer is not None else device
    pos_subs = torch.stack(subs, dim=0).to(target_device)
    
    B, P, A = pos_subs.size(0), pos_subs.size(1), pos_subs.size(2)
    
    # Store original corruption scheme to restore it later
    original_scheme = self.corruption_scheme
    original_indices = self._corruption_indices
    
    # Generate head corruptions (corrupt position 0 - head/subject)
    expected_dtype = getattr(getattr(self, "index_manager", None), "idx_dtype", torch.int32)
    if 'head' in original_scheme:
        self.corruption_scheme = ['head']
        self._corruption_indices = [0]  # head is at position 0 in (h,r,t)
        head_neg_subs = self.get_negatives(
            pos_subs, padding_atoms=P, max_arity=A - 1,
            device=target_device, num_negs=num_negs,
        )
    else:
        head_neg_subs = torch.full(
            (B, 0, P, A),
            fill_value=self.index_manager.padding_idx,
            dtype=expected_dtype,
            device=target_device,
        )
    
    # --- Generate Tail Corruptions ---
    if 'tail' in original_scheme:
        self.corruption_scheme = ['tail']
        self._corruption_indices = [2]  # tail is at position 2 in (h,r,t)
        tail_neg_subs = self.get_negatives(
            pos_subs, padding_atoms=P, max_arity=A - 1,
            device=target_device, num_negs=num_negs,
        )
    else:
        tail_neg_subs = torch.full(
            (B, 0, P, A),
            fill_value=self.index_manager.padding_idx,
            dtype=expected_dtype,
            device=target_device,
        )

    # Restore original corruption scheme
    self.corruption_scheme = original_scheme
    self._corruption_indices = original_indices
    
    if return_states:
        # Convert to Term-based states
        head_neg_terms = self.index_manager.subindices_to_terms(head_neg_subs)
        tail_neg_terms = self.index_manager.subindices_to_terms(tail_neg_subs)
        
        if B == 1: return head_neg_terms[0], tail_neg_terms[0]
        else: return head_neg_terms, tail_neg_terms
    else:
        if B == 1: return head_neg_subs.squeeze(0), tail_neg_subs.squeeze(0)
        else: return head_neg_subs, tail_neg_subs


def get_negatives_from_states(
    self: Union[BasicNegativeSamplerCustom, BasicNegativeSamplerDomain],
    states: List[List[Term]],
    device: torch.device,
    num_negs: Optional[int] = None,
    return_states: bool = True,
) -> Union[torch.Tensor, List[List[Term]]]:
    """
    Convert a list of Term-lists to sub-indices, generate negatives, and return sub-indices tensor.
    """
    # if it is only one state (List[Term]), convert it to a list of states
    if isinstance(states, Term):
        states = [[states]]
    elif isinstance(states, list) and states and isinstance(states[0], Term):
        states = [states]
    # Build sub-indices for each state
    subs = [self.index_manager.get_atom_sub_index(state) for state in states]
    target_device = self.filterer._hashes_sorted.device if self.filterer is not None else device
    pos_subs = torch.stack(subs, dim=0).to(target_device)
    # Call tensor-based sampler
    neg_subs = self.get_negatives(
        pos_subs,
        padding_atoms=pos_subs.size(1),
        max_arity=pos_subs.size(2) - 1,
        device=target_device,         # pass the same device downstream
        num_negs=num_negs,
    )
    # Convert to Term-based states
    B = neg_subs.size(0)
    if return_states:
        neg_terms = self.index_manager.subindices_to_terms(neg_subs)
        return neg_terms[0] if B == 1 else neg_terms
    else:
        return neg_subs.squeeze(0).to(device) if B == 1 else neg_subs.to(device)

# ------------------------------------------------------------------- #
# 6.  Sampler factory                                                 #
# ------------------------------------------------------------------- #
def _build_mapped_triples_from_handler(
    data_handler: DataHandler, index_manager: IndexManager
) -> torch.Tensor:
    """
    Build a (N,3) int64 tensor from DataHandler facts, using IndexManager maps.
    Skips any fact whose head/tail/relation is missing from the index maps.
    """
    rows: List[Tuple[int, int, int]] = []
    all_triples_for_filtering = getattr(data_handler, "all_known_triples", [])
    for f in all_triples_for_filtering:
        h_str = f.args[0]
        r_str = f.predicate
        t_str = f.args[1]
        h = index_manager.constant_str2idx.get(h_str, None)
        r = index_manager.predicate_str2idx.get(r_str, None)
        t = index_manager.constant_str2idx.get(t_str, None)
        if h is None or r is None or t is None:
            continue
        rows.append((h, r, t))

    if not rows:
        return torch.zeros((0, 3), dtype=torch.int64)

    arr = np.asarray(rows, dtype=np.int64)
    return torch.from_numpy(arr)


def get_sampler(
    data_handler: DataHandler, 
    index_manager: IndexManager,
    corruption_scheme: Optional[Collection[Target]] = None,
    device: torch.device = torch.device("cpu"),
) -> Union[BasicNegativeSamplerCustom, BasicNegativeSamplerDomain]:

    """
    Get sampler with optimizations enabled.
    
    Args:
        data_handler: Data handler with dataset info
        index_manager: Index manager for entity/relation mappings
        corruption_scheme: Which parts to corrupt ('head', 'tail', 'relation')
        device: Device to place sampler on
        use_optimizations: If True, use optimized samplers (default: True)
    
    Returns:
        Optimized or baseline sampler based on use_optimizations flag
    """
    from neg_sampling_optimized import (
        OptimizedDomainSampler,
        OptimizedUniformSampler,
    )
    
    # Build mapped triples
    mapped_triples_cpu = _build_mapped_triples_from_handler(data_handler, index_manager)
    
    # Choose sampler type
    if 'countries' in data_handler.dataset_name or 'ablation' in data_handler.dataset_name:
        # Build domain2idx and entity2domain
        domain2idx: Dict[str, List[int]] = {}
        skipped_entities: List[str] = []
        for domain, entities in data_handler.domain2entity.items():
            indexed_entities: List[int] = []
            for e in entities:
                if e in index_manager.constant_str2idx:
                    indexed_entities.append(index_manager.constant_str2idx[e])
                else:
                    skipped_entities.append(e)
            if indexed_entities:
                domain2idx[domain] = indexed_entities

        if skipped_entities:
            print(f"Note: Skipped {len(skipped_entities)} entities not in index: {skipped_entities[:10]}"
                  f"{'...' if len(skipped_entities) > 10 else ''}")

        entity2domain: Dict[int, str] = {}
        for domain, entities in data_handler.domain2entity.items():
            for e in entities:
                if e in index_manager.constant_str2idx:
                    entity2domain[index_manager.constant_str2idx[e]] = domain

        # Use optimized domain sampler
        num_triples = mapped_triples_cpu.shape[0]
        
        # For now, always use OptimizedDomainSampler (Phase 0+1)
        # UltraOptimizedSampler has initialization issues that need fixing
        sampler = OptimizedDomainSampler(
            mapped_triples=mapped_triples_cpu,
            domain2idx=domain2idx,
            entity2domain=entity2domain,
            filtered=True,
            corruption_scheme=list(corruption_scheme) if corruption_scheme else ['tail'],
            device=device,
        )
    else:
        # Uniform sampler for non-domain datasets
        num_entities = max(index_manager.constant_str2idx.values(), default=0) + 1
        num_relations = max(index_manager.predicate_str2idx.values(), default=0) + 1
        
        sampler = OptimizedUniformSampler(
            mapped_triples=mapped_triples_cpu,
            num_entities=num_entities,
            num_relations=num_relations,
            filtered=True,
            corruption_scheme=corruption_scheme,
            padding_idx=index_manager.padding_idx,
            device=device,
        )
    
    # Bind helper methods & attach index manager
    sampler.index_manager = index_manager
    sampler.get_negatives = types.MethodType(get_negatives, sampler)
    sampler.get_negatives_from_states = types.MethodType(get_negatives_from_states, sampler)
    sampler.get_negatives_from_states_separate = types.MethodType(get_negatives_from_states_separate, sampler)

    return sampler


def share_sampler_storage(sampler):
    """
    Move CPU tensors held by the sampler into shared memory so forked env workers
    reuse the same storage instead of receiving full pickled copies.
    """
    def _share_tensor(t: torch.Tensor):
        if t.device.type != "cpu":
            return
        if t.is_shared():
            return
        try:
            t.share_memory_()
        except RuntimeError:
            # Some tensors (e.g., non-writeable views) cannot be shared; skip them quietly
            pass

    for value in sampler.__dict__.values():
        if isinstance(value, torch.Tensor):
            _share_tensor(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    _share_tensor(item)

    filterer = getattr(sampler, "filterer", None)
    if isinstance(filterer, torch.nn.Module):
        for buffer in filterer.buffers():
            _share_tensor(buffer)
        for parameter in filterer.parameters(recurse=False):
            _share_tensor(parameter)

    sampler._shared_memory = True  # type: ignore[attr-defined]
    return sampler