import torch
try:
    # Try relative import first (when sb3/ is in sys.path)
    from sb3_utils import Term
    from sb3_dataset import DataHandler
    from sb3_index_manager import IndexManager
except ImportError:
    # Fallback to package import (when imported as sb3.sb3_neg_sampling)
    from sb3.sb3_utils import Term
    from sb3.sb3_dataset import DataHandler
    from sb3.sb3_index_manager import IndexManager
from pykeen.triples import TriplesFactory
from pykeen.sampling import BasicNegativeSampler
from typing_extensions import TypeAlias 
import math
from typing import Collection, Optional, Union,Literal, Dict, List, Tuple
import types
import warnings
from pykeen.constants import TARGET_TO_INDEX, LABEL_HEAD, LABEL_TAIL, LABEL_RELATION
import logging
import numpy as np

Target = Literal["head", "relation", "tail"]

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
        hashes = (
            (true_triples[:, 0].to(torch.int64) << 42)
            | (true_triples[:, 1].to(torch.int64) << 21)
            |  true_triples[:, 2].to(torch.int64)
        )
        self.register_buffer("_hashes_sorted", torch.sort(hashes.unique())[0])
        self._hashes = self._hashes_sorted      # ← NEW: alias for legacy access

    def forward(self, triples: torch.Tensor) -> torch.BoolTensor:
        """
        triples: (..., 3) → bool mask with the same leading shape (True = keep).
        Safe against out-of-range positions returned by searchsorted.
        """
        flat = triples.view(-1, 3)
        h    = (flat[:, 0].to(torch.int64) << 42) | (flat[:, 1].to(torch.int64) << 21) | flat[:, 2].to(torch.int64)

        pos  = torch.searchsorted(self._hashes_sorted, h)          # 1-D indices
        L    = self._hashes_sorted.numel()

        in_set = torch.zeros_like(h, dtype=torch.bool)             # default: not found
        valid  = pos < L                                           # only safe positions
        if valid.any():
            in_set[valid] = self._hashes_sorted[pos[valid]] == h[valid]

        return (~in_set).view(*triples.shape[:-1])                 # True = keep candidate


class BasicNegativeSamplerDomain(BasicNegativeSampler):
    def __init__(self,
                 mapped_triples: torch.Tensor,
                 domain2idx: Dict[str, List[int]],
                 entity2domain: Dict[int, str],
                 filtered: bool = True,
                 corruption_scheme: List[str] = ['tail'],
                 device: torch.device = torch.device("cpu")
                 ):
        """
        Domain-aware negative sampler.

        Key optimization: all domain lookups are precomputed once and stored
        as device tensors so that batch corruption is fully vectorized.
        """
        super().__init__(
            mapped_triples=mapped_triples,
            filtered=filtered,
            corruption_scheme=corruption_scheme
        )
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
            row = self.domain_padded[d, : self.domain_len[d]]
            # guard: row can be empty for corner cases
            if row.numel() == 0:
                continue
            row_long = row.to(torch.int64)
            ent2dom[row_long] = d
            # position of each entity in its domain
            pos_in_dom[row_long] = torch.arange(row.numel(), device=device, dtype=storage_dtype)

        self.ent2dom = ent2dom
        self.pos_in_dom = pos_in_dom

        # Infer number of relations from mapped triples (works with non-compact ids too)
        # Relations are stored in column 1 of mapped_triples
        self.num_relations = int(mapped_triples[:, 1].max().item()) + 1 if mapped_triples.numel() > 0 else 0

        # Validate corruption indices (0=head, 1=rel, 2=tail)
        self.corruption_scheme = corruption_scheme or ['head', 'tail']
        self._corruption_indices = [TARGET_TO_INDEX[c] for c in self.corruption_scheme]
        if any(idx not in (0, 1, 2) for idx in self._corruption_indices):
            raise ValueError(f"Invalid corruption index in scheme: {self._corruption_indices}")

    def _replace_relation_uniform_(self, batch: torch.Tensor, sel: slice) -> None:
        """Uniformly replace relations in batch[sel, 1] excluding the original id."""
        if self.num_relations <= 1:
            return
        orig = batch[sel, 1]
        # Draw from [0, num_rel-1] \ {orig}; use add-one trick
        # NB: relations here are 0-based ids in mapped_triples
        high = torch.full_like(orig, self.num_relations - 1)
        # sample per-row: floor(rand * high)
        rnd = torch.floor(torch.rand_like(orig, dtype=torch.float32) * high.to(torch.float32)).to(torch.int64)
        # shift values >= orig
        batch[sel, 1] = rnd + (rnd >= orig)

    def corrupt_batch(self, positive_batch: torch.Tensor, num_negs_per_pos: int) -> torch.Tensor:
        """
        Vectorized corruption for head/tail within the same domain and (optional) relation.
        Uses [B, K] shaped random generation for consistent RNG consumption.
        """
        B = positive_batch.shape[0]
        K = num_negs_per_pos
        device = positive_batch.device
        
        if B == 0:
            return positive_batch.unsqueeze(1).repeat(1, K, 1)
        
        # Create output tensor [B, K, 3]
        neg = positive_batch.unsqueeze(1).repeat(1, K, 1)
        
        # Split K samples across corruption indices
        n_cols = len(self._corruption_indices)
        samples_per_col = K // n_cols
        remainder = K % n_cols
        
        start_idx = 0
        for i, col in enumerate(self._corruption_indices):
            count = samples_per_col + (1 if i < remainder else 0)
            end_idx = start_idx + count
            
            if count == 0:
                continue
            
            if col == 1:
                # Relation corruption
                if self.num_relations > 1:
                    orig = neg[:, start_idx:end_idx, 1]  # [B, count]
                    rnd = torch.randint(0, self.num_relations - 1, orig.shape, device=device)
                    neg[:, start_idx:end_idx, 1] = rnd + (rnd >= orig)
            else:
                # Entity corruption within domain
                orig = neg[:, start_idx:end_idx, col]  # [B, count]
                orig_flat = orig.reshape(-1)
                valid = orig_flat > 0
                
                if valid.any():
                    orig_valid = orig_flat[valid].to(torch.int64)
                    d_ids = self.ent2dom[orig_valid].to(torch.int64)
                    pool_len = self.domain_len[d_ids]
                    pos_orig = self.pos_in_dom[orig_valid].to(torch.int64)
                    
                    can_sample = pool_len > 1
                    result = orig_valid.clone()
                    
                    if can_sample.any():
                        Lm1 = (pool_len[can_sample] - 1).float()
                        rnd = torch.floor(torch.rand(can_sample.sum(), device=device) * Lm1).long()
                        adj = rnd + (rnd >= pos_orig[can_sample])
                        result[can_sample] = self.domain_padded[d_ids[can_sample], adj].to(torch.int64)
                    
                    out_flat = orig_flat.clone()
                    out_flat[valid] = result.to(orig_flat.dtype)
                    neg[:, start_idx:end_idx, col] = out_flat.reshape(orig.shape)
            
            start_idx = end_idx
        
        return neg

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
                    rels = torch.arange(
                        self.num_relations,
                        device=triple.device,
                        dtype=triple.dtype,
                    )
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
                    d = self.ent2dom[e].item()
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

class BasicNegativeSamplerCustom(BasicNegativeSampler):
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
            num_negs_per_pos=1,
            filtered=filtered,
            corruption_scheme=corruption_scheme,
            num_entities=num_entities,
            num_relations=num_relations,
        )
        self.num_entities = num_entities
        self.pad_idx: int = int(padding_idx)
        if self.pad_idx != 0:
            logging.warning(
                "Efficient replacement assumes pad_idx = 0; "
                "fallback to slower rejection-sampling."
            )
        # Determine corruption scheme and indices
        self.corruption_scheme = corruption_scheme or (LABEL_HEAD, LABEL_TAIL)
        self._corruption_indices = [TARGET_TO_INDEX[side] for side in self.corruption_scheme]
        if any(idx not in {0, 1, 2} for idx in self._corruption_indices):
             raise ValueError(f"Invalid corruption index found in scheme: {self._corruption_indices}.")

        # Store the padding index to exclude
        self.padding_idx = padding_idx
        if self.padding_idx != 0:
             # The efficient replacement function assumes padding_idx is 0
             # Needs modification if padding_idx can be != 0
             logging.warning(f"Current efficient implementation assumes padding_idx=0, but got {padding_idx}. Adapt _efficient_replacement if needed.")


    @staticmethod
    def _efficient_replacement(
        batch: torch.Tensor,
        index: int,
        selection: slice,
        size: int,
        max_index: int,
        pad_idx: int # Pass padding index explicitly
    ) -> None:
        """
        Efficiently replace batch[selection, index] with random ints from [0, max_index-1],
        excluding the original value and the specified pad_idx.

        Assumes pad_idx = 0 for optimal efficiency in current implementation.
        """
        if max_index <= 1:
            # Cannot sample anything if max_index is 0 or 1
            logging.warning(f"Cannot replace index {index} with max_index={max_index}. Skipping.")
            return

        orig = batch[selection, index]
        orig_long = orig.to(torch.int64)

        if pad_idx == 0:
            # very fast path (unchanged)
            rng = torch.randint(1, max_index, (size,), device=batch.device, dtype=torch.int64)
            shift = (rng >= orig_long) & (orig_long > 0)
            replacement = (rng + shift.to(rng.dtype)).to(batch.dtype)
            batch[selection, index] = replacement
        else:
            # rare path – rejection sample until ok
            cand = torch.randint(0, max_index, (size,), device=batch.device, dtype=torch.int64)
            bad  = (cand == orig_long) | (cand == pad_idx)
            while bad.any():
                cand[bad] = torch.randint(0, max_index, (bad.sum(),), device=batch.device, dtype=torch.int64)
                bad = (cand == orig_long) | (cand == pad_idx)
            batch[selection, index] = cand.to(batch.dtype)


    def corrupt_batch(self, positive_batch: torch.Tensor, num_negs_per_pos: int) -> torch.Tensor:
        """
        Corrupts a batch of positive triples using the specified scheme,
        efficiently excluding the padding index (self.padding_idx, assumed 0)
        and the original triple value.
        """
        batch_shape = positive_batch.shape[:-1]

        # Clone positive batch for corruption (.repeat_interleave creates a copy)
        # Reshape to 2D: (batch_size * num_pos, 3)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(num_negs_per_pos, dim=0)

        # Total number of negatives to generate for the whole batch
        total_num_negatives = negative_batch.shape[0]

        # Determine splits for corrupting different columns roughly equally
        num_corruption_indices = len(self._corruption_indices)
        if num_corruption_indices == 0: # Should not happen with validation in init
             return negative_batch.view(*batch_shape, num_negs_per_pos, 3) # Return unchanged

        split_idx = math.ceil(total_num_negatives / num_corruption_indices)

        # Apply corruption column by column
        current_start = 0
        for index in self._corruption_indices:
            # Determine the slice of the batch to corrupt for this column
            stop = min(current_start + split_idx, total_num_negatives)
            if stop <= current_start: # No samples left for this index
                 continue
            selection = slice(current_start, stop)
            size = stop - current_start

            # Determine max index based on column (relation or entity)
            current_max_index = self.num_relations if index == 1 else self.num_entities

            # Call the modified, efficient replacement function
            self._efficient_replacement(
                batch=negative_batch,
                index=index,
                selection=selection,
                size=size,
                max_index=current_max_index,
                pad_idx=self.padding_idx # Pass the padding index
            )
            # Update start for the next iteration
            current_start = stop

        # Reshape back to (..., num_negs_per_pos, 3)
        return negative_batch.view(*batch_shape, num_negs_per_pos, 3)

    def corrupt_batch_all(self, positive_batch: torch.Tensor) -> List[torch.Tensor]:
        """
        Exhaustively enumerate every legal head / relation / tail corruption
        defined by ``self._corruption_indices`` for each triple *individually*.

        Returns
        -------
        list(torch.Tensor)
            Length = B.  The i-th item has shape (Mi, 3) with Mi ≥ 0 and
            contains **only** negatives (no positive triple rows).
        """
        device = positive_batch.device
        negatives: List[torch.Tensor] = []

        # pre-compute pools
        pool_dtype = positive_batch.dtype
        ent_pool = torch.arange(1, self.num_entities + 1, device=device, dtype=pool_dtype)
        if self.pad_idx is not None:
            ent_pool = ent_pool[ent_pool != self.pad_idx]

        rel_pool = torch.arange(self.num_relations, device=device, dtype=pool_dtype)

        for triple in positive_batch:              # loop over batch (B is usually small)
            triple_negs = []
            for col in self._corruption_indices:
                pool = rel_pool if col == 1 else ent_pool
                # exclude the positive value in this slot
                cand = pool[pool != triple[col]]
                if cand.numel() == 0:
                    continue
                # broadcast-replace
                reps = triple.repeat(cand.numel(), 1)
                reps[:, col] = cand.to(reps.dtype)
                triple_negs.append(reps)
            if triple_negs:
                negatives.append(torch.cat(triple_negs, dim=0))
            else:                                   # fully padded row
                negatives.append(triple.new_empty((0, 3)))
        return negatives

def get_negatives(
    self,
    sub_indices: torch.Tensor,
    padding_atoms: int,
    max_arity: int,
    device: torch.device,
    num_negs: Optional[int] = None,        # ← None ⇒ enumerate *all* corruptions
    debug: bool = True,                   # ← NEW: debug mode
) -> torch.Tensor:
    """
    Generate negative samples for a batch of query states.

    Parameters
    ----------
    sub_indices
        Tensor with each query encoded as (padding_atoms, max_arity+1).  
        We only look at slot 0 (= the triple).
    padding_atoms / max_arity
        Needed for shape construction of the output 
    num_negs
        * int  ➟ sample `num_negs` negatives per positive (old behaviour)  
        * None ➟ enumerate every legal corruption for every triple.
    debug
        If True, prints detailed information when actual_num_candidates < expected_num_candidates.

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

    if self.filterer._hashes_sorted.device != sub_indices.device:
        self.filterer = self.filterer.to(sub_indices.device)

    B = sub_indices.size(0)

    # -------------------------------------------------------
    # 1⃣  Extract (r,h,t) triples from the first atom slot
    # -------------------------------------------------------
    rels  = sub_indices[:, 0, 0]  # (B,)
    heads = sub_indices[:, 0, 1]
    tails = sub_indices[:, 0, 2]
    pos_batch = torch.stack([heads, rels, tails], dim=1)  # (B, 3)

    # -------------------------------------------------------
    # 2⃣  Enumerate *all* corruptions  (num_negs is None)
    # -------------------------------------------------------
    if num_negs is None:
        # ➊  let the concrete sampler enumerate candidates
        neg_batches = self.corrupt_batch_all(pos_batch)    # list length B
        lengths     = [nb.size(0) for nb in neg_batches]
        total_rows  = sum(lengths)

        if total_rows == 0:                                # nothing to pad
            max_M = 0
            neg_subs = torch.full(
                (B, 0, padding_atoms, max_arity + 1),
                fill_value=self.index_manager.padding_idx,
                dtype=expected_dtype,
                device=device,
            )
            return neg_subs

        # ➋  run the filter ONCE on the flattened tensor
        flat  = torch.cat(neg_batches, dim=0).to(expected_dtype)  # (total_rows, 3)
        mask  = self.filterer(flat)                        # True ⇒ keep

        # ➌  slice the single mask back into per-batch tensors
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

        # ➍  pad to equal length and write into output tensor
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
    # -------------------------------------------------------
    # 3⃣  Sampled negatives (robust batch-aligned path) - BATCHED
    # -------------------------------------------------------
    overshoot = 1  # No overshoot for parity with tensor sampler
    # cand: (B, overshoot*num_negs, 3)
    cand = self.corrupt_batch(
        pos_batch,
        num_negs_per_pos=overshoot * num_negs
    )
    
    # Flatten for filtering: (B*M, 3)
    M = cand.size(1)
    cand_flat = cand.view(-1, 3).to(expected_dtype)
    
    # Filter: (B*M,)
    mask_flat = self.filterer(cand_flat)
    
    # Reshape mask: (B, M)
    mask = mask_flat.view(B, M)
    
    # -------------------------------------------------------
    # Batched unique-per-row using sorting + deduplication
    # -------------------------------------------------------
    # Create hash for each candidate: h*N^2 + r*N + t where N = max_entity_id+1
    # This creates a unique 64-bit key for each triple
    N = max(cand_flat[:, 0].max().item(), cand_flat[:, 1].max().item(), 
            cand_flat[:, 2].max().item()) + 1
    N = max(N, 1)  # Ensure N >= 1
    
    # Create hashes for sorting: (B, M)
    cand_hashes = (cand[:, :, 0].long() * N * N + 
                   cand[:, :, 1].long() * N + 
                   cand[:, :, 2].long())  # (B, M)
    
    # Set invalid entries to a very large value so they sort to end
    LARGE_VAL = 2**62
    cand_hashes_masked = torch.where(mask, cand_hashes, 
                                      torch.full_like(cand_hashes, LARGE_VAL))
    
    # Sort within each row: (B, M), indices: (B, M)
    sorted_hashes, sort_indices = torch.sort(cand_hashes_masked, dim=1)
    
    # Find unique boundaries: an entry is unique if it differs from previous
    # First entry of each row is always unique (if valid)
    shifted = torch.cat([
        torch.full((B, 1), LARGE_VAL + 1, dtype=sorted_hashes.dtype, device=device),
        sorted_hashes[:, :-1]
    ], dim=1)
    is_unique = (sorted_hashes != shifted) & (sorted_hashes < LARGE_VAL)  # (B, M)
    
    # Initialize output with padding
    neg_subs = torch.full(
        (B, num_negs, padding_atoms, max_arity + 1),
        fill_value=self.index_manager.padding_idx,
        dtype=expected_dtype,
        device=device
    )
    
    # Gather sorted candidates using sort_indices
    # Expand sort_indices for gathering: (B, M, 3)
    sort_indices_exp = sort_indices.unsqueeze(-1).expand(-1, -1, 3)
    sorted_cand = torch.gather(cand.to(expected_dtype), 1, sort_indices_exp)  # (B, M, 3)
    
    # Create position indices for unique entries within each row
    # cumsum of is_unique gives 1-based position of each unique entry
    unique_positions = is_unique.long().cumsum(dim=1)  # (B, M), values in [1, unique_count]
    
    # Valid unique entries are those that are unique and within num_negs
    valid_unique = is_unique & (unique_positions <= num_negs)
    
    # Use advanced indexing to place valid unique candidates
    # Create flat views for efficient indexing
    valid_mask_flat = valid_unique.view(-1)
    target_pos_flat = (unique_positions - 1).view(-1)  # 0-indexed
    sorted_cand_flat = sorted_cand.view(-1, 3)
    
    # Create batch indices
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, M).reshape(-1)
    
    # Filter to only valid entries
    valid_batch = batch_idx[valid_mask_flat]
    valid_pos = target_pos_flat[valid_mask_flat]
    valid_vals = sorted_cand_flat[valid_mask_flat]
    
    # Create output buffer for candidates: (B, num_negs, 3)
    output_cand = torch.zeros((B, num_negs, 3), dtype=expected_dtype, device=device)
    
    # Place valid entries using advanced indexing
    if valid_batch.numel() > 0:
        output_cand[valid_batch, valid_pos] = valid_vals
    
    # Fill neg_subs from output_cand
    # output_cand is (B, num_negs, 3) in (h, r, t) format
    # neg_subs is (B, num_negs, padding_atoms, max_arity+1) in (r, h, t) format at [:, :, 0, :]
    neg_subs[:, :, 0, 1] = output_cand[:, :, 0]  # head
    neg_subs[:, :, 0, 0] = output_cand[:, :, 1]  # relation  
    neg_subs[:, :, 0, 2] = output_cand[:, :, 2]  # tail

    return neg_subs


def get_negatives_from_states(
    self: Union[BasicNegativeSamplerCustom, BasicNegativeSamplerDomain],
    states: List[List[Term]],
    device: torch.device,
    num_negs: Optional[int] = None,
    return_states: bool = True,
) -> Union[torch.Tensor, List[List[Term]]]:
    """
    Convert a list of Term-lists to sub-indices, generate negatives, and return sub-indices 

    Args:
        states: list of B query states, each a list of Term
        all_negatives: whether to return all possible negatives or a fixed number per query
    Returns:
        Tensor of shape (B, num_negs_per_pos, padding_atoms, max_arity+1)
    """
    # if it is only one state (List[Term]), convert it to a list of states
    if isinstance(states, Term):
        states = [[states]]
    elif isinstance(states, list) and states and isinstance(states[0], Term):
        states = [states]
    # Build sub-indices for each state
    subs = [self.index_manager.get_atom_sub_index(state) for state in states]
    # Stack to (B, padding_atoms, max_arity+1)
    target_device = self.filterer._hashes_sorted.device 
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
    target_device = self.filterer._hashes_sorted.device 
    pos_subs = torch.stack(subs, dim=0).to(device)
    
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

def get_sampler(
    data_handler: DataHandler, 
    index_manager: IndexManager,
    corruption_scheme: Optional[Collection[Target]] = None,
    device: torch.device = torch.device("cpu"),
    corruption_mode: bool = True,
) -> Union[BasicNegativeSamplerCustom, BasicNegativeSamplerDomain]:

    all_triples_for_filtering = data_handler.all_known_triples 
    np_facts = np.array([[f.args[0], f.predicate, f.args[1]] for f in all_triples_for_filtering], dtype=str)
    triples_factory = TriplesFactory.from_labeled_triples(triples=np_facts,
                                                        entity_to_id=index_manager.constant_str2idx,
                                                        relation_to_id=index_manager.predicate_str2idx,
                                                        compact_id=False,
                                                        create_inverse_triples=False)

    mapped_triples_cpu = triples_factory.mapped_triples.cpu()

    if not corruption_mode:
        sampler = BasicNegativeSamplerCustom(
            mapped_triples=mapped_triples_cpu,
            num_entities=len(index_manager.constant_str2idx),
            num_relations=len(index_manager.predicate_str2idx),
            corruption_scheme=corruption_scheme,
        )
    elif ('countries' in data_handler.dataset_name or 'ablation' in data_handler.dataset_name):
        if data_handler.domain2entity is None:
            raise ValueError("Domain information is required for domain-aware negative sampling.")
        # Build domain2idx, skipping entities that aren't in the index
        domain2idx = {}
        skipped_entities = []
        for domain, entities in data_handler.domain2entity.items():
            indexed_entities = []
            for e in entities:
                if e in index_manager.constant_str2idx:
                    indexed_entities.append(index_manager.constant_str2idx[e])
                else:
                    skipped_entities.append(e)
            if indexed_entities:
                domain2idx[domain] = indexed_entities
        
        if skipped_entities:
            print(f"Note: Skipped {len(skipped_entities)} entities not in index: {skipped_entities[:10]}{'...' if len(skipped_entities) > 10 else ''}")
        
        # Build entity2domain only for indexed entities
        entity2domain: Dict[int, str] = {}
        for domain, entities in data_handler.domain2entity.items():
            for e in entities:
                if e in index_manager.constant_str2idx:
                    entity2domain[index_manager.constant_str2idx[e]] = domain
        
        sampler = BasicNegativeSamplerDomain(
            mapped_triples=mapped_triples_cpu,
            domain2idx=domain2idx,
            entity2domain=entity2domain,
            filtered=True,
            corruption_scheme=corruption_scheme,
            device=device # --- ADD THIS ---
        )
    else:
        sampler = BasicNegativeSamplerCustom(   
            mapped_triples=mapped_triples_cpu, # Use CPU version for init
            num_entities=len(index_manager.constant_str2idx),
            num_relations=len(index_manager.predicate_str2idx),
            filtered=True,
            corruption_scheme=corruption_scheme,
            padding_idx=index_manager.padding_idx
        )

    # After successful initialization, move the sampler's tensors to the target device.
    sampler.mapped_triples = triples_factory.mapped_triples.to(device)

    sampler.filterer = SortedHashTripleFilter(sampler.mapped_triples).to(device)

    # add the get_negatives method and the get_negatives_from_states method to the sampler
    sampler.index_manager = index_manager
    sampler.get_negatives = types.MethodType(get_negatives, sampler)
    sampler.get_negatives_from_states = types.MethodType(get_negatives_from_states, sampler)
    sampler.get_negatives_from_states_separate = types.MethodType(get_negatives_from_states_separate, sampler)

    return sampler
