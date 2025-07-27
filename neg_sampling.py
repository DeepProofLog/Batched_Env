import torch
from utils import Term
from dataset import DataHandler
from index_manager import IndexManager
from pykeen.triples import TriplesFactory
from pykeen.sampling import BasicNegativeSampler
from typing_extensions import TypeAlias 
import math
from typing import Collection, Optional, Union,Literal, Dict, List
import types
from pykeen.constants import TARGET_TO_INDEX, LABEL_HEAD, LABEL_TAIL, LABEL_RELATION
import logging
import numpy as np

Target = Literal["head", "relation", "tail"]
LongTensor: TypeAlias = torch.LongTensor  

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
            (true_triples[:, 0].long() << 42)
            | (true_triples[:, 1].long() << 21)
            |  true_triples[:, 2].long()
        )
        self.register_buffer("_hashes_sorted", torch.sort(hashes.unique())[0])
        self._hashes = self._hashes_sorted      # ← NEW: alias for legacy access

    def forward(self, triples: torch.Tensor) -> torch.BoolTensor:
        """
        triples: (..., 3) → bool mask with the same leading shape (True = keep).
        Safe against out-of-range positions returned by searchsorted.
        """
        flat = triples.view(-1, 3)
        h    = (flat[:, 0].long() << 42) | (flat[:, 1].long() << 21) | flat[:, 2].long()

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
                 corruption_scheme: List[str] = ['tail']):
        """
        Initialize the Domain-based negative sampler.
        """
        super().__init__(
            mapped_triples=mapped_triples,
            filtered=filtered,
            corruption_scheme=corruption_scheme
        )
        self.domain2idx = domain2idx
        self.entity2domain = entity2domain
        self.idx2domain = {idx: domain for domain, idxs in domain2idx.items() for idx in idxs}
        self.domain_entities = {}
        for entity, domain in self.entity2domain.items():
            if domain not in self.domain_entities:
                self.domain_entities[domain] = []
            self.domain_entities[domain].append(entity)
        for domain in self.domain_entities:
            self.domain_entities[domain] = torch.tensor(self.domain_entities[domain], 
                                                        dtype=torch.long, 
                                                        device=mapped_triples.device)


    def corrupt_batch(self, positive_batch: LongTensor, num_negs_per_pos: int) -> LongTensor:
        batch_shape   = positive_batch.shape[:-1]
        neg_batch     = positive_batch.view(-1, 3).repeat_interleave(num_negs_per_pos, dim=0)

        # vectorise: look up domain for every entity once
        ent_ids       = torch.arange(max(self.entity2domain)+1, device=neg_batch.device)
        domain_of_ent = torch.tensor(
            [self.entity2domain.get(int(e), -1) for e in ent_ids.tolist()],
            device=neg_batch.device,
        )

        total = neg_batch.size(0)
        split = math.ceil(total / len(self._corruption_indices))

        for col, start in zip(self._corruption_indices, range(0, total, split)):
            stop   = min(start + split, total)
            slice_ = slice(start, stop)

            orig_ents   = neg_batch[slice_, col]
            dom_ids     = domain_of_ent[orig_ents]
            pools       = [self.domain_entities[self.idx2domain[int(d)]] for d in dom_ids.tolist()]

            # sample replacements in a single call per domain to avoid Python loop
            max_pool = max(len(p) for p in pools)
            indices  = torch.randint(
                high=max_pool, size=(stop - start,), device=neg_batch.device
            )
            repl     = torch.stack([
                pools[i][min(indices[i].item(), len(pools[i]) - 1)]
                for i in range(indices.numel())
            ])

            # guarantee difference
            clash      = repl == orig_ents
            while clash.any():
                new_idx          = torch.randint(high=max_pool, size=(clash.sum(),), device=neg_batch.device)
                repl[clash]      = torch.stack([
                    pools[i][min(new_idx[j].item(), len(pools[i]) - 1)]
                    for j, i in enumerate(torch.nonzero(clash, as_tuple=False)[:, 0])
                ])
                clash            = repl == orig_ents

            neg_batch[slice_, col] = repl

        return neg_batch.view(*batch_shape, num_negs_per_pos, 3)
    
    def corrupt_batch_all(self, positive_batch: torch.Tensor) -> List[torch.Tensor]:
        """
        For each positive triple, generate negatives by enumerating all entities in the same domain
        (excluding the original entity) for each corruption index.
        """
        negative_batches = []
        for triple in positive_batch:
            triple_negatives = []
            for index in self._corruption_indices:
                original_entity = triple[index].item()
                original_domain = self.entity2domain[original_entity]
                domain_candidates = self.domain_entities[original_domain]
                # Exclude the positive entity from the candidate list.
                candidates = domain_candidates[domain_candidates != original_entity]
                # Enumerate over all candidate entities.
                for candidate in candidates.tolist():
                    neg_triple = triple.clone()
                    neg_triple[index] = candidate
                    triple_negatives.append(neg_triple)
            if triple_negatives:
                negative_batches.append(torch.stack(triple_negatives, dim=0))
            else:
                negative_batches.append(torch.empty((0, 3), dtype=torch.long, device=positive_batch.device))
        return negative_batches



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
        batch: LongTensor,
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

        if pad_idx == 0:
            # very fast path (unchanged)
            rng = torch.randint(1, max_index, (size,), device=batch.device)
            shift = (rng >= orig) & (orig > 0)
            batch[selection, index] = rng + shift.long()
        else:
            # rare path – rejection sample until ok
            cand = torch.randint(0, max_index, (size,), device=batch.device)
            bad  = (cand == orig) | (cand == pad_idx)
            while bad.any():
                cand[bad] = torch.randint(0, max_index, (bad.sum(),), device=batch.device)
                bad = (cand == orig) | (cand == pad_idx)
            batch[selection, index] = cand


    def corrupt_batch(self, positive_batch: LongTensor, num_negs_per_pos: int) -> LongTensor:
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
        ent_pool = torch.arange(1, self.num_entities + 1, device=device)
        if self.pad_idx is not None:
            ent_pool = ent_pool[ent_pool != self.pad_idx]

        rel_pool = torch.arange(self.num_relations, device=device)

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
                reps[:, col] = cand
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
) -> torch.Tensor:
    """
    Generate negative samples for a batch of query states.

    Parameters
    ----------
    sub_indices
        Tensor with each query encoded as (padding_atoms, max_arity+1).  
        We only look at slot 0 (= the triple).
    padding_atoms / max_arity
        Needed for shape construction of the output tensor.
    num_negs
        * int  ➟ sample `num_negs` negatives per positive (old behaviour)  
        * None ➟ enumerate every legal corruption for every triple.

    Returns
    -------
    neg_subs : torch.Tensor
        Shape (B, M, padding_atoms, max_arity+1) where M is either
        `num_negs` (sampled mode) or the per-batch maximum when enumerating
        all corruptions.  Unused slots are padded with `self.index_manager.padding_idx`.
    """    
    
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
                dtype=torch.long,
                device=device,
            )
            return neg_subs

        # ➋  run the filter ONCE on the flattened tensor
        flat  = torch.cat(neg_batches, dim=0)              # (total_rows, 3)
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
        max_M = max(nb.size(0) for nb in filtered_batches)
        neg_subs = torch.full(
            (B, max_M, padding_atoms, max_arity + 1),
            fill_value=self.index_manager.padding_idx,
            dtype=torch.long,
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
    # 3⃣  Sampled negatives (old path, unchanged)
    # -------------------------------------------------------
    overshoot = 3
    cand = self.corrupt_batch(
        pos_batch,
        num_negs_per_pos=overshoot * num_negs
    ).view(-1, 3)                              # (B·overshoot·num_negs, 3)
    cand = cand[self.filterer(cand)]

    # Drop duplicates & true triples (if filtered=True in the sampler)
    # Keep first `num_negs` per positive & corruption side
    # ----------------------------------------------------
    chosen = cand.unique(dim=0, return_inverse=False)
    chosen = chosen[: B * num_negs]            # simple truncation

    # Reshape and pad out to fixed size
    chosen = chosen.view(B, -1, 3)             # (B, num_negs, 3)
    neg_subs = torch.full(
        (B, num_negs, padding_atoms, max_arity + 1),
        fill_value=self.index_manager.padding_idx,
        dtype=torch.long,
        device=device,
    )
    neg_subs[:, :, 0, 1] = chosen[:, :, 0]
    neg_subs[:, :, 0, 0] = chosen[:, :, 1]
    neg_subs[:, :, 0, 2] = chosen[:, :, 2]

    return neg_subs


def get_negatives_from_states(
    self: Union[BasicNegativeSamplerCustom, BasicNegativeSamplerDomain],
    states: List[List[Term]],
    device: torch.device,
    num_negs: Optional[int] = None,
    return_states: bool = True,
) -> Union[torch.Tensor, List[List[Term]]]:
    """
    Convert a list of Term-lists to sub-indices, generate negatives, and return sub-indices tensor.

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
    pos_subs = torch.stack(subs, dim=0).to(device)
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
        return neg_subs.squeeze(0) if B == 1 else neg_subs


def get_sampler(data_handler: DataHandler, 
                index_manager: IndexManager,
                corruption_scheme: Optional[Collection[Target]] = None,
                device: torch.device = torch.device("cpu"),
                )-> Union[BasicNegativeSamplerCustom, BasicNegativeSamplerDomain]:

    all_triples_for_filtering = data_handler.all_known_triples 
    np_facts = np.array([[f.args[0], f.predicate, f.args[1]] for f in all_triples_for_filtering], dtype=str)
    triples_factory = TriplesFactory.from_labeled_triples(triples=np_facts,
                                                        entity_to_id=index_manager.constant_str2idx,
                                                        relation_to_id=index_manager.predicate_str2idx,
                                                        compact_id=False,
                                                        create_inverse_triples=False)

    mapped_triples_cpu = triples_factory.mapped_triples.cpu()

    if 'countries' in data_handler.dataset_name or 'ablation' in data_handler.dataset_name:
        domain2idx = {domain: [index_manager.constant_str2idx[e] for e in entities] for domain, entities in data_handler.domain2entity.items()}
        entity2domain: Dict[int, str] = {index_manager.constant_str2idx[e]: domain for domain, entities in data_handler.domain2entity.items() for e in entities}
        sampler = BasicNegativeSamplerDomain(
                                            mapped_triples=mapped_triples_cpu, # Use CPU version for init
                                            domain2idx=domain2idx,
                                            entity2domain=entity2domain,
                                            filtered=True,
                                            corruption_scheme=corruption_scheme)
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


    # if sampler.filterer is not None:
    #     sampler.filterer = sampler.filterer.to(device)  # <-- add this

    # The BasicNegativeSamplerDomain has additional tensors that need to be moved.
    if isinstance(sampler, BasicNegativeSamplerDomain):
        for domain in sampler.domain_entities:
            sampler.domain_entities[domain] = sampler.domain_entities[domain].to(device)
    # add the get_negatives method and the get_negatives_from_states method to the sampler
    sampler.index_manager = index_manager
    sampler.get_negatives = types.MethodType(get_negatives, sampler)
    sampler.get_negatives_from_states = types.MethodType(get_negatives_from_states, sampler)


    return sampler