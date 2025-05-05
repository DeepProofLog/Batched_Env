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

Target = Literal["head", "relation", "tail"]
LongTensor: TypeAlias = torch.LongTensor  

class BasicNegativeSamplerDomain(BasicNegativeSampler):
    def __init__(self,
                 mapped_triples: torch.Tensor,
                 domain2idx: Dict[str, List[int]],
                 entity2domain: Dict[int, str],
                 num_negs_per_pos: int = 1,  # Note: not used in enumeration
                 filtered: bool = True,
                 corruption_scheme: List[str] = ['tail']):
        """
        Initialize the Domain-based negative sampler.
        """
        super().__init__(
            mapped_triples=mapped_triples,
            num_negs_per_pos=num_negs_per_pos,
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


    def corrupt_batch(self, positive_batch: LongTensor) -> LongTensor:  # noqa: D102
        batch_shape = positive_batch.shape[:-1]
        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)
        # Bind the total number of negatives to sample in this batch
        total_num_negatives = negative_batch.shape[0]

        # Equally corrupt all sides
        split_idx = int(math.ceil(total_num_negatives / len(self._corruption_indices)))
        # Do not detach, as no gradients should flow into the indices.
        for index, start in zip(self._corruption_indices, range(0, total_num_negatives, split_idx)):
            stop = min(start + split_idx, total_num_negatives)

            for i in range(start, stop):
                original_entity = negative_batch[i, index].item()
                original_domain = self.entity2domain[original_entity]

                possible_entities = self.domain_entities[original_domain]
                
                replacement_index = torch.randint(high=len(possible_entities), size=(1,), device=negative_batch.device).item()
                replacement_entity = possible_entities[replacement_index].item()
                while replacement_entity==original_entity: #make sure that the entity is different
                  replacement_index = torch.randint(high=len(possible_entities), size=(1,), device=negative_batch.device).item()
                  replacement_entity = possible_entities[replacement_index].item()
                negative_batch[i, index] = replacement_entity
        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)
    
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
            negative_batches.append(torch.stack(triple_negatives, dim=0))
        return negative_batches



class BasicNegativeSamplerCustom(BasicNegativeSampler):
    def __init__(
        self,
        mapped_triples: torch.Tensor,
        num_entities: int,
        num_relations: int,
        num_negs_per_pos: int = 1,
        filtered: bool = True,
        corruption_scheme: Optional[Collection[Target]] = None,
        padding_idx: Optional[Collection[int]] = None,
    ):
        super().__init__(
            mapped_triples=mapped_triples,
            num_negs_per_pos=num_negs_per_pos,
            filtered=filtered,
            corruption_scheme=corruption_scheme,
            num_entities=num_entities,
            num_relations=num_relations,
        )
        self.num_entities = num_entities
        self.padding_idx = set(padding_idx or ())

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
            # logging.warning(f"Cannot replace index {index} with max_index={max_index}. Skipping.")
            return

        original_values = batch[selection, index]

        # Determine the valid range size excluding the padding index
        # Example: if max_index=10, pad_idx=0 -> valid are {1..9}, size = 9
        # Example: if max_index=10, pad_idx=3 -> valid are {0,1,2, 4..9}, size = 9
        # General size = max_index - 1 (if pad_idx is within [0, max_index-1])
        # For simplicity and efficiency, we strongly assume pad_idx = 0 here.
        if pad_idx != 0:
             # Fallback to a potentially less efficient or more complex method
             # Or adapt the logic below, which becomes harder.
             raise NotImplementedError("Efficient replacement currently requires padding index to be 0.")

        # --- Optimized logic for pad_idx = 0 ---
        # We want to sample from {1, 2, ..., max_index - 1} excluding original_value

        num_valid_candidates = max_index - 1 # Size of the set {1, ..., max_index - 1}
        if num_valid_candidates <= 0: # Only index 0 exists (max_index=1)
             return

        # Sample indices from [0, num_valid_candidates - 1]
        replacement = torch.randint(
            high=num_valid_candidates,
            size=(size,),
            device=batch.device,
        )
        # Shift indices to values [1, max_index - 1]
        replacement = replacement + 1

        # Avoid original value:
        # Shift replacement up by 1 if it is >= original_value,
        # but ONLY if original_value was within the sampling range [1, max_index - 1].
        needs_shift = (replacement >= original_values) & (original_values > 0) # Check original > 0
        replacement = replacement + needs_shift.long()

        # Assign back to the selected slice
        batch[selection, index] = replacement
        # --- End of optimized logic ---


    def corrupt_batch(self, positive_batch: LongTensor) -> LongTensor:
        """
        Corrupts a batch of positive triples using the specified scheme,
        efficiently excluding the padding index (self.padding_idx, assumed 0)
        and the original triple value.
        """
        batch_shape = positive_batch.shape[:-1]

        # Clone positive batch for corruption (.repeat_interleave creates a copy)
        # Reshape to 2D: (batch_size * num_pos, 3)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)

        # Total number of negatives to generate for the whole batch
        total_num_negatives = negative_batch.shape[0]

        # Determine splits for corrupting different columns roughly equally
        num_corruption_indices = len(self._corruption_indices)
        if num_corruption_indices == 0: # Should not happen with validation in init
             return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3) # Return unchanged

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
        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)

    def corrupt_batch_all(self, positive_batch: torch.Tensor) -> List[torch.Tensor]:
        device = positive_batch.device
        # Build valid entity IDs
        all_entities = torch.arange(self.num_entities, device=device)
        if self.padding_idx:
            mask_pads = torch.ones_like(all_entities, dtype=torch.bool)
            pad = torch.tensor(list(self.padding_idx), device=device)
            mask_pads[pad] = False
            valid_entities = all_entities[mask_pads]
        else:
            valid_entities = all_entities

        batch_size, dims = positive_batch.shape
        # Prepare holders for each triple
        neg_batches: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        # For each index to corrupt, compute batch of candidates
        for idx in self._corruption_indices:
            # Expand batch: (B, M, 3)
            expanded = positive_batch.unsqueeze(1).expand(-1, valid_entities.size(0), -1).clone()
            expanded[..., idx] = valid_entities.unsqueeze(0)
            # Mask of valid corruptions: (B, M)
            orig_vals = positive_batch[:, idx].unsqueeze(1)
            keep_mask = expanded[..., idx] != orig_vals

            # Split per triple
            for i in range(batch_size):
                valid = expanded[i][keep_mask[i]]
                neg_batches[i].append(valid)

        # Concatenate per-triple negatives and return
        return [torch.cat(parts, dim=0) if parts else torch.empty((0, dims), dtype=torch.long, device=device)
                for parts in neg_batches]
    


def get_negatives(
    self: Union[BasicNegativeSamplerCustom, BasicNegativeSamplerDomain], # Added self
    inputs,
    padding_atoms: int,
    max_arity: int,
    device: torch.device,
    all_negatives: bool = False,
) -> Union[torch.Tensor, List[List[Term]]]:
    """
    Wrapper to accept either sub-indices tensor or Term-based states.

    Args:
        inputs: either a Tensor (B, padding_atoms, max_arity+1) or List[List[Term]]
        all_negatives: flag to pass to sampler
    Returns:
        Tensor of shape (B, num_negs, padding_atoms, max_arity+1)
    """

    sub_indices: torch.Tensor = inputs
    device = sub_indices.device
    B = sub_indices.size(0)
    # Extract rel, head, tail from atom slot 0
    rels = sub_indices[:, 0, 0]    # (B,)
    heads = sub_indices[:, 0, 1]   # (B,)
    tails = sub_indices[:, 0, 2]   # (B,)

    # Build positive batch: (B, 3)
    positive_batch = torch.stack([heads, rels, tails], dim=1)

    # Generate negatives
    if all_negatives:
        neg_batches = self.corrupt_batch_all(positive_batch)
    else:
        neg_batches = self.corrupt_batch(positive_batch)
    # Determine max negatives per query. If not all negatives, set to 2 (one head and one tail)
    max_negs = max(batch.size(0) for batch in neg_batches) if all_negatives else 2

    # Allocate output
    neg_subs = torch.full(
        (B, max_negs, padding_atoms, max_arity+1),
        fill_value=self.index_manager.padding_idx,
        dtype=torch.long,
        device=device
    )

    # Fill
    for i, batch in enumerate(neg_batches):
        M = batch.size(0)
        rel_i = batch[:, 1]
        head_i = batch[:, 0]
        tail_i = batch[:, 2]
        neg_subs[i, :M, 0, 0] = rel_i
        neg_subs[i, :M, 0, 1] = head_i
        neg_subs[i, :M, 0, 2] = tail_i

    return neg_subs


def get_negatives_from_states(
    self: Union[BasicNegativeSamplerCustom, BasicNegativeSamplerDomain], # Added self
    states: List[List[Term]],
    device: torch.device,
    all_negatives: bool = False,
    return_states: bool = True,
) -> torch.Tensor:
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
    elif isinstance(states, list) and isinstance(states[0], Term):
        states = [states]
    # Build sub-indices for each state
    subs = [self.index_manager.get_atom_sub_index(state) for state in states]
    # Stack to (B, padding_atoms, max_arity+1)
    pos_subs = torch.stack(subs, dim=0).to(device)
    # Call tensor-based sampler
    neg = self.get_negatives(pos_subs,
                        padding_atoms=pos_subs.size(1),
                        max_arity=pos_subs.size(2)-1,
                        device=device,
                        all_negatives=all_negatives)
    # Convert to Term-based states
    B = neg.size(0)
    if return_states:
        neg = self.index_manager.subindices_to_terms(neg)
        # print(f"Converted negatives to Term-based states {neg}")
        if B == 1:
            # If only one state, remove the first dimension
            neg = neg[0]
    else:
        if B == 1:
            # If only one state, remove the first dimension
            neg = neg.squeeze(0)
    return neg


def get_sampler(data_handler: DataHandler, 
                index_manager: IndexManager,
                triples_factory: TriplesFactory,
                corruption_scheme: Optional[Collection[Target]] = None,
                )-> Union[BasicNegativeSamplerCustom, BasicNegativeSamplerDomain]:

    if 'countries' in data_handler.dataset_name or 'ablation' in data_handler.dataset_name:
        domain2idx = {domain: [index_manager.constant_str2idx[e] for e in entities] for domain, entities in data_handler.domain2entity.items()}
        entity2domain: Dict[int, str] = {index_manager.constant_str2idx[e]: domain for domain, entities in data_handler.domain2entity.items() for e in entities}
        sampler = BasicNegativeSamplerDomain(mapped_triples=triples_factory.mapped_triples, 
                                            domain2idx=domain2idx,
                                            entity2domain=entity2domain,
                                            num_negs_per_pos=1,
                                            filtered=True,
                                            corruption_scheme=corruption_scheme)
    else:
        sampler = BasicNegativeSamplerCustom(   
            mapped_triples=triples_factory.mapped_triples, 
            num_entities=len(index_manager.constant_str2idx),
            num_relations=len(index_manager.predicate_str2idx),
            num_negs_per_pos=2,
            filtered=True,
            corruption_scheme=corruption_scheme,
            padding_idx=index_manager.padding_idx
        )

    # add the get_negatives method and the get_negatives_from_states method to the sampler
    sampler.index_manager = index_manager
    sampler.get_negatives = types.MethodType(get_negatives, sampler)
    sampler.get_negatives_from_states = types.MethodType(get_negatives_from_states, sampler)


    return sampler