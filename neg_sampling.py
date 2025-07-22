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
        batch_shape = positive_batch.shape[:-1]
        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(num_negs_per_pos, dim=0)
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
        return negative_batch.view(*batch_shape, num_negs_per_pos, 3)
    
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
        padding_idx: Optional[Collection[int]] = None,
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
        self.padding_idx = set(padding_idx or ())

        # Determine corruption scheme and indices
        self.corruption_scheme = corruption_scheme or (LABEL_HEAD, LABEL_TAIL)
        self._corruption_indices = [TARGET_TO_INDEX[side] for side in self.corruption_scheme]
        if any(idx not in {0, 1, 2} for idx in self._corruption_indices):
             raise ValueError(f"Invalid corruption index found in scheme: {self._corruption_indices}.")

        # Store the padding index to exclude
        self.padding_idx = padding_idx
        if self.padding_idx != {0}:
             # The efficient replacement function assumes padding_idx is 0
             # Needs modification if padding_idx can be != 0
             logging.warning(f"Current efficient implementation assumes padding_idx=0, but got {padding_idx}. Adapt _efficient_replacement if needed.")


    def random_replacement_(self, batch: LongTensor, index: int, selection: slice, size: int, max_index: int) -> None:
        """
        Replace a column of a batch of indices by random indices.

        :param batch: shape: `(*batch_dims, d)`
            the batch of indices
        :param index:
            the index (of the last axis) which to replace
        :param selection:
            a selection of the batch, e.g., a slice or a mask
        :param size:
            the size of the selection
        :param max_index:
            the maximum index value at the chosen position
        """
        # At least make sure to not replace the triples by the original value
        # To make sure we don't replace the {head, relation, tail} by the
        # original value we shift all values greater or equal than the original value by one up
        # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]
        replacement = torch.randint(
            high=max_index - 1,
            size=(size,),
            device=batch.device,
        )
        replacement += (replacement >= batch[selection, index]).long()
        batch[selection, index] = replacement

    def corrupt_batch(self, positive_batch: LongTensor, num_negs_per_pos: int) -> LongTensor:
        batch_shape = positive_batch.shape[:-1]

        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(num_negs_per_pos, dim=0)

        # Bind the total number of negatives to sample in this batch
        total_num_negatives = negative_batch.shape[0]

        # Equally corrupt all sides
        split_idx = int(math.ceil(total_num_negatives / len(self._corruption_indices)))

        # Do not detach, as no gradients should flow into the indices.
        for index, start in zip(self._corruption_indices, range(0, total_num_negatives, split_idx)):
            stop = min(start + split_idx, total_num_negatives)
            self.random_replacement_(
                batch=negative_batch,
                index=index,
                selection=slice(start, stop),
                size=stop - start,
                max_index=self.num_relations if index == 1 else self.num_entities,
            )

        return negative_batch.view(*batch_shape, num_negs_per_pos, 3)

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
    self: Union[BasicNegativeSamplerCustom, BasicNegativeSamplerDomain],
    inputs: torch.Tensor,
    padding_atoms: int,
    max_arity: int,
    device: torch.device,
    num_negs: Optional[int] = None,
) -> torch.Tensor:
    """
    Wrapper to accept either sub-indices tensor or Term-based states.

    Args:
        inputs: either a Tensor (B, padding_atoms, max_arity+1) or List[List[Term]]
        all_negatives: flag to pass to sampler
    Returns:
        Tensor of shape (B, num_negs, padding_atoms, max_arity+1)
    """

    sub_indices: torch.Tensor = inputs
    B = sub_indices.size(0)
    # Extract rel, head, tail from atom slot 0
    rels = sub_indices[:, 0, 0]    # (B,)
    heads = sub_indices[:, 0, 1]   # (B,)
    tails = sub_indices[:, 0, 2]   # (B,)

    # Build positive batch: (B, 3)
    positive_batch = torch.stack([heads, rels, tails], dim=1)

    if num_negs is not None:
        # --- FIXED LOGIC: Iterative Resampling ---
        # List to store the collected negatives for each positive in the batch
        collected_negatives = [[] for _ in range(B)]
        # Tensor to keep track of which positives still need negatives
        needs_negatives = torch.arange(B, device=device)
        
        # Keep generating negatives as long as there are positives that need them
        while len(needs_negatives) > 0:
            # Get the positive triples that still need more negatives
            pos_batch_to_corrupt = positive_batch[needs_negatives]
            
            # Generate ONE negative for each of the remaining positives
            # Shape: (num_still_needing, 1, 3)
            new_negs = self.corrupt_batch(pos_batch_to_corrupt, num_negs_per_pos=1)

            # Filter the new negatives if a filterer is available
            # The mask will be shape: (num_still_needing, 1)
            if self.filterer:
                new_negs = new_negs.to(self.filterer.mersenne.device)
                mask = self.filterer(new_negs)
            else:
                # If no filterer, all generated negatives are considered valid
                mask = torch.ones(new_negs.shape[:2], dtype=torch.bool, device=device)

            # Add the valid negatives to our collection
            still_needs_indices = []
            for i, neg_idx in enumerate(needs_negatives):
                # Check if the generated negative was valid
                if mask[i, 0]:
                    # Squeeze to remove the num_negs_per_pos=1 dimension
                    collected_negatives[neg_idx].append(new_negs[i].squeeze(0))
                
                # If this positive still has fewer than num_negs, we need to try again
                if len(collected_negatives[neg_idx]) < num_negs:
                    still_needs_indices.append(neg_idx)
            
            # Update the list of positives that still need negatives for the next iteration
            if still_needs_indices:
                needs_negatives = torch.tensor(still_needs_indices, dtype=torch.long, device=device)
            else:
                needs_negatives = []
        
        # At this point, each list in collected_negatives has num_negs tensors
        # Stack them into a single tensor for consistent processing.
        neg_batches = [torch.stack(negs, dim=0) for negs in collected_negatives]
    else:
        # Case 2: Generate all possible negatives within the domain/entity space
        neg_batches = self.corrupt_batch_all(positive_batch)

        # Apply the filterer to each batch in the list
        if self.filterer:
            filtered_batches = []
            for batch in neg_batches:
                if batch.numel() == 0:
                    filtered_batches.append(batch)
                    continue
                # The filterer expects a batch dimension, so we add and remove it
                # for each list item, e.g., shape (num_candidates, 3) -> (1, num_candidates, 3)
                batch = batch.to(self.filterer.mersenne.device)
                mask = self.filterer(batch.unsqueeze(0)).squeeze(0)
                filtered_batches.append(batch[mask])
            neg_batches = filtered_batches

    # --- Padding and Tensor Assembly ---
    # Determine the maximum number of negatives found across the batch after filtering
    if not neg_batches or all(b.numel() == 0 for b in neg_batches):
        max_negs = 0
    else:
        max_negs = max(batch.size(0) for batch in neg_batches)

    # Allocate the final output tensor, filled with padding
    neg_subs = torch.full(
        (B, max_negs, padding_atoms, max_arity + 1),
        fill_value=self.index_manager.padding_idx,
        dtype=torch.long,
        device=device,
    )

    # Fill the tensor with the filtered negative samples
    if max_negs > 0:
        for i, batch in enumerate(neg_batches):
            M = batch.size(0)
            if M == 0:
                continue
            # Deconstruct the (h, r, t) format back into your required sub-index format
            neg_subs[i, :M, 0, 1] = batch[:, 0]  # Head
            neg_subs[i, :M, 0, 0] = batch[:, 1]  # Relation
            neg_subs[i, :M, 0, 2] = batch[:, 2]  # Tail

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
    pos_subs = torch.stack(subs, dim=0).to(device)
    # Call tensor-based sampler
    neg_subs = self.get_negatives(pos_subs,
                                padding_atoms=pos_subs.size(1),
                                max_arity=pos_subs.size(2)-1,
                                device=device,
                                num_negs=num_negs)
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

    np_facts = np.array([[f.args[0], f.predicate, f.args[1]] for f in data_handler.facts], dtype=str)
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

    if sampler.filterer is not None:
        sampler.filterer = sampler.filterer.to(device)  # <-- add this

    # The BasicNegativeSamplerDomain has additional tensors that need to be moved.
    if isinstance(sampler, BasicNegativeSamplerDomain):
        for domain in sampler.domain_entities:
            sampler.domain_entities[domain] = sampler.domain_entities[domain].to(device)
    # add the get_negatives method and the get_negatives_from_states method to the sampler
    sampler.index_manager = index_manager
    sampler.get_negatives = types.MethodType(get_negatives, sampler)
    sampler.get_negatives_from_states = types.MethodType(get_negatives_from_states, sampler)


    return sampler