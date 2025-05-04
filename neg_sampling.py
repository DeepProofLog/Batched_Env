import torch
from dataset import DataHandler
from typing import Dict, List
from pykeen.sampling import BasicNegativeSampler
from typing_extensions import TypeAlias 
import math
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

from typing import Collection, Optional, Union,Literal
Target = Literal["head", "relation", "tail"]

class BasicNegativeSamplerCustom(BasicNegativeSampler):
    def __init__(
        self,
        mapped_triples: torch.Tensor,
        num_entities: int,
        num_negs_per_pos: int = 1,  # Note: not used in enumeration
        filtered: bool = True,
        corruption_scheme: Optional[Collection[Target]] = None,
        padding_indices: Optional[Collection[int]] = None,
    ):
        super().__init__(
            mapped_triples=mapped_triples,
            num_negs_per_pos=num_negs_per_pos,
            filtered=filtered,
            corruption_scheme=corruption_scheme
        )
        self.num_entities = num_entities
        self.padding_indices = set(padding_indices) if padding_indices else set()

    def corrupt_batch_all(self, positive_batch: torch.Tensor) -> List[torch.Tensor]:
        """
        For each positive triple, generate all possible negatives by replacing the target entity
        with every other entity in the entire entity set, excluding padding indices.
        """
        negative_batches = []
        all_entities = torch.arange(self.num_entities, device=positive_batch.device)
        valid_entities = [e for e in all_entities.tolist() if e not in self.padding_indices]
        valid_entities_tensor = torch.tensor(valid_entities, device=positive_batch.device)

        for triple in positive_batch:
            triple_negatives = []
            for index in self._corruption_indices:
                original_entity = triple[index].item()
                candidates = valid_entities_tensor[valid_entities_tensor != original_entity]
                for candidate in candidates:
                    neg_triple = triple.clone()
                    neg_triple[index] = candidate
                    triple_negatives.append(neg_triple)
            negative_batches.append(torch.stack(triple_negatives, dim=0))
        return negative_batches
    

def get_sampler(data_handler: DataHandler, 
                index_manager, 
                triples_factory,
                corruption_scheme: Optional[Collection[Target]] = None
                ):

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
            num_entities=triples_factory.num_entities,
            num_negs_per_pos=2,
            filtered=True,
            corruption_scheme=corruption_scheme,
            padding_indices={0},
        )

    return sampler