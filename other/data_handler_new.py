
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Literal
import torch

from index_manager_new import IndexManager


LongTensor = torch.LongTensor


@dataclass
class DatasetSplits:
    train_queries_idx: List[LongTensor]
    valid_queries_idx: List[LongTensor]
    test_queries_idx:  List[LongTensor]
    # Optional labels/depths if you have supervision
    train_labels: Optional[List[int]] = None
    valid_labels: Optional[List[int]] = None
    test_labels:  Optional[List[int]] = None


class DataHandler:
    """
    Pure data owner. Loads facts/rules/queries, converts to **index-only** tensors once,
    then serves tensors to env/sampler/model. Strings are kept only for debugging.
    """

    def __init__(self) -> None:
        # Raw (strings). These are optional after materialization.
        self.facts_str: List[Tuple[str, str, str]] = []
        self.rules_str: List[Tuple[Tuple[str, str, str], List[Tuple[str, str, str]]]] = []
        self.train_queries_str: List[Tuple[str, str, str]] = []
        self.valid_queries_str: List[Tuple[str, str, str]] = []
        self.test_queries_str:  List[Tuple[str, str, str]] = []

        # Index-only tensors (final products)
        self.facts_idx: Optional[LongTensor] = None           # [F, 3]
        self.rules_idx: Optional[LongTensor] = None           # [R, M, 3]
        self.rule_lens: Optional[LongTensor] = None           # [R]

        # Query splits as lists of [1,3] tensors (each query is a one-atom state)
        self.splits: Optional[DatasetSplits] = None

        # Optional: domain info for sampler (index-only, ragged)
        # maps predicate_id -> LongTensor of allowed entity ids
        self.allowed_heads_per_rel: Dict[int, LongTensor] = {}
        self.allowed_tails_per_rel: Dict[int, LongTensor] = {}

        # All known positives for filtering negatives
        self.all_known_triples_idx: Optional[LongTensor] = None   # [T, 3]

    # -----------------------------
    # Loaders (examples, replace with your own file I/O)
    # -----------------------------
    def add_fact(self, pred: str, a: str, b: str) -> None:
        self.facts_str.append((pred, a, b))

    def add_rule(self, head: Tuple[str, str, str], body: List[Tuple[str, str, str]]) -> None:
        self.rules_str.append((head, body))

    def add_query(self, split: Literal['train','valid','test'], pred: str, a: str, b: str) -> None:
        if split == 'train':
            self.train_queries_str.append((pred, a, b))
        elif split == 'valid':
            self.valid_queries_str.append((pred, a, b))
        else:
            self.test_queries_str.append((pred, a, b))

    # -----------------------------
    # Materialization (strings -> indices)
    # -----------------------------
    def materialize_indices(
        self,
        im: IndexManager,
        max_rule_atoms: int,
        device: torch.device,
        drop_strings: bool = True,
    ) -> None:
        """
        Convert all string data to index tensors using IndexManager.
        Moves resulting tensors to `device` once. Strings can be dropped to save RAM.
        """
        # facts
        facts_idx = im.state_to_tensor(self.facts_str)               # [F,3]
        im.set_facts(facts_idx)                                      # builds predicate index
        self.facts_idx = im.facts_idx                                # already on im.device
        # rules
        rules_idx, rule_lens = im.rules_to_tensor(self.rules_str, max_rule_atoms)
        self.rules_idx = rules_idx.to(device=device, dtype=torch.long, non_blocking=True)
        self.rule_lens = rule_lens.to(device=device, dtype=torch.long, non_blocking=True)

        # queries (each as [1,3])
        def make_query_list(qs: List[Tuple[str,str,str]]) -> List[LongTensor]:
            return [im.state_to_tensor([q]).to(device=device, dtype=torch.long) for q in qs]

        train_q = make_query_list(self.train_queries_str)
        valid_q = make_query_list(self.valid_queries_str)
        test_q  = make_query_list(self.test_queries_str)

        self.splits = DatasetSplits(train_q, valid_q, test_q)

        # all-known triples for filtered negative sampling
        # include facts and queries (you can add inferred facts if you want stricter filtering)
        ak = []
        if self.facts_idx is not None and self.facts_idx.numel() > 0:
            ak.append(self.facts_idx.detach().to('cpu'))
        if len(train_q) > 0:
            ak.append(torch.cat([q.squeeze(0) for q in train_q], dim=0).to('cpu'))
        if len(valid_q) > 0:
            ak.append(torch.cat([q.squeeze(0) for q in valid_q], dim=0).to('cpu'))
        if len(test_q) > 0:
            ak.append(torch.cat([q.squeeze(0) for q in test_q], dim=0).to('cpu'))
        self.all_known_triples_idx = torch.vstack(ak) if ak else torch.empty((0,3), dtype=torch.long)

        if drop_strings:
            self._drop_strings()

    def _drop_strings(self) -> None:
        self.facts_str.clear()
        self.rules_str.clear()
        self.train_queries_str.clear()
        self.valid_queries_str.clear()
        self.test_queries_str.clear()

    # -----------------------------
    # Split accessors
    # -----------------------------
    def get_split(self, split: Literal['train','valid','test']) -> List[LongTensor]:
        assert self.splits is not None, "Call materialize_indices() first"
        if split == 'train':
            return self.splits.train_queries_idx
        if split == 'valid':
            return self.splits.valid_queries_idx
        return self.splits.test_queries_idx

    # -----------------------------
    # Optional domain API (index-only)
    # -----------------------------
    def set_domain_heads(self, rel_id: int, allowed_entities: Iterable[int]) -> None:
        self.allowed_heads_per_rel[rel_id] = torch.tensor(list(allowed_entities), dtype=torch.long)

    def set_domain_tails(self, rel_id: int, allowed_entities: Iterable[int]) -> None:
        self.allowed_tails_per_rel[rel_id] = torch.tensor(list(allowed_entities), dtype=torch.long)
