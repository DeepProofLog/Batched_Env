"""
Refactored DataHandler - Clean, maintainable, and efficient.

This module provides a pure data owner that loads facts/rules/queries,
converts to index-only tensors, and serves them to environment/sampler/model.
"""

from __future__ import annotations

import os
from os.path import join
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Literal, Set
import torch

from index_manager import IndexManager
from utils import Term, Rule, get_atom_from_string, get_rule_from_string


LongTensor = torch.LongTensor


@dataclass
class MaterializedSplit:
    """Tensorized representation of a dataset split for the environment."""
    queries: torch.LongTensor        # [N, L, max_arity+1]
    labels: torch.LongTensor         # [N]
    depths: torch.LongTensor         # [N]

    def __len__(self) -> int:
        return int(self.queries.shape[0])


class DataHandler:
    """
    Pure data owner. Loads facts/rules/queries, converts to index-only tensors once,
    then serves tensors to env/sampler/model. Strings are kept only for debugging.
    
    Key improvements over original:
    - Cleaner separation: strings vs indices
    - Single materialization pass
    - No redundant conversions
    - Optional string dropping to save RAM
    """

    def __init__(
        self,
        dataset_name: str = None,
        base_path: str = "data",
        janus_file: str = None,
        train_file: str = "train.txt",
        valid_file: str = "valid.txt",
        test_file: str = "test.txt",
        rules_file: str = "rules.txt",
        facts_file: str = None,
        n_train_queries: Optional[int] = None,
        n_eval_queries: Optional[int] = None,
        n_test_queries: Optional[int] = None,
        train_depth: Optional[Set[int]] = None,
        valid_depth: Optional[Set[int]] = None,
        test_depth: Optional[Set[int]] = None,
        load_depth_info: bool = True,
        prob_facts: bool = False,
        topk_facts: Optional[int] = None,
        topk_facts_threshold: Optional[float] = None,
        filter_queries_by_rules: bool = True,
        corruption_mode: bool = False,
        deterministic: bool = False,
    ) -> None:
        """
        Initialize DataHandler and optionally load data from files.
        
        Args:
            dataset_name: Name of dataset (e.g., 'wn18rr', 'family')
            base_path: Base directory containing dataset folders
            janus_file: Prolog file (for facts)
            train_file: Training queries file
            valid_file: Validation queries file
            test_file: Test queries file
            rules_file: Rules file
            facts_file: Facts file (defaults to janus_file)
            n_train_queries: Limit on training queries
            n_eval_queries: Limit on validation queries
            n_test_queries: Limit on test queries
            train_depth: Filter training queries by depth
            valid_depth: Filter validation queries by depth
            test_depth: Filter test queries by depth
            prob_facts: Whether to load probabilistic facts
            topk_facts: Top K probabilistic facts to load
            topk_facts_threshold: Score threshold for probabilistic facts
            filter_queries_by_rules: Filter queries whose predicates don't match rule heads
            corruption_mode: Enable domain mapping for countries/ablation datasets
        """
        # Raw (strings). These are optional after materialization.
        self.facts_str: List[Tuple[str, str, str]] = []
        self.rules_str: List[Tuple[Tuple[str, str, str], List[Tuple[str, str, str]]]] = []
        self.train_queries_str: List[Tuple[str, str, str]] = []
        self.valid_queries_str: List[Tuple[str, str, str]] = []
        self.test_queries_str: List[Tuple[str, str, str]] = []
        
        # Keep original Term/Rule objects for backward compatibility
        self.facts: List[Term] = []
        self.rules: List[Rule] = []
        self.train_queries: List[Term] = []
        self.valid_queries: List[Term] = []
        self.test_queries: List[Term] = []
        
        # Depth information for queries
        self.train_depths: List[int] = []
        self.valid_depths: List[int] = []
        self.test_depths: List[int] = []
        self.train_labels: List[int] = []
        self.valid_labels: List[int] = []
        self.test_labels: List[int] = []
        self._materialized_splits: Dict[str, MaterializedSplit] = {}

        # Optional: domain info for sampler (index-only, ragged)
        # maps predicate_id -> LongTensor of allowed entity ids
        self.allowed_heads_per_rel: Dict[int, LongTensor] = {}
        self.allowed_tails_per_rel: Dict[int, LongTensor] = {}
        
        # Domain mapping for countries/ablation datasets (string-based)
        self.entity2domain: Optional[Dict[str, str]] = None
        self.domain2entity: Optional[Dict[str, List[str]]] = None

        # All known positives for filtering negatives
        self.all_known_triples_idx: Optional[LongTensor] = None   # [T, 3]

        # Metadata
        self.dataset_name = dataset_name
        self.constants: Set[str] = set()
        self.predicates: Set[str] = set()
        self.max_arity: int = 2  # Binary relations by default
        self.deterministic = deterministic  # Use sorted lists instead of sets for deterministic ordering
        
        # Probabilistic facts
        self._probabilistic_facts: List[Term] = []
        self.load_depth_info = bool(load_depth_info)
        
        # Load data if dataset specified
        if dataset_name is not None:
            self.load_dataset(
                dataset_name=dataset_name,
                base_path=base_path,
                janus_file=janus_file,
                train_file=train_file,
                valid_file=valid_file,
                test_file=test_file,
                rules_file=rules_file,
                facts_file=facts_file,
                n_train_queries=n_train_queries,
                n_eval_queries=n_eval_queries,
                n_test_queries=n_test_queries,
                train_depth=train_depth,
                valid_depth=valid_depth,
                test_depth=test_depth,
                load_depth_info=self.load_depth_info,
                prob_facts=prob_facts,
                topk_facts=topk_facts,
                topk_facts_threshold=topk_facts_threshold,
                filter_queries_by_rules=filter_queries_by_rules,
                corruption_mode=corruption_mode,
            )

    # -----------------------------
    # Data Loading
    # -----------------------------
    def load_dataset(
        self,
        dataset_name: str,
        base_path: str = "data",
        janus_file: str = None,
        train_file: str = "train.txt",
        valid_file: str = "valid.txt",
        test_file: str = "test.txt",
        rules_file: str = "rules.txt",
        facts_file: str = None,
        n_train_queries: Optional[int] = None,
        n_eval_queries: Optional[int] = None,
        n_test_queries: Optional[int] = None,
        train_depth: Optional[Set[int]] = None,
        valid_depth: Optional[Set[int]] = None,
        test_depth: Optional[Set[int]] = None,
        load_depth_info: bool = True,
        prob_facts: bool = False,
        topk_facts: Optional[int] = None,
        topk_facts_threshold: Optional[float] = None,
        filter_queries_by_rules: bool = False,
        corruption_mode: bool = False,
    ) -> None:
        """Load dataset from files."""
        dataset_path = join(base_path, dataset_name)
        
        # Load facts
        if facts_file is None:
            facts_file = janus_file
        if facts_file:
            facts_path = join(dataset_path, facts_file)
            self._load_facts_from_file(facts_path)
        
        # Load probabilistic facts if requested
        if prob_facts:
            self._load_probabilistic_facts(
                dataset_name=dataset_name,
                topk_limit=topk_facts,
                score_threshold=topk_facts_threshold,
            )
        
        # Load rules
        rules_path = join(dataset_path, rules_file)
        if os.path.exists(rules_path):
            self._load_rules_from_file(rules_path)
        
        # Load queries
        train_path = join(dataset_path, train_file)
        valid_path = join(dataset_path, valid_file)
        test_path = join(dataset_path, test_file)
        
        train_limit = None if filter_queries_by_rules else n_train_queries
        train_total = 0
        if os.path.exists(train_path):
            train_total = self._load_queries_from_file(train_path, 'train', train_limit, train_depth, load_depth_info=load_depth_info)
        
        initial_train_count = len(self.train_queries)
        
        valid_total = 0
        if os.path.exists(valid_path):
            valid_total = self._load_queries_from_file(valid_path, 'valid', n_eval_queries, valid_depth, load_depth_info=load_depth_info)
        
        test_total = 0
        if os.path.exists(test_path):
            test_total = self._load_queries_from_file(test_path, 'test', n_test_queries, test_depth, load_depth_info=load_depth_info)
        
        # Filter queries by rules if requested
        if filter_queries_by_rules and self.rules:
            self._filter_queries_by_rules()
            # Re-apply limits after filtering to mimic SB3 behavior
            if n_train_queries is not None:
                self.train_queries = self.train_queries[:n_train_queries]
                self.train_queries_str = self.train_queries_str[:n_train_queries]
                self.train_depths = self.train_depths[:n_train_queries]
                self.train_labels = self.train_labels[:n_train_queries]
        
        # Load domain mapping for countries/ablation datasets
        if corruption_mode and ('countries' in dataset_name or 'ablation' in dataset_name):
            self._load_domain_mapping(dataset_path)

        # print(f"Queries loaded - Train: {len(self.train_queries)}/{train_total}, Valid: {len(self.valid_queries)}, Test: {len(self.test_queries)}")
        
        # Discover vocabulary
        self._discover_vocabulary()

    def _load_facts_from_file(self, filepath: str) -> None:
        """Load facts from .pl or .txt file."""
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%') or line.startswith('one_step'):
                    continue
                # Skip Prolog directives
                if line.startswith(':-'):
                    continue
                # Skip lines with Prolog operators (rules, clauses)
                if ':-' in line or 'findall' in line or line in ['(', ')', ');', ',', '.']:
                    continue
                try:
                    term = get_atom_from_string(line)
                    # Only accept binary relations (arity 2)
                    if len(term.args) == 2:
                        self.facts.append(term)
                        self.facts_str.append((term.predicate, *term.args))
                except Exception:
                    continue

    def _load_rules_from_file(self, filepath: str) -> None:
        """Load rules from file.
        
        IMPORTANT: SB3 uppercases rule variable arguments. We match this behavior
        to ensure parity between implementations.
        """
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                try:
                    rule = get_rule_from_string(line)
                    
                    # Match SB3 behavior: uppercase all rule arguments
                    # (variables are uppercase by convention, but this ensures parity)
                    upper_head_args = tuple(arg.upper() for arg in rule.head.args)
                    upper_head = Term(rule.head.predicate, upper_head_args)
                    
                    upper_body = []
                    for atom in rule.body:
                        upper_args = tuple(arg.upper() for arg in atom.args)
                        upper_body.append(Term(atom.predicate, upper_args))
                    
                    updated_rule = Rule(upper_head, upper_body)
                    self.rules.append(updated_rule)
                    
                    head = (upper_head.predicate, *upper_head.args)
                    body = [(atom.predicate, *atom.args) for atom in upper_body]
                    self.rules_str.append((head, body))
                except Exception:
                    continue

    def _load_queries_from_file(
        self,
        filepath: str,
        split: Literal['train', 'valid', 'test'],
        limit: Optional[int] = None,
        depth_filter: Optional[Set[int]] = None,
        load_depth_info: bool = True,
    ) -> int:
        """Load queries from file with optional depth filtering and sidecar depth support."""
        queries = []
        queries_str = []
        depths = []

        # Prefer sidecar *_depths.txt when available
        source_path = filepath
        if load_depth_info:
            depth_path = None
            if filepath.endswith("_depths.txt") and os.path.exists(filepath):
                depth_path = filepath
            else:
                candidate = filepath.replace(".txt", "_depths.txt")
                if os.path.exists(candidate):
                    depth_path = candidate
            if depth_path is not None:
                source_path = depth_path
        
        total_candidates = 0
        with open(source_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                
                total_candidates += 1

                # Check if line contains depth info
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    # Format: "query depth"
                    query_str, depth_str = parts
                    try:
                        query_depth = int(depth_str) if depth_str != "-1" else -1
                        # Apply depth filter if specified
                        if depth_filter is not None and query_depth not in depth_filter:
                            continue
                        try:
                            term = get_atom_from_string(query_str)
                            queries.append(term)
                            queries_str.append((term.predicate, *term.args))
                            depths.append(query_depth)
                        except Exception:
                            # If parsing fails, we might ideally decrement total_candidates,
                            # but keeping it simple as "lines attempt" is usually fine.
                            # For strictness:
                            total_candidates -= 1
                            continue
                    except ValueError:
                        # Not a valid depth, skip line
                        total_candidates -= 1
                        continue
                else:
                    # No depth info, just query
                    try:
                        term = get_atom_from_string(line)
                        queries.append(term)
                        queries_str.append((term.predicate, *term.args))
                        depths.append(-1)  # Unknown depth
                    except Exception:
                        total_candidates -= 1
                        continue
                
                if limit and len(queries) >= limit:
                    break
        
        if split == 'train':
            self.train_queries = queries
            self.train_queries_str = queries_str
            self.train_depths = depths
            self.train_labels = [1] * len(queries)
        elif split == 'valid':
            self.valid_queries = queries
            self.valid_queries_str = queries_str
            self.valid_depths = depths
            self.valid_labels = [1] * len(queries)
        else:
            self.test_queries = queries
            self.test_queries_str = queries_str
            self.test_depths = depths
            self.test_labels = [1] * len(queries)
            
        return total_candidates

    def _discover_vocabulary(self) -> None:
        """Discover constants and predicates from loaded data."""
        # Extract from facts
        for pred, a, b in self.facts_str:
            self.predicates.add(pred)
            self.constants.add(a)
            self.constants.add(b)
        
        # Extract from rules (head and body)
        for (head_pred, *head_args), body in self.rules_str:
            self.predicates.add(head_pred)
            for body_pred, *body_args in body:
                self.predicates.add(body_pred)
        
        # Extract from queries
        for pred, a, b in (self.train_queries_str + self.valid_queries_str + self.test_queries_str):
            self.predicates.add(pred)
            self.constants.add(a)
            self.constants.add(b)
        
        # If deterministic, convert to sorted lists for consistent ordering
        if self.deterministic:
            self.predicates = sorted(self.predicates)
            self.constants = sorted(self.constants)
    
    def _load_probabilistic_facts(
        self,
        dataset_name: str,
        topk_limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        """Load probabilistic facts from KGE scores."""
        import os
        from collections import defaultdict
        
        # Try to find probabilistic facts file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prob_facts_path = os.path.join(base_dir, "kge_module", "top_k_scores", "files", f"kge_top_{dataset_name}.txt")
        
        if not os.path.exists(prob_facts_path):
            print(f"Warning: Probabilistic facts file {prob_facts_path} not found")
            return
        
        loaded_facts = []
        seen = set()
        
        with open(prob_facts_path, 'r', encoding='ascii') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                fact_repr = parts[0]
                try:
                    score = float(parts[1])
                except ValueError:
                    continue
                
                rank = None
                if len(parts) >= 3:
                    try:
                        rank = int(parts[2])
                    except ValueError:
                        pass
                
                # Apply filters
                if topk_limit is not None and topk_limit >= 0 and rank is not None and rank > topk_limit:
                    continue
                if score_threshold is not None and score < score_threshold:
                    continue
                
                try:
                    fact = get_atom_from_string(fact_repr)
                except Exception:
                    continue
                
                if fact in seen:
                    continue
                seen.add(fact)
                loaded_facts.append(fact)
        
        print(f"Loaded {len(loaded_facts)} probabilistic facts (topk={topk_limit}, threshold={score_threshold})")
        
        # Add to facts if not already present
        existing_facts = set(self.facts)
        new_facts = [f for f in loaded_facts if f not in existing_facts]
        if new_facts:
            self.facts.extend(new_facts)
            for fact in new_facts:
                self.facts_str.append((fact.predicate, *fact.args))
            print(f"Added {len(new_facts)} new probabilistic facts to dataset")
    
    def _filter_queries_by_rules(self) -> None:
        """Filter training queries whose predicates don't match any rule heads."""
        if not self.rules:
            return
        
        # Get all rule head predicates
        rule_head_predicates = set()
        for (head_pred, *_), _ in self.rules_str:
            rule_head_predicates.add(head_pred)
        
        # Filter training queries
        filtered_queries = []
        filtered_queries_str = []
        filtered_depths = []
        filtered_labels = []
        
        for i, (query, query_str) in enumerate(zip(self.train_queries, self.train_queries_str)):
            if query.predicate in rule_head_predicates:
                filtered_queries.append(query)
                filtered_queries_str.append(query_str)
                filtered_depths.append(self.train_depths[i] if i < len(self.train_depths) else -1)
                filtered_labels.append(self.train_labels[i] if i < len(self.train_labels) else 1)
        
        removed = len(self.train_queries) - len(filtered_queries)
        if removed > 0 and self.train_queries:
            print(f"Number of train queries excluded: {removed}. Ratio excluded: {round(removed/len(self.train_queries),3)}")
            self.train_queries = filtered_queries
            self.train_queries_str = filtered_queries_str
            self.train_depths = filtered_depths
            self.train_labels = filtered_labels
    
    def _load_domain_mapping(self, dataset_path: str) -> None:
        """Load domain mapping for countries/ablation datasets."""
        from collections import defaultdict
        
        domain_file = join(dataset_path, "domain2constants.txt")
        if not os.path.exists(domain_file):
            print(f"Warning: Domain file {domain_file} not found")
            return
        
        self.entity2domain = {}
        self.domain2entity = defaultdict(list)
        self.predicate_domain_map = {}  # predicate_str -> (head_domain, tail_domain)
        
        with open(domain_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line defines a predicate domain mapping
                # Format: "predicate:head_domain:tail_domain"
                if ':' in line and not ' ' in line.split(':')[0]:
                    parts = line.split(':')
                    if len(parts) == 3:
                        pred_str, head_dom, tail_dom = parts
                        self.predicate_domain_map[pred_str] = (head_dom, tail_dom)
                        continue
                
                # Otherwise, it's a domain definition
                # Format: "domain entity1 entity2 ..."
                parts = line.split()
                if len(parts) < 2:
                    continue
                domain = parts[0]
                entities = parts[1:]
                for entity in entities:
                    self.entity2domain[entity] = domain
                    self.domain2entity[domain].append(entity)

    # -----------------------------
    # Materialization (strings -> indices)
    # -----------------------------
    def materialize_indices(
        self,
        im: IndexManager,
        max_rule_atoms: Optional[int] = None,
        device: Optional[torch.device] = None,
        drop_strings: bool = False,
    ) -> None:
        """Convert strings to tensor form and build simple tensors for each split."""
        device = device or torch.device('cpu')
        self._materialized_splits.clear()

        if self.facts_str:
            im.set_facts(im.state_to_tensor(self.facts_str))
        else:
            im.set_facts(torch.empty((0, 3), dtype=torch.long))

        if max_rule_atoms is None:
            max_rule_atoms = max((len(body) for _, body in self.rules_str), default=0)
        max_rule_atoms = max(1, max_rule_atoms)
        if self.rules_str:
            rules_idx, rule_lens = im.rules_to_tensor(self.rules_str, max_rule_atoms)
        else:
            rules_idx = torch.empty((0, max_rule_atoms, 3), dtype=torch.long)
            rule_lens = torch.empty((0,), dtype=torch.long)
            im.rules_heads_idx = torch.empty((0, 3), dtype=torch.long)
        im.set_rules(rules_idx, rule_lens)

        width = getattr(im, 'max_arity', self.max_arity) + 1
        splits = {}
        for name, query_strs, label_list, depth_list in (
            ('train', self.train_queries_str, self.train_labels, self.train_depths),
            ('valid', self.valid_queries_str, self.valid_labels, self.valid_depths),
            ('test', self.test_queries_str, self.test_labels, self.test_depths),
        ):
            if query_strs:
                qs = im.state_to_tensor(query_strs)  # [N, 3]
                qs = qs.view(qs.shape[0], 1, qs.shape[1])
                # Only use pin_memory when moving to CUDA; CPU-only runs may fail on some builds
                if device is not None and isinstance(device, torch.device) and device.type == 'cuda' and torch.cuda.is_available():
                    qs = qs.pin_memory()
                queries_tensor = qs.to(device=device, non_blocking=(device.type == 'cuda' if isinstance(device, torch.device) else False))
            else:
                queries_tensor = torch.empty((0, 1, width), dtype=torch.long, device=device)
            labels_tensor = torch.as_tensor(label_list, dtype=torch.long, device=device)
            depths_tensor = torch.as_tensor(depth_list, dtype=torch.long, device=device)
            splits[name] = MaterializedSplit(
                queries=queries_tensor,
                labels=labels_tensor,
                depths=depths_tensor,
            )

        # Build all_known_triples_idx - ONLY from queries, NOT facts
        # SB3 behavior: self.all_known_triples = self.train_queries + self.valid_queries + self.test_queries
        ak = []
        for split in splits.values():
            if split.queries.numel() > 0:
                ak.append(split.queries[:, 0].detach().to('cpu'))
        self.all_known_triples_idx = torch.vstack(ak) if ak else torch.empty((0, 3), dtype=torch.long)

        self._materialized_splits = splits

        if self.entity2domain is not None:
            self._build_domain_maps(im)

        if drop_strings:
            self._drop_strings()

    def _drop_strings(self) -> None:
        """Drop string representations to save memory."""
        self.facts_str.clear()
        self.rules_str.clear()
        self.train_queries_str.clear()
        self.valid_queries_str.clear()
        self.test_queries_str.clear()
    
    def _build_domain_maps(self, im: IndexManager) -> None:
        """Build index-based domain maps from string-based domain mapping."""
        if self.entity2domain is None or self.domain2entity is None:
            return
        
        # Build domain -> list of entity indices
        domain_to_entity_indices = {}
        for domain, entities in self.domain2entity.items():
            entity_indices = []
            for entity in entities:
                if entity in im.constant_str2idx:
                    entity_indices.append(im.constant_str2idx[entity])
            if entity_indices:
                domain_to_entity_indices[domain] = entity_indices
        
        # Store for reference
        self.domain_to_entity_indices = domain_to_entity_indices
        
        # Build entity_idx -> domain (for sampler)
        self.entity_idx_to_domain: Dict[int, str] = {}
        for domain, entities in self.domain2entity.items():
            for entity in entities:
                if entity in im.constant_str2idx:
                    self.entity_idx_to_domain[im.constant_str2idx[entity]] = domain
        
        predicate_domain_map = self.predicate_domain_map
        
        # Apply domain constraints to predicates
        for pred_str, (head_domain, tail_domain) in predicate_domain_map.items():
            if pred_str in im.predicate_str2idx:
                pred_id = im.predicate_str2idx[pred_str]
                
                # Set head domain if available
                if head_domain in domain_to_entity_indices:
                    self.set_domain_heads(pred_id, domain_to_entity_indices[head_domain])
                
                # Set tail domain if available
                if tail_domain in domain_to_entity_indices:
                    self.set_domain_tails(pred_id, domain_to_entity_indices[tail_domain])

    # -----------------------------
    # Split accessors
    # -----------------------------
    def get_materialized_split(self, split: Literal['train', 'valid', 'test']) -> MaterializedSplit:
        """Return tensorized data for a split."""
        if split not in self._materialized_splits:
            raise ValueError("Call materialize_indices() before requesting materialized splits")
        return self._materialized_splits[split]

    # -----------------------------
    # Optional domain API (index-only)
    # -----------------------------
    def set_domain_heads(self, rel_id: int, allowed_entities: Iterable[int]) -> None:
        """Set allowed head entities for a relation."""
        self.allowed_heads_per_rel[rel_id] = torch.tensor(list(allowed_entities), dtype=torch.long)

    def set_domain_tails(self, rel_id: int, allowed_entities: Iterable[int]) -> None:
        """Set allowed tail entities for a relation."""
        self.allowed_tails_per_rel[rel_id] = torch.tensor(list(allowed_entities), dtype=torch.long)

    def get_sampler_domain_info(self) -> Tuple[Optional[Dict[str, List[int]]], Optional[Dict[int, str]]]:
        """
        Get domain information for the sampler.
        
        Returns:
            Tuple of (domain2idx, entity2domain):
            - domain2idx: Dict mapping domain name to list of entity indices
            - entity2domain: Dict mapping entity index to domain name
            
        Returns (None, None) if domain info is not available.
        Must call materialize_indices() first.
        """
        if not hasattr(self, 'domain_to_entity_indices') or self.domain_to_entity_indices is None:
            return None, None
        if not hasattr(self, 'entity_idx_to_domain') or self.entity_idx_to_domain is None:
            return None, None
        return self.domain_to_entity_indices, self.entity_idx_to_domain
