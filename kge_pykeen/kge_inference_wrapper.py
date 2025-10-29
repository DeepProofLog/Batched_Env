#!/usr/bin/env python3
"""Wrapper class to provide KGEInference-compatible interface for PyKEEN KGE models."""
from __future__ import annotations

import os
import json
import gzip
from typing import List, Optional, Tuple, Union, Sequence
import torch
import pandas as pd
import time
import sys
import types


class KGEInferencePyKEEN:
    """
    PyKEEN-based KGE inference engine compatible with the TensorFlow KGEInference interface.
    """

    def __init__(
        self,
        dataset_name: str,
        base_path: str,
        checkpoint_dir: str,
        run_signature: str,
        seed: int = 0,
        scores_file_path: Optional[str] = None,
        runtime_cache_max_entries: Optional[int] = None,
        persist_runtime_scores: bool = True,
    ):
        self.seed = seed
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.checkpoint_dir = checkpoint_dir
        self.run_signature = run_signature
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Initialize score caching
        self.atom_scores: dict = {}
        self.persist_runtime_scores = persist_runtime_scores
        
        # Workaround for torchvision compatibility issues
        self._setup_torchvision_mock()
        
        # Load model
        self.model_dir = os.path.join(checkpoint_dir, run_signature)
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"PyKEEN KGE model directory not found: {self.model_dir}")
        
        self.model, self.entity_to_id, self.relation_to_id = self._load_model()
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        
        print(f"PyKEEN KGE model loaded from {self.model_dir}")
        
        # Load pre-computed scores if provided
        if scores_file_path:
            self._load_scores(scores_file_path)

    def _setup_torchvision_mock(self):
        """Mock torchvision to prevent import errors."""
        if 'torchvision' not in sys.modules:
            sys.modules['torchvision'] = types.ModuleType('torchvision')
            sys.modules['torchvision.models'] = types.ModuleType('torchvision.models')
            sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

    def _load_model(self) -> Tuple[torch.nn.Module, dict, dict]:
        """Load PyKEEN model from directory."""
        model_path = os.path.join(self.model_dir, "trained_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PyKEEN model file not found: {model_path}")
        
        # Load model
        model = torch.load(model_path, map_location=self.device, weights_only=False)
        model.to(self.device)
        model.eval()
        
        # Load entity and relation mappings
        training_path = os.path.join(self.model_dir, "training_triples")
        
        if os.path.exists(training_path):
            if os.path.isdir(training_path):
                # Load from TSV files (newer PyKEEN format)
                entity_tsv = os.path.join(training_path, "entity_to_id.tsv.gz")
                relation_tsv = os.path.join(training_path, "relation_to_id.tsv.gz")
                
                entity_to_id = {}
                with gzip.open(entity_tsv, 'rt') as f:
                    next(f)  # Skip header
                    for line in f:
                        idx, entity = line.strip().split('\t')
                        entity_to_id[entity] = int(idx)
                
                relation_to_id = {}
                with gzip.open(relation_tsv, 'rt') as f:
                    next(f)  # Skip header
                    for line in f:
                        idx, relation = line.strip().split('\t')
                        relation_to_id[relation] = int(idx)
            else:
                # Old format - single file
                training_factory = torch.load(training_path, weights_only=False)
                entity_to_id = training_factory.entity_to_id
                relation_to_id = training_factory.relation_to_id
        else:
            # Try loading from JSON files
            entity_path = os.path.join(self.model_dir, "entity_to_id.json")
            relation_path = os.path.join(self.model_dir, "relation_to_id.json")
            
            if os.path.exists(entity_path) and os.path.exists(relation_path):
                with open(entity_path, "r") as f:
                    entity_to_id = json.load(f)
                with open(relation_path, "r") as f:
                    relation_to_id = json.load(f)
            else:
                raise FileNotFoundError("Could not find entity/relation mappings")
        
        return model, entity_to_id, relation_to_id

    def _load_scores(self, filepath: str):
        """Load pre-computed atom scores from a file."""
        if not os.path.exists(filepath) or os.path.isdir(filepath):
            print(f"Warning: Scores file not found at {filepath}. KGE will perform live inference.")
            return
        
        print(f"Loading pre-computed scores from {filepath}...")
        start_time = time.time()
        try:
            df = pd.read_csv(
                filepath, 
                sep='\t', 
                header=None, 
                names=['atom', 'score'], 
                dtype={'atom': str, 'score': float},
                engine='c'
            )
            self.atom_scores = pd.Series(df.score.values, index=df.atom).to_dict()
            end_time = time.time()
            print(f"Loaded {len(self.atom_scores)} scores in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Error loading scores file: {e}")

    def _parse_atom(self, atom_string: str) -> Tuple[str, str, str]:
        """Parse atom string in format 'relation(head,tail)' to (head, relation, tail)."""
        atom_string = atom_string.strip()
        if atom_string.endswith('.'):
            atom_string = atom_string[:-1]
        
        # Parse functional format: relation(head,tail)
        if '(' in atom_string and ')' in atom_string:
            rel_end = atom_string.index('(')
            relation = atom_string[:rel_end].strip()
            args_end = atom_string.rindex(')')
            args = atom_string[rel_end+1:args_end].strip()
            entities = [e.strip() for e in args.split(',')]
            if len(entities) == 2:
                head, tail = entities
                return head, relation, tail
        
        raise ValueError(f"Could not parse atom: {atom_string}")

    def _score_triples_batch(self, heads: List[str], relations: List[str], tails: List[str]) -> List[float]:
        """Score multiple triples using the PyKEEN model in batch mode."""
        if not heads:
            return []
        
        # Filter out triples with unknown entities/relations
        valid_indices = []
        h_ids = []
        r_ids = []
        t_ids = []
        
        for i, (h, r, t) in enumerate(zip(heads, relations, tails)):
            if h in self.entity_to_id and r in self.relation_to_id and t in self.entity_to_id:
                valid_indices.append(i)
                h_ids.append(self.entity_to_id[h])
                r_ids.append(self.relation_to_id[r])
                t_ids.append(self.entity_to_id[t])
        
        # Initialize scores with 0.0
        scores = [0.0] * len(heads)
        
        if not h_ids:
            return scores
        
        # Batch scoring
        with torch.no_grad():
            # PyKEEN expects batch of shape (batch_size, 3) with columns [head, relation, tail]
            batch_tensor = torch.tensor(
                [[h, r, t] for h, r, t in zip(h_ids, r_ids, t_ids)],
                dtype=torch.long,
                device=self.device
            )
            batch_scores = self.model.score_hrt(batch_tensor).float().cpu()
            
            # Ensure scores are 1D
            if batch_scores.dim() > 1:
                batch_scores = batch_scores.squeeze()
            
            # Handle single element case
            if batch_scores.dim() == 0:
                batch_scores = batch_scores.unsqueeze(0)
            
            batch_scores = batch_scores.tolist()
            
            # Handle case where single value becomes a scalar
            if not isinstance(batch_scores, list):
                batch_scores = [batch_scores]
        
        # Map scores back to original positions
        for idx, score in zip(valid_indices, batch_scores):
            scores[idx] = score
        
        return scores

    def predict(self, atom_string: str) -> float:
        """
        Predict the score for a single atom.
        
        Args:
            atom_string: Atom in format 'relation(head,tail)'
            
        Returns:
            Score for the atom
        """
        # Use batch prediction for consistency
        scores = self.predict_batch([atom_string])
        return scores[0] if scores else 0.0

    def predict_batch(self, atoms_for_ranking: Sequence[Union[str, Tuple]]) -> List[float]:
        """
        Batch prediction with true batch inference.
        
        Args:
            atoms_for_ranking: List of atoms (strings or tuples)
            
        Returns:
            List of scores
        """
        if not atoms_for_ranking:
            return []
        
        # Convert tuples to strings if needed
        atom_strings = []
        for atom in atoms_for_ranking:
            if isinstance(atom, tuple):
                # (relation, head, tail) -> 'relation(head,tail)'
                if len(atom) == 3:
                    r, h, t = atom
                    atom_strings.append(f"{r}({h},{t})")
                else:
                    raise ValueError(f"Invalid tuple format: {atom}")
            else:
                atom_strings.append(str(atom))
        
        # Check cache and separate cached from uncached
        cached_scores = []
        uncached_indices = []
        uncached_atoms = []
        
        for i, atom_str in enumerate(atom_strings):
            if atom_str in self.atom_scores:
                cached_scores.append((i, self.atom_scores[atom_str]))
            else:
                uncached_indices.append(i)
                uncached_atoms.append(atom_str)
        
        # Parse uncached atoms
        heads = []
        relations = []
        tails = []
        parse_valid = []
        
        for atom_str in uncached_atoms:
            try:
                h, r, t = self._parse_atom(atom_str)
                heads.append(h)
                relations.append(r)
                tails.append(t)
                parse_valid.append(True)
            except Exception:
                heads.append("")
                relations.append("")
                tails.append("")
                parse_valid.append(False)
        
        # Batch score the valid ones
        uncached_scores = self._score_triples_batch(heads, relations, tails)
        
        # Cache the results
        if self.persist_runtime_scores:
            for atom_str, score, valid in zip(uncached_atoms, uncached_scores, parse_valid):
                if valid:
                    self.atom_scores[atom_str] = score
        
        # Merge cached and uncached results
        all_scores = [0.0] * len(atom_strings)
        for idx, score in cached_scores:
            all_scores[idx] = score
        for idx, score in zip(uncached_indices, uncached_scores):
            all_scores[idx] = score
        
        return all_scores
