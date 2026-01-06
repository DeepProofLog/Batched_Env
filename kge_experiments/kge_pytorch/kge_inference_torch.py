"""KGE Inference for PyTorch models - simplified for RL integration."""
import json
import os
import subprocess
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, Sequence
import time

import pandas as pd
import torch
import numpy as np
import random

from kge_pytorch.model_torch import build_model


def get_available_gpus(memory_threshold: float = 0.1, utilization_threshold: float = 0.3) -> List[int]:
    """Get list of GPU indices that are not busy."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            return list(range(torch.cuda.device_count()))
        
        available_gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpu_id = int(parts[0])
                mem_used, mem_total, utilization = float(parts[1]), float(parts[2]), float(parts[3])
                mem_fraction = mem_used / mem_total if mem_total > 0 else 1.0
                util_fraction = utilization / 100.0
                
                if mem_fraction <= memory_threshold and util_fraction <= utilization_threshold:
                    available_gpus.append(gpu_id)
        
        return available_gpus
    except Exception:
        return list(range(torch.cuda.device_count()))


class Atom:
    """Atom parser for functional notation."""
    def __init__(self, s: str):
        import re
        a = re.sub(r'\b([(),\.])', r'\1', s.strip())
        if a.endswith("."):
            a = a[:-1]
        tokens = a.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
        self.r = tokens[0]
        self.args = tokens[1:]
    
    def toTuple(self) -> Tuple:
        return (self.r,) + tuple(self.args)


class KGEInference:
    """PyTorch KGE inference engine for RL integration."""

    def __init__(
        self,
        dataset_name: str,
        base_path: str,
        run_signature: str,
        checkpoint_dir: str = './checkpoints/',
        seed: int = 0,
        scores_file_path: Optional[str] = None,
        runtime_cache_max_entries: Optional[int] = None,
        persist_runtime_scores: bool = True,
        device: str = None,
    ):
        self.seed = seed
        self.set_seeds(self.seed)
        self.run_signature = run_signature
        self.checkpoint_dir = checkpoint_dir
        
        # Multi-GPU setup
        self.use_multi_gpu = False
        self.device = None
        self.available_gpu_ids = None
        
        if device == "cuda:all":
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                available_gpu_indices = get_available_gpus()
                if len(available_gpu_indices) > 1:
                    self.device = torch.device(f"cuda:{available_gpu_indices[0]}")
                    self.available_gpu_ids = available_gpu_indices
                    self.use_multi_gpu = True
                    print(f"PyTorch DataParallel with {len(available_gpu_indices)} GPUs: {available_gpu_indices}")
                else:
                    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cpu":
            self.device = torch.device("cpu")
        elif device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Model loaded lazily
        self.model_dir = self._resolve_model_dir(checkpoint_dir, run_signature, seed)
        self.model = None
        self.config = None
        self.entity2id = None
        self.relation2id = None
        
        # Caching
        self.atom_scores: Dict[str, float] = {}
        self.persist_runtime_scores = persist_runtime_scores
        self._tuple_cache: Dict[str, Tuple[str, ...]] = {}
        
        if runtime_cache_max_entries == 0:
            self._runtime_cache_enabled = False
            self._score_cache = None
        else:
            self._runtime_cache_enabled = True
            self._runtime_cache_max_entries = runtime_cache_max_entries
            self._score_cache = OrderedDict()
        
        if scores_file_path:
            self._load_scores(scores_file_path)
    
    def set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _load_scores(self, filepath: str):
        """Load pre-computed scores from TSV file."""
        if not os.path.exists(filepath) or os.path.isdir(filepath):
            print(f"Warning: Scores file not found at {filepath}. KGE will perform live inference.")
            return

        print(f"Loading pre-computed scores from {filepath}...")
        start_time = time.time()
        try:
            df = pd.read_csv(filepath, sep='\t', header=None, names=['atom', 'score'], 
                           dtype={'atom': str, 'score': float}, engine='c')
            self.atom_scores = pd.Series(df.score.values, index=df.atom).to_dict()
            print(f"Loaded {len(self.atom_scores)} scores in {time.time() - start_time:.2f}s.")
        except Exception as e:
            print(f"Error loading scores: {e}")
    
    def _get_runtime_cached_score(self, atom_str: str) -> Optional[float]:
        if not self._runtime_cache_enabled or self._score_cache is None:
            return None
        if atom_str in self._score_cache:
            score = self._score_cache[atom_str]
            self._score_cache.move_to_end(atom_str)
            return score
        return None

    def _store_runtime_score(self, atom_str: str, score: float) -> None:
        if self._runtime_cache_enabled and self._score_cache is not None:
            self._score_cache[atom_str] = score
            self._score_cache.move_to_end(atom_str)
            if self._runtime_cache_max_entries and len(self._score_cache) > self._runtime_cache_max_entries:
                self._score_cache.popitem(last=False)
        if self.persist_runtime_scores:
            self.atom_scores[atom_str] = score
    
    def _resolve_model_dir(self, checkpoint_dir: str, run_signature: str, seed: int) -> str:
        """Resolve run directory with backward-compatible fallbacks."""
        signature_path = os.path.expanduser(run_signature)
        if os.path.isdir(signature_path):
            return signature_path
        candidate = os.path.join(checkpoint_dir, run_signature)
        if os.path.isdir(candidate):
            return candidate
        return os.path.join(checkpoint_dir, f"{run_signature}_seed_{seed}")

    def _build_and_load_model(self) -> torch.nn.Module:
        """Build and load the PyTorch KGE model."""
        print("Building model and loading weights...")
        
        # Load config
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Load mappings
        with open(os.path.join(self.model_dir, "entity2id.json"), "r") as f:
            self.entity2id = json.load(f)
        with open(os.path.join(self.model_dir, "relation2id.json"), "r") as f:
            self.relation2id = json.load(f)
        
        # Build model
        model = build_model(
            self.config.get("model", "RotatE"),
            self.config["num_entities"],
            self.config["num_relations"],
            dim=self.config.get("dim") or self.config.get("entity_dim"),
            gamma=self.config.get("gamma", 12.0),
            p_norm=self.config.get("p", 1),
            relation_dim=self.config.get("relation_dim"),
            dropout=self.config.get("dropout", 0.0),
        )
        
        # Load weights
        weights_path = os.path.join(self.model_dir, "weights.pth")
        state = torch.load(weights_path, map_location="cpu")
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        
        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()
        
        # Multi-GPU wrapper
        if self.use_multi_gpu and self.available_gpu_ids:
            model = torch.nn.DataParallel(model, device_ids=self.available_gpu_ids)
        
        print("Weights loaded successfully.")
        return model
    
    def _normalize_atom_input(self, atom: Union[str, Tuple]) -> str:
        """Convert atom to canonical string."""
        if isinstance(atom, str):
            return atom
        if isinstance(atom, tuple):
            predicate, *args = atom
            return f"{predicate}({','.join(map(str, args))})"
        raise TypeError(f"Unsupported atom type: {type(atom)}")

    def _atom_str_to_tuple(self, atom_str: str) -> Tuple:
        """Convert atom string to tuple."""
        cached = self._tuple_cache.get(atom_str)
        if cached:
            return cached
        atom_tuple = Atom(atom_str).toTuple()
        self._tuple_cache[atom_str] = atom_tuple
        return atom_tuple
    
    def _prepare_triple_ids(self, atom_tuples: List[Tuple]) -> torch.Tensor:
        """Convert atom tuples to ID tensor."""
        ids = []
        for atom_tuple in atom_tuples:
            predicate, *args = atom_tuple
            if len(args) != 2:
                raise ValueError(f"Expected binary predicate, got {len(args)} arguments")
            head, tail = args
            if head not in self.entity2id or tail not in self.entity2id or predicate not in self.relation2id:
                raise ValueError(f"Unknown entity or relation in {atom_tuple}")
            ids.append([self.entity2id[head], self.relation2id[predicate], self.entity2id[tail]])
        return torch.tensor(ids, dtype=torch.long)
    
    def _score_atoms_via_model(self, atom_tuples: Sequence[Tuple]) -> List[float]:
        """Score atoms using KGE model."""
        if not atom_tuples:
            return []
        
        if self.model is None:
            self.model = self._build_and_load_model()
        
        try:
            triple_ids = self._prepare_triple_ids(list(atom_tuples))
        except ValueError:
            return [0.0] * len(atom_tuples)
        
        scores = []
        batch_size = 2048
        
        with torch.no_grad():
            for start in range(0, triple_ids.size(0), batch_size):
                batch = triple_ids[start : start + batch_size].to(self.device)
                batch_scores = self.model.score_triples(batch[:, 0], batch[:, 1], batch[:, 2])
                batch_scores = torch.sigmoid(batch_scores)
                scores.append(batch_scores.float().cpu())
        
        if scores:
            return torch.cat(scores, dim=0).tolist()
        return []
    
    def predict_batch(self, atoms_for_ranking: Sequence[Union[str, Tuple]]) -> List[float]:
        """
        Score a batch of atoms. First atom is assumed to be the positive sample.
        Returns list of scores in [0, 1] in the same order as input.
        """
        if not atoms_for_ranking:
            return []

        canonical_atoms = [self._normalize_atom_input(atom) for atom in atoms_for_ranking]
        scores: List[Optional[float]] = [None] * len(canonical_atoms)
        missing, missing_indices = [], []

        # Check caches
        for idx, atom_str in enumerate(canonical_atoms):
            cached = self.atom_scores.get(atom_str) or self._get_runtime_cached_score(atom_str)
            if cached is not None:
                scores[idx] = float(cached)
            else:
                missing.append(atom_str)
                missing_indices.append(idx)

        # Score missing atoms
        if missing:
            unique_missing = list(dict.fromkeys(missing))
            atom_tuples = [self._atom_str_to_tuple(a) for a in unique_missing]
            new_scores = self._score_atoms_via_model(atom_tuples)

            newly_scored = {a: float(s) for a, s in zip(unique_missing, new_scores)}
            for a, s in newly_scored.items():
                self._store_runtime_score(a, s)

            for idx, atom_str in zip(missing_indices, missing):
                scores[idx] = newly_scored[atom_str]

        return [float(s) for s in scores]
