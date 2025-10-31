"""KGE Inference for TensorFlow models - simplified for RL integration."""
import os
import sys
import json
import subprocess
from collections import OrderedDict

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))

import tensorflow as tf
import numpy as np
import random
import re
from typing import Dict, List, Tuple, Optional, Union, Sequence
import time
import pandas as pd

import ns_lib as ns
from kge_tf.kge_loader_tf import KGCDataHandler
from kge_tf.kge_model_tf import CollectiveModel
from ns_lib.utils import load_kge_weights


def get_available_gpus_tf(memory_threshold: float = 0.1, utilization_threshold: float = 0.3) -> List[int]:
    """Get list of GPU indices that are not busy."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            gpus = tf.config.list_physical_devices('GPU')
            return list(range(len(gpus)))
        
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
        gpus = tf.config.list_physical_devices('GPU')
        return list(range(len(gpus)))


class Atom:
    """Atom parser for functional notation."""
    def __init__(self, s: str):
        a = re.sub(r'\b([(),\.])', r'\1', s.strip())
        if a.endswith("."):
            a = a[:-1]
        tokens = a.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
        self.r = tokens[0]
        self.args = tokens[1:]
    
    def toTuple(self) -> Tuple:
        return (self.r,) + tuple(self.args)


class KGEInference:
    """TensorFlow KGE inference engine for RL integration."""

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
        self.strategy = None
        self.use_multi_gpu = False
        if device == "cuda:all":
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and len(gpus) > 1:
                available_gpu_indices = get_available_gpus_tf()
                if len(available_gpu_indices) > 1:
                    available_gpu_devices = [gpus[i] for i in available_gpu_indices]
                    tf.config.set_visible_devices(available_gpu_devices, 'GPU')
                    self.strategy = tf.distribute.MirroredStrategy()
                    self.use_multi_gpu = True
                    print(f"TF MirroredStrategy with {len(available_gpu_indices)} GPUs: {available_gpu_indices}")

        # Load data structures
        self.data_handler = KGCDataHandler(
            dataset_name=dataset_name, base_path=base_path, format='functional',
            domain_file='domain2constants.txt', train_file='train.txt',
            valid_file='valid.txt', test_file='test.txt', fact_file='facts.txt'
        )
        self.fol = self.data_handler.fol
        self.serializer = ns.serializer.LogicSerializerFast(
            predicates=self.fol.predicates, domains=self.fol.domains,
            constant2domain_name=self.fol.constant2domain_name
        )

        # Model loaded lazily
        self.model = None
        self.config = None

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
        tf.random.set_seed(seed)

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

    def _build_and_load_model(self) -> CollectiveModel:
        """Build and load the TF KGE model."""
        print("Building model and loading weights...")
        
        if self.config is None:
            config_path = os.path.join(self.checkpoint_dir, f"{self.run_signature}_seed_{self.seed}", "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                import argparse
                self.config = argparse.Namespace(**config_dict)
            else:
                # Fallback defaults
                import argparse
                self.config = argparse.Namespace(
                    kge='complex', resnet=True, constant_embedding_size=256,
                    predicate_embedding_size=256, kge_atom_embedding_size=4,
                    kge_regularization=0.0, kge_dropout_rate=0.0
                )
        
        build_scope = self.strategy.scope() if (self.use_multi_gpu and self.strategy) else None
        if build_scope:
            with build_scope:
                model = self._create_model()
                model = self._load_model_weights(model)
        else:
            model = self._create_model()
            model = self._load_model_weights(model)
        
        print("Weights loaded successfully.")
        return model
    
    def _create_model(self) -> CollectiveModel:
        """Create model architecture."""
        model = CollectiveModel(
            self.fol, rules=[],  kge=self.config.kge,
            kge_regularization=getattr(self.config, 'kge_regularization', 0.0),
            constant_embedding_size=self.config.constant_embedding_size,
            predicate_embedding_size=self.config.predicate_embedding_size,
            kge_atom_embedding_size=self.config.kge_atom_embedding_size,
            kge_dropout_rate=getattr(self.config, 'kge_dropout_rate', 0.0),
            reasoner_depth=0, model_name='no_reasoner',
            reasoner_atom_embedding_size=100, reasoner_formula_hidden_embedding_size=100,
            reasoner_regularization=0.0, reasoner_single_model=False, reasoner_dropout_rate=0.0,
            aggregation_type='max', signed=True, temperature=0.0, resnet=True,
            embedding_resnet=False, filter_num_heads=3, filter_activity_regularization=0.0,
            num_adaptive_constants=0, dot_product=False, cdcr_use_positional_embeddings=True,
            cdcr_num_formulas=3, r2n_prediction_type='full', distill=False,
        )
        
        # Build model
        dummy_generator = ns.dataset.DataGenerator(
            self.data_handler.get_dataset(split="test", number_negatives=0), 
            self.fol, self.serializer, engine=None, batch_size=1, ragged=False
        )
        dummy_input = next(iter(dummy_generator))[0]
        _ = model(dummy_input)
        return model
    
    def _load_model_weights(self, model: CollectiveModel) -> CollectiveModel:
        """Load weights into model."""
        name = f"{self.run_signature}_seed_{self.seed}"
        ckpt_filepath = os.path.join(self.checkpoint_dir, name, f"{name}_kge_model")
        success = load_kge_weights(model, ckpt_filepath, verbose=True)
        if not success:
            raise FileNotFoundError(f"Could not load weights from {ckpt_filepath}.weights.h5")
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

    def _score_atoms_via_model(self, atom_tuples: Sequence[Tuple]) -> List[float]:
        """Score atoms using KGE model."""
        if not atom_tuples:
            return []

        queries = [list(atom_tuples)]
        labels = [[1.0] + [0.0] * (len(atom_tuples) - 1)]

        x, _ = ns.dataset._from_strings_to_tensors(
            fol=self.fol, serializer=self.serializer, queries=queries,
            labels=labels, engine=None, ragged=False
        )

        if self.model is None:
            self.model = self._build_and_load_model()

        if self.use_multi_gpu and self.strategy:
            predictions = self._distributed_predict(x)
        else:
            predictions = self.model(x, training=False)
        
        concept_scores = predictions["concept"]
        return concept_scores.numpy()[0].tolist()
    
    def _distributed_predict(self, x):
        """Run prediction across multiple GPUs."""
        dataset = tf.data.Dataset.from_tensors(x)
        dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
        
        @tf.function
        def predict_step(inputs):
            return self.model(inputs, training=False)
        
        for batch in dist_dataset:
            predictions = self.strategy.run(predict_step, args=(batch,))
            gathered = {}
            for key, value in predictions.items():
                gathered[key] = self.strategy.gather(value, axis=0)
            return gathered
        return self.model(x, training=False)

    def predict_batch(self, atoms_for_ranking: Sequence[Union[str, Tuple]]) -> List[float]:
        """
        Score a batch of atoms. First atom is assumed to be the positive sample.
        Returns list of scores in same order as input.
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
