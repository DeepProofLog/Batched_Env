import os
import sys
import json
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))
import tensorflow as tf
import numpy as np
import random
import re
from typing import List, Tuple, Optional, Union, Sequence
import time
import pandas as pd

import ns_lib as ns
from kge_loader import KGCDataHandler
from kge_model import CollectiveModel
from ns_lib.utils import get_arg, load_kge_weights


# def get_kge_log_probs(self, obs: PyTorchObs, actions: torch.Tensor, log_prob: torch.Tensor) -> None:
#     """
#     Computes the KGE log probabilities for the actions based on the latent policy output.
#     This is used to replace the log probabilities of KGE actions with their KGE scores.
#     """
#     # Squeeze actions if it has an extra dimension
#     actions_squeezed = actions.squeeze(1) if actions.ndim > 1 else actions
    
#     # Get the predicate indices of the chosen actions
#     batch_indices = torch.arange(actions_squeezed.shape[0], device=actions_squeezed.device)
#     chosen_action_sub_indices = obs["derived_sub_indices"][batch_indices, actions_squeezed]
#     chosen_action_pred_indices = chosen_action_sub_indices[:, 0, 0]

#     # Find which of the chosen actions are KGE actions
#     kge_action_mask = torch.isin(chosen_action_pred_indices, self.kge_indices_tensor.to(chosen_action_pred_indices.device))
#     kge_batch_indices = kge_action_mask.nonzero(as_tuple=False).squeeze(-1)

#     # For those actions, get the KGE score and update the log_prob
#     if kge_batch_indices.numel() > 0:
#         for batch_idx in kge_batch_indices:
#             kge_action_sub_index = chosen_action_sub_indices[batch_idx, 0, :]
#             kge_action_str = self.index_manager.subindex_to_str(kge_action_sub_index)
#             kge_pred_str = self.index_manager.predicate_idx2str.get(kge_action_sub_index[0].item())

#             if kge_action_str and kge_pred_str:
#                 original_pred_str = kge_pred_str.removesuffix('_kge')
#                 original_atom_str = f"{original_pred_str}{kge_action_str[len(kge_pred_str):]}"
#                 score = self.kge_inference_engine.predict(original_atom_str)
#                 kge_log_prob = math.log(score + 1e-9)
#                 # print(f"Computing KGE score for action: {original_atom_str}_kge, score: {score:.5f}, log_prob: {kge_log_prob:.3f}")
                
#                 log_prob[batch_idx] = kge_log_prob
#     return log_prob

class Domain:
    """Represents a domain of constants in the First-Order Logic."""
    def __init__(self, name: str, constants: List[str], has_features: bool = False):
        self.name = name
        self.constants = constants
        self.has_features = has_features

class Predicate:
    """Represents a predicate in the First-Order Logic."""
    def __init__(self, name: str, domains: List['Domain'], has_features: bool = False):
        self.name = name
        self.domains = domains
        self.arity = len(domains)
        self.has_features = has_features

    def __repr__(self):
        args_str = ','.join([d.name for d in self.domains])
        return f'{self.name}({args_str})'

class Atom:
    """Represents an atom (a predicate applied to constants)."""
    def __init__(self, r: str = None, args: List[str] = None, s: str = None, format: str = 'functional'):
        if s is not None:
            self.read(s, format)
        else:
            self.r = r
            self.args = args

    def read(self, s: str, format: str = 'functional'):
        if format == 'functional':
            self._from_string(s)
        elif format == 'triplet':
            self._from_triplet_string(s)
        else:
            raise Exception('Unknown Atom format: %s' % format)

    def _from_string(self, a: str):
        a = re.sub(r'\b([(),\.])', r'\1', a)
        a = a.strip()
        if a.endswith("."):
            a = a[:-1]
        tokens = a.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
        self.r = tokens[0]
        self.args = [t for t in tokens[1:]]

    def _from_triplet_string(self, a: str):
        a = a.strip()
        tokens = a.split()
        assert len(tokens) == 3, str(tokens)
        self.r = tokens[1]
        self.args = [tokens[0], tokens[2]]

    def toTuple(self) -> Tuple:
        return (self.r,) + tuple(self.args)

    def __hash__(self):
        return hash((self.r, tuple(self.args)))

    def __eq__(self, other):
        return self.r == other.r and tuple(self.args) == tuple(other.args)
        
    def __repr__(self):
        args_str = ','.join(self.args)
        return f'{self.r}({args_str})'

class KGEInference:
    """
    A class to handle loading a pre-trained KGE model and performing inference.
    """
    def __init__(self, dataset_name: str, base_path: str, checkpoint_dir: str, run_signature: str, seed: int = 0, scores_file_path: str = None):
        self.seed = seed
        self.set_seeds(self.seed)
        
        self.run_signature = run_signature
        self.checkpoint_dir = checkpoint_dir

        self.data_handler = self._load_data(dataset_name, base_path)
        self.fol = self.data_handler.fol
        self.serializer = self._create_serializer()
        
        self.model = None
        self.config = None  # Store loaded config
        print("KGEInference engine initialized. Model will be built on first use.")

        self.atom_scores = {}
        if scores_file_path:
            self._load_scores(scores_file_path)

    def _load_scores(self, filepath: str):
        """Loads pre-computed atom scores from a file."""
        if not os.path.exists(filepath) or os.path.isdir(filepath):
            if os.path.isdir(filepath):
                 print(f"Warning: Scores file path '{filepath}' is a directory. KGE will perform live inference.")
            else:
                 print(f"Warning: Scores file not found at {filepath}. KGE will perform live inference.")
            return

        print(f"Loading pre-computed scores from {filepath}...")
        start_time = time.time()
        try:
            df = pd.read_csv(filepath, sep='\t', header=None, names=['atom', 'score'], dtype={'atom': str, 'score': float}, engine='c')
            self.atom_scores = pd.Series(df.score.values, index=df.atom).to_dict()
            end_time = time.time()
            print(f"Loaded {len(self.atom_scores)} scores in {end_time - start_time:.2f} seconds.")
        except pd.errors.EmptyDataError:
            print(f"Warning: Scores file at {filepath} is empty.")
        except Exception as e:
            print(f"Error loading scores file: {e}")

    def set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def _load_data(self, dataset_name: str, base_path: str) -> KGCDataHandler:
        print("Loading data...")
        return KGCDataHandler(
            dataset_name=dataset_name, base_path=base_path, format='functional',
            domain_file='domain2constants.txt', train_file='train.txt',
            valid_file='valid.txt', test_file='test.txt', fact_file='facts.txt'
        )

    def _create_serializer(self) -> ns.serializer.LogicSerializerFast:
        return ns.serializer.LogicSerializerFast(
            predicates=self.fol.predicates, domains=self.fol.domains,
            constant2domain_name=self.fol.constant2domain_name
        )
    
    def _load_config(self):
        """Load configuration from the saved config file."""
        name = f"{self.run_signature}_seed_{self.seed}"
        
        # Try multiple possible config file locations
        config_paths = [
            "./config.json",  # Current directory (like kge_eval_test.py)
            os.path.join('./../checkpoints', name, f"config.json"),
            os.path.join('./../../checkpoints', name, f"config.json"),
        ]
        
        config_filepath = None
        for path in config_paths:
            if os.path.exists(path):
                config_filepath = path
                break
        
        if config_filepath is None:
            print("Warning: No config file found. Using default parameters.")
            # Return a default config based on run signature parsing
            return self._parse_config_from_signature()
        
        print(f"Loading configuration from: {config_filepath}")
        try:
            with open(config_filepath, 'r') as f:
                config_dict = json.load(f)
            return argparse.Namespace(**config_dict)
        except Exception as e:
            print(f"Error loading config: {e}. Falling back to signature parsing.")
            return self._parse_config_from_signature()
    
    def _parse_config_from_signature(self):
        """Parse configuration from run signature as fallback."""
        parts = self.run_signature.replace('.txt', '').split('-')
        try:
            config = argparse.Namespace()
            config.kge = parts[3] if len(parts) > 3 else 'complex'
            config.resnet = parts[4].lower() == 'true' if len(parts) > 4 else True
            config.constant_embedding_size = int(parts[5]) if len(parts) > 5 else 256
            config.predicate_embedding_size = int(parts[6]) if len(parts) > 6 else 256
            config.kge_atom_embedding_size = int(parts[7]) if len(parts) > 7 else 4
            
            # Set other defaults
            config.kge_regularization = 0.0
            config.kge_dropout_rate = 0.0
            config.reasoner_atom_embedding_size = 100
            config.reasoner_formula_hidden_embedding_size = 100
            config.reasoner_regularization_factor = 0.0
            config.reasoner_single_model = False
            config.reasoner_dropout_rate = 0.0
            config.aggregation_type = 'max'
            config.signed = True
            config.temperature = 0.0
            config.embedding_resnet = False
            config.filter_num_heads = 3
            config.filter_activity_regularization = 0.0
            
            print(f"Parsed from signature: kge='{config.kge}', resnet={config.resnet}, "
                  f"const_emb={config.constant_embedding_size}, pred_emb={config.predicate_embedding_size}, "
                  f"kge_atom_emb={config.kge_atom_embedding_size}")
            return config
        except (IndexError, ValueError) as e:
            raise ValueError(f"Could not parse run_signature '{self.run_signature}'. Error: {e}")

    def _build_and_load_model(self) -> CollectiveModel:
        print("Building model and loading weights...")
        
        # Load configuration
        if self.config is None:
            self.config = self._load_config()
        
        # Build model with loaded/parsed configuration
        model = CollectiveModel(
            self.fol, rules=[], 
            kge=self.config.kge,
            kge_regularization=getattr(self.config, 'kge_regularization', 0.0),
            constant_embedding_size=self.config.constant_embedding_size,
            predicate_embedding_size=self.config.predicate_embedding_size,
            kge_atom_embedding_size=self.config.kge_atom_embedding_size,
            kge_dropout_rate=getattr(self.config, 'kge_dropout_rate', 0.0),
            reasoner_depth=0,  # KGE-only evaluation
            model_name='no_reasoner',
            reasoner_atom_embedding_size=getattr(self.config, 'reasoner_atom_embedding_size', 100),
            reasoner_formula_hidden_embedding_size=getattr(self.config, 'reasoner_formula_hidden_embedding_size', 100),
            reasoner_regularization=getattr(self.config, 'reasoner_regularization_factor', 0.0),
            reasoner_single_model=getattr(self.config, 'reasoner_single_model', False),
            reasoner_dropout_rate=getattr(self.config, 'reasoner_dropout_rate', 0.0),
            aggregation_type=getattr(self.config, 'aggregation_type', 'max'),
            signed=getattr(self.config, 'signed', True),
            temperature=getattr(self.config, 'temperature', 0.0),
            resnet=getattr(self.config, 'resnet', True),
            embedding_resnet=getattr(self.config, 'embedding_resnet', False),
            filter_num_heads=getattr(self.config, 'filter_num_heads', 3),
            filter_activity_regularization=getattr(self.config, 'filter_activity_regularization', 0.0),
            num_adaptive_constants=getattr(self.config, 'engine_num_adaptive_constants', 0),
            dot_product=getattr(self.config, 'engine_dot_product', False),
            cdcr_use_positional_embeddings=getattr(self.config, 'cdcr_use_positional_embeddings', True),
            cdcr_num_formulas=getattr(self.config, 'cdcr_num_formulas', 3),
            r2n_prediction_type=getattr(self.config, 'r2n_prediction_type', 'full'),
            distill=getattr(self.config, 'distill', False),
        )
        
        # Build model by calling it on dummy data
        dummy_generator = ns.dataset.DataGenerator(
            self.data_handler.get_dataset(split="test", number_negatives=0), 
            self.fol, self.serializer, engine=None, batch_size=1, ragged=False
        )
        dummy_input = next(iter(dummy_generator))[0]
        _ = model(dummy_input)
        
        # Load weights using the same approach as kge_eval_test.py
        name = f"{self.run_signature}_seed_{self.seed}"
        self.checkpoint_dir = '/home/castellanoontiv/checkpoints/'

        ckpt_filepath = os.path.join(self.checkpoint_dir, name, f"{name}_kge_model")
        
        print(f"Attempting to load weights from: {ckpt_filepath}.weights.h5")
        success = load_kge_weights(model, ckpt_filepath, verbose=True)
        if not success:
            ckpt_dir_path = os.path.join(self.checkpoint_dir, name)
            print(f"Available files in checkpoint directory '{ckpt_dir_path}':", 
                  os.listdir(ckpt_dir_path) if os.path.exists(ckpt_dir_path) else "Directory not found")
            raise FileNotFoundError(f"Could not load weights from {ckpt_filepath}.weights.h5")
        
        print("Weights loaded successfully.")
        return model

    def _prepare_atom(self, atom_string: str):
        atom = Atom(s=atom_string, format="functional")
        if atom.r not in self.fol.name2predicate:
            raise ValueError(f"Predicate '{atom.r}' not found in vocabulary.")
        
        queries = [[atom.toTuple()]]
        labels = [[1.0]]
        return ns.dataset._from_strings_to_tensors(
            fol=self.fol, serializer=self.serializer, queries=queries,
            labels=labels, engine=None, ragged=False
        )
        
    def _prepare_batch(self, atom_tuples: List[Tuple]):
        """Prepares a batch of atom tuples for evaluation."""
        queries = [[atom] for atom in atom_tuples]
        labels = [[1.0]] * len(queries)

        return ns.dataset._from_strings_to_tensors(
            fol=self.fol, serializer=self.serializer, queries=queries,
            labels=labels, engine=None, ragged=False
        )

    def predict(self, atom_string: str) -> float:
        """Evaluates a single atom string and returns its score."""
        if atom_string in self.atom_scores:
            return self.atom_scores[atom_string]
        
        (model_inputs, _y) = self._prepare_atom(atom_string)
        
        if self.model is None:
            self.model = self._build_and_load_model()
        
        kge_inputs = (model_inputs[0], model_inputs[1])
        atom_outputs, _ = self.model.kge_model.call(kge_inputs)
        score = atom_outputs[0][0].numpy()
        self.atom_scores[atom_string] = score
        return score

    def predict_batch(self, atoms: Sequence[Union[str, Tuple]]) -> List[float]:
        """Evaluate a batch of atoms and return their KGE scores."""
        if not atoms:
            return []

        strings, tuples = [], []
        for a in atoms:
            if isinstance(a, tuple):
                tup = a
                s = f"{tup[0]}({','.join(map(str, tup[1:]))})"
            else:
                s = a
                tup = Atom(s=s, format="functional").toTuple()
            strings.append(s)
            tuples.append(tup)

        to_eval_strs, to_eval_tuples = [], []
        for s, t in zip(strings, tuples):
            if s not in self.atom_scores:
                if s not in to_eval_strs:
                    to_eval_strs.append(s)
                    to_eval_tuples.append(t)

        if to_eval_tuples:
            model_inputs, _ = self._prepare_batch(to_eval_tuples)
            if self.model is None:
                self.model = self._build_and_load_model()
            kge_inputs = (model_inputs[0], model_inputs[1])
            batch_scores, _ = self.model.kge_model.call(kge_inputs)
            for s, score in zip(to_eval_strs, batch_scores.numpy().flatten()):
                self.atom_scores[s] = float(score)

        return [self.atom_scores[s] for s in strings]


def score_datasets(inference_engine: KGEInference, output_file: str, num_negatives: Optional[int], batch_size: int = 256):
    """
    Scores atoms from train, valid, and test sets with memory optimization.
    It processes one positive query and its negatives at a time to save memory.
    """
    print(f"Starting dataset scoring. Results will be saved to '{output_file}'")
    data_handler = inference_engine.data_handler
    
    # This set will hold all unique atoms we've scored to avoid redundant work.
    scored_atoms = set(inference_engine.atom_scores.keys())

    with open(output_file, "w") as f_out:
        # Write existing scores first
        for atom_str, score in inference_engine.atom_scores.items():
             f_out.write(f"{atom_str}\t{score:.6f}\n")

        for split in ["test"]:
            print(f"\n--- Processing '{split}' set ---")
            
            dataset = data_handler.get_dataset(split=split, number_negatives=num_negatives)
            
            # Process one positive query at a time to save memory
            start = time.time()
            for i in range(len(dataset)):
                print(f"Processing sample {i+1}/{len(dataset)} in '{split}' split...", end= '\r')
                # This gets the positive atom and its generated negatives
                queries_for_sample, _ = dataset[i]
                atoms_to_process_now = set()
                if split == 'train':
                    print(f"Negatives per query to score: {len(queries_for_sample)}") if i == 0 else None
                    atoms_to_process_now.update(queries_for_sample)
                else: # valid/test
                    print(f"Negatives per query to score: {len(queries_for_sample[0])} for head, "\
                          f"{len(queries_for_sample[1])} for tail") if i == 0 else None
                    atoms_to_process_now.update(queries_for_sample[0])
                    atoms_to_process_now.update(queries_for_sample[1])

                # Filter out atoms that have already been scored
                new_atoms_to_score = []
                for atom_tuple in atoms_to_process_now:
                    atom_str = f"{atom_tuple[0]}({','.join(map(str, atom_tuple[1:]))})"
                    if atom_str not in scored_atoms:
                        new_atoms_to_score.append(atom_tuple)

                if not new_atoms_to_score:
                    continue
                
                # Score the new atoms in batches
                for j in range(0, len(new_atoms_to_score), batch_size):
                    batch_tuples = new_atoms_to_score[j:j+batch_size]
                    try:
                        scores = inference_engine.predict_batch(batch_tuples)
                        
                        for atom_tuple, score in zip(batch_tuples, scores):
                            atom_str = f"{atom_tuple[0]}({','.join(atom_tuple[1:])})"
                            if atom_str not in scored_atoms:
                                f_out.write(f"{atom_str}\t{score:.6f}\n")
                                scored_atoms.add(atom_str)
                    except Exception as e:
                        print(f"Error scoring batch: {e}")

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start
                    start = time.time()
                    print(f"  Processed {i+1}/{len(dataset)} positive samples in {elapsed:.2f} seconds.")

            print(f"Finished scoring for '{split}' split.")

    print(f"\nAll datasets scored. Final results are in '{output_file}'.")


def run_single_query_test(inference_engine: KGEInference, atom_string: str, ignore_cache: bool):
    """
    Runs a test for a single atom, with an option to ignore the cache.
    """
    print(f"\n--- Running Single Query Test for: {atom_string} ---")
    
    # Temporarily store and clear the cache if requested
    original_scores = None
    if ignore_cache and atom_string in inference_engine.atom_scores:
        print("INFO: --ignore_cache flag is active. Temporarily clearing cache for this query.")
        original_scores = inference_engine.atom_scores.copy()
        del inference_engine.atom_scores[atom_string]

    # Perform the prediction
    start_time = time.time()
    score = inference_engine.predict(atom_string)
    end_time = time.time()

    # Restore the cache if it was cleared
    if original_scores is not None:
        inference_engine.atom_scores = original_scores
        print("INFO: Cache restored.")

    print("\n--- Test Results ---")
    if ignore_cache:
        print(f"Mode: Live Inference (Cache Ignored)")
    else:
        print(f"Mode: Standard (Cache-first)")
    
    print(f"Atom: '{atom_string}'")
    print(f"Score: {score:.8f}")
    print(f"Inference time: {end_time - start_time:.4f} seconds")


def generate_and_sort_negatives(inference_engine: KGEInference, atom_string: str):
    """
    Generates all negative corruptions for a given query, sorts them, and prints them.
    """
    print(f"\n--- Generating All Negative Corruptions for: {atom_string} ---")

    # 1. Convert string to atom tuple
    try:
        atom = Atom(s=atom_string, format="functional")
        query_tuple = atom.toTuple()
        if len(query_tuple) < 3:
             raise ValueError("Atom must be binary (have two arguments) to generate head/tail corruptions.")
    except Exception as e:
        print(f"Error parsing atom string: {e}")
        return

    # 2. Get the data handler components
    data_handler = inference_engine.data_handler
    known_facts = data_handler.ground_facts_set
    domain2constants = data_handler.domain2constants
    constant2domain = data_handler.constant2domain

    # 3. Generate all corruptions using the KGCDataHandler static method
    print("Generating corruptions...")
    corruptions_list = KGCDataHandler.create_all_corruptions(
        queries=[query_tuple],
        known_facts=known_facts,
        domain2constants=domain2constants,
        constant2domain=constant2domain,
        corrupt_mode='HEAD_AND_TAIL'
    )
    
    if not corruptions_list:
        print("Could not generate corruptions.")
        return

    # Extract head and tail corruptions
    head_corruptions = corruptions_list[0].head
    tail_corruptions = corruptions_list[0].tail
    all_negatives_tuples = set(head_corruptions + tail_corruptions)

    # 4. Convert tuples back to sorted strings for display
    print("Sorting negatives...")
    all_negatives_strings = sorted([f"{p}({args[0]},{args[1]})" for p, *args in all_negatives_tuples])

    # 5. Print the results
    print("\n--- Results ---")
    print(f"Found {len(all_negatives_strings)} unique negative corruptions for '{atom_string}'.")
    print("List of negatives (alphabetical order):")
    for neg_atom in all_negatives_strings:
        print(neg_atom)
    # print(f"all_negatives_strings: {all_negatives_strings}")


def _calculate_mrr_custom(scores: np.ndarray) -> float:
    """
    Helper to calculate reciprocal rank for a single query using the custom lexsort method.
    """
    random_keys = np.random.rand(len(scores))
    sorted_indices = np.lexsort((-random_keys, -scores))
    rank = np.where(sorted_indices == 0)[0][0] + 1
    return 1.0 / rank


def _calculate_mrr_tf_update(mrr_metric: ns.utils.MRRMetric, scores: np.ndarray):
    """
    Helper to update the state of the TF-Recommenders MRRMetric for a single query.
    """
    y_true = np.zeros_like(scores)
    y_true[0] = 1.0
    y_true_tf = tf.constant(y_true, dtype=tf.float32)[tf.newaxis, :]
    scores_tf = tf.constant(scores, dtype=tf.float32)[tf.newaxis, :]
    mrr_metric.update_state(y_true_tf, scores_tf)


def calculate_mrr(inference_engine: KGEInference, n_test_queries: Optional[int] = None):
    """
    Calculates the Mean Reciprocal Rank (MRR) on the test set using the same
    method as kge_eval_test.py - using proper data generator with filtered protocol.
    Also includes the custom MRR calculation for comparison.
    """
    print("\n--- Calculating MRR on the Test Set (Filtered Protocol) ---")
    
    # Use the same approach as kge_eval_test.py - create a proper data generator
    data_handler = inference_engine.data_handler
    fol = inference_engine.fol
    serializer = inference_engine.serializer
    
    # Get test dataset with proper filtering (same as kge_eval_test.py)
    dataset_test = data_handler.get_dataset(
        split="test",
        number_negatives=None,  # Use all negatives for standard eval (same as kge_eval_test.py)
        corrupt_mode='HEAD_AND_TAIL'  # Same as kge_eval_test.py
    )
    
    # Create data generator (same as kge_eval_test.py)
    data_gen_test = ns.dataset.DataGenerator(
        dataset_test,
        fol,
        serializer,
        engine=None,  # No grounding engine needed for KGE evaluation
        batch_size=16,  # Same as kge_eval_test.py default
        ragged=True  # Same as kge_eval_test.py
    )
    
    if n_test_queries is not None:
        print(f"INFO: Limiting to first {n_test_queries} batches for testing.")
    
    # Build model if not already built
    if inference_engine.model is None:
        inference_engine.model = inference_engine._build_and_load_model()
    
    # Initialize metric objects (same as kge_eval_test.py)
    concept_mrr = ns.utils.MRRMetric(name='concept_mrr')
    concept_h1 = ns.utils.HitsMetric(1, name='concept_hits@1')
    concept_h10 = ns.utils.HitsMetric(10, name='concept_hits@10')
    task_mrr = ns.utils.MRRMetric(name='task_mrr')
    task_h1 = ns.utils.HitsMetric(1, name='task_hits@1')
    task_h10 = ns.utils.HitsMetric(10, name='task_hits@10')

    # Additional custom MRR tracking
    reciprocal_ranks_custom = []

    # Iterate over the test dataset (same as kge_eval_test.py)
    processed_batches = 0
    for i, (x_batch, y_batch) in enumerate(data_gen_test):
        if n_test_queries is not None and processed_batches >= n_test_queries:
            break
            
        if i >= len(data_gen_test):  # Prevent infinite loops
            break
            
        print(f"Processing batch {i+1}/{len(data_gen_test)}", end='\r')
        
        # Get model predictions (same as kge_eval_test.py)
        # y_pred = inference_engine.model.predict_on_batch(x_batch)
        # y_pred = inference_engine.model.kge_model.call((x_batch[0], x_batch[1]))
        y_pred = inference_engine.model(x_batch, training=False)
        
        # Update the state of each metric (same as kge_eval_test.py)
        concept_mrr.update_state(y_batch['concept'], y_pred['concept'])
        concept_h1.update_state(y_batch['concept'], y_pred['concept'])
        concept_h10.update_state(y_batch['concept'], y_pred['concept'])
        
        task_mrr.update_state(y_batch['task'], y_pred['task'])
        task_h1.update_state(y_batch['task'], y_pred['task'])
        task_h10.update_state(y_batch['task'], y_pred['task'])
        
        # Custom MRR calculation for comparison
        concept_scores = y_pred['concept'].numpy()
        for batch_idx in range(concept_scores.shape[0]):
            scores = concept_scores[batch_idx]
            rr_custom = _calculate_mrr_custom(scores)
            reciprocal_ranks_custom.append(rr_custom)
        
        processed_batches += 1
        rolling_mrr_custom = np.mean(reciprocal_ranks_custom) if reciprocal_ranks_custom else 0.0
        print(f"Processed query {i+1}/{len(data_gen_test)}",
            f"Rolling MRR: {concept_mrr.result().numpy():.4f},",
            f"Custom MRR: {rolling_mrr_custom:.4f}", end='\r')
        
    print("\nManual evaluation loop finished.")

    # Display Results (same as kge_eval_test.py)
    print("\n" + "="*50)
    print("          MRR Calculation Complete")
    print("="*50)
    print(f"Run Signature: {inference_engine.run_signature}")
    print(f"Seed: {inference_engine.seed}")
    print("\nMetrics:")

    # Collect and print results from the metric objects (same as kge_eval_test.py)
    results = {
        'concept_mrr': concept_mrr.result().numpy(),
        'concept_hits@1': concept_h1.result().numpy(),
        'concept_hits@10': concept_h10.result().numpy(),
        'task_mrr': task_mrr.result().numpy(),
        'task_hits@1': task_h1.result().numpy(),
        'task_hits@10': task_h10.result().numpy(),
    }
    for key, value in results.items():
        print(f"  {key:<25}: {value:.4f}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="KGE Model Inference and Scoring")
    parser.add_argument('--mode', type=str, default='calculate_mrr', 
                        choices=['predict', 'calculate_mrr', 'score', 'test_single', 'generate_negatives'], 
                        help="Execution mode.")
    parser.add_argument('--atom', type=str, default='aunt(5,76)', 
                        help="The atom to use for 'predict', 'test_single', or 'generate_negatives' mode.")
    parser.add_argument('--dataset', type=str, default='family', help="Name of the dataset.")
    parser.add_argument('--base_path', type=str, default='data', help="Base path to the data directory.")
    parser.add_argument('--checkpoint_dir', type=str, default='./../../checkpoints/', help="Directory where model checkpoints are saved.")
    parser.add_argument('--run_signature', type=str, default='kinship_family-backward_0_1-no_reasoner-complex-True-256-256-4-rules.txt', help="The signature of the training run.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--scores_file', type=str, default=None, help="Optional path to a file with pre-computed atom scores.")
    parser.add_argument('--n_test_queries', type=int, default=None, help="Number of test batches for a quick MRR test.")
    parser.add_argument('--num_negatives', type=int, default=None, help="Number of negative samples per positive. Default is all.")
    parser.add_argument('--batch_size', type=int, default=2048, help="Batch size for scoring.")
    parser.add_argument('--ignore_cache', action='store_true', help="Force live inference by ignoring the scores cache for 'test_single' mode.")
    
    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    args.base_path = os.path.join(current_dir, args.base_path)

    try:
        inference_engine = KGEInference(
            dataset_name=args.dataset,
            base_path=args.base_path,
            checkpoint_dir=args.checkpoint_dir,
            run_signature=args.run_signature,
            seed=args.seed,
            scores_file_path=args.scores_file
        )

        if args.mode == 'predict':
            print(f"\n--- Running in 'predict' mode for atom: {args.atom} ---")
            score = inference_engine.predict(args.atom)
            print(f"\nFinal prediction score: {score:.4f}")
        
        elif args.mode == 'calculate_mrr':
            calculate_mrr(inference_engine, args.n_test_queries)
            
        elif args.mode == 'score':
            print("\n--- Running in 'score' mode ---")
            output_file = os.path.join(args.base_path, f'kge_scores_{args.dataset}.txt')
            score_datasets(inference_engine, output_file, args.num_negatives, args.batch_size)
            
        elif args.mode == 'test_single':
            run_single_query_test(inference_engine, args.atom, args.ignore_cache)
            
        elif args.mode == 'generate_negatives':
            generate_and_sort_negatives(inference_engine, args.atom)

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please check that the dataset name, paths, and run signature are correct.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()