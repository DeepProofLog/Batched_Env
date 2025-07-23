import os
import tensorflow as tf
import numpy as np
import random
import json
import re
import sys
from typing import List, Tuple, Optional, Union, Sequence
import time
import argparse

# Assuming ns_lib is in the python path.
# If not, you might need to add it: sys.path.append('/path/to/ns_lib_parent_dir')
import ns_lib as ns
from kge_loader import KGCDataHandler
from kge_model import CollectiveModel


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
        print("KGEInference engine initialized. Model will be built on first use.")

        # Load pre-computed scores
        self.atom_scores = {}
        if scores_file_path:
            self._load_scores(scores_file_path)

    def _load_scores(self, filepath: str):
        """Loads pre-computed atom scores from a file."""
        if not os.path.exists(filepath):
            print(f"Warning: Scores file not found at {filepath}. KGE will perform live inference for all atoms.")
            return
            
        print(f"Loading pre-computed scores from {filepath}...")
        try:
            with open(filepath, "r") as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        atom_str, score_str = parts
                        try:
                            self.atom_scores[atom_str] = float(score_str)
                        except ValueError:
                            print(f"Warning: Could not parse score for line: {line.strip()}")
            print(f"Loaded {len(self.atom_scores)} scores.")
        except Exception as e:
            print(f"Error loading scores file: {e}")

    def set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def _load_data(self, dataset_name: str, base_path: str) -> KGCDataHandler:
        print("Loading data...")
        return KGCDataHandler(
            dataset_name=dataset_name,
            base_path=base_path,
            format='functional',
            domain_file='domain2constants.txt',
            train_file='train.txt',
            valid_file='valid.txt',
            test_file='test.txt',
            fact_file='facts.txt'
        )

    def _create_serializer(self) -> ns.serializer.LogicSerializerFast:
        return ns.serializer.LogicSerializerFast(
            predicates=self.fol.predicates,
            domains=self.fol.domains,
            constant2domain_name=self.fol.constant2domain_name
        )

    def _build_and_load_model(self) -> CollectiveModel:
        print("Building model and loading weights...")
        model = CollectiveModel(
            self.fol, rules=[], kge='complex', kge_regularization=0.0,
            constant_embedding_size=200, predicate_embedding_size=200,
            kge_atom_embedding_size=100, kge_dropout_rate=0.0,
            reasoner_depth=0, model_name='no_reasoner',
            reasoner_atom_embedding_size=100, reasoner_formula_hidden_embedding_size=100,
            reasoner_regularization=0.0, reasoner_single_model=False,
            reasoner_dropout_rate=0.0, aggregation_type='max', signed=True,
            temperature=0.0, resnet=True, embedding_resnet=False,
            filter_num_heads=3, filter_activity_regularization=0.0,
            num_adaptive_constants=0, dot_product=False,
            cdcr_use_positional_embeddings=False, cdcr_num_formulas=3,
            r2n_prediction_type='full', distill=False,
        )
        
        dummy_generator = ns.dataset.DataGenerator(
            self.data_handler.get_dataset(split="test", number_negatives=0), 
            self.fol, self.serializer, engine=None, batch_size=1, ragged=False
        )
        dummy_input = next(iter(dummy_generator))[0]
        model(dummy_input)
        model.kge_model((dummy_input[0], dummy_input[1]))
        
        name = f"{self.run_signature}_seed_{self.seed}"
        ckpt_filepath = os.path.join(self.checkpoint_dir, name, f"{name}_kge_model")
        
        print(f"Attempting to load weights from: {ckpt_filepath}.weights.h5")
        if self._load_kge_weights(model, ckpt_filepath):
            print("Weights loaded successfully.")
        else:
            raise FileNotFoundError(f"Could not load weights from {ckpt_filepath}.weights.h5")
            
        return model

    def _load_kge_weights(self, model: CollectiveModel, ckpt_filepath: str) -> bool:
        h5_path = ckpt_filepath + '.weights.h5'
        if os.path.exists(h5_path):
            model.kge_model.load_weights(h5_path)
            return True
        return False

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
        labels = [[1.0]] * len(queries)  # Dummy labels

        return ns.dataset._from_strings_to_tensors(
            fol=self.fol,
            serializer=self.serializer,
            queries=queries,
            labels=labels,
            engine=None,
            ragged=False,
        )

    def predict(self, atom_string: str) -> float:
        """
        Evaluates a single atom string and returns its score.
        First checks pre-computed scores, then falls back to the model.
        """
        # Check cache first
        if atom_string in self.atom_scores:
            # print(f"Using cached score for atom: {atom_string}: {self.atom_scores[atom_string]:.4f}")
            return self.atom_scores[atom_string]
        
        # Fallback to model inference
        (model_inputs, _y) = self._prepare_atom(atom_string)
        
        if self.model is None:
            self.model = self._build_and_load_model()
        
        kge_inputs = (model_inputs[0], model_inputs[1])
        atom_outputs, _ = self.model.kge_model.call(kge_inputs)
        
        score = atom_outputs[0][0].numpy()
        
        # Optionally cache the new score for this session
        self.atom_scores[atom_string] = score
        
        return score

    def predict_batch(self,
                    atoms: Sequence[Union[str, Tuple]]) -> List[float]:
        """
        Evaluate a batch of atoms (either functional strings or tuples) and
        return their KGE scores in the original order.  Results are cached using
        the functional-style string representation, so duplicates across (and
        within) calls are computed only once.
        """
        if not atoms:
            return []

        # ---------- 1. normalise to (string, tuple) -----------------------------
        strings: List[str]  = []
        tuples:  List[Tuple] = []

        for a in atoms:
            if isinstance(a, tuple):
                tup = a
                s   = f"{tup[0]}({','.join(map(str, tup[1:]))})"
            elif isinstance(a, str):
                s   = a
                tup = Atom(s=s, format="functional").toTuple()
            else:
                raise TypeError(f"Unsupported atom type: {type(a)}")
            strings.append(s)
            tuples.append(tup)

        # ---------- 2. work out what still needs the model ----------------------
        to_eval_strs, to_eval_tuples = [], []
        for s, t in zip(strings, tuples):
            if s not in self.atom_scores:
                # only keep unique unseen atoms
                if s not in to_eval_strs:
                    to_eval_strs.append(s)
                    to_eval_tuples.append(t)

        # ---------- 3. run the model once for the uncached atoms ----------------
        if to_eval_tuples:
            model_inputs, _ = self._prepare_batch(to_eval_tuples)
            if self.model is None:
                self.model = self._build_and_load_model()

            kge_inputs = (model_inputs[0], model_inputs[1])
            batch_scores, _ = self.model.kge_model.call(kge_inputs)
            for s, score in zip(to_eval_strs, batch_scores.numpy().flatten()):
                self.atom_scores[s] = float(score)

        # ---------- 4. materialise results in original order --------------------
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


def main():
    """
    Main function to run the KGE inference script.
    """
    parser = argparse.ArgumentParser(description="KGE Model Inference and Scoring")
    parser.add_argument('--mode', type=str, default='score', choices=['predict', 'score'],
                        help="Execution mode: 'predict' for a single atom, 'score' for all dataset atoms.")
    parser.add_argument('--atom', type=str, default='locatedInCR(italy,europe)',
                        help="The atom to predict in 'predict' mode, e.g., 'predicate(const1,const2)'")
    parser.add_argument('--dataset', type=str, default='family', help="Name of the dataset.")
    parser.add_argument('--base_path', type=str, default='data', help="Base path to the data directory.")
    parser.add_argument('--checkpoint_dir', type=str, default='./../../checkpoints/',
                        help="Directory where model checkpoints are saved.")
    parser.add_argument('--run_signature', type=str,
                        default='countries_s3-backward_0_1-no_reasoner-complex-True-256-256-128-rules.txt',
                        help="The signature of the training run to load.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    # parser.add_argument('--scores_file', type=str, default=None, help="Path to a file with pre-computed atom scores.")
    parser.add_argument('--scores_file', type=str, default='./../../', help="Path to a file with pre-computed atom scores.")
    parser.add_argument('--num_negatives', type=int, default=None, help="Number of negative samples per positive. Default is all.")
    parser.add_argument('--batch_size', type=int, default=2048, help="Batch size for scoring.")

    args = parser.parse_args()

    root = './data/'
    output_file = root+'kge_scores'+ f'_{args.dataset}.txt'

    if args.dataset == "family":
        args.run_signature = 'kinship_family-backward_0_1-no_reasoner-complex-True-256-256-4-rules.txt'

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
        
        elif args.mode == 'score':
            print("\n--- Running in 'score' mode ---")
            score_datasets(inference_engine, output_file, args.num_negatives, args.batch_size)

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please check that the dataset name, paths, and run signature are correct.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == '__main__':
    # Suppress excessive TensorFlow logging.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
