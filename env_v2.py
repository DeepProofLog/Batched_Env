import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from typing import List, Optional, Tuple, Dict, Any
import random
import logging

from index_manager_v2 import IndexManager
from negative_sampling_v2 import generate_dynamic_negatives
from dataset_v2 import DataHandler
from python_unification import get_next_unification_python
from utils import Term, Rule


logging.basicConfig(level=logging.INFO) # Configure logging level

class LogicEnv_gym_batch(gym.Env):
    """
    Refactored Gymnasium environment using batched tensor operations.
    V4: Assumes batched engine, optimized filtering (no memory pruning).
    """

    def __init__(self,
                 batch_size: int,
                 index_manager: IndexManager,
                 data_handler: DataHandler, # Use refactored DataHandler
                 mode: str = 'train',
                 corruption_mode: Optional[str] = None, # 'dynamic' or None
                 train_neg_pos_ratio: int = 1,
                 seed: Optional[int] = None,
                 max_depth: int = 10,
                 # memory_pruning: bool = True, # Removed
                 end_proof_action: bool = False,
                 skip_unary_actions: bool = False, # Note: Less relevant if engine is batched
                 truncate_atoms: bool = False, # Filter derived states if they exceed padding_atoms
                 # truncate_states: bool = False, # Implicitly handled by padding_states
                 padding_states: int = 20,
                 verbose: int = 0,
                 prover_verbose: int = 0,
                 device: torch.device = torch.device("cpu"),
                 engine: str = 'python_batch', # Indicate using batched engine
                 ):

        super().__init__()

        if engine != 'python_batch':
             logging.warning(f"Engine '{engine}' not recognized. Assuming 'python_batch'.")
             engine = 'python_batch'

        self.batch_size = batch_size
        self.index_manager = index_manager
        self.data_handler = data_handler
        self.device = device
        self.verbose = verbose
        self.prover_verbose = prover_verbose
        self.engine = engine

        # Get parameters from components
        self.max_arity = self.index_manager.max_arity
        self.padding_atoms = self.index_manager.padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth

        # Store special indices
        self.true_idx = self.index_manager.true_idx
        self.false_idx = self.index_manager.false_idx
        self.end_idx = self.index_manager.end_idx
        if self.true_idx < 0 or self.false_idx < 0 or self.end_idx < 0:
             logging.warning("IndexManager did not properly initialize special predicate indices.")

        self._set_seed(seed)

        # Data and Corruption Setup
        self.facts = self.data_handler.facts # List[Term]
        self.rules = self.data_handler.rules # List[Rule]
        self.corruption_mode = corruption_mode
        self.train_neg_pos_ratio = train_neg_pos_ratio
        self.counter = 0

        # Batch-specific state
        # self.memory = [set() for _ in range(self.batch_size)] # Removed
        self.current_depth = torch.zeros(self.batch_size, dtype=torch.int, device=self.device)
        self.current_queries_terms: List[Optional[Term]] = [None] * self.batch_size
        self.current_labels = torch.zeros(self.batch_size, dtype=torch.int, device=self.device)

        # Configuration Flags
        # self.memory_pruning = memory_pruning # Removed
        self.end_proof_action = end_proof_action
        self.skip_unary_actions = skip_unary_actions # Harder to implement with batched engine
        self.truncate_atoms = truncate_atoms
        # truncate_states flag is used implicitly by padding_states limit during filtering

        # Mode and Query Lists
        assert mode in ['train', 'eval', 'eval_corr'], f"Invalid mode: {mode}"
        self.mode = mode
        self.train_queries = self.data_handler.train_queries
        self.eval_queries = self.data_handler.valid_queries # Use valid set for eval
        self.n_train_queries = len(self.train_queries)
        self.n_eval_queries = len(self.eval_queries)
        self.eval_indices = list(range(self.n_eval_queries))
        self.current_eval_batch_idx = 0

        if self.mode == 'train':
             if self.corruption_mode == "dynamic":
                  if not self.data_handler.sampler:
                       raise ValueError("Dynamic corruption requires sampler to be set in DataHandler.")
                  if not self.train_queries:
                       raise ValueError("Dynamic corruption requires train_queries.")
             elif not self.train_queries:
                  raise ValueError("Training mode requires train_queries.")

        # Define Observation and Action Spaces
        self._make_spec()

        # Initialize TensorDict structure
        self.tensordict = self._create_empty_tensordict()

    # --- Methods from V3 (keep _set_seed, _create_empty_tensordict, _make_spec, _sample_initial_batch, reset, _vectorized_done_reward, step, _get_obs_from_tensordict, _log_batch_state, render, close) ---
    # --- Note: reset and step no longer interact with self.memory ---
    def _set_seed(self, seed: Optional[int]):
        """Set the seed for the environment."""
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # The action space sampling also needs seeding if used directly
        # self.action_space.seed(self.seed)
        logging.info(f"Environment seeded with {self.seed}")

    def _create_empty_tensordict(self) -> TensorDict:
         """Creates the TensorDict structure with correct shapes and dtypes."""
         td_data = {
            "state_atom_idx": torch.zeros(self.batch_size, self.padding_atoms, dtype=torch.int64, device=self.device),
            "state_sub_idx": torch.zeros(self.batch_size, self.padding_atoms, self.max_arity + 1, dtype=torch.int64, device=self.device),
            "derived_atom_idx": torch.zeros(self.batch_size, self.padding_states, self.padding_atoms, dtype=torch.int64, device=self.device),
            "derived_sub_idx": torch.zeros(self.batch_size, self.padding_states, self.padding_atoms, self.max_arity + 1, dtype=torch.int64, device=self.device),
            "action_mask": torch.zeros(self.batch_size, self.padding_states, dtype=torch.bool, device=self.device),
            "label": torch.zeros(self.batch_size, dtype=torch.int, device=self.device),
            "done": torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
            "reward": torch.zeros(self.batch_size, dtype=torch.float32, device=self.device),
         }
         return TensorDict(td_data, batch_size=[self.batch_size], device=self.device)

    def _make_spec(self):
        """Create the batched observation and action specs using tensor representations."""
        obs_spaces = {
            'state_atom_idx': gym.spaces.Box(0, np.iinfo(np.int64).max, (self.batch_size, self.padding_atoms), dtype=np.int64),
            'state_sub_idx': gym.spaces.Box(0, np.iinfo(np.int64).max, (self.batch_size, self.padding_atoms, self.max_arity + 1), dtype=np.int64),
            'derived_atom_idx': gym.spaces.Box(0, np.iinfo(np.int64).max, (self.batch_size, self.padding_states, self.padding_atoms), dtype=np.int64),
            'derived_sub_idx': gym.spaces.Box(0, np.iinfo(np.int64).max, (self.batch_size, self.padding_states, self.padding_atoms, self.max_arity + 1), dtype=np.int64),
            'action_mask': gym.spaces.Box(0, 1, (self.batch_size, self.padding_states), dtype=np.bool_),
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.MultiDiscrete([self.padding_states] * self.batch_size)

    def _sample_initial_batch(self) -> Tuple[List[Term], List[int]]:
        """Samples a batch of initial queries (Terms) and labels for TRAINING mode."""
        batch_queries_terms: List[Term] = []
        batch_labels: List[int] = []

        if not self.train_queries:
             raise ValueError("Cannot sample for training, train_queries list is empty.")

        if self.corruption_mode == "dynamic":
            num_pos_needed = 0
            num_neg_needed = 0
            indices_for_pos = []
            indices_for_neg_base = [] # Indices of positives to be corrupted

            for _ in range(self.batch_size):
                 should_be_positive = (self.counter % (self.train_neg_pos_ratio + 1) == 0)
                 self.counter += 1
                 if should_be_positive:
                      num_pos_needed += 1
                      idx = random.randrange(self.n_train_queries)
                      indices_for_pos.append(idx)
                 else:
                      num_neg_needed += 1
                      idx = random.randrange(self.n_train_queries)
                      indices_for_neg_base.append(idx)

            # Collect positive terms to keep
            batch_queries_terms.extend([self.train_queries[i] for i in indices_for_pos])
            batch_labels.extend([1] * len(indices_for_pos))

            # Generate negatives
            if num_neg_needed > 0:
                 pos_terms_to_corrupt = [self.train_queries[i] for i in indices_for_neg_base]
                 if not self.data_handler.sampler: raise RuntimeError("DataHandler sampler is not set.")

                 generated_negatives = generate_dynamic_negatives(
                     positive_terms=pos_terms_to_corrupt,
                     index_manager=self.index_manager,
                     sampler=self.data_handler.sampler,
                     device=self.device
                 )

                 for i in range(num_neg_needed):
                      neg_term = generated_negatives[i]
                      if neg_term:
                           batch_queries_terms.append(neg_term)
                           batch_labels.append(0)
                      else:
                           logging.warning(f"Failed to generate negative for {pos_terms_to_corrupt[i]}, using original positive.")
                           batch_queries_terms.append(pos_terms_to_corrupt[i])
                           batch_labels.append(1)
        else: # No corruption
            sampled_indices = random.choices(range(self.n_train_queries), k=self.batch_size)
            batch_queries_terms = [self.train_queries[i] for i in sampled_indices]
            train_labels = getattr(self.data_handler, "train_labels", [1]*self.n_train_queries)
            batch_labels = [train_labels[i] for i in sampled_indices]

        # Final check for batch size
        if len(batch_queries_terms) != self.batch_size:
             logging.error(f"Batch size mismatch after sampling: {len(batch_queries_terms)} vs {self.batch_size}. Fixing...")
             if len(batch_queries_terms) > self.batch_size:
                  batch_queries_terms = batch_queries_terms[:self.batch_size]
                  batch_labels = batch_labels[:self.batch_size]
             else:
                  num_missing = self.batch_size - len(batch_queries_terms)
                  if not self.train_queries: raise ValueError("Cannot pad batch, no train queries.")
                  extra_indices = random.choices(range(self.n_train_queries), k=num_missing)
                  train_labels = getattr(self.data_handler, "train_labels", [1]*self.n_train_queries)
                  batch_queries_terms.extend([self.train_queries[i] for i in extra_indices])
                  batch_labels.extend([train_labels[i] for i in extra_indices])

        return batch_queries_terms, batch_labels

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              queries_to_evaluate: Optional[List[Term]] = None,
              labels_to_evaluate: Optional[List[int]] = None
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets environments, optionally using provided queries/labels."""
        if seed is not None: self._set_seed(seed)
        logging.info(f'Resetting Batch (Size: {self.batch_size})')

        batch_queries_terms: List[Term]; batch_labels: List[int]

        if queries_to_evaluate is not None:
            logging.info("Resetting with provided queries for evaluation.")
            if len(queries_to_evaluate) != self.batch_size: raise ValueError(f"Provided queries length ({len(queries_to_evaluate)}) must match batch_size ({self.batch_size}).")
            if labels_to_evaluate is None: labels_to_evaluate = [1] * self.batch_size; logging.warning("Assuming label 1 for provided queries.")
            elif len(labels_to_evaluate) != self.batch_size: raise ValueError(f"Provided labels length ({len(labels_to_evaluate)}) must match batch_size ({self.batch_size}).")

            batch_queries_terms = queries_to_evaluate
            batch_labels = labels_to_evaluate

        elif self.mode == 'eval' or self.mode == 'eval_corr':
             if not self.eval_queries: raise ValueError("Cannot reset in eval mode, eval_queries list is empty.")
             
             start_idx = self.current_eval_batch_idx * self.batch_size
             end_idx = start_idx + self.batch_size
             if start_idx >= self.n_eval_queries: start_idx = 0 
             end_idx = self.batch_size
             self.current_eval_batch_idx = 0 
             logging.warning("Eval reset wrapping around.")
             
             actual_indices = self.eval_indices[start_idx : min(end_idx, self.n_eval_queries)]
             batch_queries_terms = [self.eval_queries[i] for i in actual_indices]
             eval_labels = getattr(self.data_handler, "valid_labels", [1]*self.n_eval_queries)
             batch_labels = [eval_labels[i] for i in actual_indices]
             num_missing = self.batch_size - len(batch_queries_terms)
             if num_missing > 0:
                 logging.info(f"Padding eval batch with {num_missing} initial eval queries.")
                 if not self.eval_queries: raise ValueError("Cannot pad eval batch.")
                 batch_queries_terms.extend([self.eval_queries[0]] * num_missing); batch_labels.extend([eval_labels[0]] * num_missing)
             self.current_eval_batch_idx += 1

        elif self.mode == 'train': batch_queries_terms, batch_labels = self._sample_initial_batch()
        else: raise ValueError(f"Invalid mode for reset: {self.mode}")

        self.current_queries_terms = batch_queries_terms
        self.current_labels = torch.tensor(batch_labels, dtype=torch.int, device=self.device)
        self.current_depth.zero_()
        self.index_manager.reset_atom()
        # self.memory = [set() for _ in range(self.batch_size)] # Memory removed

        # Get initial atom and sub indices
        initial_atom_idx_list, initial_sub_idx_list = [], []
        for i in range(self.batch_size):
            query_term = batch_queries_terms[i]
            print(f"Query term: {query_term}")
            initial_state_terms = query_term if isinstance(query_term, list) else [query_term]
            if not initial_state_terms: initial_state_terms = [Term("False", [])]
            print(f"Initial state terms: {initial_state_terms}")

            atom_idx, sub_idx = self.index_manager.get_atom_sub_index(initial_state_terms)
            initial_atom_idx_list.append(atom_idx)
            initial_sub_idx_list.append(sub_idx)
            # No memory update needed here

        initial_atom_idx_batch = torch.stack(initial_atom_idx_list)
        initial_sub_idx_batch = torch.stack(initial_sub_idx_list)

        derived_atom_idx_batch, derived_sub_idx_batch, truncated_flags_batch, action_masks_batch = self.get_next_states_batch(initial_atom_idx_batch, initial_sub_idx_batch)

        self.tensordict.update_(TensorDict({
                "state_atom_idx": initial_atom_idx_batch, "state_sub_idx": initial_sub_idx_batch,
                "derived_atom_idx": derived_atom_idx_batch, "derived_sub_idx": derived_sub_idx_batch,
                "action_mask": action_masks_batch, "label": self.current_labels,
                "done": torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
                "reward": torch.zeros(self.batch_size, dtype=torch.float32, device=self.device),
            }, batch_size=[self.batch_size]))
        
        if torch.any(truncated_flags_batch): logging.info(f"Truncation occurred during reset filtering for indices: {torch.where(truncated_flags_batch)[0].tolist()}")
        obs = self._get_obs_from_tensordict(self.tensordict)
        if self.verbose > 1: self._log_batch_state("Reset")
        return obs, {}

    def get_next_states_batch(self,
                              current_atom_idx: torch.Tensor,
                              current_sub_idx: torch.Tensor,
                              dones_mask: Optional[torch.Tensor] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets the next possible states using the assumed batched engine and applies filtering.
        Fully vectorized filtering (excluding memory pruning, which is removed).
        """
        bs = current_atom_idx.shape[0]
        if dones_mask is None:
            dones_mask = torch.zeros(bs, dtype=torch.bool, device=self.device)

        # --- Call the Assumed Batched Unification Engine ---

        print(f"Current sub indices: {current_sub_idx.shape}")

        # get the sub indices for the facts and rules

        raw_derived_atom_idx, raw_derived_sub_idx, raw_action_mask = get_next_unification_python_batch(
               current_sub_idx=current_sub_idx,
               facts=facts_idx, 
               rules=rules_idx, 
               var_list=var_list,
               predicate_idx2str=self.index_manager.predicate_idx2str,
               constant_idx2str=self.index_manager.constant_idx2str,
               max_atoms=self.padding_atoms,
               device=self.device, 
               prover_verbose=self.prover_verbose
        )

        print(f"Raw derived atom indices: {raw_derived_atom_idx.shape}")
        print(f"Raw derived sub indices: {raw_derived_sub_idx.shape}")
        print(f"Raw action masks: {raw_action_mask.shape}")

        # --- Initialize Filtered Output Tensors ---
        # Start with raw results and apply masks
        filtered_derived_atom_idx = raw_derived_atom_idx
        filtered_derived_sub_idx = raw_derived_sub_idx
        filtered_action_masks = raw_action_mask
        batch_truncated_flags = torch.zeros(bs, dtype=torch.bool, device=self.device) # Track truncation due to atom limits or no valid states

        # --- Apply Filters (Vectorized, No Memory Pruning) ---

        # 1. Atom Truncation Mask
        atom_valid_mask = torch.ones_like(raw_action_mask)
        if self.truncate_atoms:
            num_atoms = (raw_derived_atom_idx > 0).sum(dim=2) # (bs, pad_states)
            atom_valid_mask = (num_atoms < self.padding_atoms)
            # Update truncation flags for states invalid due to atom limits
            batch_truncated_flags |= torch.any(~atom_valid_mask & raw_action_mask, dim=1)

        # Combine engine mask and atom truncation mask
        filtered_action_masks = filtered_action_masks & atom_valid_mask

        # 2. Handle "No Valid States" Case (Vectorized)
        valid_counts = filtered_action_masks.sum(dim=1) # (bs,)
        no_valid_mask = (valid_counts == 0) & (~dones_mask) # Only apply if not already done

        if torch.any(no_valid_mask):
            if self.verbose > 1: logging.info(f"Batch items with no valid states after filtering: {torch.where(no_valid_mask)[0].tolist()}. Inserting False.")
            # Get indices for 'False' state once
            false_term = Term("False", [])
            false_atom_idx_single, false_sub_idx_single = self.index_manager.get_atom_sub_index([false_term])
            false_atom_len = false_atom_idx_single.shape[0]

            if self.padding_atoms >= false_atom_len:
                 # Create broadcastable tensors for False state indices
                 false_atom_fill = torch.zeros(1, self.padding_atoms, dtype=torch.int64, device=self.device)
                 false_atom_fill[0, :false_atom_len] = false_atom_idx_single
                 false_sub_fill = torch.zeros(1, self.padding_atoms, self.max_arity + 1, dtype=torch.int64, device=self.device)
                 false_sub_fill[0, :false_atom_len, :] = false_sub_idx_single

                 # Use torch.where to insert False state indices for relevant batch items at the first position (index 0)
                 filtered_derived_atom_idx = torch.where(
                     no_valid_mask.view(bs, 1, 1), # Expand mask for broadcasting
                     false_atom_fill.expand(bs, -1), # Broadcast False state atom indices
                     filtered_derived_atom_idx[:, 0, :] # Keep original if not no_valid_mask
                 ).view(bs, 1, self.padding_atoms).expand(-1, self.padding_states, -1) # Hacky way to put it in pos 0 and keep shape - needs rework

                 # This needs a better way to insert into index 0 only
                 # Let's try direct assignment with mask
                 filtered_derived_atom_idx[no_valid_mask, 0, :false_atom_len] = false_atom_idx_single
                 filtered_derived_atom_idx[no_valid_mask, 0, false_atom_len:] = 0 # Zero out remaining padding
                 filtered_derived_sub_idx[no_valid_mask, 0, :false_atom_len, :] = false_sub_idx_single
                 filtered_derived_sub_idx[no_valid_mask, 0, false_atom_len:, :] = 0 # Zero out remaining padding

                 # Update action mask: only action 0 is valid now for these items
                 filtered_action_masks[no_valid_mask, 0] = True
                 filtered_action_masks[no_valid_mask, 1:] = False

                 # Mark these as truncated (because filtering led to failure state)
                 batch_truncated_flags |= no_valid_mask
            else:
                 logging.error("Cannot insert False state, padding_atoms too small.")
                 # Mark as truncated anyway if we can't insert False
                 batch_truncated_flags |= no_valid_mask


        # 3. Apply final mask to zero out invalid entries in index tensors
        # This doesn't compact, just zeros out invalid slots
        filtered_derived_atom_idx = filtered_derived_atom_idx * filtered_action_masks.unsqueeze(-1)
        filtered_derived_sub_idx = filtered_derived_sub_idx * filtered_action_masks.unsqueeze(-1).unsqueeze(-1)

        # 4. Handle "End Proof Action" (Removed for full vectorization)
        # Adding this back vectorially is complex and might require assumptions
        # about the engine or less efficient operations.

        return filtered_derived_atom_idx, filtered_derived_sub_idx, batch_truncated_flags, filtered_action_masks

    def _vectorized_done_reward(self,
                                next_atom_idx: torch.Tensor, # (bs, pad_atoms)
                                next_sub_idx: torch.Tensor,  # (bs, pad_atoms, arity+1)
                                labels: torch.Tensor        # (bs,)
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates done flags and rewards for the batch using tensor operations."""
        valid_atom_mask = (next_atom_idx > 0)
        pred_indices = next_sub_idx[:, :, 0]
        is_false = torch.any((pred_indices == self.false_idx) & valid_atom_mask, dim=1)
        is_end = torch.any((pred_indices == self.end_idx) & valid_atom_mask, dim=1) if self.end_proof_action else torch.zeros_like(is_false)
        num_valid_atoms = valid_atom_mask.sum(dim=1)
        all_valid_are_true = torch.all((pred_indices == self.true_idx) | (~valid_atom_mask), dim=1)
        is_true = (num_valid_atoms > 0) & all_valid_are_true
        terminated = is_false | is_end | is_true
        successful = is_true
        rewards = torch.where(
            terminated & successful & (labels == 1),
            torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device)
        )
        return terminated, rewards

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Performs a vectorized step for the batch."""
        if isinstance(actions, np.ndarray): actions = torch.from_numpy(actions).to(device=self.device, dtype=torch.int64)
        if actions.shape != (self.batch_size,): raise ValueError(f"Actions shape mismatch.")

        current_action_mask = self.tensordict["action_mask"]
        invalid_action_chosen_mask = ~torch.gather(current_action_mask, 1, actions.unsqueeze(1)).squeeze(1)
        invalid_action_chosen_mask |= (actions >= self.padding_states)
        if torch.any(invalid_action_chosen_mask):
             logging.warning(f"Invalid actions chosen: {torch.where(invalid_action_chosen_mask)[0].tolist()}. Clamping.")
             actions = torch.where(invalid_action_chosen_mask, 0, actions)

        action_idx_atom_gather = actions.view(self.batch_size, 1, 1).expand(-1, -1, self.padding_atoms)
        action_idx_sub_gather = actions.view(self.batch_size, 1, 1, 1).expand(-1, -1, self.padding_atoms, self.max_arity + 1)
        next_atom_idx = torch.gather(self.tensordict["derived_atom_idx"], 1, action_idx_atom_gather).squeeze(1)
        next_sub_idx = torch.gather(self.tensordict["derived_sub_idx"], 1, action_idx_sub_gather).squeeze(1)

        terminated, rewards = self._vectorized_done_reward(next_atom_idx, next_sub_idx, self.tensordict["label"])
        self.current_depth += 1
        truncated = (self.current_depth >= self.max_depth)
        terminated_final = terminated; truncated_final = truncated
        rewards = torch.where(truncated_final & ~terminated_final & (rewards == 0), 0.0, rewards)

        is_done_mask = terminated_final | truncated_final
        # Memory update loop removed
        self.current_depth = torch.where(is_done_mask, 0, self.current_depth)

        derived_atom_next, derived_sub_next, truncated_flags_next, action_masks_next = self.get_next_states_batch(next_atom_idx, next_sub_idx, terminated_final)
        truncated_final = truncated_final | truncated_flags_next
        done_combined = terminated_final | truncated_final

        self.tensordict.update_(TensorDict({
                "state_atom_idx": next_atom_idx, "state_sub_idx": next_sub_idx,
                "derived_atom_idx": derived_atom_next, "derived_sub_idx": derived_sub_next,
                "action_mask": action_masks_next, "done": done_combined, "reward": rewards,
            }, batch_size=[self.batch_size]))

        obs = self._get_obs_from_tensordict(self.tensordict)
        rewards_np = rewards.cpu().numpy(); terminateds_np = terminated_final.cpu().numpy(); truncateds_np = truncated_final.cpu().numpy()
        infos = {}
        if self.verbose > 1: self._log_batch_state(f"Step (Action: {actions.cpu().numpy()})")
        return obs, rewards_np, terminateds_np, truncateds_np, infos

    def _get_obs_from_tensordict(self, td: TensorDict) -> Dict[str, np.ndarray]:
        """Extracts numpy observations from the TensorDict based on observation_space."""
        obs = {}
        for key in self.observation_space.spaces:
            if key in td.keys(include_nested=True):
                 tensor_data = td[key]
                 if isinstance(tensor_data, torch.Tensor): obs[key] = tensor_data.cpu().numpy()
                 else: logging.error(f"Obs key '{key}' not tensor."); obs[key] = np.zeros(self.observation_space[key].shape, dtype=self.observation_space[key].dtype)
            else: raise KeyError(f"Obs key '{key}' not in TensorDict: {td.keys()}")
        return obs

    def _log_batch_state(self, stage: str):
        """Helper to log the current state of the batch (for debugging)."""
        print(f"\n--- {stage} - Batch State ---")
        print(f"Depth: {self.current_depth.cpu().numpy()}")
        print(f"Labels: {self.tensordict['label'].cpu().numpy()}")
        print(f"Dones: {self.tensordict['done'].cpu().numpy()}")
        print(f"Rewards: {self.tensordict['reward'].cpu().numpy()}")
        if self.verbose > 2:
             for i in range(self.batch_size):
                  state_terms = self.index_manager.get_term_from_indices(self.tensordict["state_atom_idx"][i], self.tensordict["state_sub_idx"][i])
                  print(f"  [Env {i}] D:{self.current_depth[i].item()} Done:{self.tensordict['done'][i].item()} R:{self.tensordict['reward'][i].item():.2f}")
                  print(f"    State: {state_terms}")
                  print(f"    Mask: {self.tensordict['action_mask'][i].cpu().numpy().astype(int)}")
                  # Memory logging removed
        print("-" * 25)

    def render(self, mode='human'): pass
    def close(self): logging.info("Closing LogicEnv_gym_batch environment.")

# Example Usage (Illustrative - Needs actual setup and BATCHED engine)
if __name__ == '__main__':
     print("--- Running Refactored Batched Environment Example V4 ---")
     # --- Dummy Setup ---
     constants = {'a', 'b', 'c', 'd'}; predicates = {'p', 'q', 'r'}; variables = {'X', 'Y', 'Z'}; rules = []
     idx_manager = IndexManager(constants=constants, predicates=predicates, variables=variables, constant_no=len(constants), predicate_no=len(predicates), variable_no=len(variables), rules=rules, padding_atoms=5, max_arity=2, device=torch.device("cpu"))
     data_handler = DataHandler(dataset_name="dummy", base_path=".")
     data_handler.train_queries = [Term('p', ['a', 'b']), Term('q', ['b', 'c']), Term('r', ['a', 'd'])]
     data_handler.valid_queries = [Term('p', ['a', 'b']), Term('q', ['b', 'c'])]

     batch_size = 4
     env = LogicEnv_gym_batch(batch_size=batch_size, index_manager=idx_manager, data_handler=data_handler, mode='train', corruption_mode=None, train_neg_pos_ratio=1, padding_states=6, max_depth=5, device=torch.device("cpu"), engine='python_batch', verbose=1, # memory_pruning=True, # Removed
                              end_proof_action=False, # Disabled for simplicity in vectorized filter
                              truncate_atoms=True, seed=42)

     # Test Reset (Provided Queries)
     eval_queries = [Term('p', ['a', 'b']), Term('q', ['b', 'c']), Term('r', ['a', 'a']), Term('p', ['d', 'c'])]
     eval_labels = [1, 1, 0, 0]
     if len(eval_queries) == batch_size:
          print("\n--- Resetting with Provided Queries ---"); obs_eval, info_eval = env.reset(queries_to_evaluate=eval_queries, labels_to_evaluate=eval_labels, seed=44); env._log_batch_state("Reset Eval")
          # Test Step
          print("\n--- Testing Step Function ---"); actions_np = env.action_space.sample(); current_mask_np = env.tensordict['action_mask'].cpu().numpy()
          for i in range(batch_size): # Simple masking
               valid_actions = np.where(current_mask_np[i])[0];
               if len(valid_actions) > 0:
                    if actions_np[i] not in valid_actions: actions_np[i] = np.random.choice(valid_actions)
               else: actions_np[i] = 0
          print(f"\n--- Taking Actions: {actions_np} ---"); obs, rewards, terminateds, truncateds, infos = env.step(actions_np)
          print(f"Rewards: {rewards}"); print(f"Terminateds: {terminateds}"); print(f"Truncateds: {truncateds}"); env._log_batch_state("After Step 1")
     else: print(f"\nSkipping provided query reset test (need {batch_size} queries).")
     env.close()
