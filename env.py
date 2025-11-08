"""
Fully Vectorized Batched Logic Environment - Clean Version

This environment uses minimal dependencies and the UnificationEngine for proving.
Supports negative sampling, end-proof actions, and skip-unary actions.
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Set, Union
from tensordict import TensorDict
from torchrl.envs import EnvBase

from atom_stringifier import AtomStringifier
from unification_engine import UnificationEngine


class BatchedVecEnv(EnvBase):
    """
    Fully vectorized batched logic environment.
    
    Key features:
    - Minimal dependencies (no data_handler, no index_manager)
    - Uses UnificationEngine for proving
    - Supports negative sampling
    - Fully vectorized operations
    """
    
    def __init__(
        self,
        batch_size: int,
        queries: List[Tensor],
        labels: List[int],
        query_depths: List[int],
        unification_engine: UnificationEngine,
        sampler=None,
        max_arity: int = 2,
        padding_idx: int = 0,
        runtime_var_start_index: int = 1,
        total_vocab_size: int = 10000,
        true_pred_idx: Optional[int] = None,
        false_pred_idx: Optional[int] = None,
        end_pred_idx: Optional[int] = None,
        atom_stringifier: Optional[AtomStringifier] = None,
        mode: str = 'train',
        corruption_mode: bool = False,
        corruption_scheme: List[str] = ['head', 'tail'],
        train_neg_ratio: float = 1.0,
        max_depth: int = 10,
        memory_pruning: bool = True,
        end_proof_action: bool = False,
        skip_unary_actions: bool = False,
        padding_atoms: int = 10,
        padding_states: int = 20,
        reward_type: int = 1,
        verbose: int = 0,
        prover_verbose: int = 0,
        device: torch.device = None,
    ):
        """
        Initialize the environment with minimal dependencies.
        
        Args:
            batch_size: Number of parallel environments
            queries: List of query tensors [1, 3]
            labels: List of labels for queries
            query_depths: List of depths for queries
            unification_engine: UnificationEngine for proving
            sampler: Negative sampler (optional)
            max_arity: Maximum arity of predicates
            padding_idx: Index used for padding
            runtime_var_start_index: Starting index for runtime variables
            total_vocab_size: Total vocabulary size
            true_pred_idx: Index for True predicate
            false_pred_idx: Index for False predicate
            end_pred_idx: Index for End predicate
            atom_stringifier: Optional stringifier for debugging
            mode: 'train' or 'eval'
            corruption_mode: Enable negative sampling
            corruption_scheme: ['head', 'tail'] for negative sampling
            train_neg_ratio: Ratio of negatives to positives
            max_depth: Maximum depth for proving
            memory_pruning: Whether to enable memory pruning
            end_proof_action: Enable end-proof action
            skip_unary_actions: Skip unary actions (facts only)
            padding_atoms: Number of atoms to pad states to
            padding_states: Number of derived states to pad to
            reward_type: Reward function type (0-4)
                0: +1 only for successful proof with label=1
                1: +1 for successful proof with label=1, -1 for successful proof with label=0
                2: +1 for successful proof with label=1 OR failed proof with label=0
                3: Asymmetric rewards with penalties for false positives
                4: Symmetric ±1 rewards for correct classifications
            verbose: Verbosity level for environment
            prover_verbose: Verbosity level for prover
            device: Device to run on
            skip_unary_actions: Skip unary predicates in actions
            padding_atoms: Maximum atoms per state
            padding_states: Maximum derived states
            verbose: Verbosity level
            prover_verbose: Prover verbosity level
            device: Device to use
        """
        # Store batch_size as int for internal use
        self.batch_size_int = batch_size
        
        # Initialize parent with torch.Size([batch_size])
        super().__init__(
            device=device if device is not None else torch.device('cpu'),
            batch_size=torch.Size([batch_size]),
        )
        
        # Store parameters
        self.mode = mode
        self.max_depth = max_depth
        # Disable memory pruning for eval mode (it's mainly for training)
        self.memory_pruning = memory_pruning if mode == 'train' else False
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.reward_type = reward_type
        self.verbose = verbose
        self.prover_verbose = prover_verbose
        
        # Store device in internal variable (EnvBase manages self.device)
        if device is None:
            self._device_internal = torch.device('cpu')
        else:
            self._device_internal = device
        
        # Store queries and labels
        self.all_queries = queries
        self.all_labels = labels
        self.all_depths = query_depths
        
        # Store parameters from arguments
        self.max_arity = max_arity
        self.padding_idx = padding_idx
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx
        self.end_pred_idx = end_pred_idx
        self.runtime_var_start_index = runtime_var_start_index
        self.total_vocab_size = total_vocab_size
        
        # Store unification engine
        self.unification_engine = unification_engine
        
        # Store atom stringifier for debugging
        self.atom_stringifier = atom_stringifier
        
        # Negative sampling
        self.corruption_mode = corruption_mode
        self.corruption_scheme = corruption_scheme
        self.train_neg_ratio = train_neg_ratio
        self.sampler = sampler
        self.neg_counter = 0
        self.negation_toggle = 0  # For alternating head/tail
        
        # Action modifiers
        self.end_proof_action = end_proof_action
        self.skip_unary_actions = skip_unary_actions
        
        # Create End tensor if needed
        if self.end_pred_idx is not None and self.end_pred_idx >= 0:
            self.end_tensor = torch.tensor(
                [[self.end_pred_idx, self.padding_idx, self.padding_idx]],
                dtype=torch.long,
                device=self._device_internal
            )
        else:
            self.end_tensor = None
        
        # Batch state - initialize empty tensors for partial reset support
        self.current_queries: Tensor = torch.full(
            (self.batch_size_int, padding_atoms, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device_internal
        )
        self.current_labels: Tensor = torch.zeros(
            self.batch_size_int,
            dtype=torch.long,
            device=self._device_internal
        )
        self.current_depths: Tensor = torch.zeros(
            self.batch_size_int,
            dtype=torch.long,
            device=self._device_internal
        )
        self.next_var_indices: Tensor = torch.full(
            (self.batch_size_int,),
            self.runtime_var_start_index,
            dtype=torch.long,
            device=self._device_internal
        )
        self.original_queries: Tensor = torch.full(
            (self.batch_size_int, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device_internal
        )
        
        # Derived states: [B, max_derived, padding_atoms, arity+1]
        self.derived_states_batch: Tensor = torch.full(
            (self.batch_size_int, padding_states, padding_atoms, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device_internal
        )
        self.derived_states_counts: Tensor = torch.zeros(
            self.batch_size_int,
            dtype=torch.long,
            device=self._device_internal
        )
        
        # Memory tracking (still need sets for membership testing)
        self.memories: List[Set] = [set() for _ in range(self.batch_size_int)]
        
        # Counter for sampling
        self.counter = 0
        
        # Define observation and action specs
        self._make_specs()
    
    def _make_specs(self):
        """Define observation and action space specs."""
        from torchrl.data import Composite, Bounded
        
        max_vocab_size = self.total_vocab_size
        
        # Observation spec - batched version
        self.observation_spec = Composite(
            sub_index=Bounded(
                low=-1,
                high=max_vocab_size,
                shape=torch.Size([self.batch_size_int, 1, self.padding_atoms, self.max_arity + 1]),
                dtype=torch.int32,
                device=self._device_internal,
            ),
            derived_sub_indices=Bounded(
                low=-1,
                high=max_vocab_size,
                shape=torch.Size([self.batch_size_int, self.padding_states, self.padding_atoms, self.max_arity + 1]),
                dtype=torch.int32,
                device=self._device_internal,
            ),
            action_mask=Bounded(
                low=0,
                high=1,
                shape=torch.Size([self.batch_size_int, self.padding_states]),
                dtype=torch.bool,
                device=self._device_internal,
            ),
            shape=torch.Size([self.batch_size_int]),
        )
        
        # Action spec - batched
        self.action_spec = Bounded(
            low=0,
            high=self.padding_states - 1,
            shape=torch.Size([self.batch_size_int]),
            dtype=torch.int64,
            device=self._device_internal,
        )
        
        # Reward spec
        self.reward_spec = Bounded(
            low=-float('inf'),
            high=float('inf'),
            shape=torch.Size([self.batch_size_int, 1]),
            dtype=torch.float32,
            device=self._device_internal,
        )
        
        # Done spec
        self.done_spec = Bounded(
            low=0,
            high=1,
            shape=torch.Size([self.batch_size_int, 1]),
            dtype=torch.bool,
            device=self._device_internal,
        )
    
    def _debug_state_to_str(self, state_idx: Tensor, oneline: bool = False) -> str:
        """Convert state indices to string for debugging."""
        if self.atom_stringifier is None:
            return f"<state with shape {state_idx.shape}>"
        if state_idx.dim() == 1:
            # Single atom [3]
            return self.atom_stringifier.atom_to_str(state_idx)
        elif state_idx.dim() == 2:
            # State [k, 3]
            return self.atom_stringifier.state_to_str(state_idx)
        else:
            return f"<state with shape {state_idx.shape}>"
    
    def _debug_states_to_str(self, states: List[Tensor]) -> str:
        """Convert list of states to string for debugging."""
        if self.atom_stringifier is None:
            return f"<{len(states)} states>"
        return "[" + ", ".join([self._debug_state_to_str(s) for s in states]) + "]"

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        """Reset all or some environments in the batch.
        
        Args:
            tensordict: Optional TensorDict with "_reset" key indicating which envs to reset.
                       If None or no "_reset" key, resets all environments.
        """

        # Determine which environments to reset
        if tensordict is not None and "_reset" in tensordict.keys():
            # Partial reset - only reset environments where _reset is True
            reset_mask = tensordict["_reset"].squeeze(-1).bool()  # [B]
            indices_to_reset = torch.where(reset_mask)[0].tolist()
            num_to_reset = len(indices_to_reset)
            print(f'\n\nPartial Reset of indices {indices_to_reset}-----------------------------') if self.verbose else None
            
            if num_to_reset == 0:
                # No environments to reset - return current observation
                return self._create_observation()
        else:
            print('\n\n-----------------------------Reset-----------------------------') if self.verbose else None
            # Full reset - reset all environments
            reset_mask = torch.ones(self.batch_size_int, dtype=torch.bool, device=self._device_internal)
            indices_to_reset = list(range(self.batch_size_int))
            num_to_reset = self.batch_size_int
        
        # Sample queries for environments that need reset
        batch_queries = []
        batch_labels = []
        batch_depths = []
        
        if self.mode == 'train':
            # Sample with replacement
            indices = torch.randint(0, len(self.all_queries), (num_to_reset,))
            batch_queries = [self.all_queries[i] for i in indices]
            batch_labels = [self.all_labels[i] for i in indices]
            batch_depths = [self.all_depths[i] for i in indices]
            
            # Apply negative sampling/corruption if enabled
            if self.corruption_mode:
                # Apply corruption based on train_neg_ratio
                # E.g., if train_neg_ratio=0.5, 1/3 of samples will be negative (counter % 3 != 0)
                # If train_neg_ratio=1.0, 1/2 of samples will be negative (counter % 2 != 0)
                neg_pos_ratio_int = int(1.0 / self.train_neg_ratio) if self.train_neg_ratio > 0 else 2
                
                for i in range(num_to_reset):
                    self.counter += 1
                    # Decide if this sample should be corrupted
                    if self.counter % (neg_pos_ratio_int + 1) != 0:
                        # Corrupt this query using new Sampler API
                        # batch_queries[i] is [1, 3] - extract to [3]
                        query_atom = batch_queries[i][0]  # [3] - first atom (predicate, arg1, arg2)
                        
                        # Use sampler.corrupt with num_negatives=1
                        corrupted = self.sampler.corrupt(
                            query_atom.unsqueeze(0),  # [1, 3]
                            num_negatives=1,
                            device=self._device_internal
                        )  # Returns [1, 1, 3]
                        
                        # Extract corrupted triple and reshape to [1, 3]
                        batch_queries[i] = corrupted[0, 0].unsqueeze(0)  # [1, 3]
                        batch_labels[i] = 0  # Corrupted query has negative label
                        
        elif self.mode == 'eval':
            # Sequential sampling
            for _ in range(num_to_reset):
                if self.counter < len(self.all_queries):
                    batch_queries.append(self.all_queries[self.counter])
                    batch_labels.append(self.all_labels[self.counter])
                    batch_depths.append(self.all_depths[self.counter])
                    self.counter += 1
                else:
                    # Wrap around if we run out
                    batch_queries.append(self.all_queries[0])
                    batch_labels.append(self.all_labels[0])
                    batch_depths.append(self.all_depths[0])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Convert queries to tensors and ensure correct shape
        queries_tensors_list = []
        for query in batch_queries:
            # All queries should be tensors already (from DataHandler)
            if not isinstance(query, Tensor):
                raise ValueError(f"Expected query to be a Tensor, got {type(query)}")
            
            # Ensure correct shape
            if query.dim() == 1:
                # [3] -> [1, 3]
                query_tensor = query.unsqueeze(0)
            elif query.dim() == 2:
                # [n_atoms, arity+1] -> already correct
                query_tensor = query
            else:
                raise ValueError(f"Unexpected query tensor shape: {query.shape}")
            
            queries_tensors_list.append(self._pad_state(query_tensor))
        
        # Update state only for environments being reset
        for i, env_idx in enumerate(indices_to_reset):
            self.current_queries[env_idx] = queries_tensors_list[i]
            self.current_labels[env_idx] = batch_labels[i]
            self.current_depths[env_idx] = 0
            self.next_var_indices[env_idx] = self.runtime_var_start_index
            self.memories[env_idx] = set()
        
        # Extract original queries for exclusion during fact unification
        # Only for newly reset environments
        first_atom_mask = self.current_queries[:, :, 0] != self.padding_idx
        first_atom_indices = first_atom_mask.long().argmax(dim=1)  # [B]
        batch_indices = torch.arange(self.batch_size_int, device=self._device_internal)
        self.original_queries = self.current_queries[batch_indices, first_atom_indices]  # [B, arity+1]

        if self.verbose >=1: print(f"\nInitial Query (Label: {self.current_labels[batch_indices]}, depth: {self.current_depths[batch_indices]}): "
                                   f"{self._debug_state_to_str(self.current_queries[batch_indices, first_atom_indices], oneline=True)}")

        # Compute derived states for ALL environments (not just reset ones)
        # This is necessary because the unification function is vectorized
        self._compute_derived_states_vec()
        
        if self.verbose >= 1:
            print(f"\nReset States:")
            for i in range(self.batch_size_int):
                if reset_mask[i]:  # Only show for reset environments
                    print(f"  Env {i}: {self._debug_state_to_str(self.current_queries[i], oneline=True)}")
                    derived_count = self.derived_states_counts[i].item()
                    derived_states_list = [self.derived_states_batch[i, j] for j in range(derived_count)]
                    print(f"  Derived States ({derived_count}): {self._debug_states_to_str(derived_states_list)}")
        
        if self.verbose >= 1:
            print(f"\n[DEBUG _reset]")
            print(f"  Reset mask: {reset_mask.tolist()}")
            print(f"  Reset indices: {indices_to_reset}")
            print(f"  After _compute_derived_states_vec:")
            print(f"  derived_states_counts: {self.derived_states_counts.tolist()}")
        
        # Create batched observations
        return self._create_observation()
    
    def _compute_derived_states_vec(self):
        """Compute derived states for all environments - FULLY VECTORIZED."""
        # Use UnificationEngine to process all queries at once!
        # Always exclude original queries from fact matching to prevent direct lookup
        all_derived, updated_var_indices = self.unification_engine.get_derived_states(
            current_states=self.current_queries,
            next_var_indices=self.next_var_indices,
            excluded_queries=self.original_queries,
            labels=self.current_labels,
            verbose=self.prover_verbose,
        )
        
        # Update variable indices
        self.next_var_indices = updated_var_indices
        
        # Process derived states for each environment
        for i in range(self.batch_size_int):
            all_derived[i] = self._process_derived_states_for_env(i, all_derived[i], apply_skip_unary=True)
        
        # Find max count
        max_derived_count = max(len(states) for states in all_derived)
        max_derived_count = min(max_derived_count, self.padding_states)
        
        # Create batched tensor [B, max_derived, padding_atoms, arity+1]
        batched_derived = torch.full(
            (self.batch_size_int, self.padding_states, self.padding_atoms, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device_internal
        )
        
        counts = torch.zeros(self.batch_size_int, dtype=torch.long, device=self._device_internal)
        
        # Fill in derived states - minimized loop
        for i, derived_states in enumerate(all_derived):
            count = min(len(derived_states), self.padding_states)
            counts[i] = count
            for j, state in enumerate(derived_states[:self.padding_states]):
                batched_derived[i, j] = self._pad_state(state)
        
        self.derived_states_batch = batched_derived
        self.derived_states_counts = counts
    
    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Execute a batch of actions - FULLY VECTORIZED."""
        actions = tensordict["action"]  # [B]
        
        if self.verbose >= 1:
            print(f'\n\n-----------------------------Step-----------------------------')     

        # Store previous state for verbose output
        if self.verbose >= 2:
            prev_states = self.current_queries.clone()
        
        # Select next state for each environment using advanced indexing - NO LOOP!
        # Create batch indices
        batch_indices = torch.arange(self.batch_size_int, device=self._device_internal)
        
        if self.verbose >= 2:
            print(f'\n=== BEFORE ACTION ===')
            for i in range(self.batch_size_int):
                print(f'\nEnv {i}:')
                print(f'  Current state: {self._debug_state_to_str(self.current_queries[i], oneline=True)}')
                print(f'  Available derived states ({self.derived_states_counts[i].item()}):{self._debug_states_to_str([self.derived_states_batch[i, j] for j in range(self.derived_states_counts[i].item())])}')
                print(f'  Selected action: {actions[i].item()}')
        
        # Select states: [B, padding_atoms, arity+1]
        selected_states = self.derived_states_batch[batch_indices, actions]
        
        # Check if action was invalid (action_idx >= count)
        invalid_mask = actions >= self.derived_states_counts  # [B]
        
        # Create false state for invalid actions
        false_state = self._create_false_state()
        
        # if there are any invalid actions, raise an error
        if invalid_mask.any():
            raise ValueError(f"Invalid actions selected: {actions[invalid_mask].tolist()} with counts {self.derived_states_counts[invalid_mask].tolist()}")
        
        self.current_queries = selected_states
        
        # Increment depths - VECTORIZED
        self.current_depths += 1
        
        # if self.verbose >= 2:
        #     print(f'\n=== AFTER ACTION ===')
        #     for i in range(self.batch_size_int):
        #         print(f'\nEnv {i}:')
        #         print(f'  Next state: {self._debug_state_to_str(self.current_queries[i], oneline=True)}')
        #         print(f'  Depth: {self.current_depths[i].item()}')
        
        # Compute rewards and dones - VECTORIZED
        rewards, dones = self._get_done_reward_vec()
        
        if self.verbose >= 2:
            print(f'\n=== REWARD & DONE ===')
            for i in range(self.batch_size_int):
                print(f'Env {i}:   Reward: {rewards[i].item():.4f}.  Done: {dones[i].item()}')
        
        # Get next derived states only for non-done environments - OPTIMIZED
        self._compute_derived_states_conditional_vec(dones)
        
        # if self.verbose >= 1:
        #     print(f"\n=== Step States ===")
        #     for i in range(self.batch_size_int):
        #         print(f"  Env {i}: {self._debug_state_to_str(self.current_queries[i], oneline=True)}")
        #         derived_count = self.derived_states_counts[i].item()
        #         derived_states_list = [self.derived_states_batch[i, j] for j in range(derived_count)]
        #         print(f"        Derived States ({derived_count}): {self._debug_states_to_str(derived_states_list)}")
        #         reward_val = rewards[i].item()
        #         done_val = dones[i].item()
        #         truncated_val = (self.current_depths[i] >= self.max_depth).item()
        #         print(f"        Step Output: Reward={reward_val}, Done={done_val}, Truncated={truncated_val}")
        
        # Create observation
        obs_dict = self._create_observation_dict()
        
        # Create TensorDict with next state
        td = TensorDict(
            {
                **obs_dict,
                "reward": rewards,
                "done": dones,
                "terminated": dones,
            },
            batch_size=self.batch_size,
            device=self._device_internal
        )
        
        return td
    
    def _compute_derived_states_conditional_vec(self, dones: Tensor):
        """Compute derived states only for non-done environments - OPTIMIZED WITH BATCHED UNIFICATION."""
        # For non-done environments, compute next states
        not_done_mask = ~dones.squeeze(-1)  # [B]
        
        # Use UnificationEngine to process all queries at once
        all_derived, updated_var_indices = self.unification_engine.get_derived_states(
            current_states=self.current_queries,
            next_var_indices=self.next_var_indices,
            excluded_queries=self.original_queries,
            labels=self.current_labels,
            verbose=self.prover_verbose,
        )
        
        # Update variable indices only for non-done
        self.next_var_indices = torch.where(
            not_done_mask,
            updated_var_indices,
            self.next_var_indices
        )
        
        # Process derived states for each environment
        for i in range(self.batch_size_int):
            if dones[i, 0].item():
                # Clear derived states for done environments
                all_derived[i] = []
            else:
                # Process derived states for non-done environments
                all_derived[i] = self._process_derived_states_for_env(i, all_derived[i], apply_skip_unary=True)
        
        # Create batched tensor
        batched_derived = torch.full(
            (self.batch_size_int, self.padding_states, self.padding_atoms, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device_internal
        )
        
        counts = torch.zeros(self.batch_size_int, dtype=torch.long, device=self._device_internal)
        
        # Fill in derived states - minimized loop
        for i, derived_states in enumerate(all_derived):
            count = min(len(derived_states), self.padding_states)
            counts[i] = count
            for j, state in enumerate(derived_states[:self.padding_states]):
                batched_derived[i, j] = self._pad_state(state)
        
        self.derived_states_batch = batched_derived
        self.derived_states_counts = counts

    def _get_done_reward_vec(self) -> Tuple[Tensor, Tensor]:
        """Compute rewards and dones for all environments - VECTORIZED."""
        # Initialize outputs
        rewards = torch.zeros(self.batch_size_int, 1, dtype=torch.float32, device=self._device_internal)
        dones = torch.zeros(self.batch_size_int, 1, dtype=torch.bool, device=self._device_internal)
        
        # Get all states [B, padding_atoms, arity+1]
        states = self.current_queries
        
        # Check for empty states (numel == 0 not applicable here, check if all padding)
        # states[:, :, 0] gives predicates [B, padding_atoms]
        non_padding_mask = states[:, :, 0] != self.padding_idx  # [B, padding_atoms]
        has_non_padding = non_padding_mask.any(dim=1)  # [B]
        
        # For states with no non-padding (empty), mark as done with reward 0
        dones[:, 0] = ~has_non_padding
        
        # For states with content, check terminal conditions
        # Get predicates for first atom (ignoring padding)
        # We need to check if all are True or any are False
        
        # For each environment, check terminal conditions
        for i in range(self.batch_size_int):
            if not has_non_padding[i]:
                # Already handled above
                continue
            
            state = states[i]  # [padding_atoms, arity+1]
            non_pad = state[:, 0] != self.padding_idx
            predicates = state[non_pad, 0]
            
            all_true = (predicates == self.true_pred_idx).all().item() if self.true_pred_idx is not None else False
            any_false = (predicates == self.false_pred_idx).any().item() if self.false_pred_idx is not None else False
            
            # Check for End state (if end_proof_action is enabled)
            is_end_terminal = False
            if self.end_proof_action and self.end_pred_idx is not None:
                # End state is terminal if it's the only state
                if len(predicates) == 1 and predicates[0].item() == self.end_pred_idx:
                    is_end_terminal = True
                    all_true = False  # End doesn't count as successful proof
                    any_false = False
            
            depth_exceeded = self.current_depths[i] >= self.max_depth
            if depth_exceeded:
                if self.verbose >= 2:
                    print(f"Env {i} exceeded max depth {self.max_depth}. Marking as done.")
            if all_true:
                if self.verbose >= 2:
                    print(f"Env {i} reached successful proof (all TRUE). Marking as done.")
            if any_false:
                if self.verbose >= 2:
                    print(f"Env {i} reached failed proof (any FALSE). Marking as done.")
            if is_end_terminal:
                if self.verbose >= 2:
                    print(f"Env {i} reached End state. Marking as done.")
                    
            done = all_true | any_false | depth_exceeded | is_end_terminal  # Use | instead of 'or' to keep as tensor
            
            dones[i, 0] = done
            
            # Compute reward based on reward_type
            successful = all_true.item() if isinstance(all_true, torch.Tensor) else all_true
            label = self.current_labels[i].item()
            
            if self.reward_type == 0:
                # Simple: +1 only for successful proof with positive label
                if done and successful and label == 1:
                    rewards[i, 0] = 1.0
            elif self.reward_type == 1:
                # +1 for true positive, -1 for false positive
                if done and successful and label == 1:
                    rewards[i, 0] = 1.0
                elif done and successful and label == 0:
                    rewards[i, 0] = -1.0
            elif self.reward_type == 2:
                # +1 for correct classification (TP or TN)
                if done and successful and label == 1:
                    rewards[i, 0] = 1.0
                elif done and not successful and label == 0:
                    rewards[i, 0] = 1.0
            elif self.reward_type == 3:
                # Asymmetric: penalize false positives more
                if label == 1:
                    if done and successful:
                        rewards[i, 0] = 1.0
                    elif done and not successful:
                        rewards[i, 0] = -0.5
                else:  # label == 0
                    if done and successful:
                        rewards[i, 0] = -1.5  # Heavy penalty for false positive
                    elif done and not successful:
                        rewards[i, 0] = 1.0  # Reward for true negative
            elif self.reward_type == 4:
                # Symmetric ±1 for correct/incorrect classification
                if label == 1:
                    if done and successful:
                        rewards[i, 0] = 1.0
                    elif done and not successful:
                        rewards[i, 0] = -1.0
                else:  # label == 0
                    if done and successful:
                        rewards[i, 0] = -1.0  # False positive
                    elif done and not successful:
                        rewards[i, 0] = 1.0  # True negative
            else:
                raise ValueError(f"Invalid reward_type: {self.reward_type}. Choose from 0-4.")
        
        return rewards, dones
    
    def _create_observation_dict(self) -> Dict:
        """Create batched observation dictionary.
        For given derived states, create a action mask that the 
        index is true if smaller than the count of derived states."""
        # Create action masks - VECTORIZED
        # Valid actions are those with index < derived_states_counts[i]
        action_indices = torch.arange(self.padding_states, device=self._device_internal)  # [padding_states]
        action_indices = action_indices.unsqueeze(0).expand(self.batch_size_int, -1)  # [B, padding_states]
        counts = self.derived_states_counts.unsqueeze(-1)  # [B, 1]
        
        action_mask = action_indices < counts  # [B, padding_states]
        
        if self.verbose >= 3:
            print(f"\n[DEBUG _create_observation_dict]")
            print(f"  derived_states_counts: {self.derived_states_counts.tolist()}")
            print(f"  action_mask:\n{action_mask}")
        
        # Return dict (not TensorDict yet - parent class handles that)
        return {
            'sub_index': self.current_queries.unsqueeze(1),  # [B, 1, padding_atoms, arity+1]
            'derived_sub_indices': self.derived_states_batch,  # [B, padding_states, padding_atoms, arity+1]
            'action_mask': action_mask,  # [B, padding_states]
        }
    
    def _create_observation(self) -> TensorDict:
        """Create full TensorDict observation."""
        obs_dict = self._create_observation_dict()
        
        # Add done and terminated (all False at reset)
        obs_dict['done'] = torch.zeros(self.batch_size_int, 1, dtype=torch.bool, device=self._device_internal)
        obs_dict['terminated'] = torch.zeros(self.batch_size_int, 1, dtype=torch.bool, device=self._device_internal)
        
        return TensorDict(
            obs_dict,
            batch_size=self.batch_size,
            device=self._device_internal
        )
    
    def _pad_state(self, state: Tensor) -> Tensor:
        """Pad a state tensor to padding_atoms."""
        # Ensure state is on the correct device
        if state.device != self._device_internal:
            state = state.to(self._device_internal)
        
        if state.shape[0] >= self.padding_atoms:
            return state[:self.padding_atoms]
        
        padding_needed = self.padding_atoms - state.shape[0]
        padding = torch.full(
            (padding_needed, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device_internal
        )
        return torch.cat([state, padding], dim=0)
    
    def _create_false_state(self) -> Tensor:
        """Create a False state for terminal condition."""
        if self.false_pred_idx is None:
            # If False predicate doesn't exist, return empty tensor
            return torch.empty((0, self.max_arity + 1), dtype=torch.long, device=self._device_internal)
        
        false_state = torch.full(
            (1, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device_internal
        )
        false_state[0, 0] = self.false_pred_idx
        return self._pad_state(false_state)
    
    def _create_end_state(self) -> Tensor:
        """Create an End state for early stopping."""
        if self.end_pred_idx is None or self.end_pred_idx == -1:
            # If End predicate doesn't exist, return empty tensor
            return torch.empty((0, self.max_arity + 1), dtype=torch.long, device=self._device_internal)
        
        end_state = torch.full(
            (1, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device_internal
        )
        end_state[0, 0] = self.end_pred_idx
        return self._pad_state(end_state)
    
    def _process_derived_states_for_env(self, env_idx: int, derived_states: List[Tensor], 
                                        apply_skip_unary: bool = True) -> List[Tensor]:
        """
        Process derived states for a single environment, applying:
        - Skip unary actions (if enabled)
        - Memory pruning (if enabled)
        - Truncate atoms
        - Truncate states
        - Add End action (if enabled)
        
        Args:
            env_idx: Index of the environment
            derived_states: List of derived state tensors
            apply_skip_unary: Whether to apply skip_unary_actions logic
            
        Returns:
            Processed list of derived state tensors
        """
        # Handle None or empty derived states
        if derived_states is None or len(derived_states) == 0:
            return []
        
        # Filter out None values
        derived_states = [ds for ds in derived_states if ds is not None]
        if len(derived_states) == 0:
            return []
        
        # SKIP UNARY ACTIONS MODULE
        if apply_skip_unary and self.skip_unary_actions:
            counter = 0
            # Loop while there's exactly one derived state and it's not terminal
            while (len(derived_states) == 1 and 
                   derived_states[0].numel() > 0 and
                   derived_states[0][0, 0].item() not in {self.true_pred_idx, self.false_pred_idx, self.end_pred_idx}):
                
                if self.verbose >= 1:
                    print(f"Skipping unary action for env {env_idx}. Current: {self._debug_state_to_str(self.current_queries[env_idx], oneline=True)}")
                
                # The single state becomes the new current state (with padding)
                self.current_queries[env_idx] = self._pad_state(derived_states[0])
                
                # Get next derived states
                batch_derived, batch_var_indices = self.unification_engine.get_derived_states(
                    current_states=self.current_queries[env_idx:env_idx+1],
                    next_var_indices=self.next_var_indices[env_idx:env_idx+1],
                    excluded_queries=self.original_queries[env_idx:env_idx+1],
                    labels=self.current_labels[env_idx:env_idx+1],
                    verbose=self.prover_verbose,
                )
                derived_states = batch_derived[0]
                self.next_var_indices[env_idx] = batch_var_indices[0]
                
                # TRUNCATE ATOMS during unary skip
                processed_states = []
                for ds_tensor in derived_states:
                    if ds_tensor.shape[0] <= self.padding_atoms:
                        processed_states.append(ds_tensor)
                    elif self.verbose >= 1:
                        print(f"Truncating state with {ds_tensor.shape[0]} atoms during unary skip.")
                derived_states = processed_states
                
                # MEMORY MODULE during unary skip
                if self.memory_pruning:
                    state_tuple = self._state_to_tuple(self.current_queries[env_idx])
                    self.memories[env_idx].add(state_tuple)
                    
                    filtered_states = []
                    for ds_tensor in derived_states:
                        ds_tuple = self._state_to_tuple(ds_tensor)
                        if ds_tuple not in self.memories[env_idx]:
                            filtered_states.append(ds_tensor)
                        elif self.verbose >= 1:
                            print(f"Memory Pruning during unary skip: State {self._debug_state_to_str(ds_tensor, oneline=True)}")
                    derived_states = filtered_states
                
                if len(derived_states) == 0:
                    if self.verbose >= 1:
                        print(f"No valid next states after processing for env {env_idx}, returning [[FALSE]].")
                    return [self._create_false_state()]
                
                # Safety break
                counter += 1
                if counter > 20:
                    if self.verbose >= 1:
                        print(f'Max iterations in skip_unary_actions reached for env {env_idx}.')
                    return [self._create_false_state()]
        
        # MEMORY MODULE
        if self.memory_pruning:
            state_tuple = self._state_to_tuple(self.current_queries[env_idx])
            self.memories[env_idx].add(state_tuple)
            
            filtered_states = []
            for ds in derived_states:
                ds_tuple = self._state_to_tuple(ds)
                if ds_tuple not in self.memories[env_idx]:
                    filtered_states.append(ds)
                elif self.verbose >= 1:
                    print(f"Memory Pruning in next derivation: State {self._debug_state_to_str(ds, oneline=True)}")
            derived_states = filtered_states if filtered_states else [self._create_false_state()]
        
        # TRUNCATE ATOMS MODULE
        processed_derived_states = []
        for ds_tensor in derived_states:
            if ds_tensor.shape[0] <= self.padding_atoms:
                processed_derived_states.append(ds_tensor)
            elif self.verbose >= 1:
                print(f"Truncating state with {ds_tensor.shape[0]} atoms (max: {self.padding_atoms}): {self._debug_state_to_str(ds_tensor, oneline=True)}")
        derived_states = processed_derived_states if processed_derived_states else [self._create_false_state()]
        
        # TRUNCATE STATES MODULE
        max_num_actions = self.padding_states
        
        # Reserve one slot for End action if enabled and there are non-terminal states
        if self.end_proof_action:
            has_non_terminal = any(
                ds.numel() > 0 and ds[0, 0].item() not in {self.true_pred_idx, self.false_pred_idx}
                for ds in derived_states
            )
            if has_non_terminal:
                max_num_actions -= 1
        
        if len(derived_states) > max_num_actions:
            if self.verbose >= 1:
                print(f"Truncating {len(derived_states)} derived states to {max_num_actions} for env {env_idx}.")
            # Sort by number of atoms (prefer simpler states)
            derived_states.sort(key=lambda t: t.shape[0])
            derived_states = derived_states[:max_num_actions]
        
        # END OF ACTION MODULE
        if self.end_proof_action:
            is_any_non_terminal = any(
                ds.numel() > 0 and ds[0, 0].item() not in {self.true_pred_idx, self.false_pred_idx}
                for ds in derived_states
            )
            if is_any_non_terminal and len(derived_states) < self.padding_states:
                # Add End action
                end_state = self._create_end_state()
                derived_states.append(end_state)
        
        return derived_states
    
    def _state_to_tuple(self, state: Tensor) -> tuple:
        """Convert state tensor to hashable tuple for memory - OPTIMIZED.
        
        Uses a fast hash computation to avoid expensive GPU-CPU transfers.
        Only calls .item() once at the end instead of for every element.
        """
        # Remove padding first
        valid_mask = state[:, 0] != self.padding_idx
        if not valid_mask.any():
            return (0,)  # Empty state
        
        valid_state = state[valid_mask]
        flat = valid_state.flatten()
        n = flat.numel()
        
        if n == 0:
            return (0,)
        
        # Simple weighted sum hash - avoid torch.arange which creates new tensor each time
        # Use cumsum for weights which is faster
        hash_val = (flat.long() * (torch.arange(n, device=flat.device, dtype=torch.long) + 1)).sum()
        hash_val = (hash_val & 0x7FFFFFFF).item()
        
        return (hash_val,)
    
    def _set_seed(self, seed: int):
        """Set random seed."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
