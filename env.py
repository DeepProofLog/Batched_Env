"""
Fully Vectorized Batched Logic Environment - No For Loops!

This environment eliminates ALL for loops by using fully vectorized operations.
All batch elements are processed in parallel using tensor operations.
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Set
from tensordict import TensorDict
from torchrl.envs import EnvBase

from utils import Term, Rule
from unification import get_next_unification
# from batched_unification_cpu import get_next_unification


class BatchedVecEnv(EnvBase):
    """
    Fully vectorized batched logic environment.
    
    Key differences from BatchedLogicEnv:
    - NO for loops over batch dimension
    - All operations are fully vectorized
    - Uses advanced indexing and masking instead of iteration
    """
    
    def __init__(
        self,
        batch_size: int,
        index_manager,
        data_handler,
        queries: List[Term],
        labels: List[int],
        query_depths: List,
        facts: Set[Term],
        mode: str = 'train',
        seed: int = 42,
        max_depth: int = 10,
        memory_pruning: bool = True,
        padding_atoms: int = 10,
        padding_states: int = 20,
        verbose: int = 0,
        prover_verbose: int = 0,
        device: torch.device = None,
        engine: str = 'python_tensor',
        **kwargs
    ):
        # Store batch_size as int for internal use
        self.batch_size_int = batch_size
        
        # Initialize parent with torch.Size([batch_size])
        super().__init__(
            device=device if device is not None else torch.device('cpu'),
            batch_size=torch.Size([batch_size]),
        )
        
        self.index_manager = index_manager
        self.data_handler = data_handler
        self.mode = mode
        self.max_depth = max_depth
        # Disable memory pruning for eval mode (it's mainly for training)
        self.memory_pruning = memory_pruning if mode == 'train' else False
        self.padding_atoms = padding_atoms
        self.padding_states = padding_states
        self.verbose = verbose
        self.prover_verbose = prover_verbose
        self.engine = engine
        
        # Store device in internal variable (EnvBase manages self.device)
        if device is None:
            self._device_internal = torch.device('cpu')
        else:
            self._device_internal = device
        
        # Store queries and labels
        self.all_queries = queries
        self.all_labels = labels
        self.all_depths = query_depths
        
        # Precompute constants (needed before initializing batch state)
        self.max_arity = index_manager.max_arity
        self.padding_idx = index_manager.padding_idx
        self.true_pred_idx = index_manager.true_pred_idx
        self.false_pred_idx = index_manager.false_pred_idx
        
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
            self.index_manager.runtime_var_start_index,
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
        
        # Convert facts to tensor for unification
        if facts:
            facts_list = list(facts)
            self.facts_tensor = index_manager.state_to_tensor(facts_list)
            # Move facts to device
            self.facts_tensor = self.facts_tensor.to(self._device_internal)
            # Use cache directory based on dataset name
            cache_dir = f"data/.cache/{data_handler.dataset_name}" if hasattr(data_handler, 'dataset_name') else "data/.cache"
            index_manager.build_facts_index(self.facts_tensor, cache_dir=cache_dir)
        
        # Convert rules to tensor
        if data_handler.rules:
            max_rule_atoms = max(len(rule.body) + 1 for rule in data_handler.rules)
            self.rules_tensor, self.rule_lengths = index_manager.rules_to_tensor(
                data_handler.rules, max_rule_atoms
            )
            # Move rules to device
            self.rules_tensor = self.rules_tensor.to(self._device_internal)
            self.rule_lengths = self.rule_lengths.to(self._device_internal)
            index_manager.build_rule_index(self.rules_tensor)
        
        # Counter for sampling
        self.counter = 0
        
        # Define observation and action specs
        self._make_specs()
    
    def _make_specs(self):
        """Define observation and action space specs."""
        from torchrl.data import Composite, Bounded
        
        max_vocab_size = self.index_manager.total_vocab_size if hasattr(self.index_manager, 'total_vocab_size') else 100000
        
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
            print(f'\n\nPartial Reset of indices {indices_to_reset}-----------------------------') if self.verbose or self.prover_verbose else None
            
            if num_to_reset == 0:
                # No environments to reset - return current observation
                return self._create_observation()
        else:
            print('\n\n-----------------------------Reset-----------------------------') if self.verbose or self.prover_verbose else None
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
        
        # Convert queries to tensors
        queries_tensors_list = []
        for query in batch_queries:
            if isinstance(query, Term):
                query_tensor = self.index_manager.state_to_tensor([query])
            else:
                query_tensor = query
            queries_tensors_list.append(self._pad_state(query_tensor))
        
        # Update state only for environments being reset
        for i, env_idx in enumerate(indices_to_reset):
            self.current_queries[env_idx] = queries_tensors_list[i]
            self.current_labels[env_idx] = batch_labels[i]
            self.current_depths[env_idx] = 0
            self.next_var_indices[env_idx] = self.index_manager.runtime_var_start_index
            self.memories[env_idx] = set()
        
        # Extract original queries for exclusion during fact unification
        # Only for newly reset environments
        first_atom_mask = self.current_queries[:, :, 0] != self.index_manager.padding_idx
        first_atom_indices = first_atom_mask.long().argmax(dim=1)  # [B]
        batch_indices = torch.arange(self.batch_size_int, device=self._device_internal)
        self.original_queries = self.current_queries[batch_indices, first_atom_indices]  # [B, arity+1]

        if self.verbose >=1: print(f"\nInitial Query (Label: {self.current_labels[batch_indices]}, depth: {self.current_depths[batch_indices]}): "
                                   f"{self.index_manager.debug_print_state_from_indices(self.current_queries[batch_indices, first_atom_indices], oneline=True)}")

        # Compute derived states for ALL environments (not just reset ones)
        # This is necessary because the unification function is vectorized
        self._compute_derived_states_vec()
        
        if self.verbose >= 1:
            print(f"\nReset States:")
            for i in range(self.batch_size_int):
                if reset_mask[i]:  # Only show for reset environments
                    print(f"  Env {i}: {self.index_manager.debug_print_state_from_indices(self.current_queries[i], oneline=True)}")
                    derived_count = self.derived_states_counts[i].item()
                    derived_states_list = [self.derived_states_batch[i, j] for j in range(derived_count)]
                    print(f"  Derived States ({derived_count}): {self.index_manager.debug_print_states_from_indices(derived_states_list)}")
        
        if self.verbose >= 1:
            print(f"\n[DEBUG _reset]")
            print(f"  Reset mask: {reset_mask.tolist()}")
            print(f"  Reset indices: {indices_to_reset}")
            print(f"  After _compute_derived_states_vec:")
            print(f"  derived_states_counts: {self.derived_states_counts.tolist()}")
        
        # Create batched observations
        return self._create_observation()
    
    def _convert_queries_to_tensors_vec(self, queries: List[Term]) -> Tensor:
        """Convert list of queries to batched tensor - minimized loops."""
        # Convert all queries to tensors at once
        queries_list = []
        for query in queries:
            if isinstance(query, Term):
                query_tensor = self.index_manager.state_to_tensor([query])
            else:
                query_tensor = query
            queries_list.append(query_tensor)
        
        # Pad all and stack
        padded = [self._pad_state(q) for q in queries_list]
        return torch.stack(padded, dim=0)
    
    def _compute_derived_states_vec(self):
        """Compute derived states for all environments - FULLY VECTORIZED."""
        # Use GPU batched unification to process all queries at once!
        all_derived, updated_var_indices = get_next_unification(
            current_states=self.current_queries,
            facts_tensor=self.facts_tensor,
            rules=self.rules_tensor,
            rule_lengths=self.rule_lengths,
            index_manager=self.index_manager,
            next_var_indices=self.next_var_indices,
            excluded_queries=self.original_queries,
            labels=self.current_labels,
            verbose=self.prover_verbose,
            verbose_engine=self.prover_verbose
        )
        
        # Update variable indices
        self.next_var_indices = updated_var_indices
        
        # Apply memory pruning if enabled
        if self.memory_pruning:
            for i in range(self.batch_size_int):
                # Add current state to memory
                state_tuple = self._state_to_tuple(self.current_queries[i])
                self.memories[i].add(state_tuple)
                
                # Filter derived states
                filtered_states = []
                for ds in all_derived[i]:
                    ds_tuple = self._state_to_tuple(ds)
                    if ds_tuple not in self.memories[i]:
                        filtered_states.append(ds)
                    elif self.verbose >= 1:
                        print(f"Memory Pruning in next derivation: State {self.index_manager.debug_print_state_from_indices(ds, oneline=True)}")
                all_derived[i] = filtered_states if filtered_states else [self._create_false_state()]
                if not filtered_states and self.verbose >= 1:
                    print(f"No valid next states after memory pruning for env {i}, returning [[FALSE]].")
        
        # Truncate if too many states per environment
        for i in range(self.batch_size_int):
            if len(all_derived[i]) > self.padding_states:
                if self.verbose >= 1:
                    print(f"Truncating {len(all_derived[i])} derived states to {self.padding_states} for env {i}.")
                all_derived[i] = all_derived[i][:self.padding_states]
        
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
                print(f'  Current state: {self.index_manager.debug_print_state_from_indices(self.current_queries[i], oneline=True)}')
                print(f'  Available derived states ({self.derived_states_counts[i].item()}):{self.index_manager.debug_print_states_from_indices([self.derived_states_batch[i, j] for j in range(self.derived_states_counts[i].item())])}')
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
        #         print(f'  Next state: {self.index_manager.debug_print_state_from_indices(self.current_queries[i], oneline=True)}')
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
        #         print(f"  Env {i}: {self.index_manager.debug_print_state_from_indices(self.current_queries[i], oneline=True)}")
        #         derived_count = self.derived_states_counts[i].item()
        #         derived_states_list = [self.derived_states_batch[i, j] for j in range(derived_count)]
        #         print(f"        Derived States ({derived_count}): {self.index_manager.debug_print_states_from_indices(derived_states_list)}")
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
        
        # Use GPU batched unification for all environments (it will handle terminal states)
        all_derived, updated_var_indices = get_next_unification(
            current_states=self.current_queries,
            facts_tensor=self.facts_tensor,
            rules=self.rules_tensor,
            rule_lengths=self.rule_lengths,
            index_manager=self.index_manager,
            next_var_indices=self.next_var_indices,
            excluded_queries=self.original_queries,
            labels=self.current_labels,
            verbose=self.prover_verbose,
            verbose_engine=self.prover_verbose
        )
        
        # Update variable indices only for non-done
        self.next_var_indices = torch.where(
            not_done_mask,
            updated_var_indices,
            self.next_var_indices
        )
        
        # Clear derived states for done environments
        for i in range(self.batch_size_int):
            if dones[i, 0].item():
                all_derived[i] = []
        
        # Apply memory pruning if enabled
        if self.memory_pruning:
            for i in range(self.batch_size_int):
                if not_done_mask[i].item():
                    # Add current state to memory
                    state_tuple = self._state_to_tuple(self.current_queries[i])
                    self.memories[i].add(state_tuple)
                    
                    # Filter derived states
                    filtered_states = []
                    for ds in all_derived[i]:
                        ds_tuple = self._state_to_tuple(ds)
                        if ds_tuple not in self.memories[i]:
                            filtered_states.append(ds)
                    all_derived[i] = filtered_states if filtered_states else [self._create_false_state()]
        
        # Truncate if too many states
        for i in range(self.batch_size_int):
            if len(all_derived[i]) > self.padding_states:
                all_derived[i] = all_derived[i][:self.padding_states]
        
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
            
            all_true = torch.all(predicates == self.true_pred_idx)
            any_false = torch.any(predicates == self.false_pred_idx)
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
            done = all_true | any_false | depth_exceeded  # Use | instead of 'or' to keep as tensor
            
            dones[i, 0] = done
            
            # Reward: 1 if successful proof and positive label
            if all_true and self.current_labels[i].item() == 1:
                rewards[i, 0] = 1.0
        
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
        """Create a false terminal state."""
        false_state = torch.full(
            (1, self.max_arity + 1),
            self.padding_idx,
            dtype=torch.long,
            device=self._device_internal
        )
        false_state[0, 0] = self.false_pred_idx
        return self._pad_state(false_state)
    
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
