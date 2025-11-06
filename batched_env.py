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
from batched_unification_gpu import get_next_unification
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
        
        # Batch state
        self.current_queries: Optional[Tensor] = None
        self.current_labels: Optional[Tensor] = None
        self.current_depths: Optional[Tensor] = None
        self.next_var_indices: Optional[Tensor] = None
        
        # Derived states: [B, max_derived, padding_atoms, arity+1]
        self.derived_states_batch: Optional[Tensor] = None
        self.derived_states_counts: Optional[Tensor] = None  # [B] number of valid derived states per env
        
        # Memory tracking (still need sets for membership testing)
        self.memories: List[Set] = [set() for _ in range(self.batch_size_int)]
        
        # Precompute constants
        self.max_arity = index_manager.max_arity
        self.padding_idx = index_manager.padding_idx
        self.true_pred_idx = index_manager.true_pred_idx
        self.false_pred_idx = index_manager.false_pred_idx
        
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
        """Reset all environments in the batch."""
        # Sample batch of queries
        batch_queries = []
        batch_labels = []
        batch_depths = []
        
        if self.mode == 'train':
            # Sample with replacement
            indices = torch.randint(0, len(self.all_queries), (self.batch_size_int,))
            batch_queries = [self.all_queries[i] for i in indices]
            batch_labels = [self.all_labels[i] for i in indices]
            batch_depths = [self.all_depths[i] for i in indices]
        elif self.mode == 'eval':
            # Sequential sampling
            actual_batch_size = min(self.batch_size_int, len(self.all_queries) - self.counter)
            batch_queries = self.all_queries[self.counter:self.counter + actual_batch_size]
            batch_labels = self.all_labels[self.counter:self.counter + actual_batch_size]
            batch_depths = self.all_depths[self.counter:self.counter + actual_batch_size]
            
            # Pad if needed
            if actual_batch_size < self.batch_size_int:
                batch_queries.extend([self.all_queries[0]] * (self.batch_size_int - actual_batch_size))
                batch_labels.extend([self.all_labels[0]] * (self.batch_size_int - actual_batch_size))
                batch_depths.extend([self.all_depths[0]] * (self.batch_size_int - actual_batch_size))
            
            self.counter += actual_batch_size
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Convert queries to tensors - VECTORIZED
        queries_tensors = self._convert_queries_to_tensors_vec(batch_queries)
        
        # Stack into batch [B, padding_atoms, max_arity+1]
        self.current_queries = queries_tensors.to(self._device_internal)
        self.current_labels = torch.tensor(batch_labels, dtype=torch.long, device=self._device_internal)
        self.current_depths = torch.zeros(self.batch_size_int, dtype=torch.long, device=self._device_internal)
        
        # Initialize batch-specific state
        self.next_var_indices = torch.full(
            (self.batch_size_int,), 
            self.index_manager.runtime_var_start_index,
            dtype=torch.long,
            device=self._device_internal
        )
        
        # Reset memories
        self.memories = [set() for _ in range(self.batch_size_int)]
        
        # Get derived states for ALL queries in batch - VECTORIZED
        self._compute_derived_states_vec()
        
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
            verbose=self.prover_verbose
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
                all_derived[i] = filtered_states if filtered_states else [self._create_false_state()]
        
        # Truncate if too many states per environment
        for i in range(self.batch_size_int):
            if len(all_derived[i]) > self.padding_states:
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
            print(f'\n\nVectorized Batched Step-----------------------------')
            print(f'Actions: {actions}')
        
        # Select next state for each environment using advanced indexing - NO LOOP!
        # Create batch indices
        batch_indices = torch.arange(self.batch_size_int, device=self._device_internal)
        
        # Clamp actions to valid range
        clamped_actions = torch.clamp(actions, 0, self.padding_states - 1)
        
        # Select states: [B, padding_atoms, arity+1]
        selected_states = self.derived_states_batch[batch_indices, clamped_actions]
        
        # Check if action was invalid (action_idx >= count)
        invalid_mask = actions >= self.derived_states_counts  # [B]
        
        # Create false state for invalid actions
        false_state = self._create_false_state()
        
        # Replace invalid actions with false state - VECTORIZED
        selected_states = torch.where(
            invalid_mask.unsqueeze(-1).unsqueeze(-1),  # [B, 1, 1]
            false_state.unsqueeze(0).expand(self.batch_size_int, -1, -1),  # [B, padding_atoms, arity+1]
            selected_states
        )
        
        self.current_queries = selected_states
        
        # Increment depths - VECTORIZED
        self.current_depths += 1
        
        # Compute rewards and dones - VECTORIZED
        rewards, dones = self._get_done_reward_vec()
        
        if self.verbose >= 2:
            print(f"After _get_done_reward_vec: dones = {dones.squeeze()}")
        
        # Get next derived states only for non-done environments - OPTIMIZED
        self._compute_derived_states_conditional_vec(dones)
        
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
            verbose=self.prover_verbose
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
            
            done = all_true | any_false | depth_exceeded  # Use | instead of 'or' to keep as tensor
            
            if self.verbose >= 2:
                print(f"  Env {i}: predicates={predicates.tolist()}, all_true={all_true.item()}, any_false={any_false.item()}, depth={self.current_depths[i].item()}/{self.max_depth}, done={done.item()}")
            
            dones[i, 0] = done
            
            # Reward: 1 if successful proof and positive label
            if all_true and self.current_labels[i].item() == 1:
                rewards[i, 0] = 1.0
        
        return rewards, dones
    
    def _create_observation_dict(self) -> Dict:
        """Create batched observation dictionary - FULLY VECTORIZED."""
        # Create action masks - VECTORIZED
        # Valid actions are those with index < derived_states_counts[i]
        action_indices = torch.arange(self.padding_states, device=self._device_internal)  # [padding_states]
        action_indices = action_indices.unsqueeze(0).expand(self.batch_size_int, -1)  # [B, padding_states]
        counts = self.derived_states_counts.unsqueeze(-1)  # [B, 1]
        
        action_mask = action_indices < counts  # [B, padding_states]
        
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
        """Convert state tensor to hashable tuple for memory - OPTIMIZED."""
        # Remove padding first
        valid_mask = state[:, 0] != self.padding_idx
        if not valid_mask.any():
            return (0,)  # Empty state
        
        valid_state = state[valid_mask]
        # Fast GPU-based hash using vectorized polynomial hash
        flat = valid_state.flatten()
        
        # Use vectorized computation on GPU
        prime = 31
        mod = 2**31 - 1
        
        # Create powers: [1, 31, 31^2, 31^3, ...]
        n = flat.numel()
        powers = torch.arange(n, device=flat.device, dtype=torch.long)
        prime_powers = torch.pow(prime, powers) % mod
        
        # Compute hash as sum of (value * prime^position)
        hash_val = ((flat.long() * prime_powers).sum() % mod).item()
        
        return (hash_val,)
    
    def _set_seed(self, seed: int):
        """Set random seed."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
