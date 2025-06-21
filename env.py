from typing import List, Optional, Tuple, Set, FrozenSet
import random
import torch
import gymnasium as gym
import numpy as np
import janus_swi as janus
from tensordict import TensorDict, NonTensorData

from dataset import DataHandler 
from index_manager import IndexManager
from utils import Rule, Term 
from python_unification import get_next_unification_pt
# from prolog_unification import get_next_unification_prolog


class LogicEnv_gym(gym.Env):
    batch_locked = False

    def __init__(self,
                index_manager: IndexManager, 
                data_handler: DataHandler,   
                queries_term: Optional[List[Term]] = None, 
                rules_term: Optional[List[Rule]] = None, 
                queries: Optional[torch.Tensor] = None,
                labels: Optional[List[int]] = None,
                facts_set: Optional[FrozenSet[Tuple[int, int, int]]] = None,
                rules: Optional[torch.Tensor] = None,
                rule_lengths: Optional[torch.Tensor] = None,
                mode: str = 'train',
                corruption_mode: Optional[str] = None,
                corruption_scheme: Optional[str] = None, # ['tail', 'head']
                train_neg_pos_ratio: int = 1,
                seed: Optional[int] = None,
                max_depth: int = 10,
                memory_pruning: bool = True,
                end_proof_action: bool = False,
                skip_unary_actions: bool = False, 
                padding_atoms: int = 10,
                padding_states: int = 20,
                verbose: int = 0,
                prover_verbose: int = 0,
                device: torch.device = torch.device("cpu"),
                engine: str = 'python_tensor', 
                # New parameter for unification strategy in python_tensor engine
                include_intermediate_rule_states: bool = False 
                ):

        super().__init__()

        self.engine = engine
        self.verbose = verbose
        self.prover_verbose = prover_verbose
        self.device = device 

        self.index_manager = index_manager
        self.index_manager.device = device 

        self.max_arity = self.index_manager.max_arity 
        self.padding_atoms = padding_atoms  
        self.padding_states = padding_states 
        self.max_depth = max_depth 

        self._set_seed(seed)
        self._make_spec() 

        # INIT SPECIAL PREDICATES
        self.true_pred_idx = self.index_manager.true_pred_idx
        self.false_pred_idx = self.index_manager.false_pred_idx
        self.end_pred_idx = self.index_manager.predicate_str2idx.get('End', -1)
        self.padding_idx = self.index_manager.padding_idx

        self.true_tensor_atom = self.index_manager.true_tensor.to(device)
        self.false_tensor_atom = self.index_manager.false_tensor.to(device)
        self.end_tensor_atom = torch.tensor([self.end_pred_idx, self.padding_idx, self.padding_idx] if self.max_arity >=2 else \
                                            ([self.end_pred_idx, self.padding_idx] if self.max_arity ==1 else \
                                             [self.end_pred_idx]), dtype=torch.long, device=self.device) if self.end_pred_idx != -1 \
                                                else torch.empty((0), dtype=torch.long, device=self.device)


        # INIT TENSORS
        self.dataset_name = data_handler.dataset_name
        self.facts_set = facts_set
        self.rules = rules.to(device)
        self.rule_lengths = rule_lengths.to(device)
        
        self.rules_term = rules_term
        self.queries_term = queries_term

        self.include_intermediate_rule_states = include_intermediate_rule_states

        # INIT ENVIRONMENT PROPERTIES
        self.memory: Set[Tuple[Tuple[int,...],...]] = set()
        self.memory_pruning = memory_pruning
        self.end_proof_action = end_proof_action
        self.skip_unary_actions = skip_unary_actions # Not directly used in this snippet, but kept
        self.next_var_index = self.index_manager.runtime_var_start_index

        self.current_query: Optional[torch.Tensor] = None
        self.current_label: Optional[int] = None

        self.mode = mode

        self.queries = queries.to(device)  
        self.labels = labels if labels is not None else []
        self.n_episodes = len(self.queries) 
        self.eval_idx = 0
    

        # CORRUPTION MODE
        self.corruption_mode = corruption_mode
        self.corruption_scheme = corruption_scheme
        self.sampler = data_handler.sampler
        self.counter = 0
        self.step_counter = 0
        self.train_neg_pos_ratio = train_neg_pos_ratio
        

    def _set_seed(self, seed:int):
        self.seed = seed if seed is not None else torch.empty((), dtype=torch.int64).random_().item()
        self.seed_gen = random.Random(self.seed) 
        np.random.seed(self.seed) 


    def _make_spec(self):
        obs_spaces = {
            'state': gym.spaces.Box(
                low=0, 
                high=np.iinfo(np.int64).max, 
                shape=(1, self.padding_atoms, self.max_arity + 1),
                dtype=np.int64,
            ),
            'derived_states': gym.spaces.Box(
                low=0,
                high=np.iinfo(np.int64).max,
                shape=(self.padding_states, self.padding_atoms, self.max_arity + 1),
                dtype=np.int64,
            ),  
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Discrete(self.padding_states)


    def reset(self, seed: Optional[int]= None, options=None):
        print('\n\nReset-----------------------------') if self.verbose or self.prover_verbose else None
        if seed is not None:
            self._set_seed(seed) # Set seed for this episode if provided

        if self.mode == 'eval':
            if self.eval_idx < self.n_episodes:
                state = self.queries[self.eval_idx]
                label = self.labels[self.eval_idx]
            else:
                state = self.false_tensor_atom.to(self.device)
                label = 0
            self.eval_idx += 1

        elif self.mode == 'eval_with_restart':
            if self.eval_idx == self.n_episodes:
                self.eval_idx = 0 

            assert self.eval_idx < self.n_episodes, f"Eval index: {self.eval_idx}, n_episodes: {self.n_episodes}. "\
                    f"Adjust the number of episodes to evaluate in the callback."
            
            state = self.queries[self.eval_idx]
            label = self.labels[self.eval_idx]
            self.eval_idx += 1

        elif self.mode == 'train':
            state, _ = self.get_random_queries(n=1)
            label = 1
            if self.corruption_mode:
                if self.counter % (int(self.train_neg_pos_ratio) + 1) != 0:
                    state_for_negatives = state.unsqueeze(0).unsqueeze(0)  # (1, n_atoms, max_arity + 1)
                    state = self.sampler.get_negatives(state_for_negatives,
                                            padding_atoms=state_for_negatives.size(1),
                                            max_arity=state_for_negatives.size(2)-1,
                                            device=self.device)
                    state = state.squeeze(0).squeeze(0)  # (n_atoms, max_arity + 1)
                    # Alternate between head and tail
                    if not hasattr(self, 'negation_toggle'):
                        self.negation_toggle = 0  # Initialize if it doesn't exist
                    if len(self.corruption_scheme) > 1:
                        state = [state[self.negation_toggle]]  
                        self.negation_toggle = 1 - self.negation_toggle  # Flip for next time get head or tail
                    
                    assert len(state) == 1, f"Length of negatives: {len(state)}"
                    state = state[0].squeeze(0)  # (n_atoms, max_arity + 1)
                    label = 0
                self.counter += 1

        else:
            raise ValueError(f"Invalid mode: {self.mode}.")

        state = state.unsqueeze(0).to(self.device) # (n_atoms, max_arity + 1)
        self.current_label = label
        self.current_query = state
        return self._reset(state, label)
    

    def _reset(self, state: torch.Tensor, label: int):
        # self.step_counter += 1
        # print(f"Counter: {self.step_counter}")
        # if self.step_counter ==50:
        #     print(stoppppp)
        # State (n_atoms, max_arity + 1)
        if self.verbose >=1: print(f"\nInitial Query (Label: {label}): {self.index_manager.debug_print_state_from_indices(state, oneline=True)}")
        self.next_var_index = self.index_manager.runtime_var_start_index
        self.current_depth = torch.tensor(0, device=self.device)

        self.memory.clear()
        state_tuple_for_memory = tuple(tuple(atom.tolist()) for atom in state) 
        self.memory.add(state_tuple_for_memory)

        derived_states, truncated_flag = self.get_next_states(state) # n_states[(n_atoms, max_arity + 1)]
        self.derived_states = derived_states # I need to pass it to step() method later to chose the next state based on the action taken.
        padded_derived_states = self._pad_derived_states(derived_states) # (padding_states, padding_atoms, max_arity + 1)
        
        obs_dict = {
            'state': state.unsqueeze(0).cpu().numpy(), # (1, n_atoms, max_arity + 1), padded to match the shape of derived_states
            'derived_states': padded_derived_states.cpu().numpy(),
        }

        if self.verbose >=1:
            print(f"\nReset State: {self.index_manager.debug_print_state_from_indices(state, oneline=True)}")
            print(f"Derived States ({len(derived_states)}): {self.index_manager.debug_print_states_from_indices(derived_states)}")
        return obs_dict, {}

    def step(self, action: int):
        '''
        Given the current state, possible next states, an action, and return the next state.
        (It should be: given the current state, and an action, return the next state, but we need to modify it for our case)
        '''
        derived_states = self.derived_states

        if action >= self.padding_states or action > len(derived_states):
            raise ValueError(
                f"Invalid action ({action}). Derived states: {derived_states}.")
        
        state_next = derived_states[action] # (padding_atoms, max_arity + 1)

        done_next, reward_next = self.get_done_reward(state_next, self.current_label)
        derived_states_next, truncated_flag = self.get_next_states(state_next)
        
        if truncated_flag:
            derived_states_next = [self.false_tensor_atom.unsqueeze(0)]


        self.current_depth += 1
        exceeded_max_depth = (self.current_depth >= self.max_depth)
        if exceeded_max_depth and self.verbose >=1: print(f'Max depth {self.max_depth} reached.')

        done_next = done_next or exceeded_max_depth or truncated_flag
        
        info = {}
        if done_next:
            self.current_query, self.current_label = None, None

        self.derived_states = derived_states_next
        padded_derived_states = self._pad_derived_states(derived_states_next) # (padding_states, padding_atoms, max_arity + 1)

        obs_dict = {
                    'state': state_next.unsqueeze(0).cpu().numpy(), # (1, n_atoms, max_arity + 1), padded to match the shape of derived_states
                    'derived_states':padded_derived_states.cpu().numpy(),
                    }

        reward_val = reward_next.cpu().item()
        done_val = bool(done_next)
        truncated_val = bool(exceeded_max_depth or truncated_flag) 

        if self.verbose >=1:
            print(f"\nStep {self.current_depth + 1}.\nState: {self.index_manager.debug_print_state_from_indices(state_next, oneline=True)}")
            print(f"Derived States ({len(derived_states_next)}): {self.index_manager.debug_print_states_from_indices(derived_states_next)}")
            print(f"Step Output: Reward={reward_val}, Done={done_val}, Truncated={truncated_val}")
        return obs_dict, reward_val, done_val, truncated_val, info


    def _pad_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Pads a state tensor to the fixed size of padding_atoms and max_arity + 1.
        Args:
            state_tensor (n_atoms, max_arity + 1): The state tensor to pad.
        Returns:
            torch.Tensor: (padding_atoms, max_arity + 1): Padded state tensor of shape."""
        
        assert state_tensor.dim() == 2, f"State tensor must be 2D, got {state_tensor.dim()}D."

        n_atoms = state_tensor.shape[0]
        if n_atoms == 0 : 
            return torch.full((self.padding_atoms, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self.device)
        
        assert n_atoms <= self.padding_atoms, f"State tensor has {n_atoms} atoms, but padding_atoms is set to {self.padding_atoms}."
        
        if n_atoms < self.padding_atoms:
            padding_needed = self.padding_atoms - n_atoms
            padding = torch.full((padding_needed, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self.device)
            return torch.cat([state_tensor, padding], dim=0)
        
        return state_tensor
    
    def _unpad_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Removes padding atoms from a state tensor.
        Args:
            state_tensor (pad_atoms, max_arity + 1): The state tensor to unpad.
        Returns:
            torch.Tensor: (n_atoms, max_arity + 1): Unpadded state tensor.
        """
        assert state_tensor.dim() == 2, f"State tensor must be 2D, got {state_tensor.dim()}D."
        
        non_padding_mask = state_tensor[:, 0] != self.padding_idx
        return state_tensor[non_padding_mask]

    def _pad_derived_states(self, derived_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Pads a tensor of derived states to the fixed size of padding_states.
        Args:
            derived_states (num_derived, padding_atoms, max_arity + 1): Tensor of derived states.
        Returns:
            torch.Tensor (padding_states, padding_atoms, max_arity + 1): Padded derived state tensor.
        """
        assert isinstance(derived_states, List), f"Expected a list, got {type(derived_states)}."
        assert all(isinstance(ds, torch.Tensor) for ds in derived_states), \
            f"All derived states must be torch tensors, got {[type(ds) for ds in derived_states]}."
        assert all(ds.dim() == 2 for ds in derived_states), \
            f"All derived states must be 2D tensors, got {[ds.dim() for ds in derived_states]}."
        assert all(ds.shape[0] <= self.padding_atoms for ds in derived_states), \
            f"All derived states must have atoms <= {self.padding_atoms}, got {[ds.shape[0] for ds in derived_states]}."

        if len(derived_states) == 0: 
            return torch.full((self.padding_states, self.padding_atoms, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self.device)

        if any(ds.shape[0] != self.padding_atoms for ds in derived_states):
            # If any derived state has fewer atoms than padding_atoms, pad it
            derived_states = [self._pad_state(ds) for ds in derived_states]
        
        # Stack the padded states into a single tensor
        derived_states = torch.stack(derived_states, dim=0)  # (num_derived, padding_atoms, max_arity + 1)

        num_derived = derived_states.shape[0]
        if num_derived < self.padding_states:
            padding_needed = self.padding_states - num_derived
            padding = torch.full((padding_needed, self.padding_atoms, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self.device)
            return torch.cat([derived_states, padding], dim=0)

        # If more states than padding allows, it should have been truncated in get_next_states
        return derived_states

    def _unpad_derived_states(self, derived_states_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Removes padding atoms and states from a tensor of derived states.
        Args:
            derived_states_tensor (padding_states, padding_atoms, max_arity + 1): The derived states tensor to unpad.
        Returns:
            List[torch.Tensor]: A list of unpadded derived state tensors (n_atoms, max_arity + 1).
        """
        assert derived_states_tensor.dim() == 3, f"Derived states tensor must be 3D, got {derived_states_tensor.dim()}D."
        
        unpadded_states = []
        for state in derived_states_tensor:
            unpadded_state = self._unpad_state(state)
            if unpadded_state.numel() > 0:
                unpadded_states.append(unpadded_state) 
        if not unpadded_states:
            unpadded_states.append(self.false_tensor_atom.unsqueeze(0))
        return unpadded_states

    def get_next_states(self, current_state: torch.Tensor) -> Tuple[List[torch.Tensor], bool]:
        """
        Computes the next states based on the current state and available rules.
        Args:
            current_state (n_atoms, max_arity + 1): The current state tensor.
        Returns:
            Tuple[List[torch.Tensor], bool]: A tuple containing:
                - List of derived state tensors (n_atoms, max_arity + 1).
                - A boolean indicating if the state was truncated.
        """
        truncated_flag = False

        if self.end_proof_action:
            if current_state.shape[0] == 1 and current_state[0,0].item() == self.end_pred_idx:
                if self.verbose >= 1: print("Current state is end predicate, returning end tensor atom.")
                return [self.end_tensor_atom.unsqueeze(0)], truncated_flag
            else:
                mask_not_end = current_state[:, 0] != self.end_pred_idx
                current_state = current_state[mask_not_end]
                if current_state.shape[0] == 0:
                    if self.verbose >= 1: print("Current state is empty after removing end predicate, returning [[FALSE]].")
                    return [self.false_tensor_atom.unsqueeze(0)], truncated_flag

        if self.engine == 'python_tensor':
            derived_states, self.next_var_index = get_next_unification_pt(
                current_state=current_state,
                fact_indexed=self.index_manager.fact_index,
                facts_set=self.facts_set,
                rules=self.rules,
                rule_lengths=self.rule_lengths,
                index_manager=self.index_manager,
                rules_term=self.rules_term, 
                excluded_fact=self.current_query[0] if self.current_label == 1 else None,
                verbose=self.prover_verbose,
                next_var_index=self.next_var_index
            ) # List[torch.Tensor]
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

        # --- SKIP UNARY ACTIONS MODULE ---
        if self.skip_unary_actions:
            counter = 0
            # Loop while there's exactly one derived state and it's not a terminal one (True, False, End).
            while (len(derived_states) == 1 and
                   derived_states[0].numel() > 0 and
                   derived_states[0][0, 0].item() not in {self.true_pred_idx, self.false_pred_idx, self.end_pred_idx}):

                if self.verbose >= 1: print(f"Skipping unary action. Current: {self.index_manager.debug_print_state_from_indices(current_state, True)}")
                current_state = derived_states[0] # The single state becomes the new current state for the next iteration.

                derived_states, self.next_var_index = get_next_unification_pt(
                    current_state=current_state,
                    fact_indexed=self.index_manager.fact_index,
                    facts_set=self.facts_set,
                    rules=self.rules,
                    rule_lengths=self.rule_lengths,
                    index_manager=self.index_manager,
                    rules_term=self.rules_term, 
                    excluded_fact=self.current_query[0] if self.current_label == 1 else None,
                    verbose=self.prover_verbose,
                    next_var_index=self.next_var_index
                )

                # TRUNCATE ATOMS MODULE
                processed_states = []
                for ds_tensor in derived_states:
                    if ds_tensor.shape[0] <= self.padding_atoms:
                        processed_states.append(ds_tensor)
                    elif self.verbose >= 1:
                        print(f"Truncating state with {ds_tensor.shape[0]} atoms during unary skip.")
                derived_states = processed_states

                # MEMORY MODULE
                if self.memory_pruning:
                    self.memory.add(tuple(tuple(atom.tolist()) for atom in current_state if 
                                          atom[0].item() not in {self.end_pred_idx, self.false_pred_idx, self.true_pred_idx})) 
                    filtered_derived_states = []
                    for ds_tensor in derived_states:
                        ds_tuple = tuple(tuple(atom.tolist()) for atom in ds_tensor)
                        if ds_tuple not in self.memory:
                            filtered_derived_states.append(ds_tensor)
                        elif self.verbose >= 1:
                            print(f"Memory Pruning in next derivation: State {self.index_manager.debug_print_state_from_indices(ds_tensor, True)}") #\
                                #   {debug_print_state_from_indices(current_state, self.index_manager, oneline=True)} --> {debug_print_states_from_indices(derived_states, self.index_manager)}")
                    derived_states = filtered_derived_states

                if len(derived_states) == 0:
                    if self.verbose >= 1: print("No valid next states after processing, returning [[FALSE]].")
                    return [self.false_tensor_atom.unsqueeze(0)], True
                # Safety break
                counter += 1
                if counter > 20:
                    if self.verbose >= 1: print('Max iterations in skip_unary_actions reached.')
                    derived_states = [self.false_tensor_atom.unsqueeze(0)]
                    truncated_flag = True
                    break

        # MEMORY MODULE
        if self.memory_pruning:
            self.memory.add(tuple(tuple(atom.tolist()) for atom in current_state if 
                                    atom[0].item() not in {self.end_pred_idx, self.false_pred_idx, self.true_pred_idx})) 
            filtered_derived_states = []
            for ds_tensor in derived_states:
                ds_tuple = tuple(tuple(atom.tolist()) for atom in ds_tensor)
                if ds_tuple not in self.memory:
                    filtered_derived_states.append(ds_tensor)
                elif self.verbose >=1:
                    print(f"Memory Pruning in next derivation: State {self.index_manager.debug_print_state_from_indices(ds_tensor, True)}") #\
                        #   {debug_print_state_from_indices(current_state, self.index_manager, oneline=True)} --> {debug_print_states_from_indices(derived_states, self.index_manager)}")
            derived_states = filtered_derived_states

        # TRUNCATE ATOMS MODULE
        # Filter out states that exceed the maximum number of atoms allowed.
        processed_derived_states = []
        for ds_tensor in derived_states:
            if ds_tensor.shape[0] <= self.padding_atoms: 
                processed_derived_states.append(ds_tensor)
            elif self.verbose >=1:
                print(f"Truncating state with {ds_tensor.shape[0]} atoms (max: {self.padding_atoms}): {self.index_manager.debug_print_state_from_indices(ds_tensor, True)}")
        derived_states = processed_derived_states

        # TRUNCATE STATES MODULE
        max_num_actions = self.padding_states
        if self.end_proof_action and any(ds.numel() > 0 and ds[0,0].item() != self.true_pred_idx and ds[0,0].item() != self.false_pred_idx for ds in derived_states):
            max_num_actions -=1 

        if len(derived_states) > max_num_actions:
            if self.verbose >=1: print(f"Truncating {len(derived_states)} derived states to {max_num_actions}.")
            derived_states.sort(key=lambda t: t.shape[0])
            derived_states = derived_states[:max_num_actions]
        
        # END OF ACTION MODULE
        if self.end_proof_action:
            is_any_non_terminal = any(
                ds.numel() > 0 and ds[0, 0].item() not in {self.true_pred_idx, self.false_pred_idx}
                for ds in derived_states
            )
            if is_any_non_terminal and len(derived_states) < self.padding_states:
                    derived_states.append(self.end_tensor_atom.unsqueeze(0))

        if not derived_states:
            if self.verbose >=1: print("No valid next states after processing, returning [[FALSE]].")
            derived_states = [self.false_tensor_atom.unsqueeze(0)]

        return derived_states, truncated_flag
    
    def get_done_reward(self, state_tensor: torch.Tensor, label: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determines if the current state is terminal and computes the reward.
        Args:            
            state_tensor (n_atoms, max_arity + 1): The current state tensor.
            label (int): The label for the current query (1 for positive, 0 for negative).  
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing: 
                - done (torch.Tensor): A boolean tensor indicating if the state is terminal.    
                - reward (torch.Tensor): A float tensor representing the reward value.
        """
        if state_tensor.numel() == 0: 
            all_atoms_true = True
            any_atom_false = False
        else:
            # --- FIX STARTS HERE ---
            # Filter out padding atoms before performing checks
            non_padding_mask = state_tensor[:, 0] != self.padding_idx

            # If the state only contains padding, it's not a successful proof.
            if not torch.any(non_padding_mask):
                return torch.tensor(True, dtype=torch.bool, device=self.device), \
                       torch.tensor(0.0, dtype=torch.float32, device=self.device)

            # Get predicates from non-padded atoms only
            predicates = state_tensor[non_padding_mask, 0]
            # --- FIX ENDS HERE ---
            
            all_atoms_true = torch.all(predicates == self.true_pred_idx).item()
            any_atom_false = torch.any(predicates == self.false_pred_idx).item()
        
        # Check for end proof action
        is_end_terminal = False
        if self.end_proof_action and state_tensor.shape[0] == 1 and state_tensor[0,0].item() == self.end_pred_idx:
            is_end_terminal = True
            all_atoms_true = False 
            any_atom_false = False 

        done = all_atoms_true or any_atom_false or is_end_terminal
        successful_proof = all_atoms_true 
        reward_val = 1 if done and successful_proof and label == 1 else 0

        return torch.tensor(done, dtype=torch.bool, device=self.device), \
               torch.tensor(reward_val, dtype=torch.float32, device=self.device)

    def get_random_queries(self, n: int = 1) -> Tuple[List[torch.Tensor], List[int]]:
        num_available_queries = len(self.queries) 

        if n > num_available_queries:
            if self.verbose >=1: print(f"Warning: Requested {n} query tensors, but only {num_available_queries} available. Sampling with replacement.")
            sampled_indices = [random.randint(0, num_available_queries - 1) for _ in range(n)]
        else:
            sampled_indices = random.sample(range(num_available_queries), n)

        sampled_queries = [self.queries[i].to(self.device) for i in sampled_indices]
        if self.labels: # Check if labels list is populated
            assert len(self.labels) == self.n_episodes, "Labels list size must match number of episodes."
            sampled_labels = [self.labels[i] for i in sampled_indices]
        else: # No labels provided initially
            sampled_labels = None
        
        if len(sampled_queries) == 1:
            return sampled_queries[0], sampled_labels[0] if sampled_labels else None

        return sampled_queries, sampled_labels


    def get_random_queries_terms(self,
                                queries_terms_list_of_list: List[List[Term]], 
                                n: int = 1, 
                                ) -> Tuple[List[List[Term]], Optional[List[int]]]:
        if not queries_terms_list_of_list:
            raise ValueError("Query list is empty, cannot sample.")
        num_available_queries = len(queries_terms_list_of_list)
        if n > num_available_queries:
            print(f"Warning: Requested {n} queries, but only {num_available_queries} available. Sampling with replacement or returning all.")
            sampled_indices = [self.seed_gen.randint(0, num_available_queries - 1) for _ in range(n)]
        else:
            sampled_indices = self.seed_gen.sample(range(num_available_queries), n)
        sampled_queries_terms = [queries_terms_list_of_list[i] for i in sampled_indices]
        sampled_labels = None
        if self.labels and len(self.labels) == len(self.initial_queries_terms):
            sampled_labels = [self.labels[i] for i in sampled_indices]
        elif n == 1 : 
            sampled_labels = [1]
        elif self.labels : 
            print("Warning: Labels list size mismatch, cannot reliably provide labels for multiple random queries.")
        return sampled_queries_terms, sampled_labels

    def close(self):
        if self.verbose: print("LogicEnv_gym closed.")

if __name__ == '__main__':

    print("--- Starting LogicEnv_gym Tensor Test ---")
    
    DEVICE = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    padding_atoms_env = 3
    padding_states_env = 5

    # --- 1. Define Raw Data ---
    constants_test = {"a", "b", "c", "john", "mary", "peter"}
    predicates_test = {"p", "q", "parent", "grandparent"}
    rules_terms_test = [
        Rule(Term("grandparent", ("X", "Z")), [Term("parent", ("X", "Y")), Term("parent", ("Y", "Z"))]),
        Rule(Term("q", ("A", "B")), [Term("p", ("A", "C"))])
    ]
    facts_terms_test = [
        Term("parent", ("john", "mary")),
        Term("parent", ("mary", "peter")),
        Term("p", ("a", "b"))
    ]
    queries_terms_test = [
        [Term("grandparent", ("john", "peter"))],
        [Term("q", ("a", "b"))],
        [Term("parent", ("a", "b"))]
    ]
    labels_for_env_test = [1, 0, 0]

    # --- 2. DataHandler holds the raw data structures. ---
    class DummyDataHandler:
        def __init__(self, rules_terms, facts_terms, queries_terms, labels):
            self.dataset_name = "test_dataset"
            self.rules_terms = rules_terms
            self.facts_terms = facts_terms
            self.test_queries_terms = queries_terms # Keep original queries for reference
            self.test_labels = labels

    data_handler_test = DummyDataHandler(rules_terms_test, facts_terms_test, queries_terms_test, labels_for_env_test)
    print("DataHandler (Mock) Initialized: Holds all raw data.")

    # --- 3. IndexManager ingests all raw data and creates all tensors. ---
    index_manager_test = IndexManager(
        constants=constants_test,
        predicates=predicates_test,
        max_total_vars=1000,
        padding_atoms=5,
        max_arity=2,
        device=DEVICE
    )
    print("IndexManager Initialized: All data has been tensorized and stored as attributes.")

    max_rule_atoms_test = max(1 + len(r.body) for r in rules_terms_test) if rules_terms_test else 1
    index_manager_test.rules, index_manager_test.rules_lengths = index_manager_test.rules_to_tensor(data_handler_test.rules_terms, max_rule_atoms=max_rule_atoms_test)
    index_manager_test.facts = index_manager_test.state_to_tensor(data_handler_test.facts_terms)
    index_manager_test.facts_set = frozenset(tuple(f.tolist()) for f in index_manager_test.facts)
    test_queries = index_manager_test.state_to_tensor(data_handler_test.test_queries_terms)

    # Build the facts index for fast lookup
    index_manager_test.build_facts_index(index_manager_test.facts)
    print("IndexManager: Facts and Rules tensors created, facts index built.")

    # --- 4. Initialize the Environment with the comprehensive IndexManager. ---
    env_test = LogicEnv_gym(
        index_manager=index_manager_test,
        data_handler=data_handler_test,
        queries_term=data_handler_test.test_queries_terms,
        rules_term=data_handler_test.rules_terms,
        queries=test_queries,
        labels=data_handler_test.test_labels,
        facts_set=index_manager_test.facts_set,
        rules=index_manager_test.rules,
        rule_lengths=index_manager_test.rules_lengths,
        mode='eval_with_restart', 
        padding_atoms=padding_atoms_env,
        padding_states=padding_states_env,
        seed=42, # Consistent seed for reproducibility
        max_depth=5,
        device=DEVICE,
        engine='python_tensor',
        include_intermediate_rule_states=False # Using new parameter
    )
    print("LogicEnv_gym Initialized for Test.")

    num_test_episodes = len(queries_terms_test)
    for episode_num in range(num_test_episodes):
        print(f"\n--- TEST EPISODE {episode_num + 1} ---")
        obs, info = env_test.reset()
        print('num derived states:', len(env_test.tensordict['derived_states']))
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        while not (terminated or truncated):
            action_to_take = 0 # Default to taking the first derived state for deterministic test
            num_derived_states = len(env_test._unpad_derived_states(env_test.tensordict['derived_states']))

            if num_derived_states == 0:
                print("  No derived states, but not done. This is unexpected. Breaking.")
                break
            
            # For the grandparent test, we want to deterministically follow the correct path
            # This is a bit of a hack for testing the logic, not how an agent would work.
            if episode_num == 0: # grandparent(john, Who)
                if step_count == 0: # After grandparent -> parent, parent
                    # We expect one state: [parent(john, DynVar_X), parent(DynVar_X, Who')]
                    # There should only be one derived state from the rule.
                    action_to_take = 0 
                elif step_count == 1: # After parent(john,Y) -> parent(mary,Y)
                    # Current state should be [parent(mary, Who')]
                    # Derived state should be [TRUE]
                    action_to_take = 0
            
            # For other episodes, or if specific path logic isn't defined, take action 0
            # or a random valid action if you prefer.
            if action_to_take >= num_derived_states:
                print(f"  Test logic error: action_to_take {action_to_take} is out of bounds for {num_derived_states} derived states. Taking 0.")
                action_to_take = 0

            print(f"  Taking Action: {action_to_take} (out of {num_derived_states} choices): "\
                  f"{env_test.index_manager.debug_print_state_from_indices(env_test.tensordict['derived_states'][action_to_take], oneline=True)}")
                    # f"tensor: {env_test.tensordict['derived_states'][action_to_take]}")
            # print(f"  Taking Action: {action_to_take} (out of {num_derived_states} choices):{env_test.tensordict['derived_states']}")
            obs, reward, terminated, truncated, info = env_test.step(action_to_take)
            total_reward += reward
            step_count += 1
            if step_count >= env_test.max_depth + 2 : 
                print("  Test safety break due to excessive steps.")
                break
        
        print(f"Episode {episode_num + 1} Finished. Total Reward: {total_reward}, Steps: {step_count}")
        expected_reward = 1 if labels_for_env_test[episode_num] == 1 else 0
        if terminated and total_reward == expected_reward:
            print(f"  SUCCESSFUL PROOF as expected (Reward: {total_reward})!")
        elif terminated and total_reward != expected_reward:
            print(f"  UNEXPECTED PROOF OUTCOME (Reward: {total_reward}, Expected: {expected_reward}).")
        elif truncated:
            print(f"  TRUNCATED (e.g. max depth). Expected reward: {expected_reward}")
    print("\n--- LogicEnv_gym Tensor Test Finished ---")