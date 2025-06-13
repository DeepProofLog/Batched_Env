from typing import List, Optional, Tuple, Set, FrozenSet
import random
import torch
import gymnasium as gym
import numpy as np
import janus_swi as janus

from dataset_idx import DataHandler 
from index_manager_idx import IndexManager, state_to_tensor_im, facts_to_tensor_im, rules_to_tensor_im, \
    debug_print_state_from_indices, debug_print_states_from_indices, queries_to_tensor_im
from utils import Rule, Term 
from python_unification_idx import get_next_unification_pt
from prolog_unification import get_next_unification_prolog


def tensor_atom_to_term_im(atom_tensor: torch.Tensor, index_manager: IndexManager) -> Term:
    """Converts a single atom tensor to a Term object."""
    if atom_tensor.shape[0] != (index_manager.max_arity + 1):
        raise ValueError(f"Atom tensor shape {atom_tensor.shape} incompatible with max_arity {index_manager.max_arity}")
    
    pred_idx = atom_tensor[0].item()
    if pred_idx == index_manager.padding_idx: 
        return Term("PAD_PRED", tuple(["PAD_ARG"] * index_manager.max_arity))

    pred_str = index_manager.get_str_for_pred_idx(pred_idx)
    
    arg_strs = []
    for i in range(1, index_manager.max_arity + 1):
        arg_idx = atom_tensor[i].item()
        if arg_idx == index_manager.padding_idx and i > 1 and atom_tensor[i-1].item() == index_manager.padding_idx:
            break 
        arg_strs.append(index_manager.get_str_for_term_idx(arg_idx))
        
    return Term(pred_str, tuple(arg_strs))

def tensor_state_to_terms_list_im(state_tensor: torch.Tensor, index_manager: IndexManager) -> List[Term]:
    """Converts a state tensor to a List of Term objects."""
    if state_tensor.numel() == 0:
        return []
    if state_tensor.dim() == 1: 
        if state_tensor.shape[0] == (index_manager.max_arity + 1):
            return [tensor_atom_to_term_im(state_tensor, index_manager)]
        else:
            raise ValueError(f"Single atom tensor has incorrect dimension {state_tensor.shape[0]}")

    terms_list = []
    for i in range(state_tensor.shape[0]):
        atom_tensor = state_tensor[i]
        if torch.all(atom_tensor == index_manager.padding_idx):
            continue
        if atom_tensor[0].item() == index_manager.padding_idx: 
            continue
        terms_list.append(tensor_atom_to_term_im(atom_tensor, index_manager))
    return terms_list


class LogicEnv_gym(gym.Env):
    batch_locked = False

    def __init__(self,
                index_manager: IndexManager, 
                data_handler: DataHandler,   
                queries_term: Optional[List[Term]] = None, 
                rules_term: Optional[List[Rule]] = None, 
                queries: Optional[torch.Tensor] = None,
                labels: Optional[List[int]] = None,
                facts: Optional[torch.Tensor] = None,
                facts_set: Optional[FrozenSet[Tuple[int, int, int]]] = None,
                rules: Optional[torch.Tensor] = None,
                rule_lengths: Optional[torch.Tensor] = None,
                mode: str = 'train',
                corruption_mode: Optional[str] = None,
                train_neg_pos_ratio: int = 1,
                seed: Optional[int] = None,
                max_depth: int = 10,
                memory_pruning: bool = True,
                end_proof_action: bool = False,
                skip_unary_actions: bool = False, 
                padding_atoms: int = 10,
                padding_states: int = 20,
                verbose: int = 1,
                prover_verbose: int = 1,
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
        self.facts = facts.to(device)
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

        self.current_query: Optional[torch.Tensor] = None
        self.current_label: Optional[int] = None

        self.mode = mode

        self.queries = queries.to(device)  
        self.labels = labels if labels is not None else []
        self.n_episodes = len(self.queries) 
        self.eval_idx = 0
    

        # CORRUPTION MODE
        self.corruption_mode = corruption_mode
        self.sampler = None
        self.corruption_counter = 0
        self.train_neg_pos_ratio = 0
        
        # JANUS INTERACTION
        self.janus_file = data_handler.janus_path if hasattr(data_handler, 'janus_path') else None
        self.janus_facts_str = getattr(data_handler, 'janus_facts_str', getattr(data_handler, 'jan_facts_str', None))

    def _set_seed(self, seed:int):
        self.seed = seed if seed is not None else torch.empty((), dtype=torch.int64).random_().item()
        self.seed_gen = random.Random(self.seed) 
        np.random.seed(self.seed) 


    def _make_spec(self):
        obs_spaces = {
            'sub_index': gym.spaces.Box(
                low=0, 
                high=np.iinfo(np.int64).max, 
                shape=(1, self.padding_atoms, self.max_arity + 1),
                dtype=np.int64,
            ),
            'derived_sub_indices': gym.spaces.Box(
                low=0,
                high=np.iinfo(np.int64).max,
                shape=(self.padding_states, self.padding_atoms, self.max_arity + 1),
                dtype=np.int64,
            ),  
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Discrete(self.padding_states)


    def reset(self, seed: Optional[int]= None, options=None):
        if self.verbose >=1: print(f'\n\nResetting environment (mode: {self.mode}, eval_idx: {self.eval_idx}/{self.n_episodes}) ---')
        if seed is not None:
            self._set_seed(seed) # Set seed for this episode if provided

        initial_query_tensor: torch.Tensor
        label: int

        if self.mode == 'eval':
            if self.n_episodes == 0:
                if self.verbose >=1: print("Eval mode: No queries available.")
                initial_query_tensor = self.false_tensor_atom.unsqueeze(0).to(self.device)
                label = 0
            elif self.eval_idx < self.n_episodes:
                initial_query_tensor = self.queries[self.eval_idx]
                label = self.labels[self.eval_idx] if self.labels and self.eval_idx < len(self.labels) else 1
            else:
                if self.verbose >=1: print("Eval mode: All queries evaluated.")
                initial_query_tensor = self.false_tensor_atom.unsqueeze(0).to(self.device)
                label = 0
            self.eval_idx += 1
        elif self.mode == 'eval_with_restart':
            if self.n_episodes == 0:
                if self.verbose >=1: print("Eval_with_restart mode: No queries available.")
                initial_query_tensor = self.false_tensor_atom.unsqueeze(0).to(self.device)
                label = 0
            else:
                if self.eval_idx >= self.n_episodes: self.eval_idx = 0
                initial_query_tensor = self.queries[self.eval_idx]
                label = self.labels[self.eval_idx] if self.labels and self.eval_idx < len(self.labels) else 1
                self.eval_idx += 1
        elif self.mode == 'train':
            if not self.queries:
                 raise ValueError("Train mode: No query tensors available for sampling.")

            # Sample a positive query tensor first
            sampled_positive_tensors, sampled_positive_labels = self.get_random_queries_tensors(n=1)
            positive_query_tensor = sampled_positive_tensors[0]
            # Use its original label if available, otherwise assume 1 for a positive sample
            # This label is for the *original* positive query. If corrupted, label becomes 0.
            original_positive_label = sampled_positive_labels[0] if sampled_positive_labels else 1


            if self.corruption_mode == "sampler" and self.sampler and \
               self.corruption_counter % (self.train_neg_pos_ratio + 1) != 0:
                # Attempt tensor-based corruption
                # Ensure positive_query_tensor is 2D: (1, num_atom_features) for a single query state
                # Our states are (num_atoms, num_atom_features)
                # get_negatives might expect a batch or a single state.
                # Let's assume sampler.get_negatives takes a single state tensor.
                try:
                    # The sampler might return a list of negative tensors, or one, or None
                    # For example: negative_samples_list: List[torch.Tensor] = self.sampler.get_negatives(positive_query_tensor, device=self.device)
                    # This is a placeholder for the actual sampler API call.
                    # Let's assume it returns a list of candidate negative tensors.
                    negative_samples_list: List[torch.Tensor] = self.sampler.get_negatives(positive_query_tensor)

                    if negative_samples_list:
                        # If multiple negatives, pick one (e.g., randomly or first)
                        # This logic might need to align with how corruption_scheme was used previously
                        # For now, just pick the first if available
                        initial_query_tensor = negative_samples_list[0].to(self.device)
                        label = 0 # This is now a negative sample
                        if self.verbose >=2: print(f"Corrupted query. Original positive label was: {original_positive_label}")
                    else: # Corruption failed to produce a negative
                        if self.verbose >=2: print("Sampler returned no negatives, using original positive query.")
                        initial_query_tensor = positive_query_tensor
                        label = original_positive_label # Use the label of the positive sample
                except Exception as e:
                    print(f"Error during sampler.get_negatives: {e}. Using original positive query.")
                    initial_query_tensor = positive_query_tensor
                    label = original_positive_label
            else: # Use the positive sample
                initial_query_tensor = positive_query_tensor
                label = original_positive_label
            self.corruption_counter = (self.corruption_counter + 1) % (self.train_neg_pos_ratio + 1) if (self.corruption_mode and self.sampler) else 0
        else:
            raise ValueError(f"Invalid mode: {self.mode}.")

        # --- Janus Interaction (remains the same, uses tensor_state_to_terms_list_im) ---
        if self.engine == 'prolog' and (self.mode == 'train' or self.consult_janus_eval) and label == 1:
            terms_for_prolog = tensor_state_to_terms_list_im(initial_query_tensor, self.index_manager)
            if terms_for_prolog:
                first_atom_term = terms_for_prolog[0]
                # Avoid retracting special predicates like ##TRUE##, ##FALSE##
                # Check against actual string values from special_preds_map_names for safety
                special_pred_strings = [self.index_manager.special_preds_map_names.get(sp, "") for sp in self.index_manager.special_preds]
                if first_atom_term.predicate not in special_pred_strings:
                    try:
                        # Optional: Check if fact exists: list(janus.query(f"{first_atom_term.prolog_str()}."))
                        janus.query_once(f"retract({first_atom_term.prolog_str()}).")
                        if self.verbose >= 2: print(f"Prolog: Retracted {first_atom_term.prolog_str()}")
                    except Exception as e:
                        if self.verbose >=1: print(f"Prolog: Failed to retract {first_atom_term.prolog_str()} - {e}")

        self.current_label = label
        self.current_query = initial_query_tensor.to(self.device) # Ensure on device

        return self._finalize_reset(self.current_query, label)


    def _finalize_reset(self, current_state: torch.Tensor, label: int):
        if self.verbose >=1: print(f"Initial Query (Label: {label}): {debug_print_state_from_indices(current_state, self.index_manager, oneline=True)}")
            
        self.current_depth = 0 
        self.memory.clear()
        state_tuple_for_memory = tuple(tuple(atom.tolist()) for atom in current_state)
        self.memory.add(state_tuple_for_memory)
        padded_current_state = self._pad_state_tensor(current_state)
        derived_state_tensors, truncated_flag = self.get_next_states_internal(current_state)
        obs_sub_index = padded_current_state.unsqueeze(0) 
        obs_derived_sub_indices = self._pad_derived_states(derived_state_tensors) 
        obs_dict = {
            'sub_index': obs_sub_index.cpu().numpy(),
            'derived_sub_indices': obs_derived_sub_indices.cpu().numpy(),
        }
        self.last_derived_state_tensors = derived_state_tensors 

        return obs_dict, {}

    def step(self, action: int):
        if self.verbose >=1: print(f"\n--- Step {self.current_depth + 1}, Action: {action} ---")
        
        if action < 0 or action >= len(self.last_derived_state_tensors):
            # This case should ideally be prevented by the agent's action masking if possible
            # If it occurs, it means the agent chose an invalid action.
            print(f"Warning: Action {action} is out of bounds for {len(self.last_derived_state_tensors)} derived states. This proof path fails.")
            next_state_tensor = self.false_tensor_atom.unsqueeze(0) 
        else:
            next_state_tensor = self.last_derived_state_tensors[action]

        if self.verbose >=1: print(f"Chosen Next State: {debug_print_state_from_indices(next_state_tensor, self.index_manager, oneline=True)}")

        done_next, reward_next = self.get_done_reward_tensor(next_state_tensor, self.current_label)
        
        derived_states_next_tensors, truncated_flag = [], False
        if not done_next: 
            derived_states_next_tensors, truncated_flag = self.get_next_states_internal(next_state_tensor)
        
        self.current_depth += 1
        exceeded_max_depth = (self.current_depth >= self.max_depth)
        if exceeded_max_depth and self.verbose >=1: print(f'Max depth {self.max_depth} reached.')

        done_final = done_next or exceeded_max_depth or truncated_flag
        
        info = {}
        if done_final:
            if self.engine == 'prolog' and self.current_label == 1 and self.current_query is not None:
                if self.current_query.numel() > 0 and self.current_query[0,0].item() != self.false_pred_idx :
                    query_terms_to_assert = tensor_state_to_terms_list_im(self.current_query, self.index_manager)
                    if query_terms_to_assert:
                        first_atom_term = query_terms_to_assert[0]
                        special_pred_strings = [self.index_manager.special_preds_map_names.get(sp, "") for sp in self.index_manager.special_preds]
                        if first_atom_term.predicate not in special_pred_strings:
                            try:
                                janus.query_once(f"asserta({first_atom_term.prolog_str()}).")
                                if self.verbose >=2: print(f"Prolog: Asserted back {first_atom_term.prolog_str()}")
                            except Exception as e:
                                if self.verbose >=1: print(f"Prolog: Failed to assert back {first_atom_term.prolog_str()} - {e}")
            self.current_query = None
            self.current_label = None

        padded_next_state_tensor = self._pad_state_tensor(next_state_tensor)
        obs_sub_index = padded_next_state_tensor.unsqueeze(0)
        obs_derived_sub_indices = self._pad_derived_states(derived_states_next_tensors)

        obs_dict = {
            'sub_index': obs_sub_index.cpu().numpy(),
            'derived_sub_indices': obs_derived_sub_indices.cpu().numpy(),
        }
        self.last_derived_state_tensors = derived_states_next_tensors
        reward_val = reward_next.cpu().item()
        done_val = bool(done_final)
        truncated_val = bool(exceeded_max_depth or truncated_flag) 

        if self.verbose >=1:
            print(f"Step Output: Reward={reward_val}, Done={done_val}, Truncated={truncated_val}")
            print(f"  Derived States for Next Obs ({len(derived_states_next_tensors)}): {debug_print_states_from_indices(derived_states_next_tensors, self.index_manager)}")
        return obs_dict, reward_val, done_val, truncated_val, info

    def _pad_state_tensor(self, state_tensor: torch.Tensor) -> torch.Tensor:
        num_atoms = state_tensor.shape[0]
        if num_atoms == 0 : 
            return torch.full((self.padding_atoms, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self.device)
        if num_atoms > self.padding_atoms:
            if self.verbose >=1: print(f"Warning: State tensor with {num_atoms} atoms exceeds padding_atoms {self.padding_atoms}. Truncating.")
            return state_tensor[:self.padding_atoms]
        if num_atoms < self.padding_atoms:
            padding_needed = self.padding_atoms - num_atoms
            padding = torch.full((padding_needed, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self.device)
            return torch.cat([state_tensor, padding], dim=0)
        return state_tensor

    def _pad_derived_states(self, derived_state_tensors: List[torch.Tensor]) -> torch.Tensor:
        if not derived_state_tensors: 
             return torch.full((self.padding_states, self.padding_atoms, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self.device)
        padded_derived_tensors = [self._pad_state_tensor(s_tensor) for s_tensor in derived_state_tensors]
        num_derived = len(padded_derived_tensors)
        stacked_derived = torch.stack(padded_derived_tensors)
        if num_derived < self.padding_states:
            padding_needed = self.padding_states - num_derived
            padding = torch.full((padding_needed, self.padding_atoms, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self.device)
            return torch.cat([stacked_derived, padding], dim=0)
        return stacked_derived

    def get_next_states_internal(self, current_state: torch.Tensor) -> Tuple[List[torch.Tensor], bool]:
        truncated_flag = False 
        if self.end_proof_action:
            is_end_state = False
            if current_state.shape[0] == 1 and current_state[0,0].item() == self.end_pred_idx:
                is_end_state = True
            if is_end_state:
                 return [self.false_tensor_atom.unsqueeze(0)], truncated_flag
            else: 
                mask_not_end = current_state[:, 0] != self.end_pred_idx
                current_state = current_state[mask_not_end]
                if current_state.shape[0] == 0: 
                    return [self.false_tensor_atom.unsqueeze(0)], truncated_flag

        derived_state_tensors: List[torch.Tensor] = []
        if self.engine == 'python_tensor':
            excluded_fact: Optional[torch.Tensor] = None
            if self.current_label == 1 and self.current_query is not None:
                if self.current_query.shape[0] == 1:
                     excluded_fact = self.current_query[0] 
                elif self.verbose >=1:
                     print("Warning: current_query has multiple atoms, cannot use as excluded_fact.")
            derived_state_tensors, _ = get_next_unification_pt(
                current_state_idx=current_state,
                facts_tensor=self.facts,
                facts_as_set=self.facts_set,
                rules_tensor=self.rules,
                rule_lengths_tensor=self.rule_lengths,
                index_manager=self.index_manager,
                original_rules_list=self.rules_term, 
                excluded_fact_idx=excluded_fact,
                include_intermediate_rule_states=self.include_intermediate_rule_states,
                verbose=self.prover_verbose 
            )
        elif self.engine == 'prolog':
            state_terms_list = tensor_state_to_terms_list_im(current_state, self.index_manager)
            if self.prover_verbose >=1: print(f"Prolog Engine - Input State (Terms): {state_terms_list}")
            self.index_manager.reset_next_var_index()
            derived_terms_list_of_list, _ = get_next_unification_prolog(
                state_terms_list, index_manager=self.index_manager, verbose=self.prover_verbose
            )
            if self.prover_verbose >=1: print(f"Prolog Engine - Output States (Terms): {derived_terms_list_of_list}")
            temp_derived_tensors = []
            for terms_list in derived_terms_list_of_list:
                var_map_for_derived_state = {}
                self.index_manager.reset_next_var_index()
                tensor_state = state_to_tensor_im(terms_list, self.index_manager, var_map_for_derived_state)
                temp_derived_tensors.append(tensor_state.to(self.device)) # Ensure on device
            derived_state_tensors = temp_derived_tensors
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
        
        if self.memory_pruning:
            current_state_tuple_for_memory = tuple(tuple(atom.tolist()) for atom in current_state)
            self.memory.add(current_state_tuple_for_memory)
            filtered_derived_states = []
            for ds_tensor in derived_state_tensors:
                ds_tuple = tuple(tuple(atom.tolist()) for atom in ds_tensor)
                if ds_tuple not in self.memory:
                    filtered_derived_states.append(ds_tensor)
                elif self.verbose >=1:
                    print(f"Memory Pruning: State {debug_print_state_from_indices(ds_tensor, self.index_manager, True)} already visited.")
            derived_state_tensors = filtered_derived_states

        processed_derived_states = []
        for ds_tensor in derived_state_tensors:
            if ds_tensor.shape[0] <= self.padding_atoms: 
                processed_derived_states.append(ds_tensor)
            elif self.verbose >=1:
                print(f"Truncating state with {ds_tensor.shape[0]} atoms (max: {self.padding_atoms}): {debug_print_state_from_indices(ds_tensor, self.index_manager, True)}")
                processed_derived_states.append(ds_tensor[:self.padding_atoms]) 
        derived_state_tensors = processed_derived_states

        max_num_actions = self.padding_states
        if self.end_proof_action and any(ds.numel() > 0 and ds[0,0].item() != self.true_pred_idx and ds[0,0].item() != self.false_pred_idx for ds in derived_state_tensors):
            max_num_actions -=1 
        if len(derived_state_tensors) > max_num_actions:
            if self.verbose >=1: print(f"Truncating {len(derived_state_tensors)} derived states to {max_num_actions}.")
            derived_state_tensors.sort(key=lambda t: t.shape[0])
            derived_state_tensors = derived_state_tensors[:max_num_actions]
        
        if self.end_proof_action:
            is_any_non_terminal = any(
                ds.numel() > 0 and ds[0,0].item() != self.true_pred_idx and ds[0,0].item() != self.false_pred_idx
                for ds in derived_state_tensors
            )
            if is_any_non_terminal and len(derived_state_tensors) < self.padding_states:
                 derived_state_tensors.append(self.end_tensor_atom.unsqueeze(0))
        if not derived_state_tensors:
            if self.verbose >=1: print("No valid next states after processing, returning [[FALSE]].")
            derived_state_tensors = [self.false_tensor_atom.unsqueeze(0)]
        return derived_state_tensors, truncated_flag
    
    def get_done_reward_tensor(self, state_tensor: torch.Tensor, label: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if state_tensor.numel() == 0: 
            all_atoms_true = True
            any_atom_false = False
        else:
            predicates = state_tensor[:, 0]
            all_atoms_true = torch.all(predicates == self.true_pred_idx).item()
            any_atom_false = torch.any(predicates == self.false_pred_idx).item()
        is_end_terminal = False
        if self.end_proof_action and state_tensor.shape[0] == 1 and state_tensor[0,0].item() == self.end_pred_idx:
            is_end_terminal = True
            all_atoms_true = False 
            any_atom_false = False 
        done = all_atoms_true or any_atom_false or is_end_terminal
        successful_proof = all_atoms_true 
        reward_val = 0
        if done:
            if successful_proof and label == 1:
                reward_val = 1
        return torch.tensor(done, dtype=torch.bool, device=self.device), \
               torch.tensor(reward_val, dtype=torch.float32, device=self.device)

    def get_random_queries_tensors(self, n: int = 1) -> Tuple[List[torch.Tensor], Optional[List[int]]]:
        if not self.queries:
            # Return a list containing a FALSE tensor if no queries are available
            false_query = self.false_tensor_atom.unsqueeze(0).to(self.device)
            if self.verbose >=1: print("Warning: get_random_queries_tensors called with no initial queries. Returning [[FALSE]].")
            return [false_query] * n, [0] * n # Label is 0 for FALSE

        num_available_queries = len(self.queries) 
        sampled_indices: List[int]
        if n > num_available_queries:
            if self.verbose >=1: print(f"Warning: Requested {n} query tensors, but only {num_available_queries} available. Sampling with replacement.")
            sampled_indices = [random.randint(0, num_available_queries - 1) for _ in range(n)]
        else:
            sampled_indices = random.sample(range(num_available_queries), n)

        sampled_queries_tensors = [self.queries[i].to(self.device) for i in sampled_indices]
        sampled_labels: Optional[List[int]] = None
        if self.labels: # Check if labels list is populated
             if len(self.labels) == self.n_episodes: # Ensure labels align with tensors
                try:
                    sampled_labels = [self.labels[i] for i in sampled_indices]
                except IndexError:
                    if self.verbose >=1: print("Warning: Label index out of bounds during sampling. Defaulting labels.")
                    sampled_labels = [1] * n # Fallback
             else: # Labels exist but don't match n_episodes
                if self.verbose >=1: print("Warning: Mismatch between number of labels and query tensors. Defaulting labels for sampled queries.")
                sampled_labels = [1] * n
        else: # No labels provided initially
            sampled_labels = [1] * n # Default to label 1 (positive)

        return sampled_queries_tensors, sampled_labels


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
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    padding_atoms_env = 5
    padding_states_env = 10

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
    index_manager_test.rules, index_manager_test.rules_lengths = rules_to_tensor_im(data_handler_test.rules_terms,max_rule_atoms=max_rule_atoms_test, index_manager=index_manager_test)
    index_manager_test.facts = facts_to_tensor_im(data_handler_test.facts_terms,index_manager_test)
    index_manager_test.facts_set = frozenset(tuple(f.tolist()) for f in index_manager_test.facts)
    test_queries = queries_to_tensor_im(data_handler_test.test_queries_terms, index_manager_test)

    # --- 4. Initialize the Environment with the comprehensive IndexManager. ---
    env_test = LogicEnv_gym(
        index_manager=index_manager_test,
        data_handler=data_handler_test,
        queries_term=data_handler_test.test_queries_terms,
        rules_term=data_handler_test.rules_terms,
        queries=test_queries,
        labels=data_handler_test.test_labels,
        facts=index_manager_test.facts,
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
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        while not (terminated or truncated):
            action_to_take = 0 # Default to taking the first derived state for deterministic test
            num_derived_states = len(env_test.last_derived_state_tensors)

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

            print(f"  Taking Action: {action_to_take} (out of {num_derived_states} choices)")
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

