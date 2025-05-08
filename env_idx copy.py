from typing import List, Optional, Tuple, Set, FrozenSet
import random
import torch
import gymnasium as gym
import numpy as np
import janus_swi as janus

from dataset_idx import DataHandler 
from index_manager_idx import IndexManager
from utils import Rule, Term # Term and Rule for initial query/rule definition
# Ensure all necessary functions are imported from python_unification_idx
from python_unification_idx import get_next_unification_pt, state_to_tensor_im, term_to_tensor_im, \
                                   facts_to_tensor_im, rules_to_tensor_im, \
                                   debug_print_state_from_indices, debug_print_states_from_indices

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
                queries: Optional[List[Term]] = None, 
                labels: Optional[List[int]] = None,
                facts_tensor: Optional[torch.Tensor] = None,
                facts_as_set_indices: Optional[FrozenSet[Tuple[int, int, int]]] = None,
                rules_tensor: Optional[torch.Tensor] = None,
                rule_lengths_tensor: Optional[torch.Tensor] = None,
                original_rules_list: Optional[List[Rule]] = None, 
                mode: str = 'train',
                corruption_mode: Optional[str] = None,
                corruption_scheme: Optional[List[str]] = None,
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

        self.dataset_name = data_handler.dataset_name
        
        self.facts_tensor = facts_tensor.to(device) if facts_tensor is not None else torch.empty((0,self.max_arity+1), dtype=torch.long, device=device)
        self.facts_as_set_indices = facts_as_set_indices if facts_as_set_indices is not None else frozenset()
        self.rules_tensor = rules_tensor.to(device) if rules_tensor is not None else torch.empty((0,1,self.max_arity+1), dtype=torch.long, device=device) 
        self.rule_lengths_tensor = rule_lengths_tensor.to(device) if rule_lengths_tensor is not None else torch.empty((0,), dtype=torch.long, device=device)
        self.original_rules_list = original_rules_list if original_rules_list is not None else []
        self.include_intermediate_rule_states = include_intermediate_rule_states


        self.corruption_mode = corruption_mode
        if self.corruption_mode:
            self.counter = 0 
            self.sampler = data_handler.sampler 
            self.triples_factory = data_handler.triples_factory 
            self.corruption_scheme = corruption_scheme
        
        self.janus_file = data_handler.janus_path
        self.janus_facts_str = data_handler.jan_facts_str if hasattr(data_handler, 'jan_facts_str') else None


        self.memory: Set[Tuple[Tuple[int,...],...]] = set() 
        self.memory_pruning = memory_pruning
        self.end_proof_action = end_proof_action 
        self.skip_unary_actions = skip_unary_actions 

        self.true_pred_idx = self.index_manager.true_pred_idx
        self.false_pred_idx = self.index_manager.false_pred_idx
        self.end_pred_idx = self.index_manager.predicate_str2idx.get('End', -1) 
        self.padding_idx = self.index_manager.padding_idx

        self.true_tensor_atom = self.index_manager.true_tensor.to(device) 
        self.false_tensor_atom = self.index_manager.false_tensor.to(device)
        self.end_tensor_atom = torch.tensor([self.end_pred_idx, self.padding_idx, self.padding_idx], dtype=torch.long, device=self.device)

        self.current_query_tensor: Optional[torch.Tensor] = None 
        self.current_label: Optional[int] = None

        self.mode = mode
        self.initial_queries_terms: List[List[Term]] = [[q] if isinstance(q, Term) else q for q in queries] if queries is not None else []
        self.labels = labels if labels is not None else []
        self.n_episodes = len(self.initial_queries_terms)
        self.eval_idx = 0
        self.consult_janus_eval = False 

        if self.mode == 'train':
            self.train_neg_pos_ratio = train_neg_pos_ratio
            if not self.initial_queries_terms:
                 raise ValueError("Training mode requires initial_queries_terms.")


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
            self._set_seed(seed)

        initial_query_as_terms: List[Term] 
        label: int

        if self.mode == 'eval':
            if self.eval_idx < self.n_episodes:
                initial_query_as_terms = self.initial_queries_terms[self.eval_idx]
                label = self.labels[self.eval_idx]
            else: 
                initial_query_as_terms = [Term(predicate='False', args=())] 
                label = 0 
            self.eval_idx += 1
        elif self.mode == 'eval_with_restart':
            if self.eval_idx >= self.n_episodes: self.eval_idx = 0 
            initial_query_as_terms = self.initial_queries_terms[self.eval_idx]
            label = self.labels[self.eval_idx]
            self.eval_idx += 1
        elif self.mode == 'train':
            raw_query_terms_list_of_list, _ = self.get_random_queries_terms(self.initial_queries_terms, n=1) 
            raw_query_terms = raw_query_terms_list_of_list[0] 
            label = 1 

            if self.corruption_mode and self.counter % (self.train_neg_pos_ratio + 1) != 0:
                neg_query_terms_outer = self.sampler.get_negatives_from_states([raw_query_terms], self.device, return_states=True) 
                
                if not hasattr(self, 'negation_toggle'): self.negation_toggle = 0
                
                selected_neg_list = []
                if neg_query_terms_outer and neg_query_terms_outer[0]: 
                    actual_negatives = neg_query_terms_outer[0] 
                    if actual_negatives:
                        if len(self.corruption_scheme) > 1 and len(actual_negatives) > self.negation_toggle :
                            selected_neg_list = [actual_negatives[self.negation_toggle]]
                            self.negation_toggle = 1 - self.negation_toggle
                        elif actual_negatives:
                            selected_neg_list = [actual_negatives[0]]
                
                if selected_neg_list:
                    initial_query_as_terms = selected_neg_list 
                    label = 0
                else: 
                    initial_query_as_terms = raw_query_terms
                    label = 1 
            else:
                initial_query_as_terms = raw_query_terms
            self.counter = (self.counter + 1) % (self.train_neg_pos_ratio + 1) if self.corruption_mode else 0
        else:
            raise ValueError(f"Invalid mode: {self.mode}.")
                
        if isinstance(initial_query_as_terms, Term): initial_query_as_terms = [initial_query_as_terms]

        if self.engine == 'prolog' and (self.mode == 'train' or self.consult_janus_eval) and label == 1:
            if initial_query_as_terms: 
                query_to_retract_term = initial_query_as_terms[0]
                # This check would be better: if query_to_retract_term in self.data_handler.facts_terms:
                # For now, we assume if it's a positive query in train/eval, we attempt retract.
                # This requires facts_terms to be available, e.g. via data_handler or IM.
                # For the purpose of this code, we'll assume the check passes or is omitted for brevity.
                janus.query_once(f"retract({query_to_retract_term.prolog_str()}).")

        self.current_label = label
        initial_var_map_reset = {} 
        self.index_manager.next_var_index = self.index_manager.variable_start_index 
        current_state_tensor = state_to_tensor_im(initial_query_as_terms, self.index_manager, initial_var_map_reset)
        current_state_tensor = current_state_tensor.to(self.device)
        self.current_query_tensor = current_state_tensor 
        return self._finalize_reset(current_state_tensor, label)

    def _finalize_reset(self, current_state_tensor: torch.Tensor, label: int):
        if self.verbose >=1: print(f"Initial Query (Label: {label}): {debug_print_state_from_indices(current_state_tensor, self.index_manager, oneline=True)}")
            
        self.current_depth = 0 
        self.memory.clear()
        state_tuple_for_memory = tuple(tuple(atom.tolist()) for atom in current_state_tensor)
        self.memory.add(state_tuple_for_memory)
        padded_current_state_tensor = self._pad_state_tensor(current_state_tensor)
        derived_state_tensors, truncated_flag = self.get_next_states_internal(current_state_tensor)
        obs_sub_index = padded_current_state_tensor.unsqueeze(0) 
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
            if self.engine == 'prolog' and self.current_label == 1 and self.current_query_tensor is not None:
                if self.current_query_tensor.shape[0] > 0:
                    query_to_assert_term = tensor_atom_to_term_im(self.current_query_tensor[0], self.index_manager)
                    janus.query_once(f"asserta({query_to_assert_term.prolog_str()}).")
            self.current_query_tensor = None
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

    def get_next_states_internal(self, current_state_tensor: torch.Tensor) -> Tuple[List[torch.Tensor], bool]:
        truncated_flag = False 
        if self.end_proof_action:
            is_end_state = False
            if current_state_tensor.shape[0] == 1 and current_state_tensor[0,0].item() == self.end_pred_idx:
                is_end_state = True
            if is_end_state:
                 return [self.false_tensor_atom.unsqueeze(0)], truncated_flag
            else: 
                mask_not_end = current_state_tensor[:, 0] != self.end_pred_idx
                current_state_tensor = current_state_tensor[mask_not_end]
                if current_state_tensor.shape[0] == 0: 
                    return [self.false_tensor_atom.unsqueeze(0)], truncated_flag

        derived_state_tensors: List[torch.Tensor] = []
        if self.engine == 'python_tensor':
            excluded_fact_tensor: Optional[torch.Tensor] = None
            if self.current_label == 1 and self.current_query_tensor is not None:
                if self.current_query_tensor.shape[0] == 1:
                     excluded_fact_tensor = self.current_query_tensor[0] 
                elif self.verbose >=1:
                     print("Warning: current_query_tensor has multiple atoms, cannot use as excluded_fact.")
            derived_state_tensors, _ = get_next_unification_pt(
                current_state_idx=current_state_tensor,
                facts_tensor=self.facts_tensor,
                facts_as_set=self.facts_as_set_indices,
                rules_tensor=self.rules_tensor,
                rule_lengths_tensor=self.rule_lengths_tensor,
                index_manager=self.index_manager,
                original_rules_list=self.original_rules_list, 
                excluded_fact_idx=excluded_fact_tensor,
                include_intermediate_rule_states=self.include_intermediate_rule_states,
                verbose=self.prover_verbose 
            )
        elif self.engine == 'prolog':
            state_terms_list = tensor_state_to_terms_list_im(current_state_tensor, self.index_manager)
            if self.prover_verbose >=1: print(f"Prolog Engine - Input State (Terms): {state_terms_list}")
            next_var_idx_prolog = self.index_manager.next_var_index 
            derived_terms_list_of_list, updated_next_var_idx_prolog = get_next_unification_prolog(
                state_terms_list,
                next_var_index=next_var_idx_prolog,
                verbose=self.prover_verbose
            )
            if self.prover_verbose >=1: print(f"Prolog Engine - Output States (Terms): {derived_terms_list_of_list}")
            temp_derived_tensors = []
            for terms_list in derived_terms_list_of_list:
                var_map_for_derived_state = {} 
                self.index_manager.next_var_index = self.index_manager.variable_start_index
                tensor_state = state_to_tensor_im(terms_list, self.index_manager, var_map_for_derived_state)
                temp_derived_tensors.append(tensor_state.to(self.device))
            derived_state_tensors = temp_derived_tensors
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
        
        if self.memory_pruning:
            current_state_tuple_for_memory = tuple(tuple(atom.tolist()) for atom in current_state_tensor)
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

    constants_test = {"a", "b", "c", "john", "mary", "peter"}
    predicates_test = {"p", "q", "parent", "grandparent"}
    rules_obj_test = [
        Rule(Term("grandparent", ("X", "Z")), [Term("parent", ("X", "Y")), Term("parent", ("Y", "Z"))]),
        Rule(Term("q", ("A", "B")), [Term("p", ("A", "C"))]) # This rule won't be used if p(A,C) doesn't lead to B
    ]
    max_total_vars_test = 1000
    padding_atoms_env = 5
    padding_states_env = 10
    
    if rules_obj_test:
        max_rule_atoms_test = max(1 + len(r.body) for r in rules_obj_test) if rules_obj_test else 1
    else:
        max_rule_atoms_test = 1 

    index_manager_test = IndexManager(
        constants=constants_test,
        predicates=predicates_test,
        max_total_vars=max_total_vars_test,
        rules=rules_obj_test, 
        padding_atoms=padding_atoms_env, 
        max_arity=2, 
        device=DEVICE
    )
    print("IndexManager Initialized.")
    print(f"  True_idx: {index_manager_test.true_pred_idx}, False_idx: {index_manager_test.false_pred_idx}")
    
    facts_terms_test = [
        Term("parent", ("john", "mary")),
        Term("parent", ("mary", "peter")),
        Term("p", ("a", "b"))
    ]

    class DummyDataHandler: 
        def __init__(self, im, rules_obj, facts_obj, max_r_atoms, dev):
            self.dataset_name = "test_dataset"
            self.rules_objects = rules_obj
            self.facts_terms = facts_obj # Store original facts
            self.sampler = None 
            self.triples_factory = None
            self.corruption_scheme = None
            self.janus_path = None
            self.janus_facts_str = None # Corrected attribute name
            self.facts_tensor = facts_to_tensor_im(self.facts_terms, index_manager_test).to(DEVICE)
            self.facts_as_set_indices = frozenset(tuple(f.tolist()) for f in self.facts_tensor)
            self.rules_tensor, self.rule_lengths_tensor = rules_to_tensor_im(
                self.rules_objects, max_r_atoms, index_manager_test
            )
            self.rules_tensor = self.rules_tensor.to(DEVICE)
            self.rule_lengths_tensor = self.rule_lengths_tensor.to(DEVICE)

    data_handler_test = DummyDataHandler(index_manager_test, rules_obj_test, facts_terms_test, max_rule_atoms_test, DEVICE)
    print("DataHandler (Mock) Initialized with Tensorized Facts/Rules.")

    queries_for_env_test = [
        [Term("grandparent", ("john", "peter"))], 
        [Term("q", ("a", "b"))],             
        [Term("parent", ("a", "b"))]            
    ]
    labels_for_env_test = [1, 0, 0] # q(a,What) is not provable with current facts/rules for q. parent(a,b) is not a fact.

    env_test = LogicEnv_gym(
        index_manager=index_manager_test,
        data_handler=data_handler_test, 
        queries=queries_for_env_test,
        labels=labels_for_env_test,
        facts_tensor=data_handler_test.facts_tensor,
        facts_as_set_indices=data_handler_test.facts_as_set_indices,
        rules_tensor=data_handler_test.rules_tensor,
        rule_lengths_tensor=data_handler_test.rule_lengths_tensor,
        original_rules_list=data_handler_test.rules_objects,
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

    num_test_episodes = len(queries_for_env_test)
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

