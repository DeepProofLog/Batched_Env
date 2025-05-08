from typing import List, Tuple, Dict, Set
import itertools
import torch
from utils import Term, Rule
class IndexManager():
    def __init__(self,
                 constants: set,
                 predicates: set,
                 max_total_vars: int, 
                 rules: List[Rule], 
                 constants_images: set = (),
                 constant_images_no: int = 0, 
                 padding_atoms: int = 10,    
                 max_arity: int = 2,         
                 device: torch.device = torch.device("cpu")):

        self.device = device
        self.constants = constants
        self.predicates = predicates
        self.special_preds_map_names = {'True': "##TRUE##", 'False': "##FALSE##", 'End': "##END##"} 
        self.special_preds = ['True', 'False', 'End'] 
        
        self.padding_idx = 0

        self.max_total_vars = max_total_vars 
        self.rules_objects = rules # Store original rule objects for verbose printing
        self.padding_atoms = padding_atoms
        self.constants_images = constants_images
        self.max_arity = max_arity 
        if self.max_arity != 2:
            print(f"Warning: IndexManager max_arity is {self.max_arity}, but code assumes 2 for 3-element term tensors.")

        self.constant_images_no = constant_images_no 

        self.constant_str2idx: Dict[str, int] = {}
        self.constant_idx2str: Dict[int, str] = {}
        self.predicate_str2idx: Dict[str, int] = {}
        self.predicate_idx2str: Dict[int, str] = {}
        self.variable_str2idx: Dict[str, int] = {} 
        self.variable_idx2str: Dict[int, str] = {} 
        
        self.create_global_idx() 

        self.variable_start_index = self.constant_no + 1
        self.variable_end_index = self.variable_start_index + self.max_total_vars - 1
        self.variable_no = self.max_total_vars 
        self.next_var_index = self.variable_start_index 

        self._create_fixed_var_idx() 

        self.dynamic_variable_start_index = self.variable_end_index + 1
        self.next_dynamic_var_idx_counter = self.dynamic_variable_start_index
        self.dynamic_variable_idx2str: Dict[int, str] = {}
        
        self.unified_term_map: Dict[str, int] = {} 
        self.unified_term_map.update(self.constant_str2idx)
        self.unified_term_map.update(self.variable_str2idx) 

        self.true_pred_idx = self.predicate_str2idx['True']
        self.false_pred_idx = self.predicate_str2idx['False']
        
        self.true_tensor = torch.tensor([self.true_pred_idx, self.padding_idx, self.padding_idx], dtype=torch.long, device=self.device)
        self.false_tensor = torch.tensor([self.false_pred_idx, self.padding_idx, self.padding_idx], dtype=torch.long, device=self.device)

        self.fact_index: Dict[Tuple, Set[Term]] = {} 

    def reset_next_var_index(self): 
        self.next_var_index = self.variable_start_index

    def create_global_idx(self):
        current_idx = 1 
        sorted_constants = sorted(list(self.constants)) 
        for term in sorted_constants:
            self.constant_str2idx[term] = current_idx
            self.constant_idx2str[current_idx] = term
            current_idx += 1
        self.constant_no = current_idx - 1

        current_idx = 1 
        
        # Ensure special predicate names defined by the user are handled correctly even if they overlap with "regular" predicates
        # Regular predicates first
        regular_predicates = sorted(list(self.predicates - set(self.special_preds)))
        for term in regular_predicates:
            if term not in self.predicate_str2idx:
                 self.predicate_str2idx[term] = current_idx
                 self.predicate_idx2str[current_idx] = term
                 current_idx += 1
        
        # Then user-defined special predicates (ensures they get specific indices if not already assigned)
        for term_user_name in self.special_preds: 
            if term_user_name not in self.predicate_str2idx: # If not already added as a regular predicate
                self.predicate_str2idx[term_user_name] = current_idx
                self.predicate_idx2str[current_idx] = term_user_name
                current_idx += 1
            # If it was already added, its index is preserved. This handles cases where a special pred name
            # might also be in the general `predicates` set.
            
        self.predicate_no = len(self.predicate_str2idx) # Total number of unique predicates


    def _create_fixed_var_idx(self):
        for i in range(self.max_total_vars):
            var_name = f"Var_{self.variable_start_index + i}" 
            var_index = self.variable_start_index + i
            self.variable_str2idx[var_name] = var_index
            self.variable_idx2str[var_index] = var_name

    def get_fresh_dynamic_var_idx(self) -> int:
        idx = self.next_dynamic_var_idx_counter
        self.dynamic_variable_idx2str[idx] = f"DynVar_{idx}" 
        self.next_dynamic_var_idx_counter += 1
        return idx

    def is_fixed_var_idx(self, idx: int) -> bool:
        return self.variable_start_index <= idx <= self.variable_end_index

    def is_dynamic_var_idx(self, idx: int) -> bool:
        return idx >= self.dynamic_variable_start_index and idx < self.next_dynamic_var_idx_counter

    def is_var_idx(self, idx: int) -> bool: 
        return self.is_fixed_var_idx(idx) or self.is_dynamic_var_idx(idx)

    def get_str_for_term_idx(self, idx: int) -> str:
        if idx == self.padding_idx: return "PAD"
        if self.is_fixed_var_idx(idx):
            return self.variable_idx2str.get(idx, f"FixedVar?({idx})")
        if self.is_dynamic_var_idx(idx):
            return self.dynamic_variable_idx2str.get(idx, f"DynVar_{idx}") 
        return self.constant_idx2str.get(idx, f"Const?({idx})")

    def get_str_for_pred_idx(self, idx: int) -> str:
        if idx == self.padding_idx: return "PAD_PRED" 
        return self.predicate_idx2str.get(idx, f"Pred?({idx})")
        
    def get_predicate_idx_by_str(self, s: str) -> int:
        try:
            return self.predicate_str2idx[s]
        except KeyError:
            raise KeyError(f"Predicate '{s}' not found in IndexManager.")

    def get_constant_idx_by_str(self, s: str) -> int:
        try:
            return self.constant_str2idx[s]
        except KeyError:
            if s in self.variable_str2idx: 
                 raise KeyError(f"String '{s}' is a pre-defined fixed variable name, not a constant.")
            raise KeyError(f"Constant '{s}' not found in IndexManager.")

    def get_atom_sub_index(self, state: List[Term]) -> torch.Tensor:
        state_len = len(state)
        if state_len > self.padding_atoms:
             raise ValueError(f"Length of processed state ({state_len}) exceeds padding_atoms ({self.padding_atoms}).")
        sub_index = torch.zeros(self.padding_atoms, self.max_arity + 1, device=self.device, dtype=torch.int64)
        predicate_map = self.predicate_str2idx
        unified_map = self.unified_term_map 
        for i, atom in enumerate(state):
            current_sub_row = sub_index[i]
            pred_str = atom.predicate
            try:
                current_sub_row[0] = predicate_map[pred_str]
            except KeyError:
                 raise KeyError(f"Predicate '{pred_str}' not found in predicate map.")
            num_args = len(atom.args)
            max_j = min(num_args, self.max_arity) 
            arg_strings = atom.args[:max_j]
            try:
                arg_indices = [unified_map[arg] for arg in arg_strings] 
            except KeyError as e:
                 raise KeyError(f"Argument '{e}' not found in constant or fixed variable maps via unified_term_map.") from e
            if arg_indices: 
                current_sub_row[1:max_j + 1] = torch.tensor(arg_indices, device=self.device, dtype=torch.int64)
        return sub_index

    def subindices_to_terms(self, idx: torch.Tensor) -> List[List[Term]]:
        B, M, P, A = idx.shape 
        if A != self.max_arity + 1:
            raise ValueError(f"Tensor last dim {A} != max_arity+1 ({self.max_arity+1})")
        
        atoms = idx[..., 0, :].reshape(-1, A) 
        mask = (atoms != self.padding_idx).all(dim=1)
        valid = atoms[mask]
        
        results: List[List[Term]] = [[] for _ in range(B * M // M if M > 0 else B)] 

        if valid.numel() == 0:
            return results

        rel_idxs = valid[:, 0].tolist()
        preds = [self.get_str_for_pred_idx(r) for r in rel_idxs]
        
        term_args_list = []
        for k in range(1, self.max_arity + 1): 
            arg_k_idxs = valid[:, k].tolist()
            term_args_list.append([self.get_str_for_term_idx(h) for h in arg_k_idxs])

        grouped_args = list(zip(*term_args_list)) 

        terms = [Term(predicate=p, args=tuple(arg_tuple[:self.max_arity])) 
                 for p, arg_tuple in zip(preds, grouped_args)]
        
        flat_positions = mask.nonzero(as_tuple=False).squeeze(1).tolist()
        for pos, term in zip(flat_positions, terms):
            batch_idx = pos // M if M > 0 else pos 
            if batch_idx < len(results):
                 results[batch_idx].append(term)
        return results

    def build_fact_index(self, facts: List[Term]) -> Dict[Tuple, Set[Term]]: 
        self.fact_index.clear()
        for fact in facts:
            predicate = fact.predicate
            args = fact.args
            constant_args_with_pos = [(i, arg) for i, arg in enumerate(args) if not is_user_variable_str(arg)]
            for i in range(len(constant_args_with_pos) + 1):
                for subset_args_with_pos in itertools.combinations(constant_args_with_pos, i):
                    sorted_subset = tuple(sorted(subset_args_with_pos, key=lambda x: x[0]))
                    key = (predicate,) + sorted_subset
                    if key not in self.fact_index:
                        self.fact_index[key] = set()
                    self.fact_index[key].add(fact)
        return self.fact_index