from typing import List, Tuple, Dict, Set, Optional
import itertools
import torch
from utils import is_variable, Term, Rule

class IndexManager():
    '''
    Manages indices for constants, predicates, and fixed variables.
    Variable indices are pre-assigned based on max_total_vars.
    Includes a unified lookup map for faster term indexing.
    '''
    def __init__(self,
                 constants: set,
                 predicates: set,
                 max_total_vars: int, # Max allowed *total* vars (pre-assigned)
                 padding_atoms: int = 10,
                 max_arity: int = 2,
                 device: torch.device = torch.device("cpu")):

        self.device = device
        self.constants = constants
        self.predicates = predicates
        self.special_preds_map_names = {'True': "##TRUE##", 'False': "##FALSE##", 'End': "##END##"} 
        self.special_preds = ['True', 'False', 'End']
        self.padding_idx = 0

        self.max_total_vars = max_total_vars # Max *pre-assigned* variables

        self.padding_atoms = padding_atoms
        self.max_arity = max_arity
        if self.max_arity != 2:
            print(f"Warning: IndexManager max_arity is {self.max_arity}, but code assumes 2 for 3-element term tensors.")

        # --- Initialize Mappings ---
        self.constant_str2idx: Dict[str, int] = {}
        self.constant_idx2str: Dict[int, str] = {}
        self.predicate_str2idx: Dict[str, int] = {}
        self.predicate_idx2str: Dict[int, str] = {}
        self.variable_str2idx: Dict[str, int] = {}
        self.variable_idx2str: Dict[int, str] = {}
        # New unified map for constants and variables
        self.unified_term_map: Dict[str, int] = {}

        # --- Global indices for Constants and Predicates ---
        self.create_global_idx()

        # --- Fixed Variable indices ---
        # Variable indices start after the last constant index
        self.variable_start_index = self.constant_no + 1
        self.variable_end_index = self.variable_start_index + self.max_total_vars - 1
        self.variable_no = self.max_total_vars # Total number of pre-assigned variables

        self.next_var_index = self.variable_start_index # Next available variable index

        self._create_fixed_var_idx() # Create mappings for fixed variables

        # --- Create Unified Term Map ---
        # Combine constant and variable maps for faster lookup
        self.unified_term_map.update(self.constant_str2idx)
        self.unified_term_map.update(self.variable_str2idx)

        self.true_pred_idx = self.predicate_str2idx['True']
        self.false_pred_idx = self.predicate_str2idx['False']
        
        self.true_tensor = torch.tensor([self.true_pred_idx, self.padding_idx, self.padding_idx], dtype=torch.long, device=self.device)
        self.false_tensor = torch.tensor([self.false_pred_idx, self.padding_idx, self.padding_idx], dtype=torch.long, device=self.device)

        # Fact index setup
        self.fact_index: Dict[Tuple, Set[Tuple[int, ...]]] = {}


    def build_facts_index(self, facts_tensor: torch.Tensor):
        """
        Builds an index for facts to allow efficient retrieval for non-ground queries.
        This mirrors the logic of the original string-based implementation.
        """
        self.fact_index.clear()
        for i in range(facts_tensor.shape[0]):
            fact = facts_tensor[i]
            # Find constant arguments (those that are not variables)
            constant_args_with_pos = []
            # We check arguments from index 1 onwards
            for j in range(1, self.max_arity + 1):
                arg_idx = fact[j].item()
                if not self.is_var_idx(arg_idx) and arg_idx != self.padding_idx:
                    constant_args_with_pos.append((j - 1, arg_idx)) # Store arg position (0, 1) and its index

            # Generate all possible subsets of constant arguments to create keys
            # A query can then find a match if its constants are a subset of a fact's constants
            for k in range(len(constant_args_with_pos) + 1):
                for subset_args_with_pos in itertools.combinations(constant_args_with_pos, k):
                    # Sort by position to create a canonical key
                    sorted_subset = tuple(sorted(subset_args_with_pos, key=lambda x: x[0]))
                    key = (fact[0].item(),) + sorted_subset
                    
                    # Ensure the set for this key exists and add the fact tuple
                    if key not in self.fact_index:
                        self.fact_index[key] = set()
                    self.fact_index[key].add(tuple(fact.tolist()))
        
        # print(f"Fact index built with {len(self.fact_index)} keys.")
        # print(f"First 10 keys: {list(self.fact_index.keys())[:10]}")
        # print(f"First 10 values: {list(self.fact_index.values())[:10]}")


    def reset_next_var_index(self):
        self.next_var_index = self.variable_start_index

    def is_var_idx(self, idx: int) -> bool:
        '''Check if the given index corresponds to a variable.'''
        return self.variable_start_index <= idx <= self.variable_end_index
    
    def get_next_var(self) -> str:
        '''Get the next available variable idx and increment the index.'''
        print('Incrementing variable index:', self.next_var_index)
        if self.next_var_index > self.variable_end_index:
            raise ValueError(f"No more available variable indices: {self.next_var_index} exceeds max {self.variable_end_index}.")
        idx = self.next_var_index
        self.variable_idx2str[idx] = f"Var_{idx}"
        self.variable_str2idx[f"Var_{idx}"] = idx
        self.next_var_index += 1
        return idx

    def get_str_for_term_idx(self, idx: int) -> str:
        '''Get the string representation for a given term index.'''
        if idx in self.constant_idx2str:
            return self.constant_idx2str[idx]
        elif idx in self.variable_idx2str:
            return self.variable_idx2str[idx]
        elif idx in self.predicate_idx2str:
            return self.predicate_idx2str[idx]
        else:
            raise KeyError(f"Index {idx} not found in any mapping.")

    def create_global_idx(self):
        '''Create global indices for constants and predicates (including specials).'''
        # --- Constants ---
        current_idx = 1

        for term in sorted(self.constants):
            self.constant_str2idx[term] = current_idx
            self.constant_idx2str[current_idx] = term
            current_idx += 1

        self.constant_no = current_idx - 1

        # --- Predicates (Regular + Special) ---
        current_idx = 1
        # Add regular predicates first
        for term in sorted(self.predicates):
            self.predicate_str2idx[term] = current_idx
            self.predicate_idx2str[current_idx] = term
            current_idx += 1

        for term in self.special_preds:
            self.predicate_str2idx[term] = current_idx
            self.predicate_idx2str[current_idx] = term
            current_idx += 1

        self.predicate_no = current_idx - 1

    def _create_fixed_var_idx(self):
        '''Create fixed indices for variables based on max_total_vars.'''
        # Create variable names "Var_i", "Var_i+1", ..., "Var_(no_constants + max_total_vars)"
        # Assign indices starting from self.variable_start_index
        for i in range(self.max_total_vars):
            var_name = f"Var_{self.variable_start_index + i}"
            var_index = self.variable_start_index + i
            self.variable_str2idx[var_name] = var_index
            self.variable_idx2str[var_index] = var_name


    def get_atom_sub_index(self, state: List[Term]) -> torch.Tensor:
        """
        Get sub-indices (predicate, args) for each atom in a state in a single pass.
        Uses a unified term map for faster argument lookup by processing arguments together.

        Args:
            state: A list of Term objects representing the logical state.

        Returns:
            sub_index: Tensor (padding_atoms, max_arity + 1) with indices.
        """
        # --- State Length Check ---
        state_len = len(state)
        if state_len > self.padding_atoms:
             raise ValueError(f"Length of processed state ({state_len}) exceeds padding_atoms ({self.padding_atoms}).")

        # --- Initialize Tensor ---
        sub_index = torch.zeros(self.padding_atoms, self.max_arity + 1, device=self.device, dtype=torch.int64)

        # --- Single Pass for Indexing ---
        # Local references to maps for efficiency
        predicate_map = self.predicate_str2idx
        unified_map = self.unified_term_map # Use the new unified map

        for i, atom in enumerate(state):
            current_sub_row = sub_index[i]

            # --- Predicate Index ---
            pred_str = atom.predicate
            try:
                current_sub_row[0] = predicate_map[pred_str]
            except KeyError:
                 raise KeyError(f"Predicate '{pred_str}' not found in predicate map.")

            # --- Argument Indices ---
            num_args = len(atom.args)
            max_j = min(num_args, self.max_arity) # Process up to max_arity

            # Extract relevant argument strings
            arg_strings = atom.args[:max_j]

            # Get indices for all relevant argument strings using the unified map
            try:
                arg_indices = [unified_map[arg] for arg in arg_strings]
            except KeyError as e:
                 raise KeyError(f"Argument '{e}' not found in constant or variable maps.") from e

            # Assign argument indices to the tensor row
            if arg_indices: # Check if there are any arguments to assign
                current_sub_row[1:max_j + 1] = torch.tensor(arg_indices, device=self.device, dtype=torch.int64)

        return sub_index

    def subindices_to_terms(self, idx: torch.Tensor) -> List[List[Term]]:

        """
        Vectorized conversion of sub-indices back to Term-based states.

        Args:
            idx: Tensor of shape (B, M, padding_atoms, max_arity+1)
        Returns:
            List of B lists, each containing Terms for non-padding atoms.
        """
        B, M, P, A = idx.shape
        # Extract the first atom slot (predicate + up to two args)
        atoms = idx[..., 0, :].reshape(B * M, A)  # shape (B*M, A)

        # Treat index 0 as padding: drop any atom where any field == 0
        mask = (atoms != self.padding_idx).all(dim=1)  # shape (B*M,)

        # Filter valid entries
        valid = atoms[mask]  # shape (N, A)
        rel_idxs = valid[:, 0].tolist()
        head_idxs = valid[:, 1].tolist()
        tail_idxs = valid[:, 2].tolist()

        # Map indices to string labels
        preds = [self.predicate_idx2str[r] for r in rel_idxs]
        heads = [self.constant_idx2str[h] for h in head_idxs]
        tails = [self.constant_idx2str[t] for t in tail_idxs]

        # Build Term objects
        terms = [Term(predicate=p, args=(h, t))
                 for p, h, t in zip(preds, heads, tails)]
        # Assign back to batches
        flat_positions = mask.nonzero(as_tuple=False).squeeze(1).tolist()  # positions in [0, B*M)
        results: List[List[Term]] = [[] for _ in range(B)]
        for pos, term in zip(flat_positions, terms):
            batch_idx = pos // M
            results[batch_idx].append(term)
        return results






# --- Global Tensor Conversion Functions using IndexManager ---
def term_to_tensor_im(term_str: Term, index_manager: IndexManager, var_map: Dict[str, int]) -> torch.Tensor:
    pred_idx = index_manager.predicate_str2idx[term_str.predicate]
    arg_indices = []
    # Ensure args is a tuple
    term_args = term_str.args if isinstance(term_str.args, tuple) else tuple(term_str.args)

    for arg_s in term_args:
        if is_variable(arg_s): 
            if arg_s not in var_map:
                var_map[arg_s] = index_manager.get_next_var()
            arg_indices.append(var_map[arg_s])
        else: 
            arg_indices.append(index_manager.constant_str2idx[arg_s])
    
    while len(arg_indices) < index_manager.max_arity:
        arg_indices.append(index_manager.padding_idx)
    
    if len(arg_indices) != index_manager.max_arity: 
        raise ValueError(f"Term {term_str} (args: {arg_indices}) does not have effective arity {index_manager.max_arity} after padding. Current arity: {len(arg_indices)}")
    return torch.tensor([pred_idx] + arg_indices, dtype=torch.long, device=index_manager.device)

def state_to_tensor_im(state_str: List[Term], index_manager: IndexManager, initial_var_map: Optional[Dict[str, int]] = None) -> torch.Tensor:
    if not state_str:
        return torch.empty((0, index_manager.max_arity + 1), dtype=torch.long, device=index_manager.device)
    
    current_var_map = initial_var_map.copy() if initial_var_map is not None else {}
    if initial_var_map is None: 
        for term_obj in state_str:
            term_args = term_obj.args if isinstance(term_obj.args, tuple) else tuple(term_obj.args)
            for arg_s in term_args:
                if is_variable(arg_s) and arg_s not in current_var_map:
                    current_var_map[arg_s] = index_manager.get_next_var()
    
    return torch.stack([term_to_tensor_im(t, index_manager, current_var_map) for t in state_str])

def facts_to_tensor_im(facts_str: List[Term], index_manager: IndexManager) -> torch.Tensor:
    """
    Convert a list of facts (Term objects) to a tensor representation.
    Each fact is converted to a tensor of shape (max_arity + 1), where the first element is the predicate index
    and the subsequent elements are the indices of the arguments.
    If a fact has fewer arguments than max_arity, it is padded with the padding index.
    If no facts are provided, an empty tensor is returned.
    """
    print(f"\nConverting {facts_str}")
    if not facts_str:
        return torch.empty((0, index_manager.max_arity + 1), dtype=torch.long, device=index_manager.device)
    for fact in facts_str:
        term_args = fact.args if isinstance(fact.args, tuple) else tuple(fact.args)
        for arg in term_args:
            if is_variable(arg): 
                raise ValueError(f"Facts should be ground. Found variable {arg} in fact {fact}")
    return torch.stack([term_to_tensor_im(t, index_manager, {}) for t in facts_str]) 

def queries_to_tensor_im(queries_str: List[List[Term]], index_manager: IndexManager) -> torch.Tensor:
    """
    Convert a list of queries (each query is a list of Term objects) to a tensor representation.
    Each query is converted to a tensor of shape (max_arity + 1), where the first element is the predicate index
    and the subsequent elements are the indices of the arguments.
    If a query has fewer arguments than max_arity, it is padded with the padding index.
    If no queries are provided, an empty tensor is returned.
    """
    if not queries_str:
        return torch.empty((0, index_manager.max_arity + 1), dtype=torch.long, device=index_manager.device)
    
    return torch.stack([facts_to_tensor_im(query, index_manager) for query in queries_str])

def rules_to_tensor_im(rules_str_list: List[Rule], max_rule_atoms: int, index_manager: IndexManager) -> Tuple[torch.Tensor, torch.Tensor]:
    if not rules_str_list:
        return torch.empty((0, max_rule_atoms, index_manager.max_arity + 1), dtype=torch.long, device=index_manager.device), \
               torch.empty((0,), dtype=torch.long, device=index_manager.device)

    rule_tensors_list = []
    rule_lengths_list = []

    for rule_obj in rules_str_list:
        rule_local_var_map: Dict[str, int] = {} 
        head_tensor = term_to_tensor_im(rule_obj.head, index_manager, rule_local_var_map)
        body_tensors = [term_to_tensor_im(t, index_manager, rule_local_var_map) for t in rule_obj.body]
        
        current_rule_atoms_list = [head_tensor] + body_tensors
        num_atoms = len(current_rule_atoms_list)
        rule_lengths_list.append(num_atoms)

        # Ensure rule atoms are on the correct device before stacking
        current_rule_atoms_list_device = [atom.to(index_manager.device) for atom in current_rule_atoms_list]
        stacked_atoms = torch.stack(current_rule_atoms_list_device)

        if num_atoms < max_rule_atoms:
            padding = torch.full((max_rule_atoms - num_atoms, index_manager.max_arity + 1), index_manager.padding_idx, dtype=torch.long, device=index_manager.device)
            padded_rule_atoms = torch.cat([stacked_atoms, padding], dim=0)
        elif num_atoms > max_rule_atoms:
            raise ValueError(f"Rule {rule_obj} has {num_atoms} atoms, exceeding max_rule_atoms {max_rule_atoms}")
        else:
            padded_rule_atoms = stacked_atoms
        rule_tensors_list.append(padded_rule_atoms)
    
    if not rule_tensors_list: 
         return torch.empty((0, max_rule_atoms, index_manager.max_arity + 1), dtype=torch.long, device=index_manager.device), \
               torch.empty((0,), dtype=torch.long, device=index_manager.device)

    return torch.stack(rule_tensors_list), torch.tensor(rule_lengths_list, dtype=torch.long, device=index_manager.device)







def debug_print_atom(atom_tensor: torch.Tensor, index_manager: IndexManager) -> str: 
    if not isinstance(atom_tensor, torch.Tensor): 
        return f"<ERROR: Expected tensor, got {type(atom_tensor)}: {str(atom_tensor)}>"
    if atom_tensor.numel() == 0: return "[]" 
    if atom_tensor.dim() > 1:
        if atom_tensor.shape[0] == 1 and atom_tensor.shape[1] == (index_manager.max_arity + 1):
            atom_tensor = atom_tensor.squeeze(0) 
        else:
            return f"<ERROR: Atom tensor has unexpected shape {atom_tensor.shape}>"
    if atom_tensor.shape[0] != (index_manager.max_arity + 1) :
         return f"<ERROR: Atom tensor has wrong size {atom_tensor.shape[0]}, expected {index_manager.max_arity + 1}>"
    pred_idx_val = atom_tensor[0].item()
    if pred_idx_val == index_manager.padding_idx: 
        is_fully_padded = torch.all(atom_tensor == index_manager.padding_idx).item()
        if is_fully_padded: return "<PAD_ATOM>"
    if pred_idx_val == index_manager.true_pred_idx: return "TRUE" 
    if pred_idx_val == index_manager.false_pred_idx: return "FALSE" 
    if pred_idx_val == index_manager.predicate_str2idx.get('End', -999): return "END" 
    pred_str = index_manager.predicate_idx2str[pred_idx_val]
    args_str_list = [] 
    for i in range(1, index_manager.max_arity + 1): 
        arg_idx_val = atom_tensor[i].item()
        if arg_idx_val == index_manager.padding_idx and not any(atom_tensor[j].item() != index_manager.padding_idx for j in range(i + 1, index_manager.max_arity + 1)):
            break
        args_str_list.append(index_manager.get_str_for_term_idx(arg_idx_val))
    return f"{pred_str}({', '.join(args_str_list)})"

def debug_print_state_from_indices(state_tensor: torch.Tensor, index_manager: IndexManager, oneline: bool = False) -> str:
    if not isinstance(state_tensor, torch.Tensor): 
        return f"<ERROR: Expected tensor for state, got {type(state_tensor)}: {str(state_tensor)}>"
    if state_tensor.numel() == 0:
        return "[]"
    if state_tensor.dim() == 1 and state_tensor.shape[0] == (index_manager.max_arity + 1): 
        state_tensor_for_iteration = state_tensor.unsqueeze(0)
    elif state_tensor.dim() == 2 and state_tensor.shape[1] == (index_manager.max_arity + 1):
        state_tensor_for_iteration = state_tensor
    else:
        return f"<ERROR: State tensor has unexpected shape {state_tensor.shape}>"
    atom_strs_list = []
    for i in range(state_tensor_for_iteration.shape[0]):
        atom_tensor = state_tensor_for_iteration[i]
        if atom_tensor[0].item() == index_manager.padding_idx and torch.all(atom_tensor == index_manager.padding_idx).item():
            continue
        atom_strs_list.append(debug_print_atom(atom_tensor, index_manager))
    if not atom_strs_list: return "[]" 
    if oneline:
        return "[" + ", ".join(atom_strs_list) + "]"
    return "\n".join(atom_strs_list)

def debug_print_states_from_indices(states_list_tensor: List[torch.Tensor], index_manager: IndexManager) -> str:
    if not states_list_tensor:
        return "[]"
    return "[" + ", ".join([debug_print_state_from_indices(s, index_manager, oneline=True) for s in states_list_tensor]) + "]"






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