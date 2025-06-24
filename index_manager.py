from typing import List, Tuple, Dict, Set, Optional, NamedTuple, FrozenSet, Any
import itertools
import torch
from utils import is_variable, Term, Rule

class IndexManager():
    '''
    Manages indices for constants, predicates, and variables.
    - Rule template variables are automatically extracted and assigned fixed indices.
    - Runtime variables for proofs are assigned dynamically.
    - Includes a unified lookup map for fast term indexing.
    '''
    def __init__(self,
                 constants: set,
                 predicates: set,
                 rules: List[Rule],
                 max_total_vars: int,
                 padding_atoms: int = 10,
                 max_arity: int = 2,
                 device: torch.device = torch.device("cpu")):

        self.device = device
        self.constants = constants
        self.predicates = predicates
        
        # New: Automatically extract template variables from the provided rules
        self.rule_template_variables = self._extract_template_variables_from_rules(rules)
        
        self.special_preds_map_names = {'True': "##TRUE##", 'False': "##FALSE##", 'End': "##END##"} 
        self.special_preds = ['True', 'False', 'End']
        self.padding_idx = 0

        self.max_total_vars = max_total_vars
        self.padding_atoms = padding_atoms
        self.max_arity = max_arity
        if self.max_arity != 2:
            print(f"Warning: IndexManager max_arity is {self.max_arity}, but code assumes 2 for 3-element term tensors.")

        # --- Initialize Mappings ---
        self.constant_str2idx: Dict[str, int] = {}
        self.constant_idx2str: Dict[int, str] = {}
        self.predicate_str2idx: Dict[str, int] = {}
        self.predicate_idx2str: Dict[int, str] = {}
        self.template_variable_str2idx: Dict[str, int] = {}
        self.template_variable_idx2str: Dict[int, str] = {}
        self.runtime_variable_str2idx: Dict[str, int] = {}
        self.runtime_variable_idx2str: Dict[int, str] = {}
        self.unified_term_map: Dict[str, int] = {}

        # --- Create Indices ---
        self.create_global_idx()
        self._create_template_var_idx()

        self.runtime_var_start_index = self.constant_no + self.template_variable_no + 1
        self.runtime_var_end_index = self.runtime_var_start_index + self.max_total_vars - 1
        self.runtime_variable_no = self.max_total_vars
        self.variable_no = self.constant_no + self.template_variable_no + self.runtime_variable_no
        self.next_runtime_var_index = self.runtime_var_start_index

        self._create_fixed_runtime_var_idx()

        # --- Create the Unified Term Map ---
        self.unified_term_map.update(self.constant_str2idx)
        self.unified_term_map.update(self.template_variable_str2idx)
        print(f"IndexManager initialized. Found {len(self.rule_template_variables)} template variables.")
        
        self.true_pred_idx = self.predicate_str2idx['True']
        self.false_pred_idx = self.predicate_str2idx['False']
        
        self.true_tensor = torch.tensor([[self.true_pred_idx, self.padding_idx, self.padding_idx]], dtype=torch.long, device=self.device)
        self.false_tensor = torch.tensor([[self.false_pred_idx, self.padding_idx, self.padding_idx]], dtype=torch.long, device=self.device)

        # --- High-Performance Indices ---
        self.fact_index: Dict[Tuple, Set[Tuple[int, ...]]] = {}
        # NEW: Rule index mapping predicate_idx -> list of rule indices
        self.rule_index: Dict[int, List[int]] = {}

    # New internal function to encapsulate the extraction logic
    def _extract_template_variables_from_rules(self, rules: List[Rule]) -> Set[str]:
        """Iterates through a list of Rule objects and extracts all unique variable names."""
        template_variables = set()
        for rule in rules:
            # Extract from the rule's head
            for arg in rule.head.args:
                if is_variable(arg):
                    template_variables.add(arg)
            # Extract from the rule's body
            for body_atom in rule.body:
                for arg in body_atom.args:
                    if is_variable(arg):
                        template_variables.add(arg)
        return template_variables


    def build_facts_index(self, facts_tensor: torch.Tensor):
        """
        Builds an index for facts to allow efficient retrieval for non-ground queries.
        This mirrors the logic of the original string-based implementation.
        Args:
            facts_tensor (torch.Tensor): A tensor of shape (num_facts, max_arity + 1)
                where each row represents a fact with the first element as the predicate index
                and the subsequent elements as argument indices.
        Creates a dictionary where:
            - Keys are tuples of the form (predicate_idx, (arg_pos1, arg_idx1), (arg_pos2, arg_idx2), ...)
            - Values are sets of tuples representing the full fact in tensor form.
        This allows for efficient retrieval of facts based on partial argument matches
        """
        self.fact_index.clear()
        for i in range(facts_tensor.shape[0]):
            fact = facts_tensor[i]
            fact_tuple = tuple(fact.tolist())
            constant_args_with_pos = []
            for j in range(1, self.max_arity + 1):
                arg_idx = fact[j].item()
                if not self.is_var_idx(arg_idx) and arg_idx != self.padding_idx:
                    constant_args_with_pos.append((j - 1, arg_idx))

            # Index all possible combinations of constants
            for k in range(len(constant_args_with_pos) + 1):
                for subset_args_with_pos in itertools.combinations(constant_args_with_pos, k):
                    sorted_subset = tuple(sorted(subset_args_with_pos, key=lambda x: x[0]))
                    key = (fact[0].item(),) + sorted_subset
                    if key not in self.fact_index:
                        self.fact_index[key] = set()
                    self.fact_index[key].add(fact_tuple)

    def build_rule_index(self, rules_tensor: torch.Tensor):
        """NEW: Builds an index for rules based on their head predicate."""
        self.rule_index.clear()
        for i in range(rules_tensor.shape[0]):
            pred_idx = rules_tensor[i, 0, 0].item()
            if pred_idx not in self.rule_index: self.rule_index[pred_idx] = []
            self.rule_index[pred_idx].append(i)

    def is_var_idx(self, idx: int) -> bool:
        '''Check if the given index corresponds to any variable (template or runtime).'''
        return (self.constant_no < idx <= self.runtime_var_end_index)

    def get_next_var(self) -> int:
        '''Get the next available *runtime* variable idx.'''
        if self.next_runtime_var_index > self.runtime_var_end_index:
            raise ValueError(f"No more available runtime variable indices.")
        idx = self.next_runtime_var_index
        var_name = f"RuntimeVar_{idx}"
        self.runtime_variable_idx2str[idx] = var_name
        self.runtime_variable_str2idx[var_name] = idx
        self.next_runtime_var_index += 1
        return idx

    def create_global_idx(self):
        '''Create global indices for constants and predicates.'''
        # Constants
        current_idx = 1
        for term in sorted(self.constants):
            self.constant_str2idx[term] = current_idx
            self.constant_idx2str[current_idx] = term
            current_idx += 1
        self.constant_no = current_idx - 1
        # Predicates
        current_idx = 1
        for term in sorted(self.predicates):
            self.predicate_str2idx[term] = current_idx
            self.predicate_idx2str[current_idx] = term
            current_idx += 1
        for term in self.special_preds:
            self.predicate_str2idx[term] = current_idx
            self.predicate_idx2str[current_idx] = term
            current_idx += 1
        self.predicate_no = current_idx - 1

    def _create_template_var_idx(self):
        '''Create fixed indices for variables found in rule templates.'''
        current_idx = self.constant_no + 1
        for var_name in sorted(list(self.rule_template_variables)):
            self.template_variable_str2idx[var_name] = current_idx
            self.template_variable_idx2str[current_idx] = var_name
            current_idx += 1
        self.template_variable_no = len(self.rule_template_variables)

    def _create_fixed_runtime_var_idx(self):
        '''Pre-allocate space for runtime variable indices.'''
        for i in range(self.max_total_vars):
            var_index = self.runtime_var_start_index + i
            var_name = f"RuntimeVar_{var_index}"
            self.runtime_variable_idx2str[var_index] = var_name
            self.runtime_variable_str2idx[var_name] = var_index
            
    def get_str_for_term_idx(self, idx: int) -> str:
        '''Get the string representation for a given index.'''
        if idx in self.constant_idx2str:
            return self.constant_idx2str[idx]
        elif idx in self.template_variable_idx2str:
            return self.template_variable_idx2str[idx]
        elif idx in self.runtime_variable_idx2str:
            return self.runtime_variable_idx2str[idx]
        elif idx in self.predicate_idx2str:
            return self.predicate_idx2str[idx]
        elif idx == self.padding_idx:
            return "<PAD>"
        else:
            raise KeyError(f"Index {idx} not found in any mapping.")

    def term_to_tensor(self, term_str: Term) -> torch.Tensor:
        """Converts a single Term object to a tensor using the pre-built unified map."""
        pred_idx = self.predicate_str2idx[term_str.predicate]
        arg_indices = []
        term_args = term_str.args if isinstance(term_str.args, tuple) else tuple(term_str.args)
        for arg_s in term_args:
            # Handle runtime variables that might not be in the initial unified_term_map
            if is_variable(arg_s) and arg_s not in self.unified_term_map:
                 # A simple way to handle new variables on the fly
                if arg_s not in self.runtime_variable_str2idx:
                    self.get_next_var() # This will assign a new index
                    self.runtime_variable_str2idx[arg_s] = self.next_runtime_var_index -1
                    self.runtime_variable_idx2str[self.next_runtime_var_index-1] = arg_s
                arg_indices.append(self.runtime_variable_str2idx[arg_s])
            else:
                 arg_indices.append(self.unified_term_map[arg_s])
        while len(arg_indices) < self.max_arity:
            arg_indices.append(self.padding_idx)
        return torch.tensor([pred_idx] + arg_indices, dtype=torch.long, device=self.device)

    def state_to_tensor(self, state_str: List[Term]) -> torch.Tensor:
        """Converts a list of Term objects (like facts or queries) to a tensor."""
        if not state_str:
            return torch.empty((0, self.max_arity + 1), dtype=torch.long, device=self.device)
        return torch.stack([self.term_to_tensor(t) for t in state_str])

    def rules_to_tensor(self, rules_str_list: List[Rule], max_rule_atoms: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts a list of Rule objects to a padded tensor."""
        if not rules_str_list:
            return torch.empty((0, max_rule_atoms, self.max_arity + 1), dtype=torch.long, device=self.device), \
                   torch.empty((0,), dtype=torch.long, device=self.device)
        rule_tensors_list = []
        rule_lengths_list = []
        for rule_obj in rules_str_list:
            head_tensor = self.term_to_tensor(rule_obj.head)
            body_tensors = [self.term_to_tensor(t) for t in rule_obj.body]
            current_rule_atoms_list = [head_tensor] + body_tensors
            num_atoms = len(current_rule_atoms_list)
            rule_lengths_list.append(num_atoms)
            stacked_atoms = torch.stack(current_rule_atoms_list)
            if num_atoms < max_rule_atoms:
                padding = torch.full((max_rule_atoms - num_atoms, self.max_arity + 1), self.padding_idx, dtype=torch.long, device=self.device)
                padded_rule_atoms = torch.cat([stacked_atoms, padding], dim=0)
            elif num_atoms > max_rule_atoms:
                raise ValueError(f"Rule {rule_obj} has {num_atoms} atoms, exceeding max_rule_atoms {max_rule_atoms}")
            else:
                padded_rule_atoms = stacked_atoms
            rule_tensors_list.append(padded_rule_atoms)
        if not rule_tensors_list: 
            return torch.empty((0, max_rule_atoms, self.max_arity + 1), dtype=torch.long, device=self.device), \
                   torch.empty((0,), dtype=torch.long, device=self.device)
        return torch.stack(rule_tensors_list), torch.tensor(rule_lengths_list, dtype=torch.long, device=self.device)



    def debug_print_atom(self, atom_tensor: torch.Tensor) -> str: 
        pred_idx_val = atom_tensor[0].item()
        pred_str = self.predicate_idx2str.get(pred_idx_val, f"<UnkPred:{pred_idx_val}>")
        args_str_list = [] 
        for i in range(1, self.max_arity + 1): 
            arg_idx_val = atom_tensor[i].item()
            if arg_idx_val == self.padding_idx:
                break
            args_str_list.append(self.get_str_for_term_idx(arg_idx_val))

        # Handle special predicates that might have no args printed
        if pred_str in self.special_preds:
            return pred_str

        return f"{pred_str}({', '.join(args_str_list)})"

    def debug_print_state_from_indices(self, state_tensor: torch.Tensor, oneline: bool = False) -> str:
        if state_tensor.numel() == 0: return "[]"
        atom_strs_list = [self.debug_print_atom(atom_tensor) for atom_tensor in state_tensor if atom_tensor[0].item() != self.padding_idx]
        if not atom_strs_list: return "[]"
        sep = ", " if oneline else "\n"
        return f"[{sep.join(atom_strs_list)}]" if oneline else sep.join(atom_strs_list)
        
    def debug_print_states_from_indices(self, states_list_tensor: List[torch.Tensor]) -> str:
        return "[" + ", ".join([self.debug_print_state_from_indices(s, oneline=True) for s in states_list_tensor]) + "]"
