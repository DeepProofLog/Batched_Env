from typing import List, Optional, Tuple, Dict, Union, Set
import itertools
import random
from utils import Term, is_variable, extract_var
import torch

class IndexManager():

    def __init__(self, 
                constants: set, 
                predicates: set,
                variables:set,
                constant_no: int,
                predicate_no: int,
                variable_no: int,
                rules: List,
                constants_images: set = (),
                constant_images_no: int = 0,
                rule_depend_var: bool = True,
                padding_atoms: int = 10,
                max_arity: int = 2,
                device: torch.device = torch.device("cpu")):
        
        self.device = device
        self.constants = constants
        self.predicates = predicates
        self.variables = variables # if rule_depend_var is the number of variables in the rules, if not, it is set in the runner
        self.constant_no = constant_no
        self.variable_no = variable_no
        self.predicate_no = predicate_no

        self.constants_images = constants_images
        self.constant_images_no = constant_images_no

        self.predicate_true_offset = 1
        self.predicate_false_offset = 2
        self.predicate_end_offset = 3
        self.predicate_special_indices = {
             'True': self.predicate_no + self.predicate_true_offset,
             'False': self.predicate_no + self.predicate_false_offset,
             'End': self.predicate_no + self.predicate_end_offset,
        }
        self.rules = rules
        self.rule_depend_var = rule_depend_var # If True, the variables are dependent on the rules, if False, the variables are set in the runner
        self.padding_atoms = padding_atoms  # Maximum number of atoms in a state
        self.max_arity = max_arity # Maximum arity of the predicates

        # LOCAL INDEXES
        self.atom_to_index = {} # Map atom to index
        self.atom_id_to_sub_id = {} # Map atom index to sub-indices of predicates and arguments
        self.next_atom_index = 1  # Next available index. 0 is reserved for padding
        if not self.rule_depend_var:
            self.variable_str2idx = {} # Map variable to index
            self.next_var_index = constant_no+1 # Next available index. 0 is reserved for padding

        self.create_global_idx()
        self.fact_index: Dict[Tuple, Set[Term]] = {}
        if self.rule_depend_var:
            self.rule_features_vars()
        # self.rules_vars = {}
        # for rule in self.rules:
        #     for atom in rule.head + rule.body:
        #         for arg in atom.args:
        #             if is_variable(arg):# add variables to the set
        #                 self.rules_vars.add(arg)

    def create_global_idx(self):
        '''Create a global index for a list of terms. Start idx counting from 1
        If there are images, reserve the first indexes for the images'''
        if self.constant_images_no>0:
            constants_wout_images = [const for const in self.constants if const not in self.constants_images]
            self.constant_str2idx = {term: i + 1 for i, term in enumerate(sorted(self.constants_images))}
            self.constant_str2idx.update({term: i + 1 + self.constant_images_no for i, term in enumerate(sorted(constants_wout_images))})
            self.constant_idx2str = {i + 1: term for i, term in enumerate(sorted(self.constants_images))}
            self.constant_idx2str.update({i + 1 + self.constant_images_no: term for i, term in enumerate(sorted(constants_wout_images))})
        else:
            self.constant_str2idx = {term: i + 1 for i, term in enumerate(sorted(self.constants))}
            self.constant_idx2str = {i + 1: term for i, term in enumerate(sorted(self.constants))}

        self.predicate_str2idx = {term: i + 1 for i, term in enumerate(sorted(self.predicates))}
        self.predicate_idx2str = {i + 1: term for i, term in enumerate(sorted(self.predicates))}

        if self.rule_depend_var:
            self.variable_str2idx = {term: i + 1 + self.constant_no for i, term in enumerate(sorted(self.variables))}
            self.variable_idx2str = {i + 1 + self.constant_no: term for i, term in enumerate(sorted(self.variables))}


    def rule_features_vars(self):
        """Create a dictionary with the features (body predicates) and variables of the rules"""
        self.rule_feats_vars = {}
        for i in range(len(self.rules)):
            rule = self.rules[i]
            if rule.head.predicate not in self.rule_feats_vars:
                self.rule_feats_vars[rule.head.predicate] = [f'RULE{i}_{arg}' for arg in rule.head.args]
            feature = ""
            vars_list = []
            for atom in rule.body:
                feature = feature+atom.predicate
                vars_list.append([f'RULE{i}_{arg}' for arg in atom.args])
            self.rule_feats_vars[feature] = vars_list

    def reset_atom(self):
        '''Reset the atom and variable dicts and indices'''
        self.atom_to_index.clear()
        self.atom_id_to_sub_id.clear()
        self.next_atom_index = 1
        if not self.rule_depend_var:
            self.variable_str2idx.clear()
            self.next_var_index = self.constant_no+1

    def substitute_variables(self, state: List[Term]) -> List[Term]:
        """Substitute variables in a state with variables defined in the rule"""
        if not ((len(state) == 1 and (state[0].predicate == 'True' or state[0].predicate == 'False')) or (not extract_var(",".join(str(s) for s in state)))):
            state_feat = "".join(atom.predicate for atom in state)
            assert state_feat in self.rule_feats_vars, f"State feature not in rule_feats_vars: {state_feat}"
            for i in range(len(state)):
                atom = state[i]
                for j in range(len(atom.args)):
                    if is_variable(atom.args[j]):
                        atom.args[j] = self.rule_feats_vars[state_feat][i][j]
        return state


    # def get_atom_sub_index(self, state: List[Term]) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Get the atom and sub index for a state. (Optimized Version)

    #     Args:
    #         state: A list of Term objects representing the current logical state.

    #     Returns:
    #         A tuple containing:
    #         - atom_index: Tensor (padding_atoms,) with the unique ID for each atom.
    #         - sub_index: Tensor (padding_atoms, max_arity + 1) with predicate
    #                      and argument indices for each atom.
    #     """
    #     state_len = len(state)
    #     if state_len > self.padding_atoms:
    #          raise ValueError(f"Length of state ({state_len}) exceeds padding_atoms ({self.padding_atoms}).")

    #     # --- Apply substitution step first if variables are rule-dependent ---
    #     if self.rule_depend_var:
    #          # substitute_variables returns a new list if changes occurred
    #          state = self.substitute_variables(state)
    #          # Update state_len if truncation happened within substitute (unlikely)
    #          state_len = len(state)
    #          if state_len > self.padding_atoms: # Check again if substitution changed length
    #              raise ValueError(f"Length of state ({state_len}) after substitution exceeds padding_atoms ({self.padding_atoms}).")

    #     # --- Handle dynamic variables if not rule-dependent ---
    #     else:
    #         for atom in state:
    #             for arg in atom.args:
    #                 assert arg not in {'True', 'False', 'End'}, f"Variable '{arg}' is a special predicate!!."
    #                 # Check if it's a variable *and* not already known *and* not special
    #                 if is_variable(arg) and \
    #                 arg not in self.variable_str2idx and \
    #                 arg not in self.predicate_special_indices:

    #                     if self.next_var_index > self.constant_no + self.variable_no:
    #                         raise ValueError(f"Exceeded the maximum number of variables ({self.variable_no}). Var: {arg}")
    #                     index = self.next_var_index
    #                     self.variable_str2idx[arg] = index
    #                     self.next_var_index += 1

    #     # --- Pre-allocate tensors ---
    #     atom_index = torch.zeros(self.padding_atoms, device=self.device, dtype=torch.int64)
    #     sub_index = torch.zeros(self.padding_atoms, self.max_arity + 1, device=self.device, dtype=torch.int64)

    #     # --- Local references to dictionaries for potential micro-optimization ---
    #     atom_to_idx_cache_get = self.atom_to_index.get
    #     atom_id_to_sub_id_cache = self.atom_id_to_sub_id
    #     predicate_specials = self.predicate_special_indices
    #     predicate_map = self.predicate_str2idx
    #     variable_map = self.variable_str2idx # Contains rule vars or dynamic vars
    #     constant_map = self.constant_str2idx

    #     for i in range(state_len): # Iterate only up to the actual state length
    #          atom = state[i]
    #          # 1. Check cache for the atom object itself using .get for efficiency
    #          cached_atom_id = atom_to_idx_cache_get(atom)

    #          if cached_atom_id is not None:
    #              atom_index[i] = cached_atom_id
    #              sub_index[i] = atom_id_to_sub_id_cache[cached_atom_id]
    #          else:
    #              new_atom_id = self.next_atom_index
    #              self.atom_to_index[atom] = new_atom_id
    #              atom_index[i] = new_atom_id
    #              self.next_atom_index += 1

    #              current_sub_row = sub_index[i]

    #              # --- Predicate Index ---
    #              pred_idx = predicate_specials.get(atom.predicate)
    #              if pred_idx is None:
    #                   try:
    #                       pred_idx = predicate_map[atom.predicate]
    #                   except KeyError:
    #                       raise KeyError(f"Predicate '{atom.predicate}' not found in predicate_str2idx. Atom: {atom}") from None
    #              current_sub_row[0] = pred_idx

    #              # --- Argument Indexing ---
    #              try:
    #                  num_args = len(atom.args)
    #                  max_j = min(num_args, self.max_arity) # Process up to max_arity args
    #                  for j in range(max_j):
    #                       arg = atom.args[j]
    #                       if is_variable(arg):
    #                           current_sub_row[j + 1] = variable_map[arg]
    #                       else:
    #                           current_sub_row[j + 1] = constant_map[arg]
    #                  # Remaining arg indices (if num_args < max_arity) remain 0 (padding)
    #              except KeyError as e:
    #                  missing_type = "variable" if is_variable(arg) else "constant"
    #                  raise KeyError(f"Argument '{e}' (type: {missing_type}) not found in index maps. Atom: {atom}") from e

    #              atom_id_to_sub_id_cache[new_atom_id] = current_sub_row

    #     return atom_index, sub_index

    def get_atom_sub_index(self, state: List[Term]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the atom and sub index for a state"""
        if self.rule_depend_var:
            state = self.substitute_variables(state)
        else:
            #Get variables
            full_state = ",".join(str(s) for s in state)
            vars = extract_var(full_state)
            for var in vars:
                if (var != "True") and (var != "False") and (var!= "End") and (var not in self.variable_str2idx):
                    if self.next_var_index > self.constant_no + self.variable_no:
                        raise ValueError(f"Exceeded the maximum number of variables: {self.variable_no}")
                    else:
                        index = self.next_var_index
                        self.variable_str2idx[var] = index
                        self.next_var_index += 1

        # Get atom_index and sub_index
        atom_index = torch.zeros(self.padding_atoms, device=self.device, dtype=torch.int64)
        sub_index = torch.zeros(self.padding_atoms, self.max_arity+1, device=self.device, dtype=torch.int64)
        assert len(state) <= self.padding_atoms, f"Length of state: {len(state)} is greater than padding_atoms: {self.padding_atoms}"
        for i, atom in enumerate(state):
            if atom not in self.atom_to_index:
                self.atom_to_index[atom] = self.next_atom_index
                atom_index[i] = self.next_atom_index
                self.next_atom_index += 1
            else:
                atom_index[i] = self.atom_to_index[atom]
            
            atom_id = atom_index[i].item()
            if atom_id not in self.atom_id_to_sub_id:
                try:
                    if atom.predicate == 'True':
                        sub_index[i, 0] = self.predicate_no + self.predicate_true_offset
                    elif atom.predicate == 'False':
                        sub_index[i, 0] = self.predicate_no + self.predicate_false_offset
                    elif atom.predicate == 'End':
                        sub_index[i, 0] = self.predicate_no + self.predicate_end_offset
                    else:
                        sub_index[i, 0] = self.predicate_str2idx[atom.predicate]
                    
                    for j, arg in enumerate(atom.args):
                        if is_variable(arg):
                            sub_index[i, j+1] = self.variable_str2idx[arg]
                        else:
                            sub_index[i, j+1] = self.constant_str2idx[arg]
                        
                except Exception as e:
                    print("The following key is not in dict:", e)
                
                self.atom_id_to_sub_id[atom_id] = sub_index[i]
            else:
                sub_index[i] = self.atom_id_to_sub_id[atom_id]

        return atom_index, sub_index 
    
    def build_fact_index(self, facts: List[Term]) -> Dict[Tuple, Set[Term]]:
        """
        Builds an index for facts based on predicate and constant arguments.

        Keys are tuples: (predicate, (pos1, val1), (pos2, val2), ...), where
        (pos, val) are the positions and values of constant arguments.
        Values are sets of facts matching the key pattern.
        """

        for fact in facts:
            predicate = fact.predicate
            args = fact.args
            constant_args_with_pos = [(i, arg) for i, arg in enumerate(args) if not is_variable(arg)]

            # Generate all combinations of constant arguments (subsets)
            for i in range(len(constant_args_with_pos) + 1):
                for subset_args_with_pos in itertools.combinations(constant_args_with_pos, i):
                    # Ensure subset_args_with_pos is sorted by position for canonical keys
                    sorted_subset = tuple(sorted(subset_args_with_pos, key=lambda x: x[0]))
                    key = (predicate,) + sorted_subset

                    if key not in self.fact_index:
                        self.fact_index[key] = set()
                    self.fact_index[key].add(fact)