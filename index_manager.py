from typing import List, Optional, Tuple, Dict, Union, Set
import itertools
import random
from utils import Term, is_variable, extract_var
import torch
class IndexManager():

    def __init__(self,
                 constants: set,
                 predicates: set,
                 variables: set, # Variables defined globally or by rules
                 constant_no: int,
                 predicate_no: int, # Number of *regular* predicates
                 variable_no: int, # Max allowed *total* vars (rule + dynamic)
                 rules: List, # Define Rule type if needed
                 constants_images: set = (),
                 constant_images_no: int = 0,
                 rule_depend_var: bool = True,
                 padding_atoms: int = 10,
                 max_arity: int = 2,
                 device: torch.device = torch.device("cpu")):

        self.device = device
        self.constants = constants
        self.predicates = predicates # Regular predicates
        self.variables = variables # Initial set of variables (from rules if rule_depend_var)
        self.constant_no = constant_no
        # Store the count of *regular* predicates before adding specials
        self.regular_predicate_count = predicate_no
        self.variable_no = variable_no # Max *total* variables allowed
        self.rules = rules
        self.constants_images = constants_images
        self.constant_images_no = constant_images_no
        self.rule_depend_var = rule_depend_var
        self.padding_atoms = padding_atoms
        self.max_arity = max_arity

        # --- Initialize Mappings ---
        self.constant_str2idx: Dict[str, int] = {}
        self.constant_idx2str: Dict[int, str] = {}
        self.predicate_str2idx: Dict[str, int] = {}
        self.predicate_idx2str: Dict[int, str] = {}
        self.variable_str2idx: Dict[str, int] = {}
        self.variable_idx2str: Dict[int, str] = {}

        # --- Create Mappings (including special predicates) ---
        self.create_global_idx() # Now handles special predicates

        # --- Variable Indexing Setup ---
        if not self.rule_depend_var:
            # Start indexing dynamic variables *after* constants
            # The initial self.variables set is ignored in this mode
            self.variable_str2idx.clear()
            self.variable_idx2str.clear()
            self.next_var_index = self.constant_no + 1
        else:
            # Pre-populate variables from the initial set (assumed from rules)
            # Indices start after constants
            for i, term in enumerate(sorted(self.variables)):
                 idx = self.constant_no + i + 1
                 self.variable_str2idx[term] = idx
                 self.variable_idx2str[idx] = term
            # Rule features might add more variables later in rule_features_vars
            self.rule_features_vars() # Call after initial setup

        # Fact index setup (if used)
        self.fact_index: Dict[Tuple, Set[Term]] = {}

    def create_global_idx(self):
        '''Create global indices for constants and predicates (including specials).'''
        # --- Constants ---
        current_idx = 1
        if self.constant_images_no > 0:
            constants_wout_images = sorted([c for c in self.constants if c not in self.constants_images])
            img_constants = sorted(list(self.constants_images))
            for term in img_constants:
                self.constant_str2idx[term] = current_idx
                self.constant_idx2str[current_idx] = term
                current_idx += 1
            for term in constants_wout_images:
                 self.constant_str2idx[term] = current_idx
                 self.constant_idx2str[current_idx] = term
                 current_idx += 1
        else:
            for term in sorted(self.constants):
                self.constant_str2idx[term] = current_idx
                self.constant_idx2str[current_idx] = term
                current_idx += 1

        # --- Predicates (Regular + Special) ---
        current_idx = 1
        # Add regular predicates first
        for term in sorted(self.predicates):
            self.predicate_str2idx[term] = current_idx
            self.predicate_idx2str[current_idx] = term
            current_idx += 1

        # Check if the provided predicate_no matches the actual count
        if current_idx - 1 != self.regular_predicate_count:
             print(f"Warning: Initial predicate_no ({self.regular_predicate_count}) does not match count of predicates in set ({current_idx - 1}). Using actual count.")
             self.regular_predicate_count = current_idx - 1

        # Add special predicates with indices starting after regular ones
        special_preds = ['True', 'False', 'End']
        for term in special_preds:
             # Ensure they don't clash with regular predicates
             if term in self.predicate_str2idx:
                  raise ValueError(f"Special predicate '{term}' conflicts with a regular predicate name!")
             self.predicate_str2idx[term] = current_idx
             self.predicate_idx2str[current_idx] = term
             current_idx += 1

        # Update total predicate count if needed (though not strictly necessary for logic)
        self.total_predicate_count = current_idx - 1


    def rule_features_vars(self):
        """
        Create rule features and ensure all rule variables are in the index.
        (Assumes self.rules is populated with objects having .head, .body, .predicate, .args)
        """
        if not self.rule_depend_var:
             return # Only run if variables depend on rules

        self.rule_feats_vars = {}
        # Example Rule structure (replace with yours):
        # class Rule:
        #    def __init__(self, head: Term, body: List[Term]): self.head = head; self.body = body

        for i, rule in enumerate(self.rules):
            rule_vars_map = {} # Maps original var name ('X') to unique rule var name ('RULEi_VAR_X')

            def get_or_create_rule_var(arg):
                if is_variable(arg):
                    unique_var_name = f'RULE{i}_VAR_{arg}'
                    rule_vars_map[arg] = unique_var_name # Store mapping for substitution
                    # Add to global variable index if not already present
                    if unique_var_name not in self.variable_str2idx:
                         # Variables indices start *after* constants
                         new_index = self.constant_no + len(self.variable_str2idx) + 1
                         # Check against the *total* variable limit
                         if len(self.variable_str2idx) >= self.variable_no:
                              raise ValueError(f"Exceeded maximum variable capacity ({self.variable_no}) while processing rule variables.")
                         self.variable_str2idx[unique_var_name] = new_index
                         self.variable_idx2str[new_index] = unique_var_name
                    return unique_var_name
                return arg # It's a constant

            # Process head and body to identify and index all variables
            head_args_subs = [get_or_create_rule_var(arg) for arg in rule.head.args]

            feature = ""
            body_vars_list_subs = []
            for atom in rule.body:
                feature += atom.predicate
                substituted_args = [get_or_create_rule_var(arg) for arg in atom.args]
                body_vars_list_subs.append(substituted_args)

            # Store info needed for substitution
            if feature not in self.rule_feats_vars:
                 self.rule_feats_vars[feature] = [] # Store list of rules matching this feature

            self.rule_feats_vars[feature].append({
                'rule_index': i,
                'var_map': rule_vars_map, # Original ('X') -> Unique ('RULEi_VAR_X')
                'body_structure': body_vars_list_subs # Structure with unique names
            })


    def reset_atom(self):
        '''Reset dynamic variable dicts and indices if not rule dependent'''
        if not self.rule_depend_var:
            # Reset only dynamic variables
            self.variable_str2idx.clear()
            self.variable_idx2str.clear()
            self.next_var_index = self.constant_no + 1 # Reset counter

    def substitute_variables(self, state: List[Term]) -> List[Term]:
        """
        Substitute variables in a state based on matching rule body features.
        Returns a *new* list of Term objects with substituted variable names.
        Handles cases where multiple rules might match a feature.
        """
        # Substitution only applies if rule_depend_var is True
        if not self.rule_depend_var:
            return state

        # Check if the state is simple True/False or ground (no variables)
        is_special_or_ground = (len(state) == 1 and state[0].predicate in {'True', 'False'}) or \
                              not any(is_variable(arg) for atom in state for arg in atom.args)
        if is_special_or_ground:
             return state

        state_feat = "".join(atom.predicate for atom in state)

        if state_feat not in self.rule_feats_vars:
             # This state doesn't match the body feature of any rule
             # print(f"Debug: State feature '{state_feat}' not found in rule_feats_vars. No substitution.")
             return state # Return original state

        matching_rules_info = self.rule_feats_vars[state_feat]

        # --- Find the specific rule that matches the state structure ---
        # This assumes the state corresponds to *one* specific rule body instance.
        # If multiple rules can have the same predicate sequence (feature),
        # we need a way to distinguish which rule's variables to use.
        # For now, let's assume the first match is the correct one,
        # or that features are unique per rule structure.
        # A more robust approach might require passing the rule context.
        found_match = False
        rule_info = None
        for r_info in matching_rules_info:
             # Check if number of atoms and arity match
             if len(state) == len(r_info['body_structure']):
                  structure_match = True
                  for i, atom in enumerate(state):
                       if len(atom.args) != len(r_info['body_structure'][i]):
                            structure_match = False
                            break
                  if structure_match:
                       rule_info = r_info
                       found_match = True
                       break # Use the first structurally matching rule

        if not found_match:
             print(f"Warning: State feature '{state_feat}' found, but state structure doesn't match any known rule body for that feature. State: {state}")
             return state # Return original state if no structural match

        # --- Perform Substitution using the matched rule's variable map ---
        var_map = rule_info['var_map'] # Original var ('X') -> Unique rule var ('RULEi_VAR_X')
        new_state = []
        for i, atom in enumerate(state):
             new_args = []
             for arg in atom.args:
                 if is_variable(arg):
                     # Substitute using the specific rule's map
                     substituted_arg = var_map.get(arg)
                     if substituted_arg is None:
                         # This shouldn't happen if rule_features_vars was correct
                         print(f"Critical Warning: Variable '{arg}' in state atom {atom} not found in matched rule map (Rule {rule_info['rule_index']}) for feature '{state_feat}'. Using original.")
                         new_args.append(arg)
                     else:
                         new_args.append(substituted_arg)
                 else:
                     new_args.append(arg) # Keep constants
             new_state.append(Term(atom.predicate, new_args))

        return new_state


    def get_atom_sub_index(self, state: List[Term]) -> torch.Tensor:
        """
        Get sub-indices (predicate, args) for each atom in a state in a single pass.
        Handles dynamic variable indexing if rule_depend_var is False.
        Uses unified predicate mapping.

        Args:
            state: A list of Term objects representing the logical state.

        Returns:
            sub_index: Tensor (padding_atoms, max_arity + 1) with indices.
        """
        # --- 1. Pre-processing (Variable Substitution if needed) ---
        # This step happens *before* the single pass, preparing the state.
        if self.rule_depend_var:
            state = self.substitute_variables(state) # Returns a potentially new list

        # --- 2. State Length Check ---
        state_len = len(state)
        if state_len > self.padding_atoms:
             raise ValueError(f"Length of processed state ({state_len}) exceeds padding_atoms ({self.padding_atoms}).")

        # --- 3. Initialize Tensor ---
        sub_index = torch.zeros(self.padding_atoms, self.max_arity + 1, device=self.device, dtype=torch.int64)

        # --- 4. Single Pass for Indexing and Dynamic Variable Update ---
        # Local references to maps for efficiency
        predicate_map = self.predicate_str2idx
        constant_map = self.constant_str2idx
        variable_map = self.variable_str2idx # Use this for lookups

        for i, atom in enumerate(state):
            current_sub_row = sub_index[i] # Reference to the tensor row

            # --- Predicate Index ---
            pred_str = atom.predicate
            current_sub_row[0] = predicate_map[pred_str]

            # --- Argument Indices & Dynamic Variable Handling ---
            num_args = len(atom.args)
            max_j = min(num_args, self.max_arity) # Process up to max_arity

            for j in range(max_j):
                arg = atom.args[j]

                if is_variable(arg):
                    # --- Variable Argument ---
                    var_idx = variable_map.get(arg)
                    if var_idx is not None:
                        # Variable already known (either from rules or previously seen dynamic)
                        current_sub_row[j + 1] = var_idx
                    else:
                        # Variable not found
                        if not self.rule_depend_var:
                            # --- Dynamic Variable Case ---
                            # Check limit against *total* number of variables seen so far
                            # Current count = len(variable_map)
                            # Index starts after constants (constant_no + 1)
                            # Max index allowed = constant_no + variable_no
                            if self.next_var_index > self.constant_no + self.variable_no:
                                raise ValueError(f"Exceeded maximum variable capacity ({self.variable_no}). Cannot add new variable '{arg}'.")

                            new_var_idx = self.next_var_index
                            variable_map[arg] = new_var_idx # Add to map *during* the pass
                            # self.variable_idx2str[new_var_idx] = arg # Optional: update reverse map
                            current_sub_row[j + 1] = new_var_idx
                            self.next_var_index += 1 # Increment for the *next* new variable
                        else:
                            # --- Rule Variable Case (Error) ---
                            # If rule_depend_var is True, all variables should have been
                            # substituted or pre-indexed. Finding an unknown one is an error.
                            raise KeyError(f"Unknown variable '{arg}' encountered in rule-dependent mode. Should have been substituted or pre-indexed. Atom: {atom}, State: {processed_state}")
                else:
                    # --- Constant Argument ---
                    try:
                        current_sub_row[j + 1] = constant_map[arg]
                    except KeyError:
                         raise KeyError(f"Constant '{arg}' not found in constant index map. Atom: {atom}, State: {processed_state}") from None

            # Arguments beyond max_arity or if num_args < max_arity remain 0 (padding)

        return sub_index

    # Keep build_fact_index if it's used elsewhere
    def build_fact_index(self, facts: List[Term]) -> Dict[Tuple, Set[Term]]:
        self.fact_index.clear()
        for fact in facts:
            predicate = fact.predicate
            args = fact.args
            constant_args_with_pos = [(i, arg) for i, arg in enumerate(args) if not is_variable(arg)]
            for i in range(len(constant_args_with_pos) + 1):
                for subset_args_with_pos in itertools.combinations(constant_args_with_pos, i):
                    sorted_subset = tuple(sorted(subset_args_with_pos, key=lambda x: x[0]))
                    key = (predicate,) + sorted_subset
                    if key not in self.fact_index:
                        self.fact_index[key] = set()
                    self.fact_index[key].add(fact)
        return self.fact_index
