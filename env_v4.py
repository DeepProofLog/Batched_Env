# --- Imports and Constants (Keep as is) ---
import torch
import gymnasium as gym
import numpy as np
from tensordict import TensorDict
from typing import List, Optional, Tuple, Dict, Any, Set, Union, NamedTuple
import random
import logging
import time # For potential timing/debugging
from utils import Term, Rule 


PADDING_VALUE = 0
UNBOUND_VAR = -998

# --- Data Structures (Term, Rule, is_variable - to give context) ---
# class Term:
#     """Represents a logical term like predicate(arg1, arg2)."""
#     def __init__(self, predicate: str, args: List[Any]):
#         self.predicate = predicate
#         self.args = args
#     def __str__(self):
#         # Handle special predicates with no args cleanly
#         if not self.args and self.predicate in ['True', 'False', 'End']:
#              return f"{self.predicate}"
#         return f"{self.predicate}({','.join(map(str, self.args))})"
#     def __repr__(self):
#         return str(self)
#     def __eq__(self, other):
#         return isinstance(other, Term) and self.predicate == other.predicate and self.args == other.args
#     def __hash__(self):
#         try:
#             hashable_args = tuple(self.args)
#         except TypeError:
#             hashable_args = tuple(map(str, self.args))
#         return hash((self.predicate, hashable_args))


# class Rule:
#     """Represents a logical rule Head :- Body."""
#     def __init__(self, head: Term, body: List[Term]):
#         self.head = head
#         self.body = body
#     def __str__(self):
#         if not self.body:
#             return f"{self.head}." # Represent facts/rules with empty body
#         body_str = ', '.join(map(str, self.body))
#         return f"{self.head} :- {body_str}"
#     def __repr__(self):
#         return str(self)
#     def get_all_terms(self) -> List[Term]:
#         """Returns head + body terms."""
#         return [self.head] + self.body
#     def get_variables(self) -> Set[str]:
#         """Extracts all variables (uppercase strings) from the rule."""
#         variables = set()
#         terms = self.get_all_terms()
#         for term in terms:
#             for arg in term.args:
#                 if is_variable(arg):
#                     variables.add(arg)
#         return variables


def is_variable(arg: Any) -> bool:
    """Checks if an argument is a variable (simple uppercase string check)."""
    return isinstance(arg, str) and arg.isupper() and len(arg) == 1 # Keep simple assumption


class UnificationResult(NamedTuple):
    """
    Structure to hold results from unification functions.
    k = maximum number of unifications to return per query
    max_subs = maximum substitutions allowed per unification
    """
    substitutions: torch.Tensor  # Shape: (bs, k, max_subs, 2)
    target_indices: torch.Tensor # Shape: (bs, k) - Index of facts or rules that unify with the query
    valid_mask: torch.Tensor     # Shape: (bs, k) - Indicates valid unifications
    target_bodies: Optional[torch.Tensor] = None # Shape: (bs, k, body_len, 3) - Only for rules


class IndexManager():
    """
    Manages global indices for constants, predicates, and variables.
    Reserves index 0 for PADDING. Constants start at 1. Predicates start at 1.
    Variables start AFTER constants (e.g., num_constants + 1).
    Converts Terms directly to sub-index tensors (P, S, O).
    """
    def __init__(self, constants: Set[str],
                 predicates: Set[str],
                 variables: Set[str], # All variables appearing anywhere
                 rules: List[Rule], # Used only for variable extraction if needed initially
                 padding_atoms: int = 10, # Max atoms (terms) per state
                 max_arity: int = 2, # Crucial: Assumes max_arity=2 for (P, S, O) format
                 device: torch.device = torch.device("cpu")):

        if max_arity != 2:
            raise ValueError("IndexManager currently requires max_arity=2 for P,S,O format.")

        self.device = device
        # Ensure special predicates are included before sorting/indexing
        self.predicates = predicates.union({'True', 'False', 'End'})
        # Sort for consistent indexing
        self.sorted_constants = sorted(list(constants))
        self.sorted_predicates = sorted(list(self.predicates))
        self.sorted_variables = sorted(list(variables)) # Store all possible variables

        self.constant_no = len(self.sorted_constants)
        self.variable_no = len(self.sorted_variables)
        self.predicate_no = len(self.sorted_predicates)

        self.rules = rules # Keep rules if needed for structure info
        self.padding_atoms = padding_atoms
        self.max_arity = max_arity # Should be 2

        # --- Global Mappings (ID 0 reserved for padding) ---
        self.constant_str2idx: Dict[str, int] = {}
        self.constant_idx2str: Dict[int, str] = {}
        self.predicate_str2idx: Dict[str, int] = {}
        self.predicate_idx2str: Dict[int, str] = {}
        self.variable_str2idx: Dict[str, int] = {}
        self.variable_idx2str: Dict[int, str] = {}

        # --- Special Predicate Indices ---
        self.true_idx: int = -1 # Will be set to positive index
        self.false_idx: int = -1 # Will be set to positive index
        self.end_idx: int = -1 # Will be set to positive index

        self._create_global_idx() # Create mappings with new positive indexing scheme

        # --- Create Global Variable Tensor (Now with Positive Indices) ---
        # Check if variable indices were correctly assigned positive values > constant_no
        if not all(idx > self.constant_no for idx in self.variable_str2idx.values()):
             min_var_idx = min(self.variable_str2idx.values()) if self.variable_str2idx else -1
             logging.warning(f"Variable indices check failed. Expected > {self.constant_no}, found min: {min_var_idx}")
             # This might indicate an issue in _create_global_idx or empty variables list

        self.vars_idx_tensor = torch.tensor(
            sorted(list(self.variable_str2idx.values())), # Use the positive indices
            dtype=torch.long,
            device=self.device
        )
        self.vars_idx_set = set(self.vars_idx_tensor.cpu().tolist())

        logging.info("IndexManager initialized (Positive Indices, 0=Padding).")
        logging.info(f"Constants: {self.constant_no} (Indices 1 to {self.constant_no})")
        logging.info(f"Predicates: {self.predicate_no} (Indices 1 to {self.predicate_no})")
        logging.info(f"Variables: {self.variable_no} (Indices {self.constant_no + 1} to {self.constant_no + self.variable_no})")
        logging.info(f"Padding Atoms (Max Terms per State): {self.padding_atoms}")
        logging.info(f"True/False/End Indices: {self.true_idx}/{self.false_idx}/{self.end_idx}")
        # logging.info(f"Variable Indices (Sample): {list(self.variable_str2idx.items())[:5]}")
        # logging.info(f"Variable Tensor Shape: {self.vars_idx_tensor.shape}")
        # logging.info(f"Variable Set (Sample): {list(self.vars_idx_set)[:5]}")

    def _create_global_idx(self):
        """Create global POSITIVE indices. 0 is reserved for padding."""
        # Constants: Positive integers starting from 1
        current_idx = 1
        for term in self.sorted_constants:
            self.constant_str2idx[term] = current_idx
            self.constant_idx2str[current_idx] = term
            current_idx += 1
        # Note: self.constant_no should equal current_idx - 1 here

        # Predicates: Positive integers starting from 1 (separate namespace)
        current_idx = 1
        for term in self.sorted_predicates:
            self.predicate_str2idx[term] = current_idx
            self.predicate_idx2str[current_idx] = term
            if term == 'True': self.true_idx = current_idx
            elif term == 'False': self.false_idx = current_idx
            elif term == 'End': self.end_idx = current_idx
            current_idx += 1
        # Note: self.predicate_no should equal current_idx - 1 here

        # Handle missing special predicates (assign next available indices)
        if self.true_idx == -1 or self.false_idx == -1 or self.end_idx == -1:
            logging.warning("Special predicates 'True', 'False', or 'End' not found during initial indexing.")
            next_pred_idx = self.predicate_no + 1 # Start after last assigned predicate index
            if 'True' not in self.predicate_str2idx:
                 self.true_idx = next_pred_idx
                 self.predicate_str2idx['True'] = self.true_idx
                 self.predicate_idx2str[self.true_idx] = 'True'
                 next_pred_idx+=1
                 logging.info(f"Assigned index {self.true_idx} to 'True'")
            if 'False' not in self.predicate_str2idx:
                 self.false_idx = next_pred_idx
                 self.predicate_str2idx['False'] = self.false_idx
                 self.predicate_idx2str[self.false_idx] = 'False'
                 next_pred_idx+=1
                 logging.info(f"Assigned index {self.false_idx} to 'False'")
            if 'End' not in self.predicate_str2idx:
                 self.end_idx = next_pred_idx
                 self.predicate_str2idx['End'] = self.end_idx
                 self.predicate_idx2str[self.end_idx] = 'End'
                 next_pred_idx+=1
                 logging.info(f"Assigned index {self.end_idx} to 'End'")
            self.predicate_no = next_pred_idx - 1 # Update total predicate count

        # Variables: Positive integers starting AFTER constants
        current_idx = self.constant_no + 1 # Start index for variables
        for term in self.sorted_variables:
            self.variable_str2idx[term] = current_idx
            self.variable_idx2str[current_idx] = term
            current_idx += 1
        # Note: self.variable_no should equal current_idx - (self.constant_no + 1)

    def _get_term_index(self, term: Any) -> int:
        """Helper to get index for a constant or variable. Returns PADDING_VALUE (0) if not found."""
        if is_variable(term):
            # Variables now have positive indices > constant_no
            idx = self.variable_str2idx.get(term, PADDING_VALUE)
            # if idx == PADDING_VALUE: logging.warning(f"Variable '{term}' not found. Returning PADDING_VALUE (0).")
        else:
            # Constants have positive indices <= constant_no
            idx = self.constant_str2idx.get(term, PADDING_VALUE)
            # if idx == PADDING_VALUE: logging.warning(f"Constant '{term}' not found. Returning PADDING_VALUE (0).")
        return idx

    def get_term_pso_indices(self, term: Term) -> Optional[Tuple[int, int, int]]:
        """Converts a Term (arity 0 or 2) to (P, S, O). Returns None if invalid. Uses 0 for padding."""
        pred_idx = self.predicate_str2idx.get(term.predicate, PADDING_VALUE) # Use 0 if predicate not found
        if pred_idx == PADDING_VALUE: # Check if predicate itself was unknown
            logging.warning(f"Predicate '{term.predicate}' not found.")
            # Return padding representation? Or None? Let's return padding for consistency.
            return (PADDING_VALUE, PADDING_VALUE, PADDING_VALUE) # Return (0, 0, 0)

        # Handle special predicates
        if pred_idx in [self.true_idx, self.false_idx, self.end_idx]:
            if not term.args: # Expect no arguments
                return (pred_idx, PADDING_VALUE, PADDING_VALUE) # (pred_idx, 0, 0)
            else:
                logging.warning(f"Special predicate '{term.predicate}' has unexpected args: {term.args}")
                return None # Invalid format

        # Handle normal arity 2 terms
        elif len(term.args) == self.max_arity:
            subj_idx = self._get_term_index(term.args[0]) # Returns 0 if arg not found
            obj_idx = self._get_term_index(term.args[1])  # Returns 0 if arg not found
            return pred_idx, subj_idx, obj_idx
        else:
             logging.warning(f"Term {term} does not have arity {self.max_arity} or 0 (for special). Cannot get PSO indices.")
             return None

    def get_state_sub_index_tensor(self, state_terms: List[Term]) -> torch.Tensor:
        """
        Converts a list of Terms representing a state into a padded sub-index tensor.
        Output shape: (padding_atoms, 3). Uses PADDING_VALUE (0) for padding.
        """
        sub_indices = torch.full((self.padding_atoms, 3), PADDING_VALUE, dtype=torch.long, device=self.device)
        # print('state_terms', state_terms) # Keep for debugging if needed
        if len(state_terms) > self.padding_atoms:
            logging.warning(f"State length {len(state_terms)} exceeds padding_atoms {self.padding_atoms}. Truncating.")
            state_terms = state_terms[:self.padding_atoms]

        for i, term in enumerate(state_terms):
            pso = self.get_term_pso_indices(term)
            if pso is not None:
                sub_indices[i] = torch.tensor(pso, dtype=torch.long, device=self.device)
            # else: leave as PADDING_VALUE (0)

        return sub_indices

    def get_term_from_indices(self, pso_tensor: torch.Tensor) -> Optional[Term]:
        """
        Reconstructs a single Term from a (P, S, O) tensor. Handles PADDING_VALUE (0).
        Input shape: (3,)
        """
        if not isinstance(pso_tensor, torch.Tensor) or pso_tensor.shape != (3,):
            logging.error(f"Invalid input shape for get_term_from_indices: {pso_tensor.shape}")
            return None

        p_idx, s_idx, o_idx = pso_tensor.tolist()

        # Handle padding atoms (all zeros)
        if p_idx == PADDING_VALUE and s_idx == PADDING_VALUE and o_idx == PADDING_VALUE:
            return None # Represent padding as None

        predicate = self.predicate_idx2str.get(p_idx)
        if predicate is None:
            if p_idx == PADDING_VALUE: # Check if predicate itself was padding
                 return None # It's a padding atom overall
            logging.warning(f"Unknown predicate index {p_idx} during reconstruction.")
            predicate = f"UNK_PRED_{p_idx}"

        # Handle special predicates (no arguments needed)
        if predicate in ['True', 'False', 'End']:
            if s_idx != PADDING_VALUE or o_idx != PADDING_VALUE:
                 logging.warning(f"Special predicate '{predicate}' has non-padding args ({s_idx}, {o_idx}). Expected padding (0).")
            return Term(predicate, [])

        # Handle normal arity 2 terms
        args = []
        for term_idx in [s_idx, o_idx]:
            if term_idx == PADDING_VALUE: # Index 0 is padding
                # logging.warning(f"Found PADDING_VALUE (0) in args for predicate {predicate} ({p_idx}).") # Less verbose
                args.append(f"PAD_{term_idx}") # Or handle as error/skip? Using string for now. Could return None.
            elif term_idx > self.constant_no: # Variable index range
                arg_str = self.variable_idx2str.get(term_idx)
                if arg_str is None: arg_str = f"UNK_VAR_{term_idx}"; logging.warning(f"Unknown variable index {term_idx}.")
                args.append(arg_str)
            elif term_idx > 0: # Constant index range (1 to constant_no)
                arg_str = self.constant_idx2str.get(term_idx)
                if arg_str is None: arg_str = f"UNK_CONST_{term_idx}"; logging.warning(f"Unknown constant index {term_idx}.")
                args.append(arg_str)
            else: # Should only be 0 (PADDING_VALUE), already handled. This case is unlikely.
                 logging.error(f"Unexpected non-positive index {term_idx} encountered for argument.")
                 args.append(f"UNK_ARG_{term_idx}")

        return Term(predicate, args)

    # --- get_state_terms_from_batch remains largely the same, relies on get_term_from_indices ---
    def get_state_terms_from_batch(self, sub_idx_batch: torch.Tensor) -> List[List[Optional[Term]]]:
        """Reconstructs List[List[Term]] from a batch of sub_idx tensors."""
        if sub_idx_batch.dim() != 3 or sub_idx_batch.shape[-1] != 3:
            raise ValueError(f"Expected sub_idx_batch shape (bs, pad_atoms, 3), got {sub_idx_batch.shape}")
        bs, pad_atoms, _ = sub_idx_batch.shape
        all_states_terms = []
        for i in range(bs):
            state_terms = []
            for j in range(pad_atoms):
                term = self.get_term_from_indices(sub_idx_batch[i, j])
                if term is not None: # Only add non-padding terms
                    state_terms.append(term)
                elif torch.any(sub_idx_batch[i, j] != PADDING_VALUE): # If it wasn't padding but failed reconstruction
                    state_terms.append(Term("RECONSTRUCTION_ERROR", [str(sub_idx_batch[i, j].tolist())]))

            all_states_terms.append(state_terms)
        return all_states_terms

    # --- facts_to_tensor uses 0 for padding implicitly via get_term_pso_indices ---
    def facts_to_tensor(self, facts: List[Term]) -> torch.Tensor:
        """Converts a list of fact Terms to a (n_facts, 3) tensor."""
        fact_indices = []
        for fact in facts:
            pso = self.get_term_pso_indices(fact)
            if pso:
                fact_indices.append(pso)
            else:
                logging.warning(f"Skipping invalid fact for tensor conversion: {fact}")
        if not fact_indices:
            return torch.empty((0, 3), dtype=torch.long, device=self.device)
        return torch.tensor(fact_indices, dtype=torch.long, device=self.device)

    # --- rules_to_tensor uses 0 for padding implicitly via get_term_pso_indices ---
    def rules_to_tensor(self, rules: List[Rule]) -> Tuple[torch.Tensor, int]:
        """Converts a list of Rules to a (n_rules, max_atoms, 3) tensor."""
        rule_tensors = []
        max_atoms_in_rules = 0
        for rule in rules:
            rule_terms = rule.get_all_terms() # Head is first
            max_atoms_in_rules = max(max_atoms_in_rules, len(rule_terms))
            term_indices = []
            for term in rule_terms:
                pso = self.get_term_pso_indices(term)
                if pso:
                    term_indices.append(pso)
                else:
                    logging.warning(f"Skipping invalid term in rule for tensor conversion: {term} in rule {rule}")
                    term_indices.append((PADDING_VALUE, PADDING_VALUE, PADDING_VALUE)) # Pad invalid term with (0,0,0)

            rule_tensors.append(term_indices)

        # Pad each rule to max_atoms_in_rules
        padded_rules = []
        if max_atoms_in_rules == 0 and rules:
             logging.error("No valid terms found in any rules for tensor conversion.")
             return torch.empty((0, 0, 3), dtype=torch.long, device=self.device), 0
        elif not rules:
             return torch.empty((0, 0, 3), dtype=torch.long, device=self.device), 0

        padding_tuple = (PADDING_VALUE, PADDING_VALUE, PADDING_VALUE)
        for term_indices in rule_tensors:
            current_len = len(term_indices)
            padding_needed = max_atoms_in_rules - current_len
            padded_indices = term_indices + [list(padding_tuple)] * padding_needed
            padded_rules.append(padded_indices)

        if not padded_rules:
             return torch.empty((0, max_atoms_in_rules, 3), dtype=torch.long, device=self.device), max_atoms_in_rules

        return torch.tensor(padded_rules, dtype=torch.long, device=self.device), max_atoms_in_rules


# --- Modified Unification Helper Functions ---

def _get_var_set_and_tensor(
    vars_idx: Union[torch.Tensor, List[int], Set[int], IndexManager],
    device: torch.device
) -> Tuple[Set[int], torch.Tensor]:
    """Ensures vars_idx is a set and a tensor on the correct device. Handles positive variable indices."""
    if isinstance(vars_idx, IndexManager):
        # IndexManager now correctly stores positive var indices in vars_idx_set/vars_idx_tensor
        return vars_idx.vars_idx_set, vars_idx.vars_idx_tensor.to(device)

    if isinstance(vars_idx, torch.Tensor):
        vars_idx_tensor = vars_idx.to(device=device, dtype=torch.long).detach()
        vars_idx_set = set(vars_idx_tensor.cpu().tolist())
    elif isinstance(vars_idx, (list, set)):
        vars_idx_set = set(vars_idx)
        vars_idx_tensor = torch.tensor(sorted(list(vars_idx_set)), dtype=torch.long, device=device) # Sort for consistency
    else:
        raise TypeError("vars_idx must be an IndexManager, tensor, list, or set")

    if vars_idx_tensor.dim() == 0:
        vars_idx_tensor = vars_idx_tensor.unsqueeze(0)

    # NEW CHECK: Ensure variable indices are positive (or empty tensor)
    if vars_idx_tensor.numel() > 0 and not torch.all(vars_idx_tensor > 0):
         logging.warning(f"Variable tensor contains non-positive values: {vars_idx_tensor[vars_idx_tensor <= 0]}. Expected positive indices.")

    return vars_idx_set, vars_idx_tensor

# --- Formatting functions (_format_term, _format_atom, _format_substitution) ---
# These primarily need updates for PADDING_VALUE=0 and checking `in vars_idx_set`
# instead of `< 0` for variables.

def _format_term(
    term_idx: int,
    vars_idx_set: Set[int],
    constant_idx2str: Optional[Dict[int, str]] = None,
    variable_idx2str: Optional[Dict[int, str]] = None,
    constant_no: Optional[int] = None # Optional: pass constant count for better context
) -> str:
    """Formats a single term index (subject or object). Handles PADDING_VALUE=0."""
    if term_idx == PADDING_VALUE: return "[pad]"
    if term_idx == UNBOUND_VAR: return "[unbound]" # Keep UNBOUND_VAR as is

    # Check if it's a variable using the provided set (now contains positive indices)
    if term_idx in vars_idx_set:
        var_str = variable_idx2str.get(term_idx, f"VAR_IDX({term_idx})") if variable_idx2str else f"VAR({term_idx})"
        return var_str

    # Check if it's a constant (positive index, potentially up to constant_no)
    if constant_idx2str and term_idx in constant_idx2str:
         # Check if the index falls within expected constant range if constant_no provided
         if constant_no is not None and term_idx > constant_no:
             logging.warning(f"Index {term_idx} found in constant_idx2str but is > constant_no ({constant_no}).")
         return constant_idx2str[term_idx]

    # Fallback if not padding, variable, or known constant
    return f"IDX({term_idx})"


def _format_atom(
    atom_tensor: torch.Tensor,
    predicate_idx2str: Optional[Dict[int, str]] = None,
    constant_idx2str: Optional[Dict[int, str]] = None,
    vars_idx_set: Optional[Set[int]] = None,
    variable_idx2str: Optional[Dict[int, str]] = None,
    constant_no: Optional[int] = None # Pass constant_no to _format_term
) -> str:
    """Formats an atom tensor (shape [3]) into a readable string. Handles PADDING_VALUE=0."""
    # ... (initial shape/type checks remain the same) ...
    if atom_tensor.is_cuda: atom_tensor = atom_tensor.cpu()
    if not isinstance(atom_tensor, torch.Tensor) or atom_tensor.dim() < 1 or atom_tensor.shape[-1] != 3: return f"[invalid atom shape: {atom_tensor.shape}]"
    if atom_tensor.dim() > 1: atom_tensor = atom_tensor.view(-1, 3)[0]
    if atom_tensor.numel() != 3: return f"[invalid atom numel: {atom_tensor.numel()}]"

    p_idx, s_idx, o_idx = atom_tensor.tolist()

    # Check for padding atom (all zeros)
    is_padding_atom = (p_idx == PADDING_VALUE and s_idx == PADDING_VALUE and o_idx == PADDING_VALUE)
    if is_padding_atom: return "[padding_atom]"

    p_str = predicate_idx2str.get(p_idx, f"PRED_IDX({p_idx})") if predicate_idx2str else f"P({p_idx})"
    if p_idx == PADDING_VALUE: p_str = "[pad_pred]" # Handle case where only predicate is padding

    # Handle special predicates (True, False, End)
    # Assuming their indices are positive now
    special_preds = {'True', 'False', 'End'} # Get from IndexManager if possible
    if p_str in special_preds:
        if s_idx != PADDING_VALUE or o_idx != PADDING_VALUE:
            s_str_err = _format_term(s_idx, vars_idx_set or set(), constant_idx2str, variable_idx2str, constant_no)
            o_str_err = _format_term(o_idx, vars_idx_set or set(), constant_idx2str, variable_idx2str, constant_no)
            return f"({p_str}, {s_str_err}, {o_str_err})[!]" # Mark as unexpected format
        else:
            return f"({p_str})" # Correct format

    vars_set = vars_idx_set if isinstance(vars_idx_set, set) else set()
    s_str = _format_term(s_idx, vars_set, constant_idx2str, variable_idx2str, constant_no)
    o_str = _format_term(o_idx, vars_set, constant_idx2str, variable_idx2str, constant_no)

    return f"({p_str}, {s_str}, {o_str})"


def _format_substitution(
    sub_pair: torch.Tensor, # Shape (2,) : [var_idx, value_idx]
    constant_idx2str: Optional[Dict[int, str]] = None,
    vars_idx_set: Optional[Set[int]] = None,
    variable_idx2str: Optional[Dict[int, str]] = None,
    constant_no: Optional[int] = None # Pass constant_no
) -> str:
    """Formats a substitution pair tensor into a readable string. Handles PADDING_VALUE=0."""
    # ... (initial shape/type checks remain the same) ...
    if sub_pair.is_cuda: sub_pair = sub_pair.cpu()
    if not isinstance(sub_pair, torch.Tensor) or sub_pair.shape != (2,): return f"[invalid sub pair: {sub_pair}]"

    var_idx, value_idx = sub_pair.tolist()

    # Check for padding substitution
    if var_idx == PADDING_VALUE: return "[no_sub]"

    vars_set = vars_idx_set if isinstance(vars_idx_set, set) else set()

    # Format variable using the variable set
    var_str = "[var?]"
    if var_idx in vars_set:
        var_str = variable_idx2str.get(var_idx, f"VAR_IDX({var_idx})") if variable_idx2str else f"VAR({var_idx})"
    else:
        # This case should ideally not happen if var_idx is correctly identified
        var_str = f"NON_VAR({var_idx})"

    # Format value using the general term formatter
    value_str = _format_term(value_idx, vars_set, constant_idx2str, variable_idx2str, constant_no)
    return f"{var_str} -> {value_str}"


# --- apply_substitutions_vectorized ---
# Logic needs updates for PADDING_VALUE=0 and using vars_idx_tensor correctly if needed
# (though direct comparison should still work)

def apply_substitutions_vectorized(
    atoms: torch.Tensor,             # Shape: (..., num_atoms, 3)
    substitutions: torch.Tensor,     # Shape: (..., max_subs, 2), [var_idx, value_idx]
    vars_idx_tensor: torch.Tensor,   # 1D Tensor of POSITIVE variable indices
    max_iterations: int = 10,        # Limit recursion/iteration depth
) -> torch.Tensor:
    """
    Applies substitutions (Var -> Value) to a tensor of atoms iteratively.
    Handles Var -> Const and Var -> Var substitutions until no changes occur or max_iterations.
    Assumes PADDING_VALUE is 0. Uses positive variable indices in vars_idx_tensor.
    """
    if atoms.numel() == 0: return atoms.clone()
    if atoms.shape[-1] != 3: raise ValueError(f"Last dimension of atoms must be 3, but got shape {atoms.shape}")
    if substitutions.numel() > 0 and substitutions.shape[-1] != 2: raise ValueError(f"Last dimension of substitutions must be 2, but got shape {substitutions.shape}")

    device = atoms.device
    substituted_atoms = atoms.clone()

    if substitutions.numel() == 0: return substituted_atoms

    substitutions = substitutions.to(device)

    # Valid substitutions have var_idx != PADDING_VALUE (0)
    valid_subs_mask = (substitutions[..., 0] != PADDING_VALUE) # Shape: (..., max_subs)
    if not torch.any(valid_subs_mask): return substituted_atoms

    changed = True
    iterations = 0
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        max_subs_dim = substitutions.shape[-2]
        for sub_idx in range(max_subs_dim):
            current_sub = substitutions[..., sub_idx, :] # Shape (..., 2)
            current_mask = valid_subs_mask[..., sub_idx] # Shape (...)

            if not torch.any(current_mask): continue

            # Use PADDING_VALUE (0) as placeholder for inactive substitutions
            var_to_replace = torch.where(current_mask, current_sub[..., 0], PADDING_VALUE)
            replacement_value = torch.where(current_mask, current_sub[..., 1], PADDING_VALUE)

            # --- Broadcasting Setup (Remains the same logic) ---
            num_atom_dims = substituted_atoms.dim()
            view_shape_list = list(var_to_replace.shape)
            view_shape_list.extend([1] * (num_atom_dims - len(view_shape_list) - 1))
            view_shape_list.append(1)
            view_shape = tuple(view_shape_list)

            var_to_replace_b = var_to_replace.view(view_shape)
            replacement_value_b = replacement_value.view(view_shape)
            current_mask_b = current_mask.view(view_shape[:-1] + (1,))

            # --- Apply Substitution to Subject (Column 1) ---
            s_match_mask = (substituted_atoms[..., 1:2] == var_to_replace_b) & current_mask_b
            if torch.any(s_match_mask):
                current_s_values = substituted_atoms[..., 1:2]
                apply_s_mask = s_match_mask & (current_s_values != replacement_value_b)
                if torch.any(apply_s_mask):
                    substituted_atoms[..., 1:2] = torch.where(apply_s_mask, replacement_value_b, current_s_values)
                    changed = True

            # --- Apply Substitution to Object (Column 2) ---
            o_match_mask = (substituted_atoms[..., 2:3] == var_to_replace_b) & current_mask_b
            if torch.any(o_match_mask):
                current_o_values = substituted_atoms[..., 2:3]
                apply_o_mask = o_match_mask & (current_o_values != replacement_value_b)
                if torch.any(apply_o_mask):
                    substituted_atoms[..., 2:3] = torch.where(apply_o_mask, replacement_value_b, current_o_values)
                    changed = True

    if iterations >= max_iterations and changed:
        logging.warning(f"Max substitution iterations ({max_iterations}) reached.")

    return substituted_atoms


# --- batch_unify_with_facts ---
# Logic needs updates for PADDING_VALUE=0 and using vars_idx_set for variable checks.

def batch_unify_with_facts(
    states_idx: torch.Tensor,             # Shape: (bs, 1, n_padding_atoms, 3)
    facts: torch.Tensor,                  # Shape: (n_facts, 3)
    vars_idx: Union[torch.Tensor, List[int], Set[int], IndexManager], # Contains POSITIVE variable indices or IndexManager
    k: int,                               # Max unifications to return
    device: Optional[torch.device] = None,
    verbose: bool = False,                # Enable verbose logging/printing
    index_manager_for_format: Optional[IndexManager] = None # Pass IndexManager instance for formatting
) -> UnificationResult:
    """
    Performs batched unification against FACTS. Uses PADDING_VALUE=0.
    Identifies variables using positive indices in vars_idx_set.
    Handles topk errors robustly.
    """
    if k <= 0: raise ValueError("k must be a positive integer")

    effective_device = device if device else states_idx.device
    states_idx = states_idx.to(effective_device)
    facts = facts.to(effective_device)

    # Get variable set/tensor (now containing positive indices)
    # This function needs to be updated for positive indices if not done already
    vars_idx_set, vars_idx_tensor = _get_var_set_and_tensor(vars_idx, effective_device)

    bs, _, n_padding_atoms, _ = states_idx.shape
    n_facts = facts.shape[0]
    # Determine actual k based on available facts
    actual_k = min(k, n_facts) if n_facts > 0 else 0

    # Initialize output tensors with padding (0)
    substitutions_out = torch.full((bs, k, 2, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
    fact_indices_out = torch.full((bs, k), PADDING_VALUE, dtype=torch.long, device=effective_device)
    valid_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=effective_device)

    # Handle edge cases: no batch items, no facts, or no query atoms
    if bs == 0 or n_facts == 0 or n_padding_atoms == 0:
        return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)

    # --- Extract First Query Atom ---
    first_queries = states_idx[:, 0, 0, :] # Shape: (bs, 3)
    # Check if the first query is just padding (all zeros)
    is_query_valid = ~torch.all(first_queries == PADDING_VALUE, dim=-1) # Shape: (bs,)

    # --- Prepare for Broadcasting ---
    queries_expanded = first_queries.unsqueeze(1) # (bs, 1, 3)
    facts_expanded = facts.unsqueeze(0)           # (1, n_facts, 3)

    # --- Check Basic Unification Conditions (Vectorized) ---
    # Predicate match
    pred_match = (queries_expanded[:, :, 0] == facts_expanded[:, :, 0]) # (bs, n_facts)

    # Subject match: Query S is variable OR Query S == Fact S
    # Check if query S index is in the set of positive variable indices
    is_query_var_s = torch.isin(queries_expanded[:, :, 1], vars_idx_tensor) # (bs, 1) -> broadcasts
    subj_match = is_query_var_s | (queries_expanded[:, :, 1] == facts_expanded[:, :, 1]) # (bs, n_facts)

    # Object match: Query O is variable OR Query O == Fact O
    is_query_var_o = torch.isin(queries_expanded[:, :, 2], vars_idx_tensor) # (bs, 1) -> broadcasts
    obj_match = is_query_var_o | (queries_expanded[:, :, 2] == facts_expanded[:, :, 2]) # (bs, n_facts)

    # Initial unification mask (predicate, subject, object must match/be variable)
    # Also ensure the query itself is valid (not padding)
    unifies_mask_initial = pred_match & subj_match & obj_match & is_query_valid.unsqueeze(1) # (bs, n_facts)

    # If no potential matches or k is 0, return padded results
    if actual_k == 0 or not torch.any(unifies_mask_initial):
        return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)

    # --- Find Top K Potential Matches ---
    scores = torch.where(unifies_mask_initial, 1.0, float('-inf'))
    try:
        # Attempt topk with the calculated actual_k
        top_scores, top_fact_indices = torch.topk(scores, k=actual_k, dim=1)

    except RuntimeError as e:
        logging.error(f"RuntimeError during fact topk: {e}. Scores shape: {scores.shape}, actual_k: {actual_k}, n_facts: {n_facts}")
        # Check if the error is the common one: k > dimension size
        if "must be less than or equal to the dimension size" in str(e) or scores.shape[1] < actual_k: # Check common error text or shape explicitly
            logging.warning(f"Adjusting k for fact topk because requested k ({actual_k}) > n_facts ({scores.shape[1]}).")
            actual_k = scores.shape[1] # Adjust k to the maximum possible
            if actual_k == 0: # If there are no facts, return empty
                logging.warning("Returning empty result because n_facts is 0 after adjustment.")
                return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)
            try:
                 # Retry topk with the adjusted k
                 top_scores, top_fact_indices = torch.topk(scores, k=actual_k, dim=1)
                 logging.info(f"Successfully retried fact topk with adjusted k={actual_k}.")
            except RuntimeError as e_retry:
                 # If retry still fails (should be very rare if k=dim_size), log and re-raise
                 logging.error(f"FATAL: Retrying fact topk failed even with adjusted k={actual_k}: {e_retry}")
                 raise e_retry # Re-raise the error from the retry attempt
        else:
            # The error was NOT the expected k > dim_size issue.
            logging.error(f"FATAL: Encountered an unexpected RuntimeError during fact topk that wasn't k > dim_size.")
            # Re-raise the original exception for clarity
            raise e

    # Mask indicating which of the top-k are actual potential matches (score > -inf)
    is_topk_potentially_valid = (top_scores > float('-inf')) # Shape: (bs, actual_k)

    # --- Check Conflicts in Top K ---
    # Find coordinates (batch_idx, k_idx) of potentially valid top-k entries
    potential_k_coords = torch.nonzero(is_topk_potentially_valid, as_tuple=True)
    n_potential_topk = len(potential_k_coords[0])

    # Initialize final masks and outputs for the top-k slice
    final_valid_mask_slice = torch.zeros((bs, actual_k), dtype=torch.bool, device=effective_device)
    gathered_subs = torch.full((bs, actual_k, 2, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
    gathered_indices = torch.full((bs, actual_k), PADDING_VALUE, dtype=torch.long, device=effective_device)

    if n_potential_topk > 0:
        valid_bs_indices, valid_k_indices = potential_k_coords
        # Get the indices of the facts corresponding to these top-k entries
        potential_fact_indices_flat = top_fact_indices[potential_k_coords] # Shape: (n_potential_topk,)

        # Gather the corresponding queries and facts
        top_k_queries_flat = first_queries[valid_bs_indices] # Shape: (n_potential_topk, 3)
        top_k_facts_flat = facts[potential_fact_indices_flat]    # Shape: (n_potential_topk, 3)

        # --- Conflict Check: Query(P, X, X) vs Fact(P, A, B) where A != B ---
        # Check if query S and O are the *same* variable (use torch.isin with positive vars_idx_tensor)
        is_query_s_var_flat = torch.isin(top_k_queries_flat[:, 1], vars_idx_tensor)
        is_query_o_var_flat = torch.isin(top_k_queries_flat[:, 2], vars_idx_tensor)
        is_same_query_var_flat = is_query_s_var_flat & is_query_o_var_flat & \
                                 (top_k_queries_flat[:, 1] == top_k_queries_flat[:, 2]) # (n_potential_topk,)

        # Check if fact S and O are different constants/terms (and not padding)
        is_diff_fact_const_flat = (top_k_facts_flat[:, 1] != top_k_facts_flat[:, 2]) # (n_potential_topk,)
        is_diff_fact_const_flat &= (top_k_facts_flat[:, 1] != PADDING_VALUE) & (top_k_facts_flat[:, 2] != PADDING_VALUE)

        # Conflict occurs if query has same var but fact has different terms
        is_conflict_flat = is_same_query_var_flat & is_diff_fact_const_flat # (n_potential_topk,)

        # Update the slice of the final valid mask - only non-conflicting ones are truly valid
        final_valid_mask_slice[potential_k_coords] = ~is_conflict_flat

        # --- Gather Substitutions for Valid, Non-Conflicting Unifications ---
        # Find coordinates of the final valid unifications within the top-k slice
        final_valid_coords = torch.nonzero(final_valid_mask_slice, as_tuple=True)
        n_final_valid = len(final_valid_coords[0])

        if n_final_valid > 0:
            # Filter down to the non-conflicting entries among the potentials
            non_conflict_mask_flat = ~is_conflict_flat # (n_potential_topk,)

            # Get the queries, facts, and fact indices for the final valid set
            final_queries = top_k_queries_flat[non_conflict_mask_flat] # (n_final_valid, 3)
            final_facts = top_k_facts_flat[non_conflict_mask_flat]    # (n_final_valid, 3)
            final_fact_idxs_out = potential_fact_indices_flat[non_conflict_mask_flat] # (n_final_valid,)

            # Place the fact indices into the output tensor at the correct (bs, k) coords
            gathered_indices[final_valid_coords] = final_fact_idxs_out

            # --- Create Substitution Pairs (Var -> Const) ---
            # Check if query S/O are variables in the final valid set (use torch.isin)
            is_s_var_final = torch.isin(final_queries[:, 1], vars_idx_tensor) # (n_final_valid,)
            is_o_var_final = torch.isin(final_queries[:, 2], vars_idx_tensor) # (n_final_valid,)

            # Subject substitution: [Query S Var, Fact S Value] if Query S is var, else [PAD, PAD]
            s_sub_pairs = torch.stack([
                torch.where(is_s_var_final, final_queries[:, 1], PADDING_VALUE),
                torch.where(is_s_var_final, final_facts[:, 1], PADDING_VALUE)
            ], dim=1) # (n_final_valid, 2)

            # Object substitution: [Query O Var, Fact O Value] if Query O is var, else [PAD, PAD]
            o_sub_pairs = torch.stack([
                torch.where(is_o_var_final, final_queries[:, 2], PADDING_VALUE),
                torch.where(is_o_var_final, final_facts[:, 2], PADDING_VALUE)
            ], dim=1) # (n_final_valid, 2)

            # Place the substitution pairs into the output tensor at the correct (bs, k) coords
            gathered_subs[final_valid_coords[0], final_valid_coords[1], 0, :] = s_sub_pairs
            gathered_subs[final_valid_coords[0], final_valid_coords[1], 1, :] = o_sub_pairs

    # --- Populate Final Output Tensors ---
    # Copy the gathered results (up to actual_k) into the full output tensors (size k)
    substitutions_out[:, :actual_k] = gathered_subs
    fact_indices_out[:, :actual_k] = gathered_indices
    valid_mask_out[:, :actual_k] = final_valid_mask_slice

    # --- Verbose Formatting ---
    if verbose and index_manager_for_format:
        # Assumes _format_atom, _format_substitution helpers are available and updated
        im = index_manager_for_format
        const_no = getattr(im, 'constant_no', None) # Get constant count if available
        print("-" * 40 + "\nDEBUG: batch_unify_with_facts (End, Positive Indices)\n" + "-" * 40)
        for b in range(min(bs, 3)): # Print first few batch items
            query_str = _format_atom(first_queries[b], getattr(im,'predicate_idx2str',{}), getattr(im,'constant_idx2str',{}), getattr(im,'vars_idx_set',set()), getattr(im,'variable_idx2str',{}), const_no)
            print(f"B={b} Query: {query_str}")
            for ki in range(k): # Iterate up to requested k
                is_valid = valid_mask_out[b, ki].item()
                fact_idx = fact_indices_out[b, ki].item()
                if fact_idx != PADDING_VALUE: # Check if a fact was actually found (index > 0)
                    fact_str = _format_atom(facts[fact_idx], getattr(im,'predicate_idx2str',{}), getattr(im,'constant_idx2str',{}), getattr(im,'vars_idx_set',set()), getattr(im,'variable_idx2str',{}), const_no)
                    subs_str = ", ".join([_format_substitution(substitutions_out[b, ki, si], getattr(im,'constant_idx2str',{}), getattr(im,'vars_idx_set',set()), getattr(im,'variable_idx2str',{}), const_no) for si in range(2) if substitutions_out[b, ki, si, 0] != PADDING_VALUE])
                    print(f"  k={ki}: Valid={is_valid}, FactIdx={fact_idx} ({fact_str}), Subs=[{subs_str}]")
                elif is_valid: # Should not happen if fact_idx is padding=0
                     print(f"  k={ki}: Valid=True, FactIdx=PADDING (Error!)")
                # else: # Don't print padded/invalid slots unless debugging all k slots
                #     if ki < actual_k: print(f"  k={ki}: Invalid/Padded")

        print("-" * 40)

    return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)


# --- Unification Helpers (get_binding, occurs_check) ---
# Need to use vars_idx_set instead of checking < 0

def get_binding(var_idx: int,
                subs_dict: Dict[int, int],
                visited: Set[int],
                vars_idx_set: Set[int], # Set of POSITIVE variable indices
                max_depth: int = 20
                ) -> Optional[int]:
    """ Finds ultimate binding. Uses vars_idx_set to identify variables. """
    # Base case: If it's not a variable in the set OR it's a variable not currently bound
    if var_idx not in vars_idx_set or var_idx not in subs_dict:
        return var_idx

    # Cycle detection / Depth limit
    if var_idx in visited: return None
    if len(visited) > max_depth: return None

    visited.add(var_idx)
    next_val = subs_dict[var_idx]
    # Recursively find the binding of the next value
    result = get_binding(next_val, subs_dict, visited, vars_idx_set, max_depth)
    visited.remove(var_idx) # Backtrack

    return result


def occurs_check(var_idx: int,
                 term_idx: int,
                 subs_dict: Dict[int, int],
                 vars_idx_set: Set[int], # Set of POSITIVE variable indices
                 max_depth: int = 20) -> bool:
    """ Simplified occurs check. Uses vars_idx_set to identify variables. """
    # If term_idx is not a variable, var_idx cannot occur in it
    if term_idx not in vars_idx_set:
        return False

    # Check the binding chain of term_idx
    current = term_idx
    visited_check = set()
    depth = 0
    while current in vars_idx_set and current in subs_dict and depth < max_depth:
        if current == var_idx: return True # Found the variable
        if current in visited_check: return True # Cycle detected, treat as occurrence
        visited_check.add(current)
        current = subs_dict[current]
        depth += 1

    # After the loop, check the final resolved value
    return current == var_idx

def batch_unify_with_rules(
    states_idx: torch.Tensor,             # Shape: (bs, 1, n_padding_atoms, 3)
    rules: torch.Tensor,                  # Shape: (n_rules, max_rule_atoms, 3) Head=0, Body=1:
    vars_idx: Union[torch.Tensor, List[int], Set[int], IndexManager], # Contains POSITIVE variable indices or IndexManager
    k: int,                               # Max rule unifications to return
    max_subs_per_rule: int = 5,           # Max substitutions tracked per unification
    device: Optional[torch.device] = None,
    verbose: bool = False,                # Main verbose flag for summary prints
    prover_verbose: int = 0,              # Controls detail level of unification debugging prints (0, 1, 2)
    index_manager_for_format: Optional[IndexManager] = None # Pass IndexManager instance for formatting
) -> UnificationResult:
    """
    Performs batched unification against RULE HEADS. Uses PADDING_VALUE=0.
    Identifies variables using positive indices in vars_idx_set.
    Calculates MGU iteratively per query-rule pair. Handles topk errors robustly.
    """
    if k <= 0: raise ValueError("k must be positive")
    if max_subs_per_rule <= 0: raise ValueError("max_subs_per_rule must be positive")

    effective_device = device if device else states_idx.device
    states_idx = states_idx.to(effective_device)
    rules = rules.to(effective_device)

    # Get variable set/tensor (now containing positive indices)
    vars_idx_set, vars_idx_tensor = _get_var_set_and_tensor(vars_idx, effective_device)

    bs, _, n_padding_atoms, _ = states_idx.shape
    if rules.numel() == 0:
        n_rules, max_rule_atoms = 0, 0
    else:
        n_rules, max_rule_atoms, _ = rules.shape
    rule_body_len = max(0, max_rule_atoms - 1)
    actual_k = min(k, n_rules) if n_rules > 0 else 0

    # --- Initialize Output Tensors (padding=0) ---
    substitutions_out = torch.full((bs, k, max_subs_per_rule, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
    rule_indices_out = torch.full((bs, k), PADDING_VALUE, dtype=torch.long, device=effective_device)
    rule_bodies_out_shape = (bs, k, rule_body_len, 3)
    rule_bodies_out = torch.full(rule_bodies_out_shape, PADDING_VALUE, dtype=torch.long, device=effective_device)
    valid_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=effective_device)

    # Handle edge cases
    if bs == 0 or n_rules == 0 or n_padding_atoms == 0 or max_rule_atoms <= 0:
        return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)

    # --- Prepare Inputs ---
    first_queries = states_idx[:, 0, 0, :] # (bs, 3)
    is_query_valid = ~torch.all(first_queries == PADDING_VALUE, dim=-1) # (bs,)

    rule_heads = rules[:, 0, :]    # (n_rules, 3)
    rule_bodies = rules[:, 1:, :]  # (n_rules, rule_body_len, 3)

    # --- Initial Predicate Match ---
    queries_expanded = first_queries.unsqueeze(1) # (bs, 1, 3)
    heads_expanded = rule_heads.unsqueeze(0)      # (1, n_rules, 3)
    # Predicate must match AND query must be valid (not padding)
    pred_match = (queries_expanded[:, :, 0] == heads_expanded[:, :, 0]) & is_query_valid.unsqueeze(1) # (bs, n_rules)

    # Precompute which query/head args are variables using torch.isin
    vars_1d = vars_idx_tensor.view(-1)
    is_query_s_var = torch.isin(first_queries[:, 1:2], vars_1d) # (bs, 1)
    is_query_o_var = torch.isin(first_queries[:, 2:3], vars_1d) # (bs, 1)
    is_head_s_var = torch.isin(rule_heads[:, 1:2], vars_1d)      # (n_rules, 1)
    is_head_o_var = torch.isin(rule_heads[:, 2:3], vars_1d)      # (n_rules, 1)

    # --- Perform MGU Calculation (Iterative per Query-Rule Pair) ---
    potential_subs_list = [[[] for _ in range(n_rules)] for _ in range(bs)]
    potential_valid = torch.zeros((bs, n_rules), dtype=torch.bool, device=effective_device)
    max_binding_depth = max_subs_per_rule * 2 + 10 # Safety depth for get_binding

    # --- Formatting Context ---
    # Use optional IndexManager instance passed for formatting
    im = index_manager_for_format
    const_no = getattr(im, 'constant_no', None) if im else None
    pred_dict = getattr(im, 'predicate_idx2str', {}) if im else {}
    const_dict = getattr(im, 'constant_idx2str', {}) if im else {}
    var_dict = getattr(im, 'variable_idx2str', {}) if im else {}


    for b in range(bs):
        if not is_query_valid[b]: continue
        if not torch.any(pred_match[b]): continue

        query = first_queries[b]
        q_p, q_s, q_o = query.tolist()
        is_q_s_var_b = is_query_s_var[b].item()
        is_q_o_var_b = is_query_o_var[b].item()

        # Debug Print Start
        if prover_verbose > 0 and b == 0 and im:
             print(f"--- MGU Start B={b}: Query = {_format_atom(query, pred_dict, const_dict, vars_idx_set, var_dict, const_no)} ---")

        for r in range(n_rules):
            if not pred_match[b, r]: continue

            head = rule_heads[r]
            h_p, h_s, h_o = head.tolist()
            is_h_s_var_r = is_head_s_var[r].item()
            is_h_o_var_r = is_head_o_var[r].item()

            # Debug Print Rule
            if prover_verbose > 1 and b == 0 and im:
                print(f"\nDEBUG: Unify B={b}, R={r} [Head: {_format_atom(head, pred_dict, const_dict, vars_idx_set, var_dict, const_no)}]")

            temp_subs = {}
            possible = True
            term_pairs = [(q_s, h_s, is_q_s_var_b, is_h_s_var_r),
                          (q_o, h_o, is_q_o_var_b, is_h_o_var_r)]

            term_pair_idx = 0
            for q_term, h_term, is_q_var, is_h_var in term_pairs:
                if not possible: break
                pair_label = "Subject" if term_pair_idx == 0 else "Object"

                # Debug Print Pair
                if prover_verbose > 1 and b == 0 and im:
                    q_term_str = _format_term(q_term, vars_idx_set, const_dict, var_dict, const_no)
                    h_term_str = _format_term(h_term, vars_idx_set, const_dict, var_dict, const_no)
                    print(f"  [{pair_label} Pair]: q={q_term_str}({q_term}), h={h_term_str}({h_term}), is_q_var={is_q_var}, is_h_var={is_h_var}")

                # 1. Resolve bindings using get_binding (uses vars_idx_set)
                q_final = get_binding(q_term, temp_subs, set(), vars_idx_set, max_binding_depth) if is_q_var else q_term
                h_final = get_binding(h_term, temp_subs, set(), vars_idx_set, max_binding_depth) if is_h_var else h_term

                if q_final is None or h_final is None: # Cycle/Depth check
                    if prover_verbose > 1 and b == 0: print(f"    Cycle/Depth detected during resolution!")
                    possible = False; break

                if q_final == h_final: # Already match
                     if prover_verbose > 1 and b == 0: print(f"    Resolved: q_final={q_final}, h_final={h_final}. Terms match, continue.")
                     term_pair_idx += 1; continue

                # Check if resolved terms are variables using vars_idx_set
                is_q_final_var = q_final in vars_idx_set
                is_h_final_var = h_final in vars_idx_set

                # Debug Print Resolved
                if prover_verbose > 1 and b == 0 and im:
                    q_final_str = _format_term(q_final, vars_idx_set, const_dict, var_dict, const_no)
                    h_final_str = _format_term(h_final, vars_idx_set, const_dict, var_dict, const_no)
                    print(f"    Resolved: q_final={q_final_str}({q_final}), h_final={h_final_str}({h_final})")
                    print(f"    is_q_final_var={is_q_final_var}, is_h_final_var={is_h_final_var}")

                # 2. Unify based on variable status
                if is_q_final_var: # Case 1: Query term is variable
                    if occurs_check(q_final, h_final, temp_subs, vars_idx_set, max_binding_depth): # Uses vars_idx_set
                        if prover_verbose > 1 and b==0 and im: print(f"    Occurs Check Failed: {_format_term(q_final, vars_idx_set, None, var_dict)} in {_format_term(h_final, vars_idx_set, const_dict, var_dict, const_no)}")
                        possible = False; break
                    if prover_verbose > 1 and b==0 and im: print(f"    Binding q_final:{_format_term(q_final, vars_idx_set, None, var_dict)}({q_final}) -> h_final:{_format_term(h_final, vars_idx_set, const_dict, var_dict, const_no)}({h_final})")
                    temp_subs[q_final] = h_final

                elif is_h_final_var: # Case 2: Head term is variable
                    if occurs_check(h_final, q_final, temp_subs, vars_idx_set, max_binding_depth): # Uses vars_idx_set
                        if prover_verbose > 1 and b==0 and im: print(f"    Occurs Check Failed: {_format_term(h_final, vars_idx_set, None, var_dict)} in {_format_term(q_final, vars_idx_set, const_dict, var_dict, const_no)}")
                        possible = False; break
                    if prover_verbose > 1 and b==0 and im: print(f"    Binding h_final:{_format_term(h_final, vars_idx_set, None, var_dict)}({h_final}) -> q_final:{_format_term(q_final, vars_idx_set, const_dict, var_dict, const_no)}({q_final})")
                    temp_subs[h_final] = q_final

                else: # Case 3: Both constants/padding, check for mismatch
                    if q_final != h_final: # Mismatch only if values are different
                         if prover_verbose > 1 and b==0 and im: print(f"    Term Mismatch: {_format_term(q_final, vars_idx_set, const_dict, var_dict, const_no)} != {_format_term(h_final, vars_idx_set, const_dict, var_dict, const_no)}")
                         possible = False; break
                    # else: q_final == h_final (both padding or same constant), handled earlier

                # Debug Print State After Pair
                if prover_verbose > 1 and b == 0 and im:
                    subs_str = {_format_term(k, vars_idx_set, None, var_dict): _format_term(v, vars_idx_set, const_dict, var_dict, const_no) for k,v in temp_subs.items()}
                    print(f"    -> Possible: {possible}, New temp_subs: {subs_str}")

                term_pair_idx += 1
            # --- End Inner Loop ---

            # Debug Print Before Finalize
            if prover_verbose > 1 and b == 0: print(f"  End Pairs Loop. Possible: {possible}")

            if possible: # Finalize substitutions if unification didn't fail
                final_subs_list = []
                processed_vars = set()
                possible_final = True

                if prover_verbose > 1 and b == 0: print(f"  Attempting Finalization. Initial Keys: {list(temp_subs.keys())}")

                # Iterate over variables initially bound
                for var_start in list(temp_subs.keys()):
                    if var_start not in vars_idx_set: continue # Should be a variable
                    if var_start in processed_vars: continue

                    # Find ultimate binding using current subs and vars_idx_set
                    final_val = get_binding(var_start, temp_subs, set(), vars_idx_set, max_binding_depth)

                    # Debug Print Finalization Step
                    if prover_verbose > 1 and b == 0 and im:
                         var_start_str = var_dict.get(var_start,"?")
                         final_val_str = _format_term(final_val, vars_idx_set, const_dict, var_dict, const_no) if final_val is not None else "None"
                         print(f"    Finalizing Key: {var_start_str}({var_start}) -> Resolved: {final_val_str}({final_val})")

                    if final_val is None: # Cycle check
                        if prover_verbose > 1 and b == 0: print(f"      Cycle/Depth detected during finalization!")
                        possible_final = False; break

                    # Add substitution only if var binds to something different
                    if var_start != final_val:
                        if prover_verbose > 1 and b == 0: print(f"      Adding Substitution: [{var_start}, {final_val}]")
                        final_subs_list.append([var_start, final_val])

                    processed_vars.add(var_start)
                    if final_val in vars_idx_set: processed_vars.add(final_val) # Mark target as processed too if var

                # Check substitution count limit
                if possible_final and len(final_subs_list) <= max_subs_per_rule:
                    if prover_verbose > 1 and b == 0: print(f"  Finalization SUCCEEDED for R={r}. Num Subs: {len(final_subs_list)}")
                    potential_valid[b, r] = True
                    potential_subs_list[b][r] = final_subs_list
                elif prover_verbose > 1 and b == 0: # Log failure reason
                    reason = "Cycle/Depth" if not possible_final else f"Too many subs ({len(final_subs_list)} > {max_subs_per_rule})"
                    print(f"  Finalization FAILED for R={r}. Reason: {reason}")
            # --- End Finalization Step ---

        # Debug Print End Rule Checks
        if prover_verbose > 0 and b == 0:
             valid_rules_indices = torch.where(potential_valid[b])[0].tolist()
             print(f"--- MGU End B={b}: Valid Rule Indices = {valid_rules_indices} ---")

    # --- Select Top K Valid Unifications ---
    if actual_k == 0:
        return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)

    scores = torch.where(potential_valid, 1.0, float('-inf'))
    num_valid_per_batch = potential_valid.sum(dim=1)
    if torch.all(num_valid_per_batch == 0): # No valid unifications found at all
        return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)

    # Ensure k does not exceed the number of potential rules
    k_for_topk = min(actual_k, n_rules)
    if k_for_topk == 0: # Should be caught above, but safety
        return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)

    try:
        # Attempt topk
        top_scores, top_indices_in_rules = torch.topk(scores, k=k_for_topk, dim=1)

    except RuntimeError as e:
        logging.error(f"RuntimeError during rule topk: {e}. Scores shape: {scores.shape}, k_for_topk: {k_for_topk}, n_rules: {n_rules}")
        # Check if the error is k > dimension size
        if "must be less than or equal to the dimension size" in str(e) or scores.shape[1] < k_for_topk:
            logging.warning(f"Adjusting k for rule topk because requested k ({k_for_topk}) > n_rules ({scores.shape[1]}).")
            k_for_topk = scores.shape[1] # Adjust k
            if k_for_topk == 0:
                logging.warning("Returning empty result because n_rules is 0 after adjustment.")
                return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)
            try:
                 # Retry topk with adjusted k
                 top_scores, top_indices_in_rules = torch.topk(scores, k=k_for_topk, dim=1)
                 logging.info(f"Successfully retried rule topk with adjusted k={k_for_topk}.")
            except RuntimeError as e_retry:
                 logging.error(f"FATAL: Retrying rule topk failed even with adjusted k={k_for_topk}: {e_retry}")
                 raise e_retry # Re-raise error from retry
        else:
            # Unexpected RuntimeError
            logging.error(f"FATAL: Encountered an unexpected RuntimeError during rule topk that wasn't k > dim_size.")
            raise e # Re-raise original exception

    # Mask indicating which of the top-k slots correspond to actual valid unifications
    final_valid_k_mask = (top_scores > float('-inf')) # Shape: (bs, k_for_topk)

    # --- Gather Results for Top K ---
    # Get batch and k coordinates of the final valid unifications
    final_valid_coords = torch.nonzero(final_valid_k_mask, as_tuple=True)
    n_final_valid = len(final_valid_coords[0])

    if n_final_valid > 0:
        final_bs_indices, final_k_indices = final_valid_coords
        # Get the original rule indices corresponding to these top-k slots
        final_rule_indices = top_indices_in_rules[final_valid_coords] # Shape: (n_final_valid,)

        # --- Populate Output Tensors ---
        # 1. Rule Indices
        rule_indices_out[final_bs_indices, final_k_indices] = final_rule_indices

        # 2. Rule Bodies (if they exist)
        if rule_body_len > 0:
            gathered_bodies = rule_bodies[final_rule_indices] # Shape: (n_final_valid, rule_body_len, 3)
            rule_bodies_out[final_bs_indices, final_k_indices] = gathered_bodies

        # 3. Substitutions
        # Create a temporary tensor to hold substitutions before scattering
        temp_subs_tensor = torch.full((n_final_valid, max_subs_per_rule, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
        for i in range(n_final_valid):
            b_final = final_bs_indices[i].item()
            rule_idx_final = final_rule_indices[i].item()
            # Retrieve the pre-calculated substitution list for this valid unification
            subs_list = potential_subs_list[b_final][rule_idx_final]
            num_subs = len(subs_list)
            if num_subs > 0:
                # Convert list to tensor and place in the temporary tensor
                subs_tensor_entry = torch.tensor(subs_list, dtype=torch.long, device=effective_device)
                temp_subs_tensor[i, :num_subs, :] = subs_tensor_entry

        # Scatter the substitutions into the output tensor at the correct (bs, k) locations
        substitutions_out[final_bs_indices, final_k_indices] = temp_subs_tensor

    # Populate the final output mask (up to k_for_topk)
    valid_mask_out[:, :k_for_topk] = final_valid_k_mask

    # Debug Print Final Mask
    if prover_verbose > 1:
         if bs > 0: # Print only if batch size is > 0
             print(f"DEBUG B=0 Final valid_mask_out[0, :k_for_topk]: {valid_mask_out[0, :k_for_topk]}")

    # Use main verbose flag for this summary print
    if verbose and prover_verbose > 0:
        print("-" * 40 + f"\nDEBUG: batch_unify_with_rules (End - B={bs}, Positive Indices) - Valid Counts: {potential_valid.sum(dim=1).tolist()}\n" + "=" * 40)

    return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)




# --- Modified apply_substitutions_and_create_next_state (Keep as is) ---
class NextStateResult(NamedTuple):
    """Structure to hold results from creating next states."""
    next_states: torch.Tensor      # Shape: (bs, k, new_max_atoms, 3)
    is_proven_mask: torch.Tensor   # Shape: (bs, k) - True if the state represents a proof

def apply_substitutions_and_create_next_state(
    states_idx: torch.Tensor,             # Shape: (bs, 1, n_padding_atoms, 3)
    unification_result: UnificationResult,# Result from fact or rule unification
    vars_idx_tensor: torch.Tensor,        # 1D Tensor of POSITIVE variable indices
    new_max_atoms: Optional[int] = None,
    max_sub_iterations: int = 10
) -> NextStateResult:
    """
    Generates next goal states. Uses PADDING_VALUE=0.
    Applies substitutions (using positive var indices) and concatenates results.
    """
    bs, _, n_padding_atoms, _ = states_idx.shape
    k = unification_result.substitutions.shape[1]
    device = states_idx.device

    substitutions = unification_result.substitutions # (bs, k, max_subs, 2)
    valid_mask = unification_result.valid_mask       # (bs, k)
    rule_bodies = unification_result.target_bodies   # (bs, k, rule_body_len, 3) or None

    has_rule_bodies = rule_bodies is not None and rule_bodies.numel() > 0 and rule_bodies.shape[-1] == 3
    rule_body_len = rule_bodies.shape[2] if has_rule_bodies else 0
    num_remaining_atoms_orig = max(0, n_padding_atoms - 1)
    if new_max_atoms is None: new_max_atoms = num_remaining_atoms_orig + rule_body_len
    if new_max_atoms < 0: new_max_atoms = 0

    # Initialize outputs (padding=0)
    next_states_out = torch.full((bs, k, new_max_atoms, 3), PADDING_VALUE, dtype=torch.long, device=device)
    is_proven_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=device)

    remaining_queries_base = states_idx[:, 0, 1:, :] if n_padding_atoms > 1 else torch.empty((bs, 0, 3), dtype=torch.long, device=device)

    # --- Process only valid unifications ---
    valid_coords = torch.nonzero(valid_mask, as_tuple=True)
    n_valid = len(valid_coords[0])

    if n_valid == 0:
        return NextStateResult(next_states_out, is_proven_mask_out)

    valid_bs_indices, valid_k_indices = valid_coords
    valid_subs = substitutions[valid_bs_indices, valid_k_indices] # (n_valid, max_subs, 2)
    valid_rem_q = remaining_queries_base[valid_bs_indices]          # (n_valid, num_remaining_atoms_orig, 3)
    valid_bodies = rule_bodies[valid_bs_indices, valid_k_indices] if has_rule_bodies else torch.empty((n_valid, 0, 3), dtype=torch.long, device=device) # (n_valid, rule_body_len, 3)

    # --- Apply Substitutions (Vectorized - uses PADDING_VALUE=0 internally) ---
    subst_rem_q_flat = apply_substitutions_vectorized(
        valid_rem_q, valid_subs, vars_idx_tensor, max_sub_iterations
    ) # Shape: (n_valid, num_remaining_atoms_orig, 3)

    subst_body_flat = torch.empty((n_valid, 0, 3), dtype=torch.long, device=device)
    if rule_body_len > 0:
        subst_body_flat = apply_substitutions_vectorized(
            valid_bodies, valid_subs, vars_idx_tensor, max_sub_iterations
        ) # Shape: (n_valid, rule_body_len, 3)

    # --- Filter Padding (0), Concatenate, Check Proven, Pad/Truncate (Iterative Part) ---
    is_proven_flat = torch.zeros(n_valid, dtype=torch.bool, device=device)

    for i in range(n_valid):
        b = valid_bs_indices[i].item()
        k_idx = valid_k_indices[i].item()

        current_body = subst_body_flat[i]
        current_rem_q = subst_rem_q_flat[i]

        # Filter out padding atoms (where ALL P, S, O are 0)
        # Using torch.any check is more robust than checking just predicate
        body_keep_mask = torch.any(current_body != PADDING_VALUE, dim=-1) # (rule_body_len,)
        rem_q_keep_mask = torch.any(current_rem_q != PADDING_VALUE, dim=-1)# (num_remaining_atoms_orig,)

        subst_body_filtered = current_body[body_keep_mask]
        subst_rem_q_filtered = current_rem_q[rem_q_keep_mask]

        concatenated_atoms = torch.cat((subst_body_filtered, subst_rem_q_filtered), dim=0)

        if concatenated_atoms.shape[0] == 0: # Proof complete if no atoms left
            is_proven_flat[i] = True

        current_len = concatenated_atoms.shape[0]
        len_to_copy = min(current_len, new_max_atoms)
        if len_to_copy > 0:
            next_states_out[b, k_idx, :len_to_copy, :] = concatenated_atoms[:len_to_copy, :]
        # Rest remains PADDING_VALUE (0)

    is_proven_mask_out[valid_bs_indices, valid_k_indices] = is_proven_flat

    return NextStateResult(next_states_out, is_proven_mask_out)


# --- Placeholder DataHandler (Keep as is) ---
class DataHandler:
    """Minimal placeholder for DataHandler."""
    def __init__(self, facts: List[Term], rules: List[Rule], train_queries: List[Term], valid_queries: List[Term]):
        self.facts = facts
        self.rules = rules
        self.train_queries = train_queries
        self.valid_queries = valid_queries
        # Assume labels are all 1 for simplicity here
        self.train_labels = [1] * len(train_queries)
        self.valid_labels = [1] * len(valid_queries)
        self.sampler = None

# --- Modified Environment (Keep as is, uses corrected unification) ---
class LogicEnv_gym_batch(gym.Env):
    """
    Gymnasium environment using batched tensor operations and integrated unification.
    Relies solely on sub-indices (P, S, O) for state representation.
    Differentiates between Proven, Failed, and Evolved states.
    """
    metadata = {'render_modes': [], 'render_fps': 1}

    def __init__(self,
                 batch_size: int,
                 index_manager: IndexManager,
                 data_handler: DataHandler,
                 mode: str = 'train',
                 corruption_mode: Optional[str] = None,
                 train_neg_pos_ratio: int = 1,
                 seed: Optional[int] = None,
                 max_depth: int = 10,
                 end_proof_action: bool = False, # Currently unused, logic handles True/False
                 skip_unary_actions: bool = False,
                 truncate_atoms: bool = False,
                 padding_states: int = 20, # Max number of derived states (actions)
                 k_facts_rules: Optional[int] = None,
                 max_subs_per_rule: int = 5,
                 max_sub_iterations: int = 10,
                 reward_proven: float = 1.0, # Reward for reaching a proven state (if label=1)
                 reward_failed: float = 0.0, # Reward for reaching a failed state (no unification)
                 reward_false: float = 0.0,  # Reward for reaching an explicit False state
                 verbose: int = 0,
                 prover_verbose: int = 0,
                 device: torch.device = torch.device("cpu"),
                 ):

        super().__init__()

        self.batch_size = batch_size
        self.index_manager = index_manager
        self.data_handler = data_handler
        self.device = device
        self.verbose = verbose
        self.prover_verbose = prover_verbose # Controls verbosity inside unification funcs

        self.max_arity = self.index_manager.max_arity
        if self.max_arity != 2: raise ValueError("Environment requires max_arity=2")
        self.padding_atoms = self.index_manager.padding_atoms
        self.padding_states = padding_states # Max actions / derived states
        self.max_depth = max_depth
        # Default k for unification is padding_states if not specified
        self.k_facts_rules = k_facts_rules if k_facts_rules is not None else self.padding_states
        self.max_subs_per_rule = max_subs_per_rule
        self.max_sub_iterations = max_sub_iterations # For apply_substitutions_vectorized

        # Get special predicate indices
        self.true_idx = self.index_manager.true_idx
        self.false_idx = self.index_manager.false_idx
        self.end_idx = self.index_manager.end_idx # Keep for potential future use
        if self.true_idx < 0 or self.false_idx < 0: # End index is optional
            raise RuntimeError("IndexManager did not properly initialize True/False predicate indices.")

        self._set_seed(seed)

        # Store data and convert to tensors
        self.facts = self.data_handler.facts
        self.rules = self.data_handler.rules
        self.facts_idx_tensor = self.index_manager.facts_to_tensor(self.facts)
        self.rules_idx_tensor, self.max_atoms_in_rules = self.index_manager.rules_to_tensor(self.rules)
        logging.info(f"Facts tensor shape: {self.facts_idx_tensor.shape}")
        logging.info(f"Rules tensor shape: {self.rules_idx_tensor.shape}, Max atoms in rule: {self.max_atoms_in_rules}")

        # Runtime state variables (per batch item)
        self.current_depth = torch.zeros(self.batch_size, dtype=torch.int, device=self.device)
        self.current_queries_terms: List[Optional[Term]] = [None] * self.batch_size # Store initial query for info


        # Configuration Flags
        self.corruption_mode = corruption_mode # For training data sampling
        self.train_neg_pos_ratio = train_neg_pos_ratio # For training data sampling
        self.end_proof_action = end_proof_action
        self.skip_unary_actions = skip_unary_actions
        self.truncate_atoms = truncate_atoms
        self.reward_proven = reward_proven
        self.reward_failed = reward_failed
        self.reward_false = reward_false

        # Mode and Query Lists for sampling/evaluation
        assert mode in ['train', 'eval', 'eval_corr'], f"Invalid mode: {mode}"
        self.mode = mode
        self.train_queries = self.data_handler.train_queries
        self.eval_queries = self.data_handler.valid_queries
        self.n_train_queries = len(self.train_queries) if self.train_queries else 0
        self.n_eval_queries = len(self.eval_queries) if self.eval_queries else 0
        self.eval_indices = list(range(self.n_eval_queries))
        self.current_eval_batch_idx = 0 # Tracks position in eval data

        # Check training requirements
        if self.mode == 'train':
            if self.corruption_mode == "dynamic":
                 if not self.data_handler.sampler: raise ValueError("Dynamic corruption requires sampler.")
                 if not self.train_queries: raise ValueError("Dynamic corruption requires train_queries.")
            elif not self.train_queries: raise ValueError("Training mode requires train_queries.")

        # Define observation/action spaces and initialize TensorDict
        self._make_spec()
        self.tensordict = self._create_empty_tensordict()

    def _set_seed(self, seed: Optional[int]):
        """Set the seed for the environment."""
        super().reset(seed=seed)
        if seed is None: seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # logging.info(f"Environment seeded with {seed}")

    def _create_empty_tensordict(self) -> TensorDict:
        """Creates the TensorDict structure including new fields."""
        td_data = {
            # Current state goal stack (sub-indices)
            "state_sub_idx": torch.full((self.batch_size, self.padding_atoms, 3), PADDING_VALUE, dtype=torch.int64, device=self.device),
            # Potential next states derived from current state (actions)
            "derived_sub_idx": torch.full((self.batch_size, self.padding_states, self.padding_atoms, 3), PADDING_VALUE, dtype=torch.int64, device=self.device),
            # Mask indicating which derived states are valid actions
            "action_mask": torch.zeros(self.batch_size, self.padding_states, dtype=torch.bool, device=self.device),
            # Mask indicating if a valid action leads directly to a proven state (empty goal stack)
            "proven_mask": torch.zeros(self.batch_size, self.padding_states, dtype=torch.bool, device=self.device),
            # Flag set during state derivation if *no* unification was possible
            "failed_unification": torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
            # Original query label (e.g., 1 for provable, 0 for not) - remains constant
            "label": torch.zeros(self.batch_size, dtype=torch.int, device=self.device),
            # Standard RL fields
            "done": torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
            "reward": torch.zeros(self.batch_size, dtype=torch.float32, device=self.device),
        }
        return TensorDict(td_data, batch_size=[self.batch_size], device=self.device)

    def _make_spec(self):
        """Create the batched observation and action specs including new fields."""
        # Observation space includes state and masks needed for decision making
        obs_spaces = {
            'state_sub_idx': gym.spaces.Box(np.iinfo(np.int64).min, np.iinfo(np.int64).max, (self.batch_size, self.padding_atoms, 3), dtype=np.int64),
            'derived_sub_idx': gym.spaces.Box(np.iinfo(np.int64).min, np.iinfo(np.int64).max, (self.batch_size, self.padding_states, self.padding_atoms, 3), dtype=np.int64),
            'action_mask': gym.spaces.Box(0, 1, (self.batch_size, self.padding_states), dtype=np.bool_),
            'proven_mask': gym.spaces.Box(0, 1, (self.batch_size, self.padding_states), dtype=np.bool_), # Include proven mask in observation
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        # Action space: Choose one of the `padding_states` derived states
        # self.action_space = gym.spaces.MultiDiscrete([self.padding_states] * self.batch_size)
        self.action_space = gym.spaces.Discrete(self.padding_states)

    def _sample_initial_batch(self) -> Tuple[List[Term], List[int]]:
        """Samples a batch of initial queries (Terms) and labels for TRAINING mode."""
        batch_queries_terms: List[Term] = []
        batch_labels: List[int] = []

        if not self.train_queries:
            raise ValueError("Cannot sample for training, train_queries list is empty.")

        # Simplified sampling: Randomly choose from train_queries
        # (Add dynamic corruption logic here if needed based on self.corruption_mode)
        if self.corruption_mode == "dynamic":
            raise NotImplementedError("Dynamic negative sampling logic needs to be implemented/verified.")
        else: # No corruption or static corruption assumed handled by data_handler labels
            sampled_indices = random.choices(range(self.n_train_queries), k=self.batch_size)
            batch_queries_terms = [self.train_queries[i] for i in sampled_indices]
            # Get labels (default to 1 if not provided)
            train_labels = getattr(self.data_handler, "train_labels", [1]*self.n_train_queries)
            batch_labels = [train_labels[i] for i in sampled_indices]

        # Ensure final batch size is correct (e.g., if n_train_queries < batch_size)
        if len(batch_queries_terms) != self.batch_size:
            logging.warning(f"Batch size mismatch after sampling: {len(batch_queries_terms)} vs {self.batch_size}. Padding/Truncating...")
            if len(batch_queries_terms) > self.batch_size:
                batch_queries_terms = batch_queries_terms[:self.batch_size]
                batch_labels = batch_labels[:self.batch_size]
            else: # Pad with first query if needed
                num_missing = self.batch_size - len(batch_queries_terms)
                if not self.train_queries: raise ValueError("Cannot pad batch, no train queries.")
                first_query = self.train_queries[0]
                first_label = getattr(self.data_handler, "train_labels", [1]*self.n_train_queries)[0]
                batch_queries_terms.extend([first_query] * num_missing)
                batch_labels.extend([first_label] * num_missing)

        return batch_queries_terms, batch_labels

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None, # Standard gym argument, not used here
              queries_to_evaluate: Optional[List[Term]] = None, # Allow passing specific queries
              labels_to_evaluate: Optional[List[int]] = None   # Allow passing specific labels
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]: # Returns obs, info
        """Resets environments, optionally using provided queries/labels or sampling based on mode."""
        super().reset(seed=seed)
        if seed is not None: self._set_seed(seed)
        logging.info(f'Resetting Batch (Size: {self.batch_size})')

        batch_queries_terms: List[Term]
        batch_labels: List[int]

        # --- Logic for selecting initial queries ---
        if queries_to_evaluate is not None:
            # Use provided queries/labels (e.g., for specific evaluation)
            logging.info("Resetting with provided queries for evaluation.")
            if len(queries_to_evaluate) != self.batch_size: raise ValueError(f"Provided queries length ({len(queries_to_evaluate)}) must match batch_size ({self.batch_size}).")
            if labels_to_evaluate is None: labels_to_evaluate = [1] * self.batch_size; logging.warning("Assuming label 1 for provided queries.")
            elif len(labels_to_evaluate) != self.batch_size: raise ValueError(f"Provided labels length ({len(labels_to_evaluate)}) must match batch_size ({self.batch_size}).")
            batch_queries_terms = queries_to_evaluate
            batch_labels = labels_to_evaluate
            self.mode = 'eval' # Treat as evaluation if specific queries are given

        elif self.mode == 'eval' or self.mode == 'eval_corr':
            # Cycle through evaluation queries
            if not self.eval_queries: raise ValueError("Cannot reset in eval mode, eval_queries list is empty.")
            start_idx = self.current_eval_batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            # Wrap around if end of eval data is reached
            if start_idx >= self.n_eval_queries:
                start_idx = 0; end_idx = self.batch_size; self.current_eval_batch_idx = 0
                logging.warning("Eval reset wrapping around to the beginning of eval data.")

            # Get the slice of eval data for this batch
            actual_indices = self.eval_indices[start_idx : min(end_idx, self.n_eval_queries)]
            batch_queries_terms = [self.eval_queries[i] for i in actual_indices]
            eval_labels = getattr(self.data_handler, "valid_labels", [1]*self.n_eval_queries)
            batch_labels = [eval_labels[i] for i in actual_indices]

            # Pad if the last batch is smaller than batch_size
            num_missing = self.batch_size - len(batch_queries_terms)
            if num_missing > 0:
                logging.info(f"Padding eval batch with {num_missing} initial eval queries.")
                if not self.eval_queries: raise ValueError("Cannot pad eval batch, list is empty.")
                first_eval_query = self.eval_queries[0]
                first_eval_label = eval_labels[0]
                batch_queries_terms.extend([first_eval_query] * num_missing)
                batch_labels.extend([first_eval_label] * num_missing)
            self.current_eval_batch_idx += 1 # Move to the next batch for the next reset

        elif self.mode == 'train':
            # Sample training queries
            batch_queries_terms, batch_labels = self._sample_initial_batch()
        else:
            raise ValueError(f"Invalid mode for reset: {self.mode}")

        # --- Initialize Environment State ---
        self.current_queries_terms = batch_queries_terms # Store for info
        current_labels = torch.tensor(batch_labels, dtype=torch.int, device=self.device)
        self.current_depth.zero_() # Reset depth counter

        # Convert initial query Terms to sub-index tensors
        initial_sub_idx_list = []
        for i in range(self.batch_size):
            query_term = batch_queries_terms[i]
            if isinstance(query_term, Term):
                initial_state_terms = [query_term]
            elif isinstance(query_term, list):
                initial_state_terms = query_term
            else:
                raise ValueError("query_term must be a Term or a list of Terms", f"got {type(query_term)}.")
            # Handle potentially empty initial state?
            if not initial_state_terms:
                 logging.warning(f"Empty initial state for batch item {i}. Using 'False'.")
                 initial_state_terms = [Term("False", [])] # Default to False if query is bad
            print(f"Initial state for batch item {i}: {initial_state_terms}")
            sub_idx = self.index_manager.get_state_sub_index_tensor(initial_state_terms)
            initial_sub_idx_list.append(sub_idx)

        initial_sub_idx_batch = torch.stack(initial_sub_idx_list) # (bs, padding_atoms, 3)

        # --- Pre-calculate First Set of Possible Actions/States ---
        # This determines the initial action mask, proven mask, etc.
        derived_sub_idx_batch, truncated_flags_batch, action_masks_batch, proven_masks_batch, failed_unif_batch = self.get_next_states_batch(initial_sub_idx_batch)

        # --- Update TensorDict with Initial State ---
        self.tensordict.update_(TensorDict({
                "state_sub_idx": initial_sub_idx_batch,
                "derived_sub_idx": derived_sub_idx_batch,
                "action_mask": action_masks_batch,
                "proven_mask": proven_masks_batch,
                "failed_unification": failed_unif_batch, # Crucial: was unification possible at start?
                "label": current_labels,
                "done": torch.zeros(self.batch_size, dtype=torch.bool, device=self.device), # Start not done
                "reward": torch.zeros(self.batch_size, dtype=torch.float32, device=self.device), # Start with zero reward
            }, batch_size=[self.batch_size]))

        if torch.any(truncated_flags_batch): # Log if issues occurred during initial state derivation
            logging.warning(f"Truncation or filtering issues occurred during reset for indices: {torch.where(truncated_flags_batch)[0].tolist()}")

        # Prepare observation and info dict for return
        obs = self._get_obs_from_tensordict(self.tensordict)
        info = {} # Standard Gymnasium practice: return empty info dict on reset

        if self.verbose > 1: self._log_batch_state("Reset")
        return obs, info


    def get_next_states_batch(self,
                              current_sub_idx: torch.Tensor, # Shape (bs, pad_atoms, 3)
                              dones_mask: Optional[torch.Tensor] = None # Shape (bs,) - To avoid processing done states
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets the next possible states using the integrated unification engine (Vectorized).
        Gathers all valid next states (from facts then rules), keeps up to padding_states,
        and determines proof/failure status using vectorized operations.
        **NEW**: Explicitly handles cases where the current state is exactly 'True' or 'False'.
                 In such cases, it sets the next state to be the same ('True'/'False')
                 and makes it the only valid action. This applies even if dones_mask is True
                 for the transition calculation, ensuring the observation upon termination is correct.
        Output shapes:
         - derived_sub_idx: (bs, padding_states, padding_atoms, 3)
         - truncated_flags: (bs,) - Indicates if filtering caused issues (e.g., > padding_states valid actions)
         - action_masks: (bs, padding_states) - Valid actions
         - proven_masks: (bs, padding_states) - Actions leading to proof
         - failed_unification_mask: (bs,) - True if no unification possible from current state OR current state is False
        """
        bs = current_sub_idx.shape[0]
        if dones_mask is None:
            dones_mask = torch.zeros(bs, dtype=torch.bool, device=self.device)

        # --- Check for Terminal True/False States FIRST (regardless of dones_mask) ---
        # A state is terminal True/False if the first atom is True/False and the rest are padding
        first_atom_pred = current_sub_idx[:, 0, 0] # Predicate index of the first atom (bs,)
        # Check if *all* subsequent atoms (from index 1 onwards) are padding
        # Handle edge case where padding_atoms might be 1
        if self.padding_atoms > 1:
             rest_is_padding = torch.all(current_sub_idx[:, 1:] == PADDING_VALUE, dim=(1, 2)) # (bs,)
        else:
             rest_is_padding = torch.ones(bs, dtype=torch.bool, device=self.device) # If only 1 atom allowed, rest is always "padding"

        is_terminal_true = (first_atom_pred == self.true_idx) & rest_is_padding
        is_terminal_false = (first_atom_pred == self.false_idx) & rest_is_padding
        is_already_terminal = is_terminal_true | is_terminal_false

        # --- Initialize outputs ---
        filtered_derived_sub_idx = torch.full((bs, self.padding_states, self.padding_atoms, 3), PADDING_VALUE, dtype=torch.long, device=self.device)
        filtered_action_masks = torch.zeros((bs, self.padding_states), dtype=torch.bool, device=self.device)
        filtered_proven_masks = torch.zeros((bs, self.padding_states), dtype=torch.bool, device=self.device)
        batch_truncated_flags = torch.zeros(bs, dtype=torch.bool, device=self.device) # Assume no truncation initially
        failed_unification_mask = torch.zeros(bs, dtype=torch.bool, device=self.device)

        # --- Handle Non-Terminal States via Unification ---
        # Identify states that need standard unification processing (not terminal AND not already done)
        needs_unification_mask = ~is_already_terminal & ~dones_mask
        if torch.any(needs_unification_mask):
            states_to_unify = current_sub_idx[needs_unification_mask]
            bs_unify = states_to_unify.shape[0]
            k = self.k_facts_rules # Cache k value
            k_total = 2 * k
            current_state_reshaped = states_to_unify.unsqueeze(1) # (bs_unify, 1, pad_atoms, 3)

            # --- Run Unification ---
            fact_results = batch_unify_with_facts(
                states_idx=current_state_reshaped, facts=self.facts_idx_tensor,
                vars_idx=self.index_manager, k=k, device=self.device,
                verbose=self.prover_verbose > 1, # Pass verbose flag
                # CORRECT Argument: Pass the IndexManager instance for formatting
                index_manager_for_format=self.index_manager if self.prover_verbose > 1 else None
            )
            rule_results = batch_unify_with_rules(
                states_idx=current_state_reshaped, rules=self.rules_idx_tensor,
                vars_idx=self.index_manager, k=k, max_subs_per_rule=self.max_subs_per_rule,
                device=self.device, verbose=self.prover_verbose > 0, prover_verbose=self.prover_verbose,
                # CORRECT Argument: Pass the IndexManager instance for formatting
                index_manager_for_format=self.index_manager if self.prover_verbose > 0 else None
            )

            # --- Apply Substitutions and Generate Next States ---
            next_state_fact_result = apply_substitutions_and_create_next_state(
                states_idx=current_state_reshaped, unification_result=fact_results,
                vars_idx_tensor=self.index_manager.vars_idx_tensor, new_max_atoms=self.padding_atoms,
                max_sub_iterations=self.max_sub_iterations
            )
            next_state_rule_result = apply_substitutions_and_create_next_state(
                states_idx=current_state_reshaped, unification_result=rule_results,
                vars_idx_tensor=self.index_manager.vars_idx_tensor, new_max_atoms=self.padding_atoms,
                max_sub_iterations=self.max_sub_iterations
            )

            # --- Combine Potential Results ---
            combined_states = torch.cat((next_state_fact_result.next_states, next_state_rule_result.next_states), dim=1)
            combined_valid = torch.cat((fact_results.valid_mask, rule_results.valid_mask), dim=1)
            combined_proven = torch.cat((next_state_fact_result.is_proven_mask, next_state_rule_result.is_proven_mask), dim=1)

            # --- Prioritize and Select Top `padding_states` Valid Actions ---
            valid_counts_per_batch = combined_valid.sum(dim=1) # (bs_unify,)
            num_to_keep_per_batch = torch.minimum(valid_counts_per_batch, torch.tensor(self.padding_states, device=self.device)) # (bs_unify,)

            ramp = torch.arange(k_total, device=self.device).float() * -1e-5
            scores = combined_valid.float() + ramp.unsqueeze(0) # (bs_unify, 2*k)
            _, top_indices = torch.topk(scores, k=k_total, dim=1) # (bs_unify, 2*k)
            indices_to_gather = top_indices[:, :self.padding_states] # (bs_unify, padding_states)

            # --- Gather the Prioritized Results ---
            idx_gather_expanded = indices_to_gather.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.padding_atoms, 3)
            gathered_states = torch.gather(combined_states, 1, idx_gather_expanded) # (bs_unify, padding_states, pad_atoms, 3)
            gathered_proven = torch.gather(combined_proven, 1, indices_to_gather) # (bs_unify, padding_states)

            # --- Create Action Mask based on actual number kept ---
            range_tensor = torch.arange(self.padding_states, device=self.device).unsqueeze(0) # (1, padding_states)
            base_action_mask = (range_tensor < num_to_keep_per_batch.unsqueeze(1)) # (bs_unify, padding_states)

            # --- Determine Failure and Truncation for this subset ---
            subset_failed_unification = (valid_counts_per_batch == 0) # (bs_unify,) - No valid unifications found
            subset_truncated = (valid_counts_per_batch > self.padding_states) # (bs_unify,)

            # --- Insert 'False' state for unification failures within this subset ---
            false_state_tensor = None
            false_fits = False
            if torch.any(subset_failed_unification):
                false_term = Term("False", [])
                false_sub_idx_single = self.index_manager.get_state_sub_index_tensor([false_term]) # (pad_atoms, 3)
                false_state_tensor = false_sub_idx_single.unsqueeze(0) # (1, pad_atoms, 3)
                false_fits = self.padding_atoms >= false_sub_idx_single.shape[0]

                if false_fits:
                    if self.verbose > 0: logging.info(f"Vectorized (Subset): Inserting False for failed unification.")
                    gathered_states = torch.where(subset_failed_unification.view(bs_unify, 1, 1, 1), false_state_tensor, gathered_states)
                    base_action_mask = torch.where(subset_failed_unification.unsqueeze(1),
                                                   torch.cat([torch.ones(bs_unify, 1, dtype=torch.bool, device=self.device),
                                                              torch.zeros(bs_unify, self.padding_states-1, dtype=torch.bool, device=self.device)], dim=1),
                                                   base_action_mask)
                    gathered_proven = torch.where(subset_failed_unification.unsqueeze(1), False, gathered_proven)
                else:
                    logging.error(f"Cannot insert False state, padding_atoms ({self.padding_atoms}) too small.")
                    subset_truncated = subset_truncated | subset_failed_unification

            # --- Scatter results back to the main tensors using needs_unification_mask ---
            filtered_derived_sub_idx[needs_unification_mask] = gathered_states
            filtered_action_masks[needs_unification_mask] = base_action_mask
            # Apply mask to proven flags when scattering
            filtered_proven_masks[needs_unification_mask] = gathered_proven & base_action_mask
            batch_truncated_flags[needs_unification_mask] = subset_truncated
            failed_unification_mask[needs_unification_mask] = subset_failed_unification

        # --- Handle Terminal States (True/False) ---
        # This block executes regardless of dones_mask and overwrites previous results if needed.

        # Handle Terminal True
        if torch.any(is_terminal_true):
            true_term = Term("True", [])
            true_sub_idx_single = self.index_manager.get_state_sub_index_tensor([true_term])
            true_state_tensor = true_sub_idx_single.unsqueeze(0) # (1, pad_atoms, 3)
            true_fits = self.padding_atoms >= true_sub_idx_single.shape[0]

            if true_fits:
                filtered_derived_sub_idx[is_terminal_true, 0] = true_state_tensor
                filtered_derived_sub_idx[is_terminal_true, 1:] = PADDING_VALUE
                filtered_action_masks[is_terminal_true, 0] = True
                filtered_action_masks[is_terminal_true, 1:] = False
                filtered_proven_masks[is_terminal_true, :] = False
                failed_unification_mask[is_terminal_true] = False
                batch_truncated_flags[is_terminal_true] = False
            else:
                 logging.error(f"Cannot insert True state, padding_atoms ({self.padding_atoms}) too small.")
                 batch_truncated_flags[is_terminal_true] = True

        # Handle Terminal False
        if torch.any(is_terminal_false):
            # Define false_state_tensor if not already defined by unification failure block
            if 'false_state_tensor' not in locals() or false_state_tensor is None:
                 false_term = Term("False", [])
                 false_sub_idx_single = self.index_manager.get_state_sub_index_tensor([false_term]) # (pad_atoms, 3)
                 false_state_tensor = false_sub_idx_single.unsqueeze(0) # (1, pad_atoms, 3)
                 false_fits = self.padding_atoms >= false_sub_idx_single.shape[0]
            # else: false_state_tensor and false_fits already exist from the unification block

            if false_fits:
                filtered_derived_sub_idx[is_terminal_false, 0] = false_state_tensor
                filtered_derived_sub_idx[is_terminal_false, 1:] = PADDING_VALUE
                filtered_action_masks[is_terminal_false, 0] = True
                filtered_action_masks[is_terminal_false, 1:] = False
                filtered_proven_masks[is_terminal_false, :] = False
                failed_unification_mask[is_terminal_false] = True # Reaching False is a failure state
                batch_truncated_flags[is_terminal_false] = False
            else:
                 logging.error(f"Cannot insert False state, padding_atoms ({self.padding_atoms}) too small.")
                 batch_truncated_flags[is_terminal_false] = True

        # --- Optional Atom Truncation Check (Apply to final results) ---
        # This logic remains the same, applied after terminal state handling
        if self.truncate_atoms:
            active_state_mask = filtered_action_masks.unsqueeze(-1).unsqueeze(-1)
            num_actual_atoms = torch.where(filtered_action_masks,
                                          (filtered_derived_sub_idx[..., 0] != PADDING_VALUE).sum(dim=2),
                                          0)
            too_long_mask = (num_actual_atoms > self.padding_atoms) & filtered_action_masks

            if torch.any(too_long_mask):
                batch_truncated_flags = batch_truncated_flags | torch.any(too_long_mask, dim=1)
                filtered_action_masks = filtered_action_masks & ~too_long_mask
                filtered_proven_masks = filtered_proven_masks & filtered_action_masks
                filtered_derived_sub_idx = torch.where(too_long_mask.unsqueeze(-1).unsqueeze(-1),
                                                      PADDING_VALUE,
                                                      filtered_derived_sub_idx)

                # Re-check for newly failed states AFTER truncation AND ensure not already terminal
                newly_failed_mask = (filtered_action_masks.sum(dim=1) == 0) & (~failed_unification_mask) & (~dones_mask) & (~is_already_terminal)
                if torch.any(newly_failed_mask):
                     failed_unification_mask = failed_unification_mask | newly_failed_mask
                     if self.verbose > 0: logging.info(f"Vectorized: Batch items failed post-truncation: {torch.where(newly_failed_mask)[0].tolist()}. Inserting False.")
                     # Re-apply False state insertion logic
                     if 'false_state_tensor' not in locals() or false_state_tensor is None:
                          false_term = Term("False", [])
                          false_sub_idx_single = self.index_manager.get_state_sub_index_tensor([false_term])
                          false_state_tensor = false_sub_idx_single.unsqueeze(0)
                          false_fits = self.padding_atoms >= false_sub_idx_single.shape[0]

                     if false_fits:
                          failed_indices_new = torch.where(newly_failed_mask)[0]
                          filtered_derived_sub_idx[failed_indices_new, 0] = false_state_tensor
                          filtered_derived_sub_idx[failed_indices_new, 1:] = PADDING_VALUE
                          filtered_action_masks[failed_indices_new, 0] = True
                          filtered_action_masks[failed_indices_new, 1:] = False
                          filtered_proven_masks[failed_indices_new, :] = False
                     else:
                           batch_truncated_flags = batch_truncated_flags | newly_failed_mask

        # --- Final cleanup: Ensure PADDING_VALUE where action mask is False ---
        # We keep the action mask generated by terminal handling, even if dones_mask was True.
        final_action_mask_expanded = filtered_action_masks.unsqueeze(-1).unsqueeze(-1)
        filtered_derived_sub_idx = torch.where(final_action_mask_expanded, filtered_derived_sub_idx, PADDING_VALUE)
        filtered_proven_masks = filtered_proven_masks & filtered_action_masks # Ensure consistency

        # Reset failed unification flag for environments that were marked done in the input mask
        # (They are finished, not currently failed)
        failed_unification_mask[dones_mask] = False

        # NOTE: We no longer clear action/proven masks based on input dones_mask here.
        # The termination is handled by the `done` flag returned by the `step` function.

        return filtered_derived_sub_idx, batch_truncated_flags, filtered_action_masks, filtered_proven_masks, failed_unification_mask
        # Note: The derived_sub_idx_out is already padded with PADDING_VALUE, so no need to set it again here.
    # --- Modified _vectorized_done_reward (Simplified) ---
    def _vectorized_check_false(self,
                                next_sub_idx: torch.Tensor, # (bs, pad_atoms, 3)
                               ) -> torch.Tensor:
        """Checks if the state contains an explicit 'False' atom."""
        pred_indices = next_sub_idx[:, :, 0] # (bs, pad_atoms)
        is_padding_pred = (pred_indices == PADDING_VALUE)
        is_false = torch.any((pred_indices == self.false_idx) & (~is_padding_pred), dim=1)
        return is_false
# --- Modified step (v4 - Force terminal observation) ---
    def step(self, actions: Union[np.ndarray, torch.Tensor]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Performs a vectorized step, calculating done/reward based on proof/failure status.
        **NEW**: Explicitly sets the next observation's derived state and action mask
                 if termination occurs due to reaching a final 'True' or 'False' state,
                 ensuring the observation reflects the desired self-loop.
        """
        if isinstance(actions, np.ndarray): actions = torch.from_numpy(actions).to(device=self.device, dtype=torch.int64)
        if actions.shape != (self.batch_size,): raise ValueError(f"Actions shape mismatch: {actions.shape} vs {(self.batch_size,)}")

        # --- Retrieve necessary info from CURRENT TensorDict ---
        current_derived_sub_idx = self.tensordict["derived_sub_idx"]
        current_action_mask = self.tensordict["action_mask"]
        current_proven_mask = self.tensordict["proven_mask"]
        current_failed_unification = self.tensordict["failed_unification"] # Failure status leading to current options
        current_labels = self.tensordict["label"]
        current_dones = self.tensordict["done"] # Track already done environments

        # --- Validate and Select Next State based on Action ---
        actions = torch.clamp(actions, 0, self.padding_states - 1)
        action_valid_on_mask = torch.gather(current_action_mask, 1, actions.unsqueeze(1)).squeeze(1)
        ignore_validity_check = current_dones
        action_truly_invalid = ~action_valid_on_mask & ~ignore_validity_check

        if torch.any(action_truly_invalid):
             logging.warning(f"Invalid actions chosen by agent after clamping for indices: {torch.where(action_truly_invalid)[0].tolist()}. Selecting corresponding derived state (likely padding).")

        action_idx_sub_gather = actions.view(self.batch_size, 1, 1, 1).expand(-1, -1, self.padding_atoms, 3)
        next_sub_idx = torch.gather(current_derived_sub_idx, 1, action_idx_sub_gather).squeeze(1)

        # --- Determine Termination Conditions based on the chosen action and state ---
        is_chosen_state_proven = torch.gather(current_proven_mask, 1, actions.unsqueeze(1)).squeeze(1)
        terminated_proven = is_chosen_state_proven & ~current_dones

        terminated_failed = current_failed_unification & ~current_dones

        # Check the state *we are transitioning into* for True/False atoms
        # Use the same definition as in get_next_states_batch: first atom is T/F, rest is padding
        first_atom_pred_next = next_sub_idx[:, 0, 0]
        if self.padding_atoms > 1:
             rest_is_padding_next = torch.all(next_sub_idx[:, 1:] == PADDING_VALUE, dim=(1, 2))
        else:
             rest_is_padding_next = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

        is_next_state_terminal_true = (first_atom_pred_next == self.true_idx) & rest_is_padding_next
        is_next_state_terminal_false = (first_atom_pred_next == self.false_idx) & rest_is_padding_next

        # Terminate if the state itself IS False/True (covers explicit False/True atoms)
        terminated_reached_false = is_next_state_terminal_false & ~current_dones
        terminated_reached_true = is_next_state_terminal_true & ~current_dones # Terminate on True as well

        # Combine termination conditions
        terminated_this_step = terminated_proven | terminated_failed | terminated_reached_false | terminated_reached_true

        # --- Determine Truncation Conditions ---
        self.current_depth += (~current_dones).int()
        truncated_depth = (self.current_depth >= self.max_depth) & (~current_dones)
        # truncated_filter determined later

        # --- Calculate Reward ---
        rewards = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        rewards = torch.where(terminated_proven & (current_labels == 1), self.reward_proven, rewards)
        # Use terminated_failed OR terminated_reached_false for failure reward
        # (Avoid double penalty if failed unification led to False state)
        is_failure_termination = terminated_failed | terminated_reached_false
        rewards = torch.where(is_failure_termination, self.reward_failed, rewards)
        # If we need a specific reward for True, add it here (currently treated like non-failure)
        # rewards = torch.where(terminated_reached_true, SOME_REWARD, rewards)


        # --- Update State, Done, Reward in TensorDict FIRST ---
        terminated_final = current_dones | terminated_this_step
        truncated_final = truncated_depth # Initial truncation
        done_combined = terminated_final | truncated_final

        self.current_depth = torch.where(terminated_this_step | truncated_depth, 0, self.current_depth)

        self.tensordict.update_(TensorDict({
                 "state_sub_idx": next_sub_idx,
                 "done": done_combined,
                 "reward": rewards,
            }, batch_size=[self.batch_size]))

        # --- Get Next Possible States/Actions from the *new* current state ---
        # Call get_next_states_batch normally
        derived_sub_next, trunc_filter_next, action_masks_next, proven_masks_next, failed_unif_next = self.get_next_states_batch(
             self.tensordict["state_sub_idx"],
             self.tensordict["done"] # Pass the newly calculated done status
        )

        # --- *** OVERRIDE FOR TERMINAL TRUE/FALSE STATES *** ---
        # If termination happened *this step* because we reached True or False,
        # force the derived state/action mask to show the self-loop.

        # Indices that terminated specifically by reaching True *this step*
        newly_terminated_true = terminated_reached_true # terminated_reached_true already excludes current_dones

        # Indices that terminated specifically by reaching False *this step*
        newly_terminated_false = terminated_reached_false # already excludes current_dones

        # Handle True termination observation override
        if torch.any(newly_terminated_true):
             true_term = Term("True", [])
             true_sub_idx_single = self.index_manager.get_state_sub_index_tensor([true_term])
             if self.padding_atoms >= true_sub_idx_single.shape[0]:
                  true_state_tensor = true_sub_idx_single.unsqueeze(0)
                  derived_sub_next[newly_terminated_true, 0] = true_state_tensor
                  derived_sub_next[newly_terminated_true, 1:] = PADDING_VALUE
                  action_masks_next[newly_terminated_true, 0] = True
                  action_masks_next[newly_terminated_true, 1:] = False
                  proven_masks_next[newly_terminated_true, :] = False
                  failed_unif_next[newly_terminated_true] = False # Reaching True isn't failure
                  trunc_filter_next[newly_terminated_true] = False # No truncation here
             else:
                  logging.error("Cannot create True self-loop observation, padding too small.")
                  trunc_filter_next[newly_terminated_true] = True # Mark as truncated if cannot represent

        # Handle False termination observation override
        if torch.any(newly_terminated_false):
             false_term = Term("False", [])
             false_sub_idx_single = self.index_manager.get_state_sub_index_tensor([false_term])
             if self.padding_atoms >= false_sub_idx_single.shape[0]:
                  false_state_tensor = false_sub_idx_single.unsqueeze(0)
                  derived_sub_next[newly_terminated_false, 0] = false_state_tensor
                  derived_sub_next[newly_terminated_false, 1:] = PADDING_VALUE
                  action_masks_next[newly_terminated_false, 0] = True
                  action_masks_next[newly_terminated_false, 1:] = False
                  proven_masks_next[newly_terminated_false, :] = False
                  failed_unif_next[newly_terminated_false] = True # Reaching False is failure
                  trunc_filter_next[newly_terminated_false] = False # No truncation here
             else:
                  logging.error("Cannot create False self-loop observation, padding too small.")
                  trunc_filter_next[newly_terminated_false] = True # Mark as truncated if cannot represent

        # --- Update Truncation and Final Done Status (incorporating overrides) ---
        truncated_filter = trunc_filter_next & ~terminated_final # Filtering truncation only matters if not already terminated
        truncated_final = truncated_final | truncated_filter     # Combine depth and filter truncation
        done_combined = terminated_final | truncated_final       # Re-calculate final done flag

        # Reset depth again if newly truncated by filtering (and not already done previously)
        self.current_depth = torch.where(truncated_filter & ~terminated_final, 0, self.current_depth)


        # --- Update TensorDict with Derived Info and Final Done Status ---
        self.tensordict.update_(TensorDict({
                 "derived_sub_idx": derived_sub_next,
                 "action_mask": action_masks_next,
                 "proven_mask": proven_masks_next,
                 "failed_unification": failed_unif_next,
                 "done": done_combined, # Update done flag again
            }, batch_size=[self.batch_size]))

        # --- Prepare Return Values ---
        obs = self._get_obs_from_tensordict(self.tensordict)
        rewards_np = rewards.cpu().numpy()
        terminateds_np = terminated_this_step.cpu().numpy()
        truncateds_np = truncated_final.cpu().numpy()

        # Adjust infos to reflect termination cause more accurately
        infos = {"proven": terminated_proven.cpu().numpy(),
                 "failed_unification": terminated_failed.cpu().numpy(), # Terminated due to previous unif failure
                 "reached_false": terminated_reached_false.cpu().numpy(), # Terminated by reaching False state
                 "reached_true": terminated_reached_true.cpu().numpy()} # Terminated by reaching True state

        if self.verbose > 1: self._log_batch_state(f"Step (Action: {actions.cpu().numpy()})")
        return obs, rewards_np, terminateds_np, truncateds_np, infos

    def _get_obs_from_tensordict(self, td: TensorDict) -> Dict[str, np.ndarray]:
        """Extracts numpy observations from the TensorDict based on observation_space."""
        obs = {}
        # Keys defined in _make_spec's obs_spaces
        obs_keys = ['state_sub_idx', 'derived_sub_idx', 'action_mask', 'proven_mask']
        for key in obs_keys:
            if key in td.keys(include_nested=True):
                tensor_data = td[key]
                if isinstance(tensor_data, torch.Tensor):
                    obs[key] = tensor_data.cpu().numpy()
                else:
                    logging.error(f"Obs key '{key}' is not a tensor: {type(tensor_data)}.")
                    space = self.observation_space[key]
                    obs[key] = np.zeros(space.shape, dtype=space.dtype) # Fallback
            else:
                raise KeyError(f"Observation key '{key}' not found in TensorDict keys: {td.keys()}")
        return obs

    def _log_batch_state(self, stage: str):
        """Helper to log the current state of the batch (for debugging)."""
        print(f"\n--- {stage} - Batch State ---")
        print(f"Depth: {self.current_depth.cpu().numpy()}")
        print(f"Labels: {self.tensordict['label'].cpu().numpy()}")
        print(f"Failed Unif: {self.tensordict['failed_unification'].cpu().numpy().astype(int)}") # Show failure flag
        print(f"Dones: {self.tensordict['done'].cpu().numpy().astype(int)}")
        print(f"Rewards: {self.tensordict['reward'].cpu().numpy()}")
        if self.verbose > 2:
            for i in range(min(self.batch_size, 3)): # Print first few envs
                state_terms = self.index_manager.get_state_terms_from_batch(self.tensordict["state_sub_idx"][i].unsqueeze(0))[0]
                print(f"  [Env {i}] D:{self.current_depth[i].item()} Fail:{self.tensordict['failed_unification'][i].item()} Done:{self.tensordict['done'][i].item()} R:{self.tensordict['reward'][i].item():.2f} L:{self.tensordict['label'][i].item()}")
                print(f"    State: {[str(t) for t in state_terms if t]}")
                action_mask_np = self.tensordict['action_mask'][i].cpu().numpy().astype(int)
                proven_mask_np = self.tensordict['proven_mask'][i].cpu().numpy().astype(int)
                print(f"    Action Mask: {action_mask_np}")
                print(f"    Proven Mask: {proven_mask_np}") # Show proven mask

                if self.verbose > 3:
                     print(f"    Derived States (Actions):")
                     derived_states_terms_batch = self.index_manager.get_state_terms_from_batch(self.tensordict["derived_sub_idx"][i])
                     for action_idx in range(self.padding_states):
                         if self.tensordict['action_mask'][i, action_idx]:
                             derived_terms = derived_states_terms_batch[action_idx]
                             proven_flag = "[Proven]" if self.tensordict['proven_mask'][i, action_idx] else ""
                             print(f"      Action {action_idx}: {[str(t) for t in derived_terms if t]} {proven_flag}")
        print("-" * 25)

    def render(self,i):
        """Renders the state of the first environment in the batch."""
        if self.batch_size > 0:
            print(f"--- Rendering Env {i} (Depth: {self.current_depth[i].item()}) ---")
            state_terms = self.index_manager.get_state_terms_from_batch(self.tensordict["state_sub_idx"][i].unsqueeze(0))[0]
            print(f"  State: {[str(t) for t in state_terms if t]}")
            print(f"  Label: {self.tensordict['label'][i].item()}")
            print(f"  Failed Unification: {self.tensordict['failed_unification'][i].item()}")
            print(f"  Done: {self.tensordict['done'][i].item()}")
            print(f"  Reward: {self.tensordict['reward'][i].item():.2f}")
            action_mask_np = self.tensordict['action_mask'][i].cpu().numpy().astype(int)
            proven_mask_np = self.tensordict['proven_mask'][i].cpu().numpy().astype(int)
            print(f"  Action Mask: {action_mask_np}")
            print(f"  Proven Mask: {proven_mask_np}")
            print(f"  Possible Next States (Valid Actions):")
            derived_states_terms_batch = self.index_manager.get_state_terms_from_batch(self.tensordict["derived_sub_idx"][i])
            valid_action_found = False
            for action_idx in range(self.padding_states):
                if self.tensordict['action_mask'][i, action_idx]:
                    valid_action_found = True
                    derived_terms = derived_states_terms_batch[action_idx]
                    proven_flag = "[Proven]" if self.tensordict['proven_mask'][i, action_idx] else ""
                    print(f"    Action {action_idx}: {[str(t) for t in derived_terms if t]} {proven_flag}")
            if not valid_action_found:
                print("    [No valid actions]")
        else:
            print("Batch size is 0, cannot render.")

    def close(self):
        logging.info("Closing LogicEnv_gym_batch environment.")


# --- Example Usage Setup (Modified Rewards) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s') # Simpler format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Define Sample Data ---
    constants = {"a", "b", "c", "d", "p1", "p2", "p3"}
    predicates = {"p", "q", "parent", "ancestor"}
    # Rule1: ancestor(X,Y) :- parent(X,Y)
    rule1 = Rule(Term("ancestor", ["X", "Y"]), [Term("parent", ["X", "Y"])])
    # Rule2: ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)
    rule2 = Rule(Term("ancestor", ["X", "Z"]), [Term("parent", ["X", "Y"]), Term("ancestor", ["Y", "Z"])])
    # Query1: ancestor(p1, V) - Should succeed eventually finding p2, p3
    query1 = Term("ancestor", ["p1", "V"])
    # Query2: p(a, X) - Should succeed finding b
    query2 = Term("p", ["a", "X"])
    # Query3: q(b, V) - Should succeed finding c
    query3 = Term("q", ["b", "V"])
    # Query4: ancestor(p1, p3) - Should succeed
    query4 = Term("ancestor", ["p1", "p3"])
    # Query5: p(X, c) - Should fail (no matching fact/rule head)
    query5 = Term("p", ["X", "c"])
    # Query6: parent(p1, p1) - Should fail (no matching fact)
    query6 = Term("parent", ["p1", "p1"])


    all_vars = set()
    for r in [rule1, rule2]: all_vars.update(r.get_variables())
    for q in [query1, query2, query3, query4, query5, query6]:
        for arg in q.args:
            if is_variable(arg): all_vars.add(arg)
    print(f"All Variables: {all_vars}")


    # --- Initialize Components ---
    idx_manager = IndexManager(
        constants=constants, predicates=predicates, variables=all_vars,
        rules=[rule1, rule2], padding_atoms=6, # Increased padding slightly
        max_arity=2, device=device
    )

    facts_list = [
        Term("parent", ["p1", "p2"]), Term("parent", ["p2", "p3"]),
        Term("p", ["a", "b"]), Term("q", ["b", "c"]),
    ]
    rules_list = [rule1, rule2]
    # Use a mix of provable and unprovable queries for training/testing
    train_queries_list = [query1, query2, query5] # Mix of success/fail
    valid_queries_list = [query4, query6] # Provable and unprovable

    data_handler = DataHandler(facts_list, rules_list, train_queries_list, valid_queries_list)

    # --- Initialize Environment ---
    BATCH_SIZE = 2 # Test with 2 environments
    env = LogicEnv_gym_batch(
        batch_size=BATCH_SIZE,
        index_manager=idx_manager,
        data_handler=data_handler,
        mode='eval', # Use eval to cycle through valid_queries
        padding_states=10,
        max_depth=6, # Allow enough depth for ancestor
        reward_proven=1, # Reward for proven state
        reward_failed=0, # Penalty for reaching False state
        reward_false=0,  # Penalty for reaching False state
        verbose=2,
        prover_verbose=0, # Set to 1 or 2 for detailed unification logs
        device=device
    )

    # --- Simple Env Interaction Loop ---
    print("\n--- Starting Env Interaction (Eval Mode) ---")
    NUM_EPISODES = len(valid_queries_list) // BATCH_SIZE + (1 if len(valid_queries_list) % BATCH_SIZE else 0)

    for episode in range(NUM_EPISODES):
        print(f"\n===== EPISODE {episode + 1} =====")
        obs, info = env.reset()
        print("Initial State:")
        [env.render(i) for i in range(BATCH_SIZE)] # Render env 0

        total_rewards = np.zeros(BATCH_SIZE)
        is_done = np.zeros(BATCH_SIZE, dtype=bool)

        for step in range(env.max_depth + 1): # Allow one extra step to see final state
            print(f"\n--- Step {step} ---")
            # Choose first valid action (simple deterministic policy)
            action_mask_batch = obs['action_mask']
            actions = []
            for i in range(BATCH_SIZE):
                if is_done[i]: # Don't choose action if already done
                    actions.append(0) # Placeholder action
                    continue
                valid_actions = np.where(action_mask_batch[i])[0]
                if len(valid_actions) > 0:
                    action = valid_actions[0] # Choose first valid action
                else:
                    action = 0 # Should already be done if no valid actions
                    # print(f"[Env {i}] No valid actions found in mask (already done?). Choosing 0.")
                actions.append(action)
            actions_np = np.array(actions, dtype=np.int64)

            # Only step environments that are not done
            active_envs_mask = ~is_done
            if not np.any(active_envs_mask):
                 print("All environments finished.")
                 break

            # We step all envs, but only care about results for active ones
            obs, rewards, terminated, truncated, infos = env.step(actions_np)

            # Update totals and done status only for active envs before the step
            total_rewards[active_envs_mask] += rewards[active_envs_mask]
            step_dones = terminated | truncated
            newly_done = step_dones & active_envs_mask
            is_done |= step_dones # Update overall done status

            print(f"Chosen Actions: {actions_np}")
            print(f"Step Results:")
            print(f"  Rewards: {rewards}")
            print(f"  Terminated: {terminated.astype(int)}")
            print(f"  Truncated: {truncated.astype(int)}")
            print(f"  Infos['proven']: {infos['proven'].astype(int)}")
            print(f"  Infos['failed_unification']: {infos['failed_unification'].astype(int)}")
            print(f"  Infos['reached_false']: {infos['reached_false'].astype(int)}")
            print(f"  Current Dones: {is_done.astype(int)}")

            # Render env 0 if it wasn't done before this step
            if active_envs_mask[0]:
                 print("\nState After Step (Env 0):")
                 [env.render(i) for i in range(BATCH_SIZE)] # Render env 0

            if np.all(is_done):
                print(f"\nAll environments finished episode at step {step}.")
                break
        print(f"\n===== EPISODE {episode + 1} END =====")
        print(f"Total Rewards: {total_rewards}")


    env.close()
    print("\n--- Env Interaction Finished ---")