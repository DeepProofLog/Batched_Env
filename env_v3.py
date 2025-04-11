import torch
import gymnasium as gym
import numpy as np
from tensordict import TensorDict
from typing import List, Optional, Tuple, Dict, Any, Set, Union, NamedTuple
import random
import logging

# --- Constants from Unification ---
PADDING_VALUE = -999 # Value for padding tensors
UNBOUND_VAR = -998     # Placeholder for unbound variables (primarily for formatting/debugging)

# --- Data Structures ---
class Term:
    """Represents a logical term like predicate(arg1, arg2)."""
    def __init__(self, predicate: str, args: List[Any]):
        self.predicate = predicate
        self.args = args
    def __str__(self):
        return f"{self.predicate}({','.join(map(str, self.args))})"
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        return isinstance(other, Term) and self.predicate == other.predicate and self.args == other.args
    def __hash__(self):
        try:
            hashable_args = tuple(self.args)
        except TypeError:
            hashable_args = tuple(map(str, self.args))
            # logging.warning(f"Unhashable args in Term hash: {self.args}. Converted to strings.") # Optional logging
        return hash((self.predicate, hashable_args))

class Rule:
    """Represents a logical rule Head :- Body."""
    def __init__(self, head: Term, body: List[Term]):
        self.head = head
        self.body = body
    def __str__(self):
        body_str = ', '.join(map(str, self.body))
        return f"{self.head} :- {body_str}"
    def __repr__(self):
        return str(self)
    def get_all_terms(self) -> List[Term]:
        """Returns head + body terms."""
        return [self.head] + self.body
    def get_variables(self) -> Set[str]:
        """Extracts all variables (uppercase strings) from the rule."""
        variables = set()
        terms = self.get_all_terms()
        for term in terms:
            for arg in term.args:
                if is_variable(arg):
                    variables.add(arg)
        return variables

def is_variable(arg: Any) -> bool:
    """Checks if an argument is a variable (simple uppercase string check)."""
    return isinstance(arg, str) and arg.isupper() and len(arg) == 1

class UnificationResult(NamedTuple):
    """Structure to hold results from unification functions."""
    substitutions: torch.Tensor  # Shape: (bs, k, max_subs, 2)
    target_indices: torch.Tensor # Shape: (bs, k) - Index of fact or rule
    valid_mask: torch.Tensor     # Shape: (bs, k) - Indicates valid unifications
    target_bodies: Optional[torch.Tensor] = None # Shape: (bs, k, body_len, 3) - Only for rules

# --- Index Manager (Provided, slightly adapted) ---
class IndexManager():
    """
    Manages global and local indices for constants, predicates, variables, and atoms.
    Refactored for better integration with batched tensor operations.
    """
    def __init__(self, constants: Set[str],
                 predicates: Set[str],
                 variables: Set[str], # All variables appearing anywhere (queries, facts, rules)
                 rules: List[Rule], # Used to extract rule variables if needed
                 padding_atoms: int = 10,
                 max_arity: int = 2, # Crucial: Assumes max_arity=2 for (P, S, O) format
                 device: torch.device = torch.device("cpu")):

        if max_arity != 2:
            raise ValueError("IndexManager currently requires max_arity=2 for P,S,O format.")

        self.device = device
        self.constants = constants
        self.predicates = predicates.union({'True', 'False', 'End'}) # Ensure special predicates are included
        self.variables = variables # Store all possible variables
        self.constant_no = len(constants)
        self.variable_no = len(variables)
        self.predicate_no = len(self.predicates) # Recalculate based on the union

        self.rules = rules # Keep rules if needed for variable extraction or structure
        self.padding_atoms = padding_atoms
        self.max_arity = max_arity # Should be 2

        # --- Global Mappings (ID 0 reserved for padding/unknown, Negative for variables) ---
        self.constant_str2idx: Dict[str, int] = {}
        self.constant_idx2str: Dict[int, str] = {}
        self.predicate_str2idx: Dict[str, int] = {}
        self.predicate_idx2str: Dict[int, str] = {}
        self.variable_str2idx: Dict[str, int] = {}
        self.variable_idx2str: Dict[int, str] = {}

        # --- Special Predicate Indices ---
        self.true_idx: int = -1
        self.false_idx: int = -1
        self.end_idx: int = -1

        # --- Local (Episode-Specific) Mappings ---
        # Map hash(Term) -> atom_index for unique atom identification within an episode
        self.term_hash_to_atom_idx: Dict[int, int] = {}
        # Map atom_index -> sub_indices_tensor (P, S, O) for caching
        self.atom_idx_to_sub_idx: Dict[int, torch.Tensor] = {}

        self.next_atom_idx: int = 1 # Start atom indices from 1

        self._create_global_idx()

        # --- Create Global Variable Tensor ---
        # Ensure variables have negative indices
        if not all(idx < 0 for idx in self.variable_str2idx.values()):
             raise ValueError("Variable indices must be negative.")
        self.vars_idx_tensor = torch.tensor(
            sorted(list(self.variable_str2idx.values())),
            dtype=torch.long,
            device=self.device
        )
        self.vars_idx_set = set(self.vars_idx_tensor.cpu().tolist())


        logging.info("IndexManager initialized.")
        logging.info(f"Constants: {self.constant_no}, Predicates: {self.predicate_no}, Variables: {self.variable_no}")
        logging.info(f"True/False/End Indices: {self.true_idx}/{self.false_idx}/{self.end_idx}")
        logging.info(f"Variable Indices (Sample): {list(self.variable_str2idx.items())[:5]}")
        logging.info(f"Variable Tensor Shape: {self.vars_idx_tensor.shape}")

    def _create_global_idx(self):
        """Create global indices for constants, predicates, and variables."""
        # Constants: Positive integers starting from 1
        current_idx = 1
        for term in sorted(self.constants):
            self.constant_str2idx[term] = current_idx
            self.constant_idx2str[current_idx] = term
            current_idx += 1

        # Predicates: Positive integers starting from 1 (separate namespace)
        current_idx = 1
        for term in sorted(self.predicates):
            self.predicate_str2idx[term] = current_idx
            self.predicate_idx2str[current_idx] = term
            if term == 'True': self.true_idx = current_idx
            elif term == 'False': self.false_idx = current_idx
            elif term == 'End': self.end_idx = current_idx
            current_idx += 1

        if self.true_idx == -1 or self.false_idx == -1 or self.end_idx == -1:
            logging.warning("Special predicates 'True', 'False', or 'End' not found during indexing.")
            # Assign placeholder indices if missing - adjust if necessary
            next_pred_idx = len(self.predicate_str2idx) + 1
            if 'True' not in self.predicate_str2idx: self.true_idx = next_pred_idx; next_pred_idx+=1
            if 'False' not in self.predicate_str2idx: self.false_idx = next_pred_idx; next_pred_idx+=1
            if 'End' not in self.predicate_str2idx: self.end_idx = next_pred_idx; next_pred_idx+=1


        # Variables: Negative integers starting from -1
        current_idx = -1
        for term in sorted(self.variables):
            self.variable_str2idx[term] = current_idx
            self.variable_idx2str[current_idx] = term
            current_idx -= 1

    def reset_atom(self):
        """Reset episode-specific indices (atoms)."""
        self.term_hash_to_atom_idx = {}
        self.atom_idx_to_sub_idx = {}
        self.next_atom_idx = 1

    def _get_term_index(self, term: Any) -> int:
        """Helper to get index for a constant or variable."""
        if is_variable(term):
            if term not in self.variable_str2idx:
                # This should ideally not happen if variables set is complete
                logging.warning(f"Variable '{term}' not found in global index. Returning PADDING_VALUE {PADDING_VALUE}.")
                return PADDING_VALUE # Use the specific padding value
            return self.variable_str2idx[term]
        else:
            # Constant lookup
            if term not in self.constant_str2idx:
                logging.warning(f"Constant '{term}' not found in global index. Returning PADDING_VALUE {PADDING_VALUE}.")
                return PADDING_VALUE # Use the specific padding value
            return self.constant_str2idx[term]

    def get_term_pso_indices(self, term: Term) -> Optional[Tuple[int, int, int]]:
        """Converts a Term (arity 2) to (predicate_idx, subject_idx, object_idx). Returns None if invalid."""
        if term.predicate in ['True', 'False', 'End']:
             # Handle special predicates - represent them uniquely if needed, e.g., (pred_idx, 0, 0)
             pred_idx = self.predicate_str2idx.get(term.predicate)
             if pred_idx is not None:
                 return (pred_idx, PADDING_VALUE, PADDING_VALUE) # Or (pred_idx, 0, 0) if 0 is not used elsewhere
             else:
                 logging.warning(f"Special predicate '{term.predicate}' has no index.")
                 return None
        elif len(term.args) != self.max_arity: # Check against max_arity (should be 2)
            logging.warning(f"Term {term} does not have arity {self.max_arity}. Cannot get PSO indices.")
            return None

        pred_idx = self.predicate_str2idx.get(term.predicate)
        subj_idx = self._get_term_index(term.args[0])
        obj_idx = self._get_term_index(term.args[1])

        if pred_idx is None: # Should have been caught by check above, but safety
             logging.warning(f"Predicate '{term.predicate}' not found.")
             return None

        # PADDING_VALUE indicates an unknown term during _get_term_index
        if subj_idx == PADDING_VALUE or obj_idx == PADDING_VALUE:
             # Don't return None here, let the PADDING_VALUE propagate
             pass

        return pred_idx, subj_idx, obj_idx

    def get_atom_sub_index(self, state: List[Term]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get padded atom indices and sub-indices (P, S, O format) for a single state.
        Manages local atom indices. Uses caching for sub-indices.
        Output sub_idx shape: (padding_atoms, 3)
        """
        atom_indices = torch.zeros(self.padding_atoms, dtype=torch.int64, device=self.device)
        # Sub indices now store P, S, O directly
        sub_indices = torch.full((self.padding_atoms, 3), PADDING_VALUE, dtype=torch.int64, device=self.device)

        if len(state) > self.padding_atoms:
            logging.warning(f"State length {len(state)} exceeds padding_atoms {self.padding_atoms}. Truncating.")
            state = state[:self.padding_atoms]

        for i, atom in enumerate(state):
            term_hash = hash(atom)

            # --- Get Atom Index ---
            if term_hash in self.term_hash_to_atom_idx:
                atom_idx = self.term_hash_to_atom_idx[term_hash]
            else:
                atom_idx = self.next_atom_idx
                self.term_hash_to_atom_idx[term_hash] = atom_idx
                self.next_atom_idx += 1
            atom_indices[i] = atom_idx

            # --- Get Sub Indices (P, S, O) ---
            if atom_idx in self.atom_idx_to_sub_idx:
                sub_indices[i] = self.atom_idx_to_sub_idx[atom_idx]
            else:
                pso = self.get_term_pso_indices(atom)
                if pso is not None:
                    current_sub_idx = torch.tensor(pso, dtype=torch.int64, device=self.device)
                    sub_indices[i] = current_sub_idx
                    self.atom_idx_to_sub_idx[atom_idx] = current_sub_idx.clone() # Cache
                # else: leave as PADDING_VALUE (already initialized)

        return atom_indices, sub_indices

    def get_term_from_indices(self, pso_tensor: torch.Tensor) -> Optional[Term]:
        """
        Reconstructs a single Term from a (P, S, O) tensor.
        Input shape: (3,)
        """
        if not isinstance(pso_tensor, torch.Tensor) or pso_tensor.shape != (3,):
            logging.error(f"Invalid input shape for get_term_from_indices: {pso_tensor.shape}")
            return None

        p_idx, s_idx, o_idx = pso_tensor.tolist()

        # Handle padding atoms
        if p_idx == PADDING_VALUE and s_idx == PADDING_VALUE and o_idx == PADDING_VALUE:
            return None # Represent padding as None

        predicate = self.predicate_idx2str.get(p_idx)
        if predicate is None:
            logging.warning(f"Unknown predicate index {p_idx} during reconstruction.")
            predicate = f"UNK_PRED_{p_idx}"

        # Handle special predicates (no arguments needed)
        if predicate in ['True', 'False', 'End']:
            return Term(predicate, [])

        args = []
        for term_idx in [s_idx, o_idx]:
            if term_idx == PADDING_VALUE:
                 # This might indicate an issue if P is not padding but S/O are
                 logging.warning(f"Found PADDING_VALUE ({PADDING_VALUE}) in args for predicate {predicate} ({p_idx}).")
                 args.append(f"PAD_{term_idx}") # Or handle as error/skip?
            elif term_idx < 0: # Variable
                 arg_str = self.variable_idx2str.get(term_idx)
                 if arg_str is None:
                      logging.warning(f"Unknown variable index {term_idx}.")
                      arg_str = f"UNK_VAR_{term_idx}"
                 args.append(arg_str)
            elif term_idx > 0: # Constant
                 arg_str = self.constant_idx2str.get(term_idx)
                 if arg_str is None:
                      logging.warning(f"Unknown constant index {term_idx}.")
                      arg_str = f"UNK_CONST_{term_idx}"
                 args.append(arg_str)
            else: # Index 0 - Undefined or specific meaning? Assume unknown for now.
                 logging.warning(f"Index 0 encountered for argument. Treating as UNK_ARG_0.")
                 args.append("UNK_ARG_0")

        return Term(predicate, args)

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
                    # Add placeholder if reconstruction failed but wasn't padding
                    state_terms.append(Term("RECONSTRUCTION_ERROR", [str(sub_idx_batch[i, j].tolist())]))

            all_states_terms.append(state_terms)
        return all_states_terms


    def get_state_repr_for_memory(self, atom_idx_tensor: torch.Tensor) -> Tuple[int, ...]:
        """Creates a hashable representation (tuple of non-zero atom indices)."""
        if atom_idx_tensor.dim() != 1: raise ValueError(f"Input tensor must be 1D. Got {atom_idx_tensor.shape}")
        mem_repr = tuple(idx.item() for idx in atom_idx_tensor if idx.item() != 0)
        return mem_repr

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
                    term_indices.append((PADDING_VALUE, PADDING_VALUE, PADDING_VALUE)) # Pad invalid term within rule

            rule_tensors.append(term_indices)

        # Pad each rule to max_atoms_in_rules
        padded_rules = []
        if max_atoms_in_rules == 0 and rules: # Handle case where all rules were invalid
             logging.error("No valid terms found in any rules for tensor conversion.")
             return torch.empty((0, 0, 3), dtype=torch.long, device=self.device), 0
        elif not rules:
             return torch.empty((0, 0, 3), dtype=torch.long, device=self.device), 0


        for term_indices in rule_tensors:
            current_len = len(term_indices)
            padding_needed = max_atoms_in_rules - current_len
            padded_indices = term_indices + [[PADDING_VALUE, PADDING_VALUE, PADDING_VALUE]] * padding_needed
            padded_rules.append(padded_indices)

        if not padded_rules:
             return torch.empty((0, max_atoms_in_rules, 3), dtype=torch.long, device=self.device), max_atoms_in_rules

        return torch.tensor(padded_rules, dtype=torch.long, device=self.device), max_atoms_in_rules


# --- Unification Functions (Provided) ---

def _get_var_set_and_tensor(
    vars_idx: Union[torch.Tensor, List[int], Set[int], IndexManager],
    device: torch.device
) -> Tuple[Set[int], torch.Tensor]:
    """Ensures vars_idx is a set and a tensor on the correct device."""
    # Simplified: Assumes IndexManager already provides the tensor and set correctly
    if isinstance(vars_idx, IndexManager):
        return vars_idx.vars_idx_set, vars_idx.vars_idx_tensor.to(device)

    # Original logic as fallback if tensor/list/set is passed directly
    if isinstance(vars_idx, torch.Tensor):
        vars_idx_tensor = vars_idx.to(device=device, dtype=torch.long).detach()
        vars_idx_set = set(vars_idx_tensor.cpu().tolist()) # Convert to list on CPU for set
    elif isinstance(vars_idx, (list, set)):
        vars_idx_set = set(vars_idx)
        vars_idx_tensor = torch.tensor(list(vars_idx_set), dtype=torch.long, device=device)
    else:
        raise TypeError("vars_idx must be an IndexManager, tensor, list, or set")
    # Ensure vars_idx_tensor is at least 1D
    if vars_idx_tensor.dim() == 0:
        vars_idx_tensor = vars_idx_tensor.unsqueeze(0)
    # Ensure all variables are negative
    if not torch.all(vars_idx_tensor < 0):
         logging.warning(f"Variable tensor contains non-negative values: {vars_idx_tensor[vars_idx_tensor >= 0]}. This might conflict with constants.")
         # raise ValueError("Variable indices in tensor must be negative.") # Stricter check
    return vars_idx_set, vars_idx_tensor

def _format_term(
    term_idx: int,
    vars_idx_set: Set[int],
    constant_idx2str: Optional[Dict[int, str]] = None,
    variable_idx2str: Optional[Dict[int, str]] = None # Added for clarity
) -> str:
    """Formats a single term (subject or object)."""
    if term_idx == PADDING_VALUE:
        return "[pad]"
    if term_idx == UNBOUND_VAR: # Should not appear in final results ideally
        return "[unbound]"
    if term_idx in vars_idx_set:
        var_str = variable_idx2str.get(term_idx, f"VAR_IDX({term_idx})") if variable_idx2str else f"VAR({term_idx})"
        return var_str
    if constant_idx2str and term_idx in constant_idx2str:
        return constant_idx2str[term_idx]
    # Default to string representation (covers 0 and other constants/indices)
    # Check if it's a known variable index even if not in vars_idx_set (e.g., during formatting)
    if variable_idx2str and term_idx in variable_idx2str:
        return variable_idx2str[term_idx] + "[?]" # Mark if found in var map but not expected set

    return f"IDX({term_idx})" # More explicit for unknown positive indices

def _format_atom(
    atom_tensor: torch.Tensor,
    predicate_idx2str: Optional[Dict[int, str]] = None,
    constant_idx2str: Optional[Dict[int, str]] = None,
    vars_idx_set: Optional[Set[int]] = None, # Pass the set directly
    variable_idx2str: Optional[Dict[int, str]] = None # Added for clarity
) -> str:
    """Formats an atom tensor (shape [3]) into a readable string."""
    if atom_tensor.is_cuda: atom_tensor = atom_tensor.cpu()

    if not isinstance(atom_tensor, torch.Tensor) or atom_tensor.dim() < 1 or atom_tensor.shape[-1] != 3:
        return f"[invalid atom shape: {atom_tensor.shape}]"
    if atom_tensor.dim() > 1: atom_tensor = atom_tensor.view(-1, 3)[0]
    if atom_tensor.numel() != 3: return f"[invalid atom numel: {atom_tensor.numel()}]"

    p_idx, s_idx, o_idx = atom_tensor.tolist()

    is_padding_atom = (p_idx == PADDING_VALUE and s_idx == PADDING_VALUE and o_idx == PADDING_VALUE)
    if is_padding_atom: return "[padding_atom]"

    # Format predicate
    p_str = predicate_idx2str.get(p_idx, f"PRED_IDX({p_idx})") if predicate_idx2str else f"P({p_idx})"

    # Ensure vars_idx_set is a set
    vars_set = vars_idx_set if isinstance(vars_idx_set, set) else set()

    # Format subject and object
    s_str = _format_term(s_idx, vars_set, constant_idx2str, variable_idx2str)
    o_str = _format_term(o_idx, vars_set, constant_idx2str, variable_idx2str)

    return f"({p_str}, {s_str}, {o_str})"

def _format_substitution(
    sub_pair: torch.Tensor, # Shape (2,) : [var_idx, value_idx]
    constant_idx2str: Optional[Dict[int, str]] = None,
    vars_idx_set: Optional[Set[int]] = None, # Pass the set directly
    variable_idx2str: Optional[Dict[int, str]] = None # Added for clarity
) -> str:
    """Formats a substitution pair tensor into a readable string."""
    if sub_pair.is_cuda: sub_pair = sub_pair.cpu()

    if not isinstance(sub_pair, torch.Tensor) or sub_pair.shape != (2,):
        return f"[invalid sub pair: {sub_pair}]"

    var_idx, value_idx = sub_pair.tolist()

    if var_idx == PADDING_VALUE: return "[no_sub]"

    # Ensure vars_idx_set is a set
    vars_set = vars_idx_set if isinstance(vars_idx_set, set) else set()

    # Format the variable part (should always be a variable)
    var_str = "[var?]"
    if var_idx < 0 : # Variables must be negative
        var_str = variable_idx2str.get(var_idx, f"VAR_IDX({var_idx})") if variable_idx2str else f"VAR({var_idx})"
    else:
        var_str = f"NON_VAR({var_idx})" # Error case

    # Format the value part using the term formatter
    value_str = _format_term(value_idx, vars_set, constant_idx2str, variable_idx2str)

    return f"{var_str} -> {value_str}"


def apply_substitutions_vectorized(
    atoms: torch.Tensor,               # Shape: (..., num_atoms, 3) e.g. (bs, k, num_atoms, 3) or (num_atoms, 3)
    substitutions: torch.Tensor,       # Shape: (..., max_subs, 2), [var_idx, value_idx] - Must broadcast correctly with atoms
    vars_idx_tensor: torch.Tensor,     # 1D Tensor of variable indices
    max_iterations: int = 10,          # Limit recursion/iteration depth
) -> torch.Tensor:
    """
    Applies substitutions (Var -> Value) to a tensor of atoms iteratively using vectorized operations.
    Handles Var -> Const and Var -> Var substitutions until no changes occur or max_iterations.
    Assumes `substitutions` can broadcast to the leading dimensions of `atoms`.
    """
    # --- Input Validation and Setup ---
    if atoms.numel() == 0: return atoms.clone()
    if atoms.shape[-1] != 3: raise ValueError(f"Last dimension of atoms must be 3, but got shape {atoms.shape}")
    if substitutions.numel() > 0 and substitutions.shape[-1] != 2: raise ValueError(f"Last dimension of substitutions must be 2, but got shape {substitutions.shape}")

    device = atoms.device
    substituted_atoms = atoms.clone()

    # --- Filter and Prepare Substitutions ---
    if substitutions.numel() == 0: return substituted_atoms

    valid_subs_mask = (substitutions[..., 0] != PADDING_VALUE) # Shape: (..., max_subs)
    if not torch.any(valid_subs_mask): return substituted_atoms

    # --- Iterative Substitution Application ---
    changed = True
    iterations = 0
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        # atoms_before_iter = substituted_atoms.clone() # Debugging

        max_subs_dim = substitutions.shape[-2]
        for sub_idx in range(max_subs_dim):
            current_sub = substitutions[..., sub_idx, :] # Shape (..., 2)
            current_mask = valid_subs_mask[..., sub_idx] # Shape (...)

            if not torch.any(current_mask): continue

            # Use PADDING_VALUE where mask is False to avoid unintended matches
            var_to_replace = torch.where(current_mask, current_sub[..., 0], PADDING_VALUE)
            replacement_value = torch.where(current_mask, current_sub[..., 1], PADDING_VALUE)

            # --- Vectorized Application ---
            num_atom_dims = substituted_atoms.dim()
            # Calculate view_shape for broadcasting var/value/mask to atoms shape
            # Example: atoms (bs, k, n, 3), var (bs, k) -> view_shape (bs, k, 1, 1)
            view_shape_list = list(var_to_replace.shape)
            view_shape_list.extend([1] * (num_atom_dims - len(view_shape_list) -1)) # Add 1s for atom dims (e.g., n_atoms)
            view_shape_list.append(1) # Add final 1 for the P/S/O dimension
            view_shape = tuple(view_shape_list)

            var_to_replace_b = var_to_replace.view(view_shape)
            replacement_value_b = replacement_value.view(view_shape)
            current_mask_b = current_mask.view(view_shape)

            # Match masks shape: (..., n_atoms, 1) - selecting S/O column
            s_match_mask = (substituted_atoms[..., 1:2] == var_to_replace_b) & current_mask_b # Check Subject (index 1)
            o_match_mask = (substituted_atoms[..., 2:3] == var_to_replace_b) & current_mask_b # Check Object (index 2)

            # Apply substitution using torch.where, only if the value actually changes
            if torch.any(s_match_mask):
                current_s_values = substituted_atoms[..., 1:2]
                apply_s_mask = s_match_mask & (current_s_values != replacement_value_b)
                if torch.any(apply_s_mask):
                    substituted_atoms[..., 1:2] = torch.where(apply_s_mask, replacement_value_b, current_s_values)
                    changed = True

            if torch.any(o_match_mask):
                current_o_values = substituted_atoms[..., 2:3]
                apply_o_mask = o_match_mask & (current_o_values != replacement_value_b)
                if torch.any(apply_o_mask):
                    substituted_atoms[..., 2:3] = torch.where(apply_o_mask, replacement_value_b, current_o_values)
                    changed = True

    if iterations >= max_iterations and changed:
        logging.warning(f"Max substitution iterations ({max_iterations}) reached in apply_substitutions_vectorized.")

    return substituted_atoms


def batch_unify_with_facts(
    states_idx: torch.Tensor,            # Shape: (bs, 1, n_padding_atoms, 3)
    facts: torch.Tensor,                 # Shape: (n_facts, 3)
    vars_idx: Union[torch.Tensor, List[int], Set[int], IndexManager], # Variable indices or IndexManager
    k: int,                              # Max number of unifications to return
    device: Optional[torch.device] = None,
    verbose: bool = False,
    predicate_idx2str: Optional[Dict[int, str]] = None,
    constant_idx2str: Optional[Dict[int, str]] = None,
    variable_idx2str: Optional[Dict[int, str]] = None # Added for formatting
) -> UnificationResult:
    """
    Performs batched unification for the first query against FACTS. (Vectorized)
    Returns simple substitutions (Query Var -> Fact Const) for up to K unifying facts.
    Detects conflicts where a query like (P, VarX, VarX) tries to unify with (P, A, B).
    """
    # --- Setup ---
    if k <= 0: raise ValueError("k must be a positive integer")

    effective_device = device if device else states_idx.device
    states_idx = states_idx.to(effective_device)
    facts = facts.to(effective_device)
    vars_idx_set, vars_idx_tensor = _get_var_set_and_tensor(vars_idx, effective_device)

    bs, _, n_padding_atoms, _ = states_idx.shape
    n_facts = facts.shape[0]

    if verbose:
        print("-" * 40 + "\nDEBUG: batch_unify_with_facts (Start)\n" + "-" * 40)
        print(f"BS: {bs}, N_Facts: {n_facts}, K: {k}, Device: {effective_device}")
        print(f"Vars Set: {vars_idx_set}")
        print(f"Vars Tensor Shape: {vars_idx_tensor.shape}, Dim: {vars_idx_tensor.dim()}")

    actual_k = min(k, n_facts) if n_facts > 0 else 0

    # Initialize outputs (using max_subs=2 for facts: one for S, one for O)
    substitutions_out = torch.full((bs, k, 2, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
    fact_indices_out = torch.full((bs, k), PADDING_VALUE, dtype=torch.long, device=effective_device)
    valid_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=effective_device)

    if bs == 0 or n_facts == 0 or n_padding_atoms == 0:
        if verbose: print("DEBUG: Early exit (no batch items, facts, or atoms).")
        return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)

    # --- Extract Queries & Check Validity ---
    first_queries = states_idx[:, 0, 0, :] # Shape: (bs, 3)
    is_query_valid = ~torch.all(first_queries == PADDING_VALUE, dim=-1) # Shape: (bs,)

    if verbose and bs > 0:
        query0_str = _format_atom(first_queries[0], predicate_idx2str, constant_idx2str, vars_idx_set, variable_idx2str)
        print(f"\nDEBUG: First Query [0]: {query0_str}")
        print(f"DEBUG: Is Valid Query [0]: {is_query_valid[0].item()}")

    # --- Calculate Potential Unification Mask (Vectorized) ---
    queries_expanded = first_queries.unsqueeze(1) # (bs, 1, 3)
    facts_expanded = facts.unsqueeze(0)           # (1, n_facts, 3)

    # Check if query subject/object are variables (bs, 1)
    # Ensure vars_idx_tensor is 1D for isin
    is_query_var_s = torch.isin(queries_expanded[:, :, 1], vars_idx_tensor.view(-1))
    is_query_var_o = torch.isin(queries_expanded[:, :, 2], vars_idx_tensor.view(-1))

    # Check matches allowing for variables (bs, n_facts)
    pred_match = (queries_expanded[:, :, 0] == facts_expanded[:, :, 0])
    subj_match = is_query_var_s | (queries_expanded[:, :, 1] == facts_expanded[:, :, 1])
    obj_match = is_query_var_o | (queries_expanded[:, :, 2] == facts_expanded[:, :, 2])

    # Initial unification mask: matches AND query is valid (bs, n_facts)
    unifies_mask_initial = pred_match & subj_match & obj_match & is_query_valid.unsqueeze(1)

    # --- Select Top K Potential Unifications ---
    if actual_k == 0:
        if verbose: print("\nDEBUG: actual_k=0, skipping topk.")
        return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)

    scores = torch.where(unifies_mask_initial,
                         torch.tensor(1.0, device=effective_device),
                         torch.tensor(float('-inf'), device=effective_device))

    try:
        top_scores, top_fact_indices = torch.topk(scores, k=actual_k, dim=1) # (bs, actual_k)
    except RuntimeError as e:
       print(f"Error during fact topk: {e}")
       print(f"Scores shape: {scores.shape}, actual_k: {actual_k}, n_facts: {n_facts}")
       # Handle case where actual_k > n_facts if scores dim is smaller than requested k
       if scores.shape[1] < actual_k:
           print("Attempting topk with available facts.")
           actual_k = scores.shape[1]
           if actual_k == 0: return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)
           top_scores, top_fact_indices = torch.topk(scores, k=actual_k, dim=1)
       else:
           return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)


    is_topk_potentially_valid = (top_scores > float('-inf')) # (bs, actual_k)

    # --- Gather Data and Perform Conflict Check (Vectorized) ---
    potential_k_coords = torch.nonzero(is_topk_potentially_valid, as_tuple=True)
    n_potential_topk = len(potential_k_coords[0])

    final_valid_mask_slice = torch.zeros((bs, actual_k), dtype=torch.bool, device=effective_device)
    gathered_subs = torch.full((bs, actual_k, 2, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
    gathered_indices = torch.full((bs, actual_k), PADDING_VALUE, dtype=torch.long, device=effective_device)

    if n_potential_topk > 0:
        valid_bs_indices, valid_k_indices = potential_k_coords
        potential_fact_indices_flat = top_fact_indices[potential_k_coords] # (n_potential_topk,)

        # --- Conflict Check ---
        top_k_queries_flat = first_queries[valid_bs_indices] # (n_potential_topk, 3)
        top_k_facts_flat = facts[potential_fact_indices_flat] # (n_potential_topk, 3)

        is_query_s_var_flat = torch.isin(top_k_queries_flat[:, 1], vars_idx_tensor.view(-1))
        is_query_o_var_flat = torch.isin(top_k_queries_flat[:, 2], vars_idx_tensor.view(-1))
        is_same_query_var_flat = is_query_s_var_flat & is_query_o_var_flat & \
                                 (top_k_queries_flat[:, 1] == top_k_queries_flat[:, 2])
        is_diff_fact_const_flat = (top_k_facts_flat[:, 1] != top_k_facts_flat[:, 2])
        is_conflict_flat = is_same_query_var_flat & is_diff_fact_const_flat

        final_valid_mask_slice[potential_k_coords] = ~is_conflict_flat

        if verbose and bs > 0:
            print("\nDEBUG: Fact Conflict Check [Batch Element 0]:")
            indices_for_batch_0 = (valid_bs_indices == 0).nonzero(as_tuple=True)[0]
            if len(indices_for_batch_0) > 0:
                k_indices_0 = valid_k_indices[indices_for_batch_0]
                fact_indices_0 = potential_fact_indices_flat[indices_for_batch_0]
                conflicts_0 = is_conflict_flat[indices_for_batch_0]
                for i in range(len(indices_for_batch_0)):
                    k_slot, fact_idx, is_conflict = k_indices_0[i].item(), fact_indices_0[i].item(), conflicts_0[i].item()
                    query_str = _format_atom(top_k_queries_flat[indices_for_batch_0[i]], predicate_idx2str, constant_idx2str, vars_idx_set, variable_idx2str)
                    fact_str = _format_atom(top_k_facts_flat[indices_for_batch_0[i]], predicate_idx2str, constant_idx2str, vars_idx_set, variable_idx2str)
                    print(f"  - K-Slot {k_slot}, Fact {fact_idx}: Q={query_str}, F={fact_str} -> Conflict: {is_conflict}")
            else: print("  No facts in Top K for batch 0 to check.")
            print(f"  Final Valid Mask Slice [0]: {final_valid_mask_slice[0].tolist()}")


        # --- Populate Final Outputs for Non-Conflicting Items ---
        final_valid_coords = torch.nonzero(final_valid_mask_slice, as_tuple=True)
        n_final_valid = len(final_valid_coords[0])

        if n_final_valid > 0:
            # Select the non-conflicting entries from the flat tensors
            # Create a boolean mask from ~is_conflict_flat to index the original potential items
            non_conflict_mask_flat = ~is_conflict_flat # Shape: (n_potential_topk,)

            final_queries = top_k_queries_flat[non_conflict_mask_flat] # (n_final_valid, 3)
            final_facts = top_k_facts_flat[non_conflict_mask_flat]     # (n_final_valid, 3)
            final_fact_idxs_out = potential_fact_indices_flat[non_conflict_mask_flat] # (n_final_valid,)

            gathered_indices[final_valid_coords] = final_fact_idxs_out

            # Populate substitutions (vectorized)
            is_s_var_final = torch.isin(final_queries[:, 1], vars_idx_tensor.view(-1)) # (n_final_valid,)
            is_o_var_final = torch.isin(final_queries[:, 2], vars_idx_tensor.view(-1)) # (n_final_valid,)

            s_sub_pairs = torch.stack([
                torch.where(is_s_var_final, final_queries[:, 1], PADDING_VALUE), # Var
                torch.where(is_s_var_final, final_facts[:, 1], PADDING_VALUE)    # Const
            ], dim=1) # (n_final_valid, 2)
            o_sub_pairs = torch.stack([
                torch.where(is_o_var_final, final_queries[:, 2], PADDING_VALUE), # Var
                torch.where(is_o_var_final, final_facts[:, 2], PADDING_VALUE)    # Const
            ], dim=1) # (n_final_valid, 2)

            # Scatter into gathered_subs using final_valid_coords
            gathered_subs[final_valid_coords[0], final_valid_coords[1], 0, :] = s_sub_pairs
            gathered_subs[final_valid_coords[0], final_valid_coords[1], 1, :] = o_sub_pairs

    # Assign gathered results to the final output tensors, padding correctly
    substitutions_out[:, :actual_k] = gathered_subs
    fact_indices_out[:, :actual_k] = gathered_indices
    valid_mask_out[:, :actual_k] = final_valid_mask_slice

    if verbose:
        print("\nDEBUG: Final Fact Substitutions [Batch Element 0]:")
        if bs > 0:
            found_any = False
            for k_idx in range(actual_k):
                if valid_mask_out[0, k_idx]:
                    found_any = True
                    fact_id = fact_indices_out[0, k_idx].item()
                    s_sub_str = _format_substitution(substitutions_out[0, k_idx, 0], constant_idx2str, vars_idx_set, variable_idx2str)
                    o_sub_str = _format_substitution(substitutions_out[0, k_idx, 1], constant_idx2str, vars_idx_set, variable_idx2str)
                    print(f"  - K-Slot {k_idx} (Fact {fact_id}): Sub S: {s_sub_str}, Sub O: {o_sub_str}")
            if not found_any: print("  No valid substitutions.")
        print("-" * 40 + "\nDEBUG: batch_unify_with_facts (End)\n" + "-" * 40)

    return UnificationResult(substitutions_out, fact_indices_out, valid_mask_out)


def batch_unify_with_rules(
    states_idx: torch.Tensor,            # Shape: (bs, 1, n_padding_atoms, 3)
    rules: torch.Tensor,                 # Shape: (n_rules, max_rule_atoms, 3) Head=0, Body=1:
    vars_idx: Union[torch.Tensor, List[int], Set[int], IndexManager], # Variable indices or IndexManager
    k: int,                              # Max number of rule unifications
    max_subs_per_rule: int = 5,          # Max substitutions tracked per unification
    device: Optional[torch.device] = None,
    verbose: bool = False,
    predicate_idx2str: Optional[Dict[int, str]] = None,
    constant_idx2str: Optional[Dict[int, str]] = None,
    variable_idx2str: Optional[Dict[int, str]] = None # Added for formatting
) -> UnificationResult:
    """
    Performs batched unification for the first query against RULE HEADS.
    Calculates the Most General Unifier (MGU) substitutions needed.
    Handles Var->Const, Const->Var, Var->Var bindings and detects conflicts/cycles.
    NOTE: The core MGU calculation per (query, rule_head) pair remains iterative.
    """
    # --- Setup ---
    if k <= 0: raise ValueError("k must be a positive integer")
    if max_subs_per_rule <= 0: raise ValueError("max_subs_per_rule must be positive")

    effective_device = device if device else states_idx.device
    states_idx = states_idx.to(effective_device)
    rules = rules.to(effective_device)
    vars_idx_set, vars_idx_tensor = _get_var_set_and_tensor(vars_idx, effective_device)

    bs, _, n_padding_atoms, _ = states_idx.shape
    if rules.numel() == 0: # Handle empty rules tensor
        n_rules, max_rule_atoms = 0, 0
    else:
        n_rules, max_rule_atoms, _ = rules.shape
    rule_body_len = max(0, max_rule_atoms - 1)

    if verbose:
        print("\n" + "=" * 40 + "\nDEBUG: batch_unify_with_rules (Start)\n" + "=" * 40)
        print(f"BS: {bs}, N_Rules: {n_rules}, Max Rule Atoms: {max_rule_atoms}, K: {k}, Max Subs: {max_subs_per_rule}")
        print(f"Device: {effective_device}, Vars Set: {vars_idx_set}")
        print(f"Vars Tensor Shape: {vars_idx_tensor.shape}, Dim: {vars_idx_tensor.dim()}")

    actual_k = min(k, n_rules) if n_rules > 0 else 0

    # Initialize output tensors
    substitutions_out = torch.full((bs, k, max_subs_per_rule, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
    rule_indices_out = torch.full((bs, k), PADDING_VALUE, dtype=torch.long, device=effective_device)
    rule_bodies_out_shape = (bs, k, rule_body_len, 3)
    rule_bodies_out = torch.full(rule_bodies_out_shape, PADDING_VALUE, dtype=torch.long, device=effective_device)
    valid_mask_out = torch.zeros((bs, k), dtype=torch.bool, device=effective_device)

    if bs == 0 or n_rules == 0 or n_padding_atoms == 0 or max_rule_atoms <= 0:
        if verbose: print("DEBUG: Early exit (no batch items, rules, atoms, or rule atoms).")
        return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)

    # --- Extract Queries, Rule Heads/Bodies & Check Validity ---
    first_queries = states_idx[:, 0, 0, :] # (bs, 3)
    is_query_valid = ~torch.all(first_queries == PADDING_VALUE, dim=-1) # (bs,)

    rule_heads = rules[:, 0, :]    # (n_rules, 3)
    rule_bodies = rules[:, 1:, :] # (n_rules, rule_body_len, 3)

    if verbose and bs > 0:
        b_idx_print = 0 # Print state 0
        query_print_str = _format_atom(first_queries[b_idx_print], predicate_idx2str, constant_idx2str, vars_idx_set, variable_idx2str)
        print(f"\nDEBUG: First Query [b={b_idx_print}]: {query_print_str}")
        print(f"DEBUG: Is Valid Query [b={b_idx_print}]: {is_query_valid[b_idx_print].item()}")
        if n_rules > 0: print(f"DEBUG: First Rule Head [r=0]: {_format_atom(rule_heads[0], predicate_idx2str, constant_idx2str, vars_idx_set, variable_idx2str)}")

    # --- Initial Predicate Match (Vectorized) ---
    queries_expanded = first_queries.unsqueeze(1) # (bs, 1, 3)
    heads_expanded = rule_heads.unsqueeze(0)      # (1, n_rules, 3)
    pred_match = (queries_expanded[:, :, 0] == heads_expanded[:, :, 0]) & is_query_valid.unsqueeze(1) # (bs, n_rules)

    # --- Per-Query, Per-Rule Unification (MGU calculation - Iterative Core) ---
    potential_subs_list = [[[] for _ in range(n_rules)] for _ in range(bs)] # List of lists of [var, val] pairs
    potential_valid = torch.zeros((bs, n_rules), dtype=torch.bool, device=effective_device)

    is_query_s_var = torch.isin(first_queries[:, 1:2], vars_idx_tensor.view(-1)) # (bs, 1)
    is_query_o_var = torch.isin(first_queries[:, 2:3], vars_idx_tensor.view(-1)) # (bs, 1)
    is_head_s_var = torch.isin(rule_heads[:, 1:2], vars_idx_tensor.view(-1))    # (n_rules, 1)
    is_head_o_var = torch.isin(rule_heads[:, 2:3], vars_idx_tensor.view(-1))    # (n_rules, 1)

    # --- MGU Helper Function (defined inside for scope access) ---
    max_depth_get_binding = max_subs_per_rule * 2 + n_padding_atoms + n_rules # Safety depth
    def get_binding(var, subs_dict, visited):
        """Finds ultimate binding, detects cycles using visited set."""
        # Using vars_idx_set directly here. Ensure it's accessible.
        if var not in vars_idx_set or var not in subs_dict: return var
        if var in visited: return None # Cycle detected
        visited.add(var)
        if len(visited) > max_depth_get_binding:
            visited.remove(var); return None # Depth limit / potential cycle

        next_val = subs_dict[var]
        result = get_binding(next_val, subs_dict, visited)
        visited.remove(var) # Backtrack
        return result
    # --- End MGU Helper ---

    # --- Main MGU Loop (Iterative) ---
    for b in range(bs):
        if not torch.any(pred_match[b]): continue # Skip if no predicate matches for this query

        query = first_queries[b] # (3,)
        q_p, q_s, q_o = query.tolist()
        is_q_s_var_b = is_query_s_var[b].item()
        is_q_o_var_b = is_query_o_var[b].item()

        for r in range(n_rules):
            if not pred_match[b, r]: continue # Skip if specific predicate doesn't match

            head = rule_heads[r] # (3,)
            h_p, h_s, h_o = head.tolist()
            is_h_s_var_r = is_head_s_var[r].item()
            is_h_o_var_r = is_head_o_var[r].item()

            # --- MGU Logic for (b, r) ---
            temp_subs = {} # {var_idx: value_idx}
            possible = True
            term_pairs = [(q_s, h_s, is_q_s_var_b, is_h_s_var_r),
                          (q_o, h_o, is_q_o_var_b, is_h_o_var_r)]

            for q_term, h_term, is_q_var, is_h_var in term_pairs:
                if not possible: break

                # Resolve terms using current bindings
                q_final = get_binding(q_term, temp_subs, set()) if is_q_var else q_term
                h_final = get_binding(h_term, temp_subs, set()) if is_h_var else h_term

                if q_final is None or h_final is None: possible = False; break # Cycle/Error

                is_q_final_var = q_final in vars_idx_set
                is_h_final_var = h_final in vars_idx_set

                # --- Unification Cases ---
                if q_final == h_final: continue
                elif is_q_final_var and is_h_final_var: # Var-Var
                    # Check for conflicts before binding h_final -> q_final
                    h_binding_check = get_binding(h_final, temp_subs, set())
                    q_binding_check = get_binding(q_final, temp_subs, set())
                    if h_binding_check is None or q_binding_check is None: possible = False; break # Cycle
                    if (h_binding_check != h_final and h_binding_check != q_final) or \
                       (q_binding_check != q_final and q_binding_check != h_final):
                        possible = False; break # Conflict
                    temp_subs[h_final] = q_final # Bind head var to query var (or other consistent choice)
                elif is_q_final_var: # Var(Query) - Const/Var(Head)
                    q_binding_check = get_binding(q_final, temp_subs, set())
                    if q_binding_check is None: possible = False; break # Cycle
                    if q_binding_check != q_final and q_binding_check != h_final: possible = False; break # Conflict
                    temp_subs[q_final] = h_final
                elif is_h_final_var: # Const/Var(Query) - Var(Head)
                    h_binding_check = get_binding(h_final, temp_subs, set())
                    if h_binding_check is None: possible = False; break # Cycle
                    if h_binding_check != h_final and h_binding_check != q_final: possible = False; break # Conflict
                    temp_subs[h_final] = q_final
                else: # Const-Const mismatch
                    possible = False; break
            # --- End MGU Logic for term pair ---

            # --- Finalize and Store Substitutions ---
            if possible:
                final_subs_list = []
                processed_vars = set()
                possible_final = True
                for var_start in list(temp_subs.keys()): # Iterate original keys
                    if var_start in processed_vars: continue
                    final_val = get_binding(var_start, temp_subs, set())
                    if final_val is None: possible_final = False; break # Cycle/Error
                    if var_start != final_val: # Add non-trivial binding
                        final_subs_list.append([var_start, final_val])
                    processed_vars.add(var_start)
                    if final_val in temp_subs: processed_vars.add(final_val) # Mark value if it was also a key

                if possible_final and len(final_subs_list) <= max_subs_per_rule:
                    potential_valid[b, r] = True
                    potential_subs_list[b][r] = final_subs_list
                # else: unification failed final check or too many subs

    # --- Select Top K Rules (Vectorized) ---
    if actual_k == 0:
        if verbose: print("\nDEBUG: actual_k=0, skipping topk.")
        return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)

    scores = torch.where(potential_valid,
                         torch.tensor(1.0, device=effective_device),
                         torch.tensor(float('-inf'), device=effective_device))

    try:
        top_scores, top_indices_in_rules = torch.topk(scores, k=actual_k, dim=1) # (bs, actual_k)
    except RuntimeError as e:
       print(f"Error during rule topk: {e}")
       print(f"Scores shape: {scores.shape}, actual_k: {actual_k}, n_rules: {n_rules}")
       # Handle case where actual_k > n_rules if scores dim is smaller than requested k
       if scores.shape[1] < actual_k:
           print("Attempting topk with available rules.")
           actual_k = scores.shape[1]
           if actual_k == 0: return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)
           top_scores, top_indices_in_rules = torch.topk(scores, k=actual_k, dim=1)
       else:
           return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)


    final_valid_k_mask = (top_scores > float('-inf')) # (bs, actual_k)

    if verbose and bs > 0:
        b_idx_print = 0
        print("\nDEBUG: TopK Rule Results [b={b_idx_print}]:")
        print(f"  Potential Valid Mask[b={b_idx_print}]: {potential_valid[b_idx_print].tolist()}")
        print(f"  Scores[{b_idx_print}]: {scores[b_idx_print].tolist()}")
        print(f"  Top Scores[{b_idx_print}]: {top_scores[b_idx_print].tolist()}")
        print(f"  Top Indices[{b_idx_print}]: {top_indices_in_rules[b_idx_print].tolist()}")
        print(f"  Final Valid Mask[{b_idx_print}]: {final_valid_k_mask[b_idx_print].tolist()}")


    # --- Gather Final Results ---
    final_valid_coords = torch.nonzero(final_valid_k_mask, as_tuple=True) # (final_bs_indices, final_k_indices)
    n_final_valid = len(final_valid_coords[0])

    if n_final_valid > 0:
        final_bs_indices, final_k_indices = final_valid_coords
        final_rule_indices = top_indices_in_rules[final_valid_coords] # (n_final_valid,)

        # --- Scatter Rule Indices and Bodies ---
        rule_indices_out[final_valid_coords] = final_rule_indices
        if rule_body_len > 0:
            gathered_bodies = rule_bodies[final_rule_indices] # (n_final_valid, rule_body_len, 3)
            rule_bodies_out[final_bs_indices, final_k_indices] = gathered_bodies

        # --- Gather and Scatter Substitutions (Iterative part) ---
        temp_subs_tensor = torch.full((n_final_valid, max_subs_per_rule, 2), PADDING_VALUE, dtype=torch.long, device=effective_device)
        for i in range(n_final_valid):
            b = final_bs_indices[i].item()
            # k_idx = final_k_indices[i].item() # Index in the k dimension (0 to actual_k-1)
            rule_idx = final_rule_indices[i].item() # Index in the original rules tensor
            subs_list = potential_subs_list[b][rule_idx]
            num_subs = len(subs_list)
            if num_subs > 0:
                subs_tensor_entry = torch.tensor(subs_list, dtype=torch.long, device=effective_device)
                temp_subs_tensor[i, :num_subs, :] = subs_tensor_entry

        # Scatter the temporary tensor into the final output tensor
        substitutions_out[final_bs_indices, final_k_indices] = temp_subs_tensor

    # Update the main output mask based on topk results
    valid_mask_out[:, :actual_k] = final_valid_k_mask

    if verbose:
        b_idx_print = 0
        print(f"\nDEBUG: Rule Top-{actual_k} Selection & Results [Batch Element {b_idx_print}]:")
        if bs > 0:
            if torch.any(valid_mask_out[b_idx_print]):
                valid_indices_print = rule_indices_out[b_idx_print][valid_mask_out[b_idx_print]].tolist()
                print(f"  Selected Rule Indices (Valid in Top K): {valid_indices_print}")
                for k_idx in range(actual_k):
                    if valid_mask_out[b_idx_print, k_idx]:
                        rule_idx = rule_indices_out[b_idx_print, k_idx].item()
                        subs_tensor_out = substitutions_out[b_idx_print, k_idx]
                        body_tensor_out = rule_bodies_out[b_idx_print, k_idx]
                        rule_head_str = "[Rule OOB]"
                        if 0 <= rule_idx < n_rules:
                            rule_head_str = _format_atom(rule_heads[rule_idx], predicate_idx2str, constant_idx2str, vars_idx_set, variable_idx2str)

                        print(f"  - K-Slot {k_idx}: Unifies with Rule {rule_idx} (Head: {rule_head_str})")
                        subs_strs = [_format_substitution(subs_tensor_out[i], constant_idx2str, vars_idx_set, variable_idx2str)
                                     for i in range(subs_tensor_out.shape[0]) if subs_tensor_out[i, 0].item() != PADDING_VALUE]
                        print(f"    Subs: {{{', '.join(subs_strs)}}}")
                        print(f"    Rule Body Atoms (Original):")
                        has_body = False
                        if body_tensor_out.shape[0] > 0:
                            for atom_idx in range(body_tensor_out.shape[0]):
                                body_atom = body_tensor_out[atom_idx]
                                if body_atom[0].item() != PADDING_VALUE:
                                    print(f"      {_format_atom(body_atom, predicate_idx2str, constant_idx2str, vars_idx_set, variable_idx2str)}")
                                    has_body = True
                        if not has_body: print("      [No body atoms or all padding]")
            else:
                print("  No valid rule unifications for this batch element.")
        print("-" * 40 + "\nDEBUG: batch_unify_with_rules (End)\n" + "=" * 40)

    return UnificationResult(substitutions_out, rule_indices_out, valid_mask_out, rule_bodies_out)


def apply_substitutions_and_create_next_state(
    states_idx: torch.Tensor,             # Shape: (bs, 1, n_padding_atoms, 3)
    unification_result: UnificationResult,# Result from fact or rule unification
    vars_idx_tensor: torch.Tensor,        # 1D Tensor of variable indices
    new_max_atoms: Optional[int] = None,  # Max atoms in the output state, None to calculate based on input
    max_sub_iterations: int = 10          # Iteration limit for apply_substitutions_vectorized
) -> torch.Tensor:
    """
    Generates next goal states based on successful unifications (fact or rule).
    Applies substitutions vectorially to remaining original goals and (if applicable) rule bodies.
    Concatenates results and pads/truncates.
    Output shape: (bs, k, new_max_atoms, 3)
    """
    bs, _, n_padding_atoms, _ = states_idx.shape
    k = unification_result.substitutions.shape[1] # k used in unification
    device = states_idx.device

    # Extract components from UnificationResult
    substitutions = unification_result.substitutions # (bs, k, max_subs, 2)
    valid_mask = unification_result.valid_mask       # (bs, k)
    rule_bodies = unification_result.target_bodies   # (bs, k, rule_body_len, 3) or None

    has_rule_bodies = rule_bodies is not None and rule_bodies.numel() > 0 and rule_bodies.shape[-1] == 3
    rule_body_len = rule_bodies.shape[2] if has_rule_bodies else 0

    # Determine the number of atoms in the output state
    num_remaining_atoms_orig = max(0, n_padding_atoms - 1)
    if new_max_atoms is None:
        new_max_atoms = num_remaining_atoms_orig + rule_body_len
    if new_max_atoms < 0: new_max_atoms = 0 # Ensure non-negative

    # Initialize the output tensor for next states
    next_states = torch.full((bs, k, new_max_atoms, 3), PADDING_VALUE, dtype=torch.long, device=device)

    # Get original remaining queries (all atoms except the first one)
    # Shape: (bs, num_remaining_atoms_orig, 3)
    remaining_queries_base = states_idx[:, 0, 1:, :] if n_padding_atoms > 1 else torch.empty((bs, 0, 3), dtype=torch.long, device=device)

    # Find coordinates of valid unifications
    valid_coords = torch.nonzero(valid_mask, as_tuple=True) # (valid_bs_indices, valid_k_indices)
    n_valid = len(valid_coords[0])

    if n_valid == 0:
        return next_states # No valid unifications, return padded tensor

    valid_bs_indices, valid_k_indices = valid_coords

    # --- Prepare Tensors for Vectorized Substitution ---
    # Gather data corresponding to valid unifications
    valid_subs = substitutions[valid_bs_indices, valid_k_indices] # (n_valid, max_subs, 2)
    valid_rem_q = remaining_queries_base[valid_bs_indices]        # (n_valid, num_remaining_atoms_orig, 3)
    valid_bodies = rule_bodies[valid_bs_indices, valid_k_indices] if has_rule_bodies else torch.empty((n_valid, 0, 3), dtype=torch.long, device=device) # (n_valid, rule_body_len, 3)

    # --- Apply Substitutions Vectorially ---
    # Need to unsqueeze subs to broadcast against atoms dim: (n_valid, 1, max_subs, 2)
    valid_subs_unsqueezed = valid_subs.unsqueeze(1)

    subst_rem_q_flat = apply_substitutions_vectorized(
        valid_rem_q, valid_subs_unsqueezed, vars_idx_tensor, max_sub_iterations
    ) # Output: (n_valid, num_remaining_atoms_orig, 3)

    subst_body_flat = torch.empty((n_valid, 0, 3), dtype=torch.long, device=device)
    if rule_body_len > 0:
        subst_body_flat = apply_substitutions_vectorized(
            valid_bodies, valid_subs_unsqueezed, vars_idx_tensor, max_sub_iterations
        ) # Output: (n_valid, rule_body_len, 3)

    # --- Filter Padding/Zeros and Concatenate (Iterative approach needed here) ---
    # Vectorizing concatenation of variable length tensors after filtering is complex.
    for i in range(n_valid):
        b = valid_bs_indices[i].item()
        k_idx = valid_k_indices[i].item()

        current_body = subst_body_flat[i]   # (rule_body_len, 3)
        current_rem_q = subst_rem_q_flat[i] # (num_remaining_atoms_orig, 3)

        # Filter padding atoms
        body_keep_mask = torch.any(current_body != PADDING_VALUE, dim=-1)
        rem_q_keep_mask = torch.any(current_rem_q != PADDING_VALUE, dim=-1)

        subst_body_filtered = current_body[body_keep_mask]
        subst_rem_q_filtered = current_rem_q[rem_q_keep_mask]

        # Concatenate filtered atoms (body first, then remaining query)
        concatenated_atoms = torch.cat((subst_body_filtered, subst_rem_q_filtered), dim=0)

        # --- Pad/Truncate to final size ---
        current_len = concatenated_atoms.shape[0]
        len_to_copy = min(current_len, new_max_atoms)
        if len_to_copy > 0:
            next_states[b, k_idx, :len_to_copy, :] = concatenated_atoms[:len_to_copy, :]
        # The rest remains PADDING_VALUE due to initialization

    return next_states


# --- Placeholder DataHandler ---
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
        self.sampler = None # Add sampler if dynamic corruption is used


# --- Integrated Environment ---
class LogicEnv_gym_batch(gym.Env):
    """
    Gymnasium environment using batched tensor operations and integrated unification.
    """
    metadata = {'render_modes': [], 'render_fps': 1} # Add metadata

    def __init__(self,
                 batch_size: int,
                 index_manager: IndexManager,
                 data_handler: DataHandler, # Use refactored DataHandler
                 mode: str = 'train',
                 corruption_mode: Optional[str] = None, # 'dynamic' or None
                 train_neg_pos_ratio: int = 1,
                 seed: Optional[int] = None,
                 max_depth: int = 10,
                 end_proof_action: bool = False,
                 skip_unary_actions: bool = False, # Note: Less relevant if engine is batched
                 truncate_atoms: bool = False, # Filter derived states if they exceed padding_atoms
                 padding_states: int = 20, # Max number of derived states (actions)
                 k_facts_rules: Optional[int] = None, # K for unification (defaults to padding_states)
                 max_subs_per_rule: int = 5, # From unification
                 max_sub_iterations: int = 10, # From unification
                 verbose: int = 0,
                 prover_verbose: int = 0, # Passed to unification functions
                 device: torch.device = torch.device("cpu"),
                 # engine: str = 'python_batch', # No longer needed, unification is integrated
                 ):

        super().__init__()

        self.batch_size = batch_size
        self.index_manager = index_manager
        self.data_handler = data_handler
        self.device = device
        self.verbose = verbose
        self.prover_verbose = prover_verbose
        # self.engine = engine # Removed

        # Get parameters from components
        self.max_arity = self.index_manager.max_arity
        if self.max_arity != 2: raise ValueError("Environment requires max_arity=2")
        self.padding_atoms = self.index_manager.padding_atoms
        self.padding_states = padding_states
        self.max_depth = max_depth
        self.k_facts_rules = k_facts_rules if k_facts_rules is not None else self.padding_states
        self.max_subs_per_rule = max_subs_per_rule
        self.max_sub_iterations = max_sub_iterations


        # Store special indices
        self.true_idx = self.index_manager.true_idx
        self.false_idx = self.index_manager.false_idx
        self.end_idx = self.index_manager.end_idx
        if self.true_idx < 0 or self.false_idx < 0 or self.end_idx < 0:
            logging.warning("IndexManager did not properly initialize special predicate indices.")

        self._set_seed(seed)

        # Data and Corruption Setup
        self.facts = self.data_handler.facts # List[Term]
        self.rules = self.data_handler.rules # List[Rule]
        self.corruption_mode = corruption_mode
        self.train_neg_pos_ratio = train_neg_pos_ratio
        self.counter = 0

        # --- Pre-process facts and rules into tensors ---
        self.facts_idx_tensor = self.index_manager.facts_to_tensor(self.facts)
        self.rules_idx_tensor, self.max_atoms_in_rules = self.index_manager.rules_to_tensor(self.rules)
        logging.info(f"Facts tensor shape: {self.facts_idx_tensor.shape}")
        logging.info(f"Rules tensor shape: {self.rules_idx_tensor.shape}, Max atoms in rule: {self.max_atoms_in_rules}")
        # ---

        # Batch-specific state
        self.current_depth = torch.zeros(self.batch_size, dtype=torch.int, device=self.device)
        self.current_queries_terms: List[Optional[Term]] = [None] * self.batch_size
        self.current_labels = torch.zeros(self.batch_size, dtype=torch.int, device=self.device)

        # Configuration Flags
        self.end_proof_action = end_proof_action
        self.skip_unary_actions = skip_unary_actions
        self.truncate_atoms = truncate_atoms

        # Mode and Query Lists
        assert mode in ['train', 'eval', 'eval_corr'], f"Invalid mode: {mode}"
        self.mode = mode
        self.train_queries = self.data_handler.train_queries
        self.eval_queries = self.data_handler.valid_queries # Use valid set for eval
        self.n_train_queries = len(self.train_queries)
        self.n_eval_queries = len(self.eval_queries)
        self.eval_indices = list(range(self.n_eval_queries))
        self.current_eval_batch_idx = 0

        if self.mode == 'train':
            if self.corruption_mode == "dynamic":
                 if not self.data_handler.sampler: raise ValueError("Dynamic corruption requires sampler.")
                 if not self.train_queries: raise ValueError("Dynamic corruption requires train_queries.")
            elif not self.train_queries: raise ValueError("Training mode requires train_queries.")

        # Define Observation and Action Spaces
        self._make_spec()

        # Initialize TensorDict structure
        self.tensordict = self._create_empty_tensordict()

    def _set_seed(self, seed: Optional[int]):
        """Set the seed for the environment."""
        # self.seed = seed # Deprecated in newer gym
        super().reset(seed=seed) # Use the official way
        if seed is None: seed = random.randint(0, 2**32 - 1) # Generate if None
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logging.info(f"Environment seeded with {seed}")

    def _create_empty_tensordict(self) -> TensorDict:
        """Creates the TensorDict structure with correct shapes and dtypes."""
        td_data = {
            # Current state (P, S, O format)
            "state_sub_idx": torch.full((self.batch_size, self.padding_atoms, 3), PADDING_VALUE, dtype=torch.int64, device=self.device),
            # Derived states (P, S, O format) - Action space
            "derived_sub_idx": torch.full((self.batch_size, self.padding_states, self.padding_atoms, 3), PADDING_VALUE, dtype=torch.int64, device=self.device),
            "action_mask": torch.zeros(self.batch_size, self.padding_states, dtype=torch.bool, device=self.device),
            "label": torch.zeros(self.batch_size, dtype=torch.int, device=self.device),
            "done": torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
            "reward": torch.zeros(self.batch_size, dtype=torch.float32, device=self.device),
            # Removed atom_idx as sub_idx contains the primary info now
            # "state_atom_idx": torch.zeros(self.batch_size, self.padding_atoms, dtype=torch.int64, device=self.device),
            # "derived_atom_idx": torch.zeros(self.batch_size, self.padding_states, self.padding_atoms, dtype=torch.int64, device=self.device),
        }
        return TensorDict(td_data, batch_size=[self.batch_size], device=self.device)

    def _make_spec(self):
        """Create the batched observation and action specs using tensor representations."""
        # Observation space uses sub_idx (P,S,O)
        obs_spaces = {
            'state_sub_idx': gym.spaces.Box(np.iinfo(np.int64).min, np.iinfo(np.int64).max, (self.batch_size, self.padding_atoms, 3), dtype=np.int64),
            'derived_sub_idx': gym.spaces.Box(np.iinfo(np.int64).min, np.iinfo(np.int64).max, (self.batch_size, self.padding_states, self.padding_atoms, 3), dtype=np.int64),
            'action_mask': gym.spaces.Box(0, 1, (self.batch_size, self.padding_states), dtype=np.bool_),
            # Removed atom_idx from observation space
            # 'state_atom_idx': gym.spaces.Box(0, np.iinfo(np.int64).max, (self.batch_size, self.padding_atoms), dtype=np.int64),
            # 'derived_atom_idx': gym.spaces.Box(0, np.iinfo(np.int64).max, (self.batch_size, self.padding_states, self.padding_atoms), dtype=np.int64),
        }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        # Action space selects one of the derived states
        self.action_space = gym.spaces.MultiDiscrete([self.padding_states] * self.batch_size)


    def _sample_initial_batch(self) -> Tuple[List[Term], List[int]]:
        """Samples a batch of initial queries (Terms) and labels for TRAINING mode."""
        # (Logic remains the same as provided, using self.train_queries, self.train_neg_pos_ratio etc.)
        batch_queries_terms: List[Term] = []
        batch_labels: List[int] = []

        if not self.train_queries:
            raise ValueError("Cannot sample for training, train_queries list is empty.")

        if self.corruption_mode == "dynamic":
            # Dynamic negative sampling logic (requires data_handler.sampler)
            # ... (Keep the logic from the original env file) ...
             raise NotImplementedError("Dynamic negative sampling logic needs to be copied/verified.")
        else: # No corruption
            sampled_indices = random.choices(range(self.n_train_queries), k=self.batch_size)
            batch_queries_terms = [self.train_queries[i] for i in sampled_indices]
            train_labels = getattr(self.data_handler, "train_labels", [1]*self.n_train_queries)
            batch_labels = [train_labels[i] for i in sampled_indices]

        # Final check for batch size (remains the same)
        if len(batch_queries_terms) != self.batch_size:
            logging.error(f"Batch size mismatch after sampling: {len(batch_queries_terms)} vs {self.batch_size}. Fixing...")
            # ... (Keep the fixing logic) ...
            if len(batch_queries_terms) > self.batch_size:
                 batch_queries_terms = batch_queries_terms[:self.batch_size]
                 batch_labels = batch_labels[:self.batch_size]
            else:
                 num_missing = self.batch_size - len(batch_queries_terms)
                 if not self.train_queries: raise ValueError("Cannot pad batch, no train queries.")
                 extra_indices = random.choices(range(self.n_train_queries), k=num_missing)
                 train_labels = getattr(self.data_handler, "train_labels", [1]*self.n_train_queries)
                 batch_queries_terms.extend([self.train_queries[i] for i in extra_indices])
                 batch_labels.extend([train_labels[i] for i in extra_indices])


        return batch_queries_terms, batch_labels

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              queries_to_evaluate: Optional[List[Term]] = None,
              labels_to_evaluate: Optional[List[int]] = None
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets environments, optionally using provided queries/labels."""
        super().reset(seed=seed) # Call parent reset for seeding
        if seed is not None: self._set_seed(seed) # Also set local random state if needed
        logging.info(f'Resetting Batch (Size: {self.batch_size})')

        batch_queries_terms: List[Term]; batch_labels: List[int]

        # --- Logic for selecting initial queries (train/eval/provided) ---
        # (Keep the logic from the original env file)
        if queries_to_evaluate is not None:
            logging.info("Resetting with provided queries for evaluation.")
            if len(queries_to_evaluate) != self.batch_size: raise ValueError(f"Provided queries length ({len(queries_to_evaluate)}) must match batch_size ({self.batch_size}).")
            if labels_to_evaluate is None: labels_to_evaluate = [1] * self.batch_size; logging.warning("Assuming label 1 for provided queries.")
            elif len(labels_to_evaluate) != self.batch_size: raise ValueError(f"Provided labels length ({len(labels_to_evaluate)}) must match batch_size ({self.batch_size}).")
            batch_queries_terms = queries_to_evaluate
            batch_labels = labels_to_evaluate
        elif self.mode == 'eval' or self.mode == 'eval_corr':
            if not self.eval_queries: raise ValueError("Cannot reset in eval mode, eval_queries list is empty.")
            start_idx = self.current_eval_batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            if start_idx >= self.n_eval_queries:
                 start_idx = 0 # Wrap around
                 end_idx = self.batch_size
                 self.current_eval_batch_idx = 0
                 logging.warning("Eval reset wrapping around.")

            actual_indices = self.eval_indices[start_idx : min(end_idx, self.n_eval_queries)]
            batch_queries_terms = [self.eval_queries[i] for i in actual_indices]
            eval_labels = getattr(self.data_handler, "valid_labels", [1]*self.n_eval_queries)
            batch_labels = [eval_labels[i] for i in actual_indices]
            num_missing = self.batch_size - len(batch_queries_terms)
            if num_missing > 0:
                 logging.info(f"Padding eval batch with {num_missing} initial eval queries.")
                 if not self.eval_queries: raise ValueError("Cannot pad eval batch.")
                 # Pad with the first eval query/label
                 first_eval_query = self.eval_queries[0]
                 first_eval_label = eval_labels[0]
                 batch_queries_terms.extend([first_eval_query] * num_missing)
                 batch_labels.extend([first_eval_label] * num_missing)
            self.current_eval_batch_idx += 1
        elif self.mode == 'train':
            batch_queries_terms, batch_labels = self._sample_initial_batch()
        else:
            raise ValueError(f"Invalid mode for reset: {self.mode}")
        # --- End query selection logic ---

        self.current_queries_terms = batch_queries_terms
        self.current_labels = torch.tensor(batch_labels, dtype=torch.int, device=self.device)
        self.current_depth.zero_()
        self.index_manager.reset_atom() # Reset local atom indices

        # Get initial sub indices (P, S, O)
        initial_sub_idx_list = []
        for i in range(self.batch_size):
            query_term = batch_queries_terms[i]
            # Initial state is just the query itself
            initial_state_terms = [query_term] if isinstance(query_term, Term) else query_term # Handle if query is already a list?
            if not initial_state_terms: initial_state_terms = [Term("False", [])] # Handle empty initial state

            # Use get_atom_sub_index which returns (pad_atoms, 3) tensor
            _, sub_idx = self.index_manager.get_atom_sub_index(initial_state_terms)
            initial_sub_idx_list.append(sub_idx)

        initial_sub_idx_batch = torch.stack(initial_sub_idx_list) # (bs, pad_atoms, 3)

        # Get next possible states based on the initial state
        derived_sub_idx_batch, truncated_flags_batch, action_masks_batch = self.get_next_states_batch(initial_sub_idx_batch)

        # Update TensorDict with initial state and derived states
        self.tensordict.update_(TensorDict({
                 "state_sub_idx": initial_sub_idx_batch,
                 "derived_sub_idx": derived_sub_idx_batch,
                 "action_mask": action_masks_batch,
                 "label": self.current_labels,
                 "done": torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
                 "reward": torch.zeros(self.batch_size, dtype=torch.float32, device=self.device),
             }, batch_size=[self.batch_size]))

        if torch.any(truncated_flags_batch):
            logging.info(f"Truncation occurred during reset filtering for indices: {torch.where(truncated_flags_batch)[0].tolist()}")

        obs = self._get_obs_from_tensordict(self.tensordict)
        if self.verbose > 1: self._log_batch_state("Reset")
        return obs, {} # Return obs and info dict

    def get_next_states_batch(self,
                              current_sub_idx: torch.Tensor, # Shape (bs, pad_atoms, 3)
                              dones_mask: Optional[torch.Tensor] = None # Shape (bs,)
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets the next possible states using the integrated unification engine and applies filtering.
        Output shapes:
         - derived_sub_idx: (bs, padding_states, padding_atoms, 3)
         - truncated_flags: (bs,)
         - action_masks: (bs, padding_states)
        """
        bs = current_sub_idx.shape[0]
        if dones_mask is None:
            dones_mask = torch.zeros(bs, dtype=torch.bool, device=self.device)

        # --- Prepare Input for Unification ---
        # Reshape current state for unification functions: (bs, 1, pad_atoms, 3)
        current_state_reshaped = current_sub_idx.unsqueeze(1)

        # --- Run Unification ---
        # Fact Unification
        fact_results = batch_unify_with_facts(
            states_idx=current_state_reshaped,
            facts=self.facts_idx_tensor,
            vars_idx=self.index_manager, # Pass IndexManager to get vars tensor/set
            k=self.k_facts_rules,
            device=self.device,
            verbose=self.prover_verbose > 1,
            predicate_idx2str=self.index_manager.predicate_idx2str,
            constant_idx2str=self.index_manager.constant_idx2str,
            variable_idx2str=self.index_manager.variable_idx2str
        )

        # Rule Unification
        rule_results = batch_unify_with_rules(
            states_idx=current_state_reshaped,
            rules=self.rules_idx_tensor,
            vars_idx=self.index_manager, # Pass IndexManager
            k=self.k_facts_rules,
            max_subs_per_rule=self.max_subs_per_rule,
            device=self.device,
            verbose=self.prover_verbose > 0, # More verbose for rules maybe
            predicate_idx2str=self.index_manager.predicate_idx2str,
            constant_idx2str=self.index_manager.constant_idx2str,
            variable_idx2str=self.index_manager.variable_idx2str
        )

        # --- Apply Substitutions and Generate Next States ---
        next_states_from_facts = apply_substitutions_and_create_next_state(
            states_idx=current_state_reshaped,
            unification_result=fact_results,
            vars_idx_tensor=self.index_manager.vars_idx_tensor,
            new_max_atoms=self.padding_atoms, # Output states should match env padding
            max_sub_iterations=self.max_sub_iterations
        ) # Shape: (bs, k_facts_rules, padding_atoms, 3)

        next_states_from_rules = apply_substitutions_and_create_next_state(
            states_idx=current_state_reshaped,
            unification_result=rule_results,
            vars_idx_tensor=self.index_manager.vars_idx_tensor,
            new_max_atoms=self.padding_atoms, # Output states should match env padding
            max_sub_iterations=self.max_sub_iterations
        ) # Shape: (bs, k_facts_rules, padding_atoms, 3)

        # --- Combine Fact and Rule Results ---
        # Concatenate along the 'k' dimension (dim=1)
        combined_next_states = torch.cat((next_states_from_facts, next_states_from_rules), dim=1)
        combined_valid_masks = torch.cat((fact_results.valid_mask, rule_results.valid_mask), dim=1)
        # Shape: (bs, 2 * k_facts_rules, padding_atoms, 3)
        # Shape: (bs, 2 * k_facts_rules)

        # --- Truncate/Pad to self.padding_states ---
        total_k = combined_next_states.shape[1]
        k_to_keep = min(total_k, self.padding_states)

        # Use combined_valid_masks to select top valid states if total_k > padding_states
        # For simplicity now, just truncate. A better approach might prioritize valid ones.
        raw_derived_sub_idx = torch.full((bs, self.padding_states, self.padding_atoms, 3), PADDING_VALUE, dtype=torch.long, device=self.device)
        raw_action_mask = torch.zeros((bs, self.padding_states), dtype=torch.bool, device=self.device)

        raw_derived_sub_idx[:, :k_to_keep, :, :] = combined_next_states[:, :k_to_keep, :, :]
        raw_action_mask[:, :k_to_keep] = combined_valid_masks[:, :k_to_keep]


        # --- Initialize Filtered Output Tensors ---
        filtered_derived_sub_idx = raw_derived_sub_idx.clone() # Use sub_idx directly
        filtered_action_masks = raw_action_mask.clone()
        batch_truncated_flags = torch.zeros(bs, dtype=torch.bool, device=self.device)

        # --- Apply Filters (Vectorized, No Memory Pruning) ---

        # 1. Atom Truncation Mask (Check if number of *actual* atoms exceeds limit)
        # An atom is actual if its predicate is not PADDING_VALUE
        atom_valid_mask = torch.ones_like(raw_action_mask) # Default to valid
        if self.truncate_atoms:
            # Count non-padding predicates in each derived state
            num_actual_atoms = (raw_derived_sub_idx[:, :, :, 0] != PADDING_VALUE).sum(dim=2) # (bs, pad_states)
            atom_valid_mask = (num_actual_atoms <= self.padding_atoms) # Check if count is within limit
            # Update truncation flags for states invalid due to atom limits
            batch_truncated_flags |= torch.any(~atom_valid_mask & raw_action_mask, dim=1)

        # Combine engine mask and atom truncation mask
        filtered_action_masks = filtered_action_masks & atom_valid_mask

        # 2. Handle "No Valid States" Case (Vectorized)
        valid_counts = filtered_action_masks.sum(dim=1) # (bs,)
        no_valid_mask = (valid_counts == 0) & (~dones_mask) # Only apply if not already done

        if torch.any(no_valid_mask):
            if self.verbose > 1: logging.info(f"Batch items with no valid states after filtering: {torch.where(no_valid_mask)[0].tolist()}. Inserting False.")
            # Get indices for 'False' state once
            false_term = Term("False", [])
            # Use get_atom_sub_index which returns (pad_atoms, 3)
            _, false_sub_idx_single = self.index_manager.get_atom_sub_index([false_term])
            false_atom_len = (false_sub_idx_single[:, 0] != PADDING_VALUE).sum().item() # Count actual atoms in False state

            if self.padding_atoms >= false_atom_len:
                # Insert False state at the first action slot (index 0) for affected batch items
                filtered_derived_sub_idx[no_valid_mask, 0, :, :] = false_sub_idx_single

                # Update action mask: only action 0 is valid now for these items
                filtered_action_masks[no_valid_mask, 0] = True
                filtered_action_masks[no_valid_mask, 1:] = False
                # Mark these as truncated (because filtering led to failure state)
                batch_truncated_flags |= no_valid_mask
            else:
                logging.error("Cannot insert False state, padding_atoms too small.")
                batch_truncated_flags |= no_valid_mask # Mark as truncated anyway

        # 3. Apply final mask to zero out invalid entries in index tensors (use PADDING_VALUE)
        # Expand mask for broadcasting: (bs, pad_states, 1, 1)
        mask_expanded = filtered_action_masks.unsqueeze(-1).unsqueeze(-1)
        # Where mask is False, fill with PADDING_VALUE, otherwise keep original
        filtered_derived_sub_idx = torch.where(mask_expanded, filtered_derived_sub_idx, PADDING_VALUE)

        # 4. Handle "End Proof Action" (Optional - Add if needed)
        # ...

        return filtered_derived_sub_idx, batch_truncated_flags, filtered_action_masks


    def _vectorized_done_reward(self,
                                next_sub_idx: torch.Tensor,  # (bs, pad_atoms, 3)
                                labels: torch.Tensor        # (bs,)
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates done flags and rewards for the batch using sub_idx tensor."""
        # Check if *any* atom in the state has predicate 'False' or 'End'
        pred_indices = next_sub_idx[:, :, 0] # (bs, pad_atoms)
        is_padding_pred = (pred_indices == PADDING_VALUE)

        is_false = torch.any((pred_indices == self.false_idx) & (~is_padding_pred), dim=1)
        is_end = torch.any((pred_indices == self.end_idx) & (~is_padding_pred), dim=1) if self.end_proof_action else torch.zeros_like(is_false)

        # Check if *all* non-padding atoms have predicate 'True'
        # Mask for non-padding atoms: (bs, pad_atoms)
        non_padding_mask = ~is_padding_pred
        # Check if predicate is True for non-padding atoms
        is_true_pred_mask = (pred_indices == self.true_idx) & non_padding_mask
        # Count non-padding atoms and true atoms per batch item
        num_non_padding = non_padding_mask.sum(dim=1)
        num_true = is_true_pred_mask.sum(dim=1)
        # Condition: At least one non-padding atom exists, and all non-padding atoms are True
        is_true = (num_non_padding > 0) & (num_true == num_non_padding)

        terminated = is_false | is_end | is_true
        successful = is_true # Success only if state is exclusively True atoms

        # Reward: +1 for reaching True state with label 1, 0 otherwise
        rewards = torch.where(
            terminated & successful & (labels == 1),
            torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device)
        )
        # Optional: Penalty for reaching False state?
        # rewards = torch.where(terminated & is_false, torch.tensor(-1.0, device=self.device), rewards)

        return terminated, rewards

    def step(self, actions: Union[np.ndarray, torch.Tensor]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Performs a vectorized step for the batch."""
        if isinstance(actions, np.ndarray): actions = torch.from_numpy(actions).to(device=self.device, dtype=torch.int64)
        if actions.shape != (self.batch_size,): raise ValueError(f"Actions shape mismatch: {actions.shape} vs {(self.batch_size,)}")

        # --- Validate and Select Next State based on Action ---
        current_action_mask = self.tensordict["action_mask"]
        # Clamp actions to be within bounds [0, padding_states - 1]
        actions = torch.clamp(actions, 0, self.padding_states - 1)
        # Check if the *clamped* action is valid according to the mask
        invalid_action_chosen_mask = ~torch.gather(current_action_mask, 1, actions.unsqueeze(1)).squeeze(1)

        if torch.any(invalid_action_chosen_mask):
            logging.warning(f"Invalid actions chosen by agent after clamping for indices: {torch.where(invalid_action_chosen_mask)[0].tolist()}. Defaulting to action 0 (if valid) or False state.")
            # Default to action 0 if it's valid, otherwise force a 'False' state transition?
            # For now, let the invalid action proceed but it might lead to a padding state.
            # A stricter approach would be needed here, e.g., finding the first valid action.
            # Let's select the state corresponding to the (potentially invalid) action.
            pass # Allow selection, subsequent steps handle padding/False states

        # Gather the chosen next state's sub_idx based on the action
        # Action selects along dim 1 of derived_sub_idx
        action_idx_sub_gather = actions.view(self.batch_size, 1, 1, 1).expand(-1, -1, self.padding_atoms, 3)
        next_sub_idx = torch.gather(self.tensordict["derived_sub_idx"], 1, action_idx_sub_gather).squeeze(1) # (bs, pad_atoms, 3)

        # --- Check for Termination and Calculate Reward ---
        terminated, rewards = self._vectorized_done_reward(next_sub_idx, self.tensordict["label"])

        # --- Check for Truncation (Max Depth) ---
        self.current_depth += 1
        truncated_depth = (self.current_depth >= self.max_depth)

        # --- Get Next Possible States from the *new* current state ---
        # Only compute next states for environments that are not already terminated
        compute_next_mask = ~terminated
        # FIX: Add the missing fill_value argument
        derived_sub_next = torch.full_like(self.tensordict["derived_sub_idx"], fill_value=PADDING_VALUE) # Initialize with padding
        truncated_filter = torch.zeros_like(truncated_depth)
        action_masks_next = torch.zeros_like(self.tensordict["action_mask"])

        if torch.any(compute_next_mask):
             # Select only the states that need next state computation
             states_to_process = next_sub_idx[compute_next_mask]
             dones_for_process = terminated[compute_next_mask] # Pass dones to avoid inserting False again

             derived_sub_proc, trunc_filter_proc, action_masks_proc = self.get_next_states_batch(
                 states_to_process,
                 dones_for_process
             )
             # Scatter results back
             derived_sub_next[compute_next_mask] = derived_sub_proc
             truncated_filter[compute_next_mask] = trunc_filter_proc
             action_masks_next[compute_next_mask] = action_masks_proc


        # Final done flags
        truncated_final = truncated_depth | truncated_filter # Truncated if max depth OR filtering failed
        done_combined = terminated | truncated_final

        # Reset depth for environments that are done
        self.current_depth = torch.where(done_combined, 0, self.current_depth)

        # --- Update TensorDict ---
        self.tensordict.update_(TensorDict({
                 "state_sub_idx": next_sub_idx, # The state chosen by the action is the new current state
                 "derived_sub_idx": derived_sub_next, # Possible states derived from the new current state
                 "action_mask": action_masks_next,
                 "done": done_combined,
                 "reward": rewards,
                 # label remains the same
             }, batch_size=[self.batch_size]))

        # --- Prepare Return Values ---
        obs = self._get_obs_from_tensordict(self.tensordict)
        rewards_np = rewards.cpu().numpy()
        terminateds_np = terminated.cpu().numpy() # Return logical termination
        truncateds_np = truncated_final.cpu().numpy() # Return truncation (depth or filter fail)
        infos = {} # Add any extra info if needed

        if self.verbose > 1: self._log_batch_state(f"Step (Action: {actions.cpu().numpy()})")
        return obs, rewards_np, terminateds_np, truncateds_np, infos

    def _get_obs_from_tensordict(self, td: TensorDict) -> Dict[str, np.ndarray]:
        """Extracts numpy observations from the TensorDict based on observation_space."""
        obs = {}
        # Make sure observation space keys match the TensorDict keys used for observation
        obs_keys = ['state_sub_idx', 'derived_sub_idx', 'action_mask']
        for key in obs_keys:
            if key in td.keys(include_nested=True):
                tensor_data = td[key]
                if isinstance(tensor_data, torch.Tensor):
                    obs[key] = tensor_data.cpu().numpy()
                else:
                    logging.error(f"Obs key '{key}' is not a tensor in TensorDict: {type(tensor_data)}.")
                    # Create dummy data matching the expected space shape/type
                    space = self.observation_space[key]
                    obs[key] = np.zeros(space.shape, dtype=space.dtype)
            else:
                # This indicates a mismatch between _make_spec and _create_empty_tensordict or update_
                raise KeyError(f"Observation key '{key}' defined in spec not found in TensorDict keys: {td.keys()}")
        return obs


    def _log_batch_state(self, stage: str):
        """Helper to log the current state of the batch (for debugging)."""
        print(f"\n--- {stage} - Batch State ---")
        print(f"Depth: {self.current_depth.cpu().numpy()}")
        print(f"Labels: {self.tensordict['label'].cpu().numpy()}")
        print(f"Dones: {self.tensordict['done'].cpu().numpy()}")
        print(f"Rewards: {self.tensordict['reward'].cpu().numpy()}")
        if self.verbose > 2:
            for i in range(min(self.batch_size, 3)): # Print first few envs
                # Reconstruct terms from sub_idx
                state_terms = self.index_manager.get_state_terms_from_batch(self.tensordict["state_sub_idx"][i].unsqueeze(0))[0]
                print(f"  [Env {i}] D:{self.current_depth[i].item()} Done:{self.tensordict['done'][i].item()} R:{self.tensordict['reward'][i].item():.2f}")
                print(f"    State: {[str(t) for t in state_terms if t]}") # Print non-None terms
                print(f"    Mask: {self.tensordict['action_mask'][i].cpu().numpy().astype(int)}")
                # Log derived states if very verbose
                if self.verbose > 3:
                     print(f"    Derived States (Actions):")
                     derived_states_terms = self.index_manager.get_state_terms_from_batch(self.tensordict["derived_sub_idx"][i])
                     for action_idx, derived_terms in enumerate(derived_states_terms):
                         if self.tensordict['action_mask'][i, action_idx]: # Only print valid actions
                             print(f"      Action {action_idx}: {[str(t) for t in derived_terms if t]}")

        print("-" * 25)

    def render(self):
        # Render the environment to the screen (optional)
        # Could print the state of the first environment in the batch
        if self.batch_size > 0:
             print(f"--- Rendering Env 0 (Depth: {self.current_depth[0].item()}) ---")
             state_terms = self.index_manager.get_state_terms_from_batch(self.tensordict["state_sub_idx"][0].unsqueeze(0))[0]
             print(f"  State: {[str(t) for t in state_terms if t]}")
             print(f"  Label: {self.tensordict['label'][0].item()}")
             print(f"  Done: {self.tensordict['done'][0].item()}")
             print(f"  Reward: {self.tensordict['reward'][0].item():.2f}")
             print(f"  Action Mask: {self.tensordict['action_mask'][0].cpu().numpy().astype(int)}")
             print(f"  Possible Next States (Valid Actions):")
             derived_states_terms = self.index_manager.get_state_terms_from_batch(self.tensordict["derived_sub_idx"][0])
             valid_action_found = False
             for action_idx, derived_terms in enumerate(derived_states_terms):
                 if self.tensordict['action_mask'][0, action_idx]:
                     valid_action_found = True
                     print(f"    Action {action_idx}: {[str(t) for t in derived_terms if t]}")
             if not valid_action_found:
                 print("    [No valid actions]")
        else:
             print("Batch size is 0, cannot render.")


    def close(self):
        logging.info("Closing LogicEnv_gym_batch environment.")


# --- Example Usage Setup (Minimal) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Define Sample Data ---
    constants = {"a", "b", "c", "d", "p1", "p2", "p3"}
    predicates = {"p", "q", "parent", "ancestor"}
    # Extract variables from rules and potential queries
    rule1 = Rule(Term("ancestor", ["X", "Y"]), [Term("parent", ["X", "Y"])])
    rule2 = Rule(Term("ancestor", ["X", "Z"]), [Term("parent", ["X", "Y"]), Term("ancestor", ["Y", "Z"])])
    query1 = Term("ancestor", ["p1", "V"])
    query2 = Term("p", ["a", "X"])

    all_vars = set()
    for r in [rule1, rule2]: all_vars.update(r.get_variables())
    for q in [query1, query2]:
        for arg in q.args:
            if is_variable(arg): all_vars.add(arg)

    print(f"All Variables: {all_vars}")


    # --- Initialize Components ---
    idx_manager = IndexManager(
        constants=constants,
        predicates=predicates,
        variables=all_vars,
        rules=[rule1, rule2],
        padding_atoms=5,
        max_arity=2,
        device=device
    )

    facts_list = [
        Term("parent", ["p1", "p2"]),
        Term("parent", ["p2", "p3"]),
        Term("p", ["a", "b"]),
        Term("q", ["b", "c"]),
    ]
    rules_list = [rule1, rule2]
    train_queries_list = [query1, query2, Term("q", ["b", "V"])]
    valid_queries_list = [Term("ancestor", ["p1", "p3"])] # Should be provable

    data_handler = DataHandler(facts_list, rules_list, train_queries_list, valid_queries_list)

    # --- Initialize Environment ---
    BATCH_SIZE = 2
    env = LogicEnv_gym_batch(
        batch_size=BATCH_SIZE,
        index_manager=idx_manager,
        data_handler=data_handler,
        mode='train', # Use train to sample initial states
        padding_states=10, # Max 10 actions
        max_depth=5,
        verbose=2, # Set verbosity level
        prover_verbose=1, # Set unification verbosity
        device=device
    )

    # --- Simple Env Interaction Loop ---
    print("\n--- Starting Env Interaction ---")
    obs, info = env.reset()
    print("Initial Observation Received.")

    for step in range(env.max_depth * 2): # Run for a few steps
        print(f"\n--- Step {step} ---")
        # Choose a random valid action for each env in the batch
        action_mask_batch = obs['action_mask'] # (bs, pad_states)
        actions = []
        for i in range(BATCH_SIZE):
            valid_actions = np.where(action_mask_batch[i])[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                action = 0 # Choose action 0 if no actions are valid (should lead to done state)
                print(f"[Env {i}] No valid actions, choosing action 0.")
            actions.append(action)
        actions_np = np.array(actions, dtype=np.int64)
        print(f"Chosen Actions: {actions_np}")

        obs, rewards, terminated, truncated, infos = env.step(actions_np)
        print(f"Step {step} Results:")
        print(f"  Rewards: {rewards}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")

        dones = terminated | truncated
        if np.all(dones):
            print(f"\nAll environments finished at step {step}.")
            # Optionally reset here if needed for continuous runs
            # obs, info = env.reset()
            # print("Environments Reset.")
            break

    env.close()
    print("\n--- Env Interaction Finished ---")
