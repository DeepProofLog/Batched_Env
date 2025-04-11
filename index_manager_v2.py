import torch
from typing import List, Tuple, Set, Dict, Any, Optional
import logging

# Assuming Term and is_variable are defined elsewhere
class Term:
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
        # Ensure args are hashable (e.g., convert lists to tuples if they contain unhashable types)
        try:
            hashable_args = tuple(self.args)
        except TypeError:
            # Handle unhashable arguments if necessary, e.g., by converting them
            # For simplicity, assuming args are typically strings or numbers
            hashable_args = tuple(map(str, self.args))
            logging.warning(f"Unhashable args in Term hash: {self.args}. Converted to strings.")
        return hash((self.predicate, hashable_args))

def is_variable(arg: Any) -> bool:
    # Simple check if it's an uppercase string, adjust as needed
    return isinstance(arg, str) and arg.isupper() and len(arg) == 1

class IndexManager():
    """
    Manages global and local indices for constants, predicates, variables, and atoms.
    Refactored for better integration with batched tensor operations.
    """
    def __init__(self, constants: Set[str],
                 predicates: Set[str],
                 variables: Set[str],
                 constant_no: int,
                 predicate_no: int,
                 variable_no: int,
                 rules: List, # Assuming rules are parsed into some structure if needed
                 constants_images: Set[str] = (),
                 constant_images_no: int = 0,
                 rule_depend_var: bool = True, # Keep for compatibility, but less relevant if not substituting
                 padding_atoms: int = 10,
                 max_arity: int = 2,
                 device: torch.device = torch.device("cpu")):

        self.device = device
        self.constants = constants
        self.predicates = predicates.union({'True', 'False', 'End'}) # Ensure special predicates are included
        self.variables = variables
        self.constant_no = constant_no
        self.variable_no = variable_no
        # Adjust predicate_no to account for special predicates
        self.predicate_no = len(self.predicates) # Recalculate based on the union

        self.constants_images = constants_images
        self.constant_images_no = constant_images_no

        self.rules = rules
        self.rule_depend_var = rule_depend_var
        self.padding_atoms = padding_atoms
        self.max_arity = max_arity

        # --- Global Mappings (ID 0 reserved for padding) ---
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
        # Map atom_index -> sub_indices_tensor for caching
        self.atom_idx_to_sub_idx: Dict[int, torch.Tensor] = {}
        # Map variable_string -> variable_index within an episode (if not rule_depend_var)
        self.ep_variable_str2idx: Dict[str, int] = {}

        self.next_atom_idx: int = 1 # Start atom indices from 1
        self.next_var_idx: int = -1 # Initialized in reset_atom if needed

        self._create_global_idx()
        # Removed rule_features_vars as variable substitution is complex and maybe less needed with direct indexing

        logging.info("IndexManager initialized.")
        logging.info(f"Constants: {self.constant_no}, Predicates: {self.predicate_no}, Variables: {self.variable_no}")
        logging.info(f"True/False/End Indices: {self.true_idx}/{self.false_idx}/{self.end_idx}")

    def _create_global_idx(self):
        """Create global indices for constants, predicates, and potentially rule-dependent variables."""
        # Constants (handle images if present)
        if self.constant_images_no > 0:
            constants_wout_images = [c for c in self.constants if c not in self.constants_images]
            current_idx = 1
            for term in sorted(self.constants_images):
                self.constant_str2idx[term] = current_idx
                self.constant_idx2str[current_idx] = term
                current_idx += 1
            for term in sorted(constants_wout_images):
                self.constant_str2idx[term] = current_idx
                self.constant_idx2str[current_idx] = term
                current_idx += 1
        else:
            for i, term in enumerate(sorted(self.constants)):
                self.constant_str2idx[term] = i + 1
                self.constant_idx2str[i + 1] = term

        # Predicates (including special ones)
        for i, term in enumerate(sorted(self.predicates)):
            idx = i + 1
            self.predicate_str2idx[term] = idx
            self.predicate_idx2str[idx] = term
            if term == 'True': self.true_idx = idx
            elif term == 'False': self.false_idx = idx
            elif term == 'End': self.end_idx = idx

        if self.true_idx == -1 or self.false_idx == -1 or self.end_idx == -1:
             logging.warning("Special predicates 'True', 'False', or 'End' not found during indexing.")
             # Assign placeholder indices if missing, though this indicates a potential issue
             if 'True' not in self.predicate_str2idx: self.true_idx = self.predicate_no + 1
             if 'False' not in self.predicate_str2idx: self.false_idx = self.predicate_no + 2
             if 'End' not in self.predicate_str2idx: self.end_idx = self.predicate_no + 3


        # Variables (rule-dependent or global pool)
        if self.rule_depend_var:
            # Assign indices relative to constants
            var_start_idx = self.constant_no + 1
            for i, term in enumerate(sorted(self.variables)):
                idx = var_start_idx + i
                self.variable_str2idx[term] = idx
                self.variable_idx2str[idx] = term
        # else: variables are indexed per episode in reset_atom/get_atom_sub_index

    def reset_atom(self):
        """Reset episode-specific indices (atoms and potentially variables)."""
        self.term_hash_to_atom_idx = {}
        self.atom_idx_to_sub_idx = {}
        self.next_atom_idx = 1
        if not self.rule_depend_var:
            self.ep_variable_str2idx = {}
            # Variables indices start after constants
            self.next_var_idx = self.constant_no + 1

    def _get_term_index(self, term: Any) -> int:
        """Helper to get index for a constant or variable within the current episode context."""
        if is_variable(term):
            if self.rule_depend_var:
                if term not in self.variable_str2idx:
                     raise ValueError(f"Rule-dependent variable '{term}' not found in global index.")
                return self.variable_str2idx[term]
            else:
                # Episode-dependent variable indexing
                if term not in self.ep_variable_str2idx:
                    if self.next_var_idx > self.constant_no + self.variable_no:
                        # Log error instead of raising? Or allow dynamic expansion?
                        logging.error(f"Exceeded maximum number of variables ({self.variable_no}) with new variable '{term}'. Assigning dummy index.")
                        # Assign a dummy index or handle error appropriately
                        return 0 # Return padding index as fallback
                    else:
                        self.ep_variable_str2idx[term] = self.next_var_idx
                        self.next_var_idx += 1
                return self.ep_variable_str2idx[term]
        else:
            # Constant lookup
            if term not in self.constant_str2idx:
                # Handle unknown constants - log warning and return padding?
                logging.warning(f"Constant '{term}' not found in global index. Returning padding index 0.")
                return 0 # Padding index
            return self.constant_str2idx[term]

    def get_atom_sub_index(self, state: List[Term]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get padded atom indices and sub-indices for a single state (list of Terms).
        Manages local atom and variable indices for the current episode.
        Uses caching for sub-indices.
        """
        atom_indices = torch.zeros(self.padding_atoms, dtype=torch.int64, device=self.device)
        sub_indices = torch.zeros(self.padding_atoms, self.max_arity + 1, dtype=torch.int64, device=self.device)

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

            # --- Get Sub Indices (use cache if available) ---
            if atom_idx in self.atom_idx_to_sub_idx:
                sub_indices[i] = self.atom_idx_to_sub_idx[atom_idx]
            else:
                # Calculate sub-indices
                current_sub_idx = torch.zeros(self.max_arity + 1, dtype=torch.int64, device=self.device)
                try:
                    # Predicate Index
                    if atom.predicate == 'True': current_sub_idx[0] = self.true_idx
                    elif atom.predicate == 'False': current_sub_idx[0] = self.false_idx
                    elif atom.predicate == 'End': current_sub_idx[0] = self.end_idx
                    elif atom.predicate in self.predicate_str2idx:
                        current_sub_idx[0] = self.predicate_str2idx[atom.predicate]
                    else:
                        logging.warning(f"Unknown predicate '{atom.predicate}' in state. Assigning padding index 0.")
                        current_sub_idx[0] = 0 # Padding index

                    # Argument Indices
                    for j, arg in enumerate(atom.args):
                        if j >= self.max_arity:
                            logging.warning(f"Atom {atom} has more args than max_arity {self.max_arity}. Truncating args.")
                            break
                        current_sub_idx[j + 1] = self._get_term_index(arg)

                    sub_indices[i] = current_sub_idx
                    # Cache the result
                    self.atom_idx_to_sub_idx[atom_idx] = current_sub_idx.clone() # Store a clone

                except Exception as e:
                    logging.error(f"Error processing atom {atom}: {e}", exc_info=True)
                    # Assign padding index 0 on error
                    sub_indices[i, 0] = 0

        return atom_indices, sub_indices

    def get_term_from_indices(self, atom_idx_tensor: torch.Tensor, sub_idx_tensor: torch.Tensor) -> List[Term]:
        """
        Reconstructs a List[Term] state from padded atom and sub-index tensors for a SINGLE state.
        Stops when atom_idx is 0 (padding).
        """
        state: List[Term] = []
        if atom_idx_tensor.dim() != 1 or sub_idx_tensor.dim() != 2:
             raise ValueError(f"Input tensors must be 1D (atom_idx) and 2D (sub_idx). Got {atom_idx_tensor.shape} and {sub_idx_tensor.shape}")

        for i in range(self.padding_atoms):
            atom_idx = atom_idx_tensor[i].item()
            if atom_idx == 0: # Stop at padding
                break

            sub_indices = sub_idx_tensor[i]
            pred_idx = sub_indices[0].item()

            try:
                predicate = self.predicate_idx2str.get(pred_idx, f"UNK_PRED_{pred_idx}")
                if predicate.startswith("UNK_PRED"): logging.warning(f"Unknown predicate index {pred_idx} during reconstruction.")

                args = []
                for j in range(1, self.max_arity + 1):
                    arg_idx = sub_indices[j].item()
                    if arg_idx == 0: # Stop if argument is padding
                        # This assumes args are padded contiguously. If not, need different logic.
                        break

                    # Determine if it's a constant or variable based on index range
                    if arg_idx in self.constant_idx2str:
                        args.append(self.constant_idx2str[arg_idx])
                    elif arg_idx in self.variable_idx2str: # Check global rule-dependent vars first
                         args.append(self.variable_idx2str[arg_idx])
                    else:
                         # Check episode-local vars if applicable
                         found_local = False
                         if not self.rule_depend_var:
                              for var_str, var_idx in self.ep_variable_str2idx.items():
                                   if var_idx == arg_idx:
                                        args.append(var_str)
                                        found_local = True
                                        break
                         if not found_local:
                              logging.warning(f"Unknown argument index {arg_idx} for predicate {predicate}. Treating as UNK_ARG.")
                              args.append(f"UNK_ARG_{arg_idx}")

                state.append(Term(predicate, args))
            except Exception as e:
                 logging.error(f"Error reconstructing Term for atom index {atom_idx}, pred index {pred_idx}: {e}", exc_info=True)
                 # Add a placeholder or skip on error
                 state.append(Term("ERROR", [f"idx_{atom_idx}"]))


        return state

    def get_state_repr_for_memory(self, atom_idx_tensor: torch.Tensor) -> Tuple[int, ...]:
        """
        Creates a hashable representation (tuple of non-zero atom indices)
        for a single state's atom_idx tensor, suitable for memory set storage.
        """
        if atom_idx_tensor.dim() != 1:
             raise ValueError(f"Input tensor must be 1D. Got {atom_idx_tensor.shape}")
        # Filter out padding (0) and convert to tuple
        mem_repr = tuple(idx.item() for idx in atom_idx_tensor if idx.item() != 0)
        return mem_repr

    # --- Methods for Batched Negative Sampling ---
    def map_term_to_triple_indices(self, term: Term) -> Optional[Tuple[int, int, int]]:
        """Converts a Term (assuming arity 2) to (head_idx, pred_idx, tail_idx)."""
        if term.predicate in ['True', 'False', 'End'] or len(term.args) != 2:
             # logging.warning(f"Cannot map non-binary term to triple indices: {term}")
             return None # Or handle differently if needed

        pred_idx = self.predicate_str2idx.get(term.predicate)
        head_idx = self.constant_str2idx.get(term.args[0]) # Assuming args are constants
        tail_idx = self.constant_str2idx.get(term.args[1])

        if pred_idx is None or head_idx is None or tail_idx is None:
             logging.warning(f"Could not find indices for all parts of term {term} -> ({head_idx}, {pred_idx}, {tail_idx})")
             return None
        return head_idx, pred_idx, tail_idx

    def map_indices_to_term(self, head_idx: int, pred_idx: int, tail_idx: int) -> Optional[Term]:
        """Converts (head_idx, pred_idx, tail_idx) back to a Term."""
        predicate = self.predicate_idx2str.get(pred_idx)
        head = self.constant_idx2str.get(head_idx)
        tail = self.constant_idx2str.get(tail_idx)

        if predicate is None or head is None or tail is None:
             logging.warning(f"Could not find strings for all indices ({head_idx}, {pred_idx}, {tail_idx})")
             return None
        return Term(predicate, [head, tail])
