from typing import List, Tuple, Dict, Set
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
                 rules: List[Rule], # Rules might be needed for other logic, but not for var indexing here
                 constants_images: set = (),
                 constant_images_no: int = 0,
                 padding_atoms: int = 10,
                 max_arity: int = 2,
                 device: torch.device = torch.device("cpu")):

        self.device = device
        self.constants = constants
        self.predicates = predicates
        
        # Dynamically create a KGE-specific version for each predicate
        self.kge_preds = {f"{p}_kge" for p in self.predicates}
        self.special_preds = ['True', 'False', 'End'] + sorted(list(self.kge_preds))
        self.padding_idx = 0

        self.max_total_vars = max_total_vars # Max *pre-assigned* variables

        self.rules = rules # Stored but not used for variable indexing in this version
        self.padding_atoms = padding_atoms
        self.constants_images = constants_images
        self.max_arity = max_arity
        self.constant_images_no = constant_images_no

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

        self.fact_index: Dict[Tuple, Set[Term]] = {}

    def reset_next_var_index(self):
        self.next_var_index = self.variable_start_index

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

        self.constant_no = current_idx - 1

        # --- Predicates (Regular + Special) ---
        current_idx = 1
        # Add regular predicates first
        for term in sorted(self.predicates):
            self.predicate_str2idx[term] = current_idx
            self.predicate_idx2str[current_idx] = term
            current_idx += 1

        # Add special predicates (True, False, End, and all _kge versions)
        for term in self.special_preds:
            if term not in self.predicate_str2idx:
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
        unified_map = self.unified_term_map

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

    def subindices_to_terms(
        self,
        idx: torch.Tensor
    ) -> List[List[Term]]:
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

    def subindex_to_str(self, sub_index: torch.Tensor) -> str:
        """Converts a single sub_index tensor to a string representation of a Term."""
        pred_idx = sub_index[0].item()
        if pred_idx == 0:
            return "" 
        
        # This map needs to contain all possible terms, including variables
        idx2term_map = {**self.constant_idx2str, **self.variable_idx2str}
        
        predicate = self.predicate_idx2str.get(pred_idx)
        if not predicate:
            return ""

        arg_indices = sub_index[1:].tolist()
        args = []
        for arg_idx in arg_indices:
            if arg_idx == 0:
                break
            arg_str = idx2term_map.get(arg_idx)
            if arg_str:
                args.append(arg_str)
        
        args_str = ','.join(args)
        return f"{predicate}({args_str})"


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
