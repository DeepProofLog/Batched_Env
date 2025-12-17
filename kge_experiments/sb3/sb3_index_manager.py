from typing import List, Tuple, Dict, Set, Union
import itertools
import torch
import numpy as np
try:
    # Try relative import first (when sb3/ is in sys.path)
    from sb3_utils import is_variable, Term, Rule
except ImportError:
    # Fallback to package import (when imported as sb3.sb3_index_manager)
    from sb3.sb3_utils import is_variable, Term, Rule
from functools import lru_cache

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
                 device: torch.device = torch.device("cpu"),
                 include_kge_predicates: bool = False):

        self.device = device
        self.idx_dtype = torch.int32
        self.constants = constants
        self.predicates = predicates
        
        # Dynamically create a KGE-specific version for each predicate
        self.kge_preds = {f"{p}_kge" for p in self.predicates}
        self.special_preds = ['True', 'False', 'Endf']
        if include_kge_predicates:
            self.special_preds += sorted(list(self.kge_preds))
        self.padding_idx = 0

        self.max_total_vars = max_total_vars # Max *pre-assigned* variables

        self.rules = rules
        # Pre-index rules by predicate to speed up unification
        self.rules_by_pred = {}
        for r in rules:
            self.rules_by_pred.setdefault(r.head.predicate, []).append(r)
        
        # Template variable indices will be computed after create_global_idx()
        self.template_var_str2idx: Dict[str, int] = {}
            
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

        # --- Compute template variable indices (to match tensor engine behavior) ---
        # The tensor engine assigns indices to variables in rules starting from constant_no + 1
        # We need to compute the same max index to ensure runtime variables start at the same point
        self._compute_template_variable_indices()

        # --- Fixed Variable indices ---
        # Variable indices start after the last constant index
        self.variable_start_index = self.constant_no + 1
        self.variable_end_index = self.variable_start_index + self.max_total_vars - 1
        self.variable_no = self.max_total_vars # Total number of pre-assigned variables

        # self.next_var_index = self.variable_start_index # Next available variable index

        self._create_fixed_var_idx() # Create mappings for fixed variables

        # --- Create Unified Term Map ---
        # Combine constant and variable maps for faster lookup
        self.unified_term_map.update(self.constant_str2idx)
        self.unified_term_map.update(self.variable_str2idx)

        self.fact_index: Dict[Tuple, Set[Term]] = {}

    # def reset_next_var_index(self):
    #     self.next_var_index = self.variable_start_index

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

        self.constant_no = len(self.constant_str2idx)

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

    def _compute_template_variable_indices(self):
        """Compute template variable indices from rules (to match tensor engine behavior).
        
        The tensor engine assigns indices to variables in rules:
        - X, Z, Y, K, etc. get indices starting from constant_no + 1
        - Variables are indexed in the order they first appear in rules
        
        This method builds the same mapping so that next_var_start_for_proofs
        can be computed identically to the tensor engine.
        """
        next_template_idx = self.constant_no + 1
        
        # Process rules in the same order as tensor engine (by rule order)
        for rule in self.rules:
            # Process head arguments
            for arg in rule.head.args:
                if isinstance(arg, str) and is_variable(arg) and arg not in self.template_var_str2idx:
                    self.template_var_str2idx[arg] = next_template_idx
                    next_template_idx += 1
            
            # Process body arguments
            for term in rule.body:
                for arg in term.args:
                    if isinstance(arg, str) and is_variable(arg) and arg not in self.template_var_str2idx:
                        self.template_var_str2idx[arg] = next_template_idx
                        next_template_idx += 1
        
        # Store the max template variable index and compute next_var_start_for_proofs
        self.max_template_var_idx = next_template_idx - 1 if self.template_var_str2idx else self.constant_no
        self.next_var_start_for_proofs = self.max_template_var_idx + 1

    def _create_fixed_var_idx(self):
        '''Create fixed indices for variables based on max_total_vars.'''
        # Create variable names "Var_i", "Var_i+1", ..., "Var_(no_constants + max_total_vars)"
        # Assign indices starting from self.variable_start_index
        for i in range(self.max_total_vars):
            var_name = f"Var_{self.variable_start_index + i}"
            var_index = self.variable_start_index + i
            self.variable_str2idx[var_name] = var_index
            self.variable_idx2str[var_index] = var_name


    def state_to_tuple(self, state: List[Term]) -> Tuple[int, ...]:
        """
        Convert a list of Term objects to a flat, hashable tuple of indices:

            (pred_idx, arg1_idx, arg2_idx, ..., PAD,  pred_idx, ...)

        The tuple has fixed length = padding_atoms * (max_arity + 1) so it can be
        used directly as an LRU-cache key.
        """
        flat: List[int] = []
        pmap, amap   = self.predicate_str2idx, self.unified_term_map
        pad, A, P    = self.padding_idx, self.max_arity, self.padding_atoms

        for atom in state[:P]:
            flat.append(pmap[atom.predicate])
            # copy args (and pad if arity < max_arity)
            for arg in atom.args[:A]:
                flat.append(amap[arg])
            flat.extend([pad] * (A - len(atom.args)))

        # pad missing atoms
        missing_atoms = P - len(state)
        flat.extend([pad] * missing_atoms * (A + 1))
        return tuple(flat)


    @lru_cache(maxsize=131_072)
    def _state_tuple_to_subidx(self, key: Tuple[int, ...]) -> torch.Tensor:
        """Convert a state tuple key back to a sub-index """
        # `as_tensor()` avoids an extra copy when the cache hits
        flat = torch.as_tensor(key, dtype=self.idx_dtype, device=self.device)
        return flat.view(self.padding_atoms, self.max_arity + 1)

    def get_atom_sub_index(self, state: List[Term]) -> torch.Tensor:
        # zero-copy lookup; DO NOT mutate the returned tensor
        return self._state_tuple_to_subidx(self.state_to_tuple(state))


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

    @torch._dynamo.disable
    def subindex_to_str(self, sub_index: torch.Tensor, truncate: bool = False) -> str:
        indices = sub_index.tolist()          # e.g. [14, 732, 11, 0, 0]
        pred_idx = indices[0]                 # <-- use the Python int
        if pred_idx == self.padding_idx:      # simpler pad check
            return ""

        predicate = self.predicate_idx2str.get(pred_idx)
        if predicate is None:
            return ""

        if truncate:
            predicate = (predicate[:3] + "_kge") if "_kge" in predicate else predicate[:3]

        args = []
        for arg_idx in indices[1:]:
            if arg_idx == self.padding_idx:
                break
            arg_str = self.constant_idx2str.get(arg_idx) or self.variable_idx2str.get(arg_idx)
            if arg_str:
                args.append(arg_str[:8] if truncate else arg_str)

        return f"{predicate}({','.join(args)})"

    def state_subindex_to_str(self, state_subindex: Union[torch.Tensor, np.ndarray],truncate=False) -> str:
        """Converts a state sub_index tensor (a list of atoms) to a single string."""
        # Ensure the input is a torch tensor before processing
        if isinstance(state_subindex, np.ndarray):
            state_subindex = torch.from_numpy(state_subindex).to(self.device)
        terms = []
        # state_subindex has shape (padding_atoms, max_arity + 1)
        for atom_subindex in state_subindex:
            # An atom_subindex row is all zeros for padding
            if torch.all(atom_subindex == self.padding_idx):
                break
            
            term_str = self.subindex_to_str(atom_subindex,truncate=truncate)
            if term_str:
                terms.append(term_str)
        
        return ", ".join(terms)

    def build_fact_index(self, 
                         facts: List[Term],
                         deterministic: bool = False
                         ) -> Dict[Tuple, List[Term]]:
        """Build an inverted index over facts supporting partial-argument lookup.

        The index key format is:
            (predicate, (pos0, const0), (pos1, const1), ...)
        where each ``(pos, const)`` pair indicates a fixed (non-variable) argument position
        and its ground constant. For every fact we generate keys for **all** subsets of its
        fixed-argument positions, enabling queries such as:
            - all facts of a predicate
            - facts of a predicate with the 1st argument fixed, etc.

        Parameters
        ----------
        facts : List[Term]
            Iterable of ground facts to index.

        Returns
        -------
        Dict[Tuple, List[Term]]
            Mapping from subset-keys to the list of matching facts (sorted for determinism).
        """
        self.fact_index.clear()
        
        # Build facts_idx tensor sorted by (pred_idx, head_idx, tail_idx) to match new implementation
        facts_indexed = []
        for fact in facts:
            pred_idx = self.predicate_str2idx[fact.predicate]
            head_idx = self.constant_str2idx[fact.args[0]]
            tail_idx = self.constant_str2idx[fact.args[1]]
            facts_indexed.append((pred_idx, head_idx, tail_idx, fact))
        
        # Sort by (pred_idx, head_idx, tail_idx) - matches new implementation's sorting
        facts_indexed.sort(key=lambda x: (x[0], x[1], x[2]))
        
        # Store sorted facts_idx tensor and sorted facts list
        self.facts_idx = torch.tensor(
            [[f[0], f[1], f[2]] for f in facts_indexed],
            dtype=torch.long, device=self.device
        )
        self.facts_sorted = [f[3] for f in facts_indexed]
        
        # Build the inverted index using sorted facts
        for fact in self.facts_sorted:
            predicate = fact.predicate
            args = fact.args
            constant_args_with_pos = [(i, arg) for i, arg in enumerate(args) if not is_variable(arg)]
            for i in range(len(constant_args_with_pos) + 1):
                for subset_args_with_pos in itertools.combinations(constant_args_with_pos, i):
                    sorted_subset = tuple(sorted(subset_args_with_pos, key=lambda x: x[0]))
                    key = (predicate,) + sorted_subset
                    if key not in self.fact_index:
                        if deterministic:
                            self.fact_index[key] = []
                        else:
                            self.fact_index[key] = set()
                    if deterministic:
                        self.fact_index[key].append(fact)
                    else:
                        self.fact_index[key].add(fact)
        
        # Sort all fact lists by index for deterministic ordering (matches new implementation)
        if deterministic:
            for key in self.fact_index:
                # Sort by (pred_idx, head_idx, tail_idx) to match new implementation exactly
                self.fact_index[key].sort(key=lambda f: (
                    self.predicate_str2idx[f.predicate],
                    self.constant_str2idx[f.args[0]] if len(f.args) > 0 else 0,
                    self.constant_str2idx[f.args[1]] if len(f.args) > 1 else 0
                ))
        return self.fact_index
