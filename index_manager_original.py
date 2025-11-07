from typing import List, Tuple, Dict, Set, Union, Optional
import itertools
import torch
import numpy as np
from utils import is_variable, Term, Rule
from functools import lru_cache
import torch

class IndexManager():
    '''
    Manages indices for constants, predicates, and variables (both template and runtime).
    Includes unified lookup maps for faster term indexing and tensor-based fact indexing.
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
        self.idx_dtype = torch.int32
        self.constants = constants
        self.predicates = predicates
        
        # Extract template variables from rules
        self.rule_template_variables = self._extract_template_variables_from_rules(rules)
        
        # Dynamically create a KGE-specific version for each predicate
        self.kge_preds = {f"{p}_kge" for p in self.predicates}
        self.special_preds = ['True', 'False', 'Endf', 'Endt'] + sorted(list(self.kge_preds))
        self.padding_idx = 0

        self.max_total_vars = max_total_vars # Max *pre-assigned* variables

        self.rules = rules
        # Pre-index rules by predicate to speed up unification
        self.rules_by_pred = {}
        for r in rules:
            self.rules_by_pred.setdefault(r.head.predicate, []).append(r)
            
        self.padding_atoms = padding_atoms
        self.constants_images = constants_images
        self.max_arity = max_arity
        self.constant_images_no = constant_images_no

        # --- Initialize Mappings ---
        self.constant_str2idx: Dict[str, int] = {}
        self.constant_idx2str: Dict[int, str] = {}
        self.predicate_str2idx: Dict[str, int] = {}
        self.predicate_idx2str: Dict[int, str] = {}
        self.template_variable_str2idx: Dict[str, int] = {}
        self.template_variable_idx2str: Dict[int, str] = {}
        self.runtime_variable_str2idx: Dict[str, int] = {}
        self.runtime_variable_idx2str: Dict[int, str] = {}
        # New unified map for constants and variables
        self.unified_term_map: Dict[str, int] = {}

        # --- Global indices for Constants and Predicates ---
        self.create_global_idx()

        # --- Template Variable indices ---
        self._create_template_var_idx()

        # --- Runtime Variable indices ---
        # Variable indices start after constants and template variables
        self.runtime_var_start_index = self.constant_no + self.template_variable_no + 1
        self.runtime_var_end_index = self.runtime_var_start_index + self.max_total_vars - 1
        self.runtime_variable_no = self.max_total_vars
        self.variable_no = self.constant_no + self.template_variable_no + self.runtime_variable_no

        self._create_fixed_runtime_var_idx() # Create mappings for runtime variables

        # --- Create Unified Term Map ---
        # Combine constant and variable maps for faster lookup
        self.unified_term_map.update(self.constant_str2idx)
        self.unified_term_map.update(self.template_variable_str2idx)

        # --- High-Performance Indices ---
        self.fact_index: Dict[Tuple, Union[Set[Term], torch.Tensor]] = {}
        self.rule_index: Dict[int, List[int]] = {}
        
        # Tensor-based fact storage
        self.facts_tensor: Optional[torch.Tensor] = None
        self.sorted_facts_tensor: Optional[torch.Tensor] = None
        self.predicate_range_map: Optional[torch.Tensor] = None
        
        # Special predicate tensors
        self.true_pred_idx = self.predicate_str2idx.get('True', -1)
        self.false_pred_idx = self.predicate_str2idx.get('False', -1)
        
        if self.true_pred_idx != -1:
            self.true_tensor = torch.tensor([[self.true_pred_idx, self.padding_idx, self.padding_idx]], dtype=torch.long, device=self.device)
        else:
            self.true_tensor = torch.empty((1, 3), dtype=torch.long, device=self.device)
            
        if self.false_pred_idx != -1:
            self.false_tensor = torch.tensor([[self.false_pred_idx, self.padding_idx, self.padding_idx]], dtype=torch.long, device=self.device)
        else:
            self.false_tensor = torch.empty((1, 3), dtype=torch.long, device=self.device)

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
        self.constant_no = current_idx - 1

        # --- Predicates ---
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
        """Convert a state tuple key back to a sub-index tensor."""
        # `as_tensor()` avoids an extra copy when the cache hits
        flat = torch.as_tensor(key, dtype=self.idx_dtype, device=self.device)
        return flat.view(self.padding_atoms, self.max_arity + 1)

    def get_atom_sub_index(self, state: List[Term]) -> torch.Tensor:
        # zero-copy lookup; DO NOT mutate the returned tensor
        return self._state_tuple_to_subidx(self.state_to_tuple(state))


    # def get_atom_sub_index(self, state: List[Term]) -> torch.Tensor:
    #     """
    #     Get sub-indices (predicate, args) for each atom in a state in a single pass.
    #     Uses a unified term map for faster argument lookup by processing arguments together.

    #     Args:
    #         state: A list of Term objects representing the logical state.

    #     Returns:
    #         sub_index: Tensor (padding_atoms, max_arity + 1) with indices.
    #     """
    #     # --- State Length Check ---
    #     state_len = len(state)
    #     if state_len > self.padding_atoms:
    #          raise ValueError(f"Length of processed state ({state_len}) exceeds padding_atoms ({self.padding_atoms}).")

    #     # --- Initialize Tensor ---
    #     sub_index = torch.zeros(self.padding_atoms, self.max_arity + 1, device=self.device, dtype=self.idx_dtype)

    #     # --- Single Pass for Indexing ---
    #     # Local references to maps for efficiency
    #     predicate_map = self.predicate_str2idx
    #     unified_map = self.unified_term_map

    #     for i, atom in enumerate(state):
    #         # --- Predicate Index ---

    #         try:
    #             sub_index[i, 0] = predicate_map[atom.predicate]
    #         except KeyError:
    #              raise KeyError(f"Predicate '{atom.predicate}' not found in predicate map.")

    #         # --- Argument Indices ---
    #         num_args = len(atom.args)
    #         if num_args > 0:
    #             max_j = min(num_args, self.max_arity)
    #             try:
    #                 # Directly assign the slice from a list comprehension
    #                 sub_index[i, 1:max_j + 1] = torch.tensor(
    #                     [unified_map[arg] for arg in atom.args[:max_j]],
    #                     device=self.device,
    #                     dtype=self.idx_dtype
    #                 )
    #             except KeyError as e:
    #                  raise KeyError(f"Argument '{e}' not in constant or variable maps.") from e

    #     return sub_index

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

    def build_fact_index(self, facts: List[Term]) -> Dict[Tuple, Set[Term]]:
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
        Dict[Tuple, Set[Term]]
            Mapping from subset-keys to the set of matching facts.
        """
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

    # ========================================================================
    # TENSOR-BASED METHODS
    # ========================================================================

    def is_var_idx(self, idx: Union[int, torch.Tensor]) -> Union[bool, torch.Tensor]:
        """
        Check if the given index or tensor of indices corresponds to any variable (template or runtime).
        """
        is_after_constants = self.constant_no < idx
        is_before_end = idx <= self.runtime_var_end_index
        return is_after_constants & is_before_end

    def get_next_var(self) -> int:
        '''Get the next available runtime variable index (not used in tensor ops, but kept for compatibility).'''
        # This is mainly for the old string-based code
        if not hasattr(self, 'next_runtime_var_index'):
            self.next_runtime_var_index = self.runtime_var_start_index
        if self.next_runtime_var_index > self.runtime_var_end_index:
            raise ValueError(f"No more available runtime variable indices.")
        idx = self.next_runtime_var_index
        var_name = f"RuntimeVar_{idx}"
        self.runtime_variable_idx2str[idx] = var_name
        self.runtime_variable_str2idx[var_name] = idx
        self.unified_term_map[var_name] = idx  # Add to unified map
        self.next_runtime_var_index += 1
        return idx

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
                if arg_s not in self.runtime_variable_str2idx:
                    if not hasattr(self, 'next_runtime_var_index'):
                        self.next_runtime_var_index = self.runtime_var_start_index
                    self.get_next_var()
                    self.runtime_variable_str2idx[arg_s] = self.next_runtime_var_index - 1
                    self.runtime_variable_idx2str[self.next_runtime_var_index - 1] = arg_s
                    self.unified_term_map[arg_s] = self.next_runtime_var_index - 1  # Add to unified map
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
        
        # OPTIMIZED: Use vectorized conversion for large lists
        if len(state_str) > 100:  # Threshold for vectorization
            return self._state_to_tensor_vectorized(state_str)
        
        return torch.stack([self.term_to_tensor(t) for t in state_str])
    
    def _state_to_tensor_vectorized(self, state_str: List[Term]) -> torch.Tensor:
        """Vectorized conversion of Terms to tensor - much faster for large lists."""
        import numpy as np
        
        num_terms = len(state_str)
        
        # Use numpy for faster construction
        result = np.full(
            (num_terms, self.max_arity + 1),
            self.padding_idx,
            dtype=np.int64
        )
        
        # Vectorized lookup of predicates - use list comprehension but convert to numpy once
        pred_indices = np.array([self.predicate_str2idx[term.predicate] for term in state_str], dtype=np.int64)
        result[:, 0] = pred_indices
        
        # Vectorized lookup of arguments - process column by column
        for arg_pos in range(self.max_arity):
            # Extract all arguments at this position in one go
            arg_indices = []
            for term in state_str:
                term_args = term.args if isinstance(term.args, tuple) else tuple(term.args)
                if arg_pos < len(term_args):
                    arg_s = term_args[arg_pos]
                    # Use unified map for fast lookup (dictionary lookup is O(1))
                    arg_idx = self.unified_term_map.get(arg_s, self.padding_idx)
                    arg_indices.append(arg_idx)
                else:
                    arg_indices.append(self.padding_idx)
            
            result[:, arg_pos + 1] = np.array(arg_indices, dtype=np.int64)
        
        # Convert to torch tensor and move to target device
        return torch.from_numpy(result).to(device=self.device, dtype=torch.long)

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

    def build_facts_index(self, facts_tensor: torch.Tensor, cache_dir: str = None):
        """
        Builds an index that replicates the fast, old, string-based version for tensor-based facts.
        The index is a dictionary where keys are composite tuples representing every
        possible query pattern and values are tensors of the fact IDs that match that pattern.
        
        OPTIMIZED VERSION: Uses vectorized operations and minimizes GPU-CPU transfers.
        Index tensors are kept on CPU since they're only used for indexing/lookup operations.
        The actual facts_tensor stays on GPU for computation.
        
        Args:
            facts_tensor: Tensor of facts to index
            cache_dir: Optional directory to cache the index (disabled for now due to performance issues)
        """
        self.facts_tensor = facts_tensor.to(self.device)
        
        # Build the index - caching disabled for performance
        self.fact_index: Dict[Tuple, torch.Tensor] = {}
        
        # Build predicate range map for fallback
        self._build_predicate_range_map()

        # Build the Composite Key Index using optimized operations
        temp_index: Dict[Tuple, List[int]] = {}

        # Extract all data at once to minimize GPU-CPU transfers
        # Work with CPU tensors for index building
        if self.facts_tensor.is_cuda:
            facts_cpu = self.facts_tensor.cpu()
        else:
            facts_cpu = self.facts_tensor
        
        # Convert to numpy for fast iteration
        facts_np = facts_cpu.numpy()
        
        # Vectorized check for variables and padding
        is_var_mask = (facts_np > self.constant_no) & (facts_np <= self.runtime_var_end_index)
        is_padding_mask = (facts_np == self.padding_idx)
        
        # Process all facts
        for fact_id in range(len(facts_np)):
            fact = facts_np[fact_id]
            predicate_idx = int(fact[0])
            
            # Find all constant arguments in the current fact using masks
            constant_args_with_pos = []
            for arg_pos in range(self.max_arity):
                arg_val = int(fact[arg_pos + 1])
                if not is_var_mask[fact_id, arg_pos + 1] and not is_padding_mask[fact_id, arg_pos + 1]:
                    constant_args_with_pos.append((arg_pos, arg_val))

            # Generate a key for every combination of constants (including zero constants)
            for k in range(len(constant_args_with_pos) + 1):
                for subset_args in itertools.combinations(constant_args_with_pos, k):
                    key = (predicate_idx,) + subset_args
                    
                    if key not in temp_index:
                        temp_index[key] = []
                    temp_index[key].append(fact_id)

        # Convert all lists to tensors on CPU (they're only used for indexing)
        # This avoids 190k GPU transfers and is much faster
        import numpy as np
        for key, val_list in temp_index.items():
            val_array = np.array(val_list, dtype=np.int64)
            # Keep index on CPU - it's only used for lookups, not computation
            self.fact_index[key] = torch.from_numpy(val_array)

    def _build_predicate_range_map(self):
        """Helper to build the sorted tensor and range map for rule unification fallback."""
        if self.facts_tensor is None or self.facts_tensor.numel() == 0:
            self.sorted_facts_tensor = torch.empty((0, self.max_arity + 1), dtype=torch.long, device=self.device)
            self.predicate_range_map = torch.zeros((self.predicate_no + 2, 2), dtype=torch.long, device=self.device)
            return

        sorted_indices = torch.argsort(self.facts_tensor[:, 0], stable=True)
        self.sorted_facts_tensor = self.facts_tensor[sorted_indices]
        
        pred_indices = self.sorted_facts_tensor[:, 0]
        unique_preds, counts = torch.unique_consecutive(pred_indices, return_counts=True)
        
        self.predicate_range_map = torch.zeros((self.predicate_no + 2, 2), dtype=torch.long, device=self.device)
        ends = torch.cumsum(counts, dim=0)
        starts = ends - counts
        self.predicate_range_map[unique_preds, 0] = starts
        self.predicate_range_map[unique_preds, 1] = ends

    def build_rule_index(self, rules_tensor: torch.Tensor):
        """Builds an index for rules based on their head predicate."""
        self.rule_index.clear()
        for i in range(rules_tensor.shape[0]):
            pred_idx = rules_tensor[i, 0, 0].item()
            if pred_idx not in self.rule_index:
                self.rule_index[pred_idx] = []
            self.rule_index[pred_idx].append(i)

    def debug_print_atom(self, atom_tensor: torch.Tensor) -> str:
        """Debug helper to convert an atom tensor back to a string."""
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
        """Debug helper to convert a state tensor back to a string."""
        if state_tensor.numel() == 0:
            return "[]"
        atom_strs_list = [self.debug_print_atom(atom_tensor) for atom_tensor in state_tensor if atom_tensor[0].item() != self.padding_idx]
        if not atom_strs_list:
            return "[]"
        sep = ", " if oneline else "\n"
        return f"[{sep.join(atom_strs_list)}]" if oneline else sep.join(atom_strs_list)

    def debug_print_states_from_indices(self, states_list_tensor: List[torch.Tensor]) -> str:
        """Debug helper to convert a list of state tensors back to a string."""
        return "[" + ", ".join([self.debug_print_state_from_indices(s, oneline=True) for s in states_list_tensor]) + "]"
