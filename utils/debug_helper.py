import torch
from torch import Tensor
from typing import Optional, Sequence, Tuple, List, Dict
from tensordict import TensorDict


class DebugHelper:
    """
    Reusable debug helper class for logging and state inspection.
    Can be used across different environment classes and files.
    """

    def __init__(
        self,
        verbose: int = 0,
        debug_prefix: str = "[DebugHelper]",
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        padding_idx: int = 0,
        # Atom stringifier parameters
        n_constants: Optional[int] = None,
        idx2constant: Optional[Sequence[str]] = None,
        idx2predicate: Optional[Sequence[str]] = None,
        idx2template_var: Optional[Sequence[str]] = None,
    ):
        """
        Initialize the debug helper.

        Args:
            verbose: Verbosity level (0 = silent, higher = more verbose)
            debug_prefix: Prefix for log messages
            batch_size: Batch size for tensor operations
            device: Torch device
            padding_idx: Index used for padding
            n_constants: Number of constants for atom stringification
            idx2constant: Mapping from constant indices to strings
            idx2predicate: Mapping from predicate indices to strings
            idx2template_var: Mapping from template variable indices to strings
        """
        self.verbose = int(verbose)
        self._debug_prefix = debug_prefix
        self.batch_size_int = batch_size
        self._device = device
        self.padding_idx = padding_idx
        
        # Atom stringifier parameters
        self.n_constants = n_constants
        self.constant_no = n_constants  # Alias
        self.idx2constant = idx2constant
        self.idx2predicate = idx2predicate
        self.idx2template_var = idx2template_var

    def set_verbose(self, level: int) -> None:
        """Set the verbosity level."""
        self.verbose = int(level)

    def is_enabled(self, level: int = 1) -> bool:
        """Check if debugging is enabled at the given level."""
        return self.verbose >= level

    def _log(self, level: int, message: str) -> None:
        """Log a message if verbosity level is sufficient."""
        if self.verbose >= level:
            print(f"{self._debug_prefix} {message}")

    def atom_to_str(self, atom_idx: torch.LongTensor) -> str:
        """Convert an atom index tensor to a string representation."""
        p, a, b = [int(x) for x in atom_idx.tolist()]
        ps = self.idx2predicate[p] if self.idx2predicate and 0 <= p < len(self.idx2predicate) else str(p)

        def term_str(t: int) -> str:
            if t == self.padding_idx:
                return "_0"
            if self.n_constants is not None and 1 <= t <= self.n_constants:
                if self.idx2constant and 0 <= t < len(self.idx2constant):
                    return self.idx2constant[t]
                return f"c{t}"
            return f"Var_{t}"

        args: list[str] = []
        for arg in (a, b):
            if arg == self.padding_idx:
                continue
            args.append(term_str(arg))

        if ps in {"True", "False"}:
            return f"{ps}()"

        args_str = ", ".join(args)
        return f"{ps}({args_str})"

    def state_to_str(self, state_idx: torch.LongTensor) -> str:
        """Convert a state index tensor to a string representation, excluding padding."""
        if state_idx.numel() == 0:
            return "<empty>"
        if state_idx.dim() == 3:
            state_idx = state_idx.squeeze(0)  # Remove batch dimension
        # Filter out padding atoms
        non_pad_mask = state_idx[:, 0] != self.padding_idx
        non_pad_atoms = state_idx[non_pad_mask]
        if non_pad_atoms.numel() == 0:
            return "<empty>"
        parts = [self.atom_to_str(row) for row in non_pad_atoms]
        return ", ".join(parts)

    def _format_atoms(self, state: torch.Tensor) -> list[str]:
        """Format a state tensor into a list of atom strings or indices."""
        mask = state[:, 0] != self.padding_idx
        atoms = state[mask]
        if atoms.numel() == 0:
            return []
        if self.n_constants is not None and self.idx2constant is not None and self.idx2predicate is not None:
            return [self.atom_to_str(atom) for atom in atoms]
        return atoms.tolist()

    def _dump_states(
        self,
        label: str,
        current_queries: torch.Tensor,
        derived_states_batch: torch.Tensor,
        derived_states_counts: torch.Tensor,
        current_depths: Optional[torch.Tensor] = None,
        proof_depths: Optional[torch.Tensor] = None,
        current_labels: Optional[torch.Tensor] = None,
        rows: Optional[torch.Tensor] = None,
        level: int = 2,
        action_mask: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None
    ) -> None:
        """Dump current and derived states for specified rows (or all)."""
        if self.verbose < level:
            return  # Skip all work when not verbose enough
        
        print("\n")
        if rows is None:
            if self.batch_size_int is not None:
                rows = torch.arange(self.batch_size_int, device=self._device)
            else:
                rows = torch.arange(current_queries.shape[0], device=current_queries.device)
        rows_list = rows.tolist()
        self._log(level, f"{label}: inspecting rows {rows_list}")
        for idx in rows_list:
            state = current_queries[idx]
            derived_count = int(derived_states_counts[idx].item())
            state_atoms = self._format_atoms(state)
            depth_info = ""
            if current_depths is not None:
                depth_info += f"depth={int(current_depths[idx])} "
            if proof_depths is not None:
                depth_info += f"proof_depth={int(proof_depths[idx])} "
            if current_labels is not None:
                depth_info += f"label={int(current_labels[idx])} "
            self._log(level, f"  Idx {idx} {depth_info}query={state_atoms}")

            # Include action mask info if provided
            if action_mask is not None:
                mask_for_idx = action_mask[idx].tolist()
                valid_indices = [i for i, m in enumerate(mask_for_idx) if m]
                preview = valid_indices[: min(len(valid_indices), 10)]
                self._log(
                    level,
                    f"    Action mask: valid actions: {len(valid_indices)}"
                    f"{' -> ' + str(preview) if preview else ''}",
                )

            # Include done/rewards info if provided
            if done is not None:
                done_for_idx = done[idx].item()
                self._log(level, f"    Done: {done_for_idx}")
            if rewards is not None:
                reward_for_idx = rewards[idx].item()
                self._log(level, f"    Reward: {reward_for_idx}")

            derived_atoms = []
            for d in range(derived_count):
                derived_atoms.append(self._format_atoms(derived_states_batch[idx, d]))
            self._log(level, f"    Derived: {derived_atoms}\n")

    def print_states(self, title: str, states_tensor: torch.Tensor, 
                     counts: Optional[torch.Tensor] = None, 
                     padding_idx: Optional[int] = None,
                     verbose: Optional[int] = 1) -> None:
        """Print states in human-readable format."""
        if verbose is None:
            verbose = self.verbose
        if verbose <= 0:
            return  # Skip all work when not verbose
        
        pad = padding_idx if padding_idx is not None else self.padding_idx
        
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        if states_tensor.dim() == 3:  # [B, M, 3]
            for i in range(states_tensor.shape[0]):
                state = states_tensor[i]
                valid = state[:, 0] != pad
                if valid.any():
                    atoms = state[valid]
                    atoms_str = [self.atom_to_str(atom) for atom in atoms]
                    print(f"  State {i}: [{', '.join(atoms_str)}]")
                else:
                    print(f"  State {i}: <empty>")
        elif states_tensor.dim() == 4:  # [B, K, M, 3]
            for i in range(states_tensor.shape[0]):
                count = counts[i].item() if counts is not None else states_tensor.shape[1]
                if count > 0:
                    print(f"  Batch {i} ({count} states):")
                    for j in range(min(count, states_tensor.shape[1])):
                        state = states_tensor[i, j]
                        valid = state[:, 0] != pad
                        if valid.any():
                            atoms = state[valid]
                            atoms_str = [self.atom_to_str(atom) for atom in atoms]
                            print(f"    [{j}]: [{', '.join(atoms_str)}]")
                        else:
                            print(f"    [{j}]: <empty>")

    def _merge_reset_obs(self, base_td: TensorDict, reset_td: TensorDict, done_mask: torch.Tensor) -> TensorDict:
        """Merge base and reset tensordicts based on done mask."""
        rows = done_mask.nonzero(as_tuple=False).view(-1)
        if rows.numel() == 0:
            return base_td
        merged = base_td.clone()
        for key in reset_td.keys():
            reset_val = reset_td.get(key)
            if isinstance(reset_val, TensorDict):
                base_val = merged.get(key) if key in merged.keys() else reset_val.clone()
                merged.set(key, self._merge_reset_obs(base_val, reset_val, done_mask))
            else:
                if key in merged.keys():
                    target = merged.get(key).clone()
                else:
                    target = reset_val.clone()
                target[rows] = reset_val[rows]
                merged.set(key, target)
        return merged


    # ---- Methods for canonicalization and sorting for comparison with string envs ----


    # --- Canonical string representation ---

    def canonical_state_to_str(self, state: Tensor) -> str:
        """
        Canonicalize a tensor state and convert to string for comparisons.
        
        Unified canonicalization logic (matches string engine):
        1. Sort atoms alphabetically by (predicate, arg_types)
           - All variables are treated as equivalent for sorting (type 'VAR')
           - Constants are distinguished by their values (type 'CONST')
        2. After sorting, rename variables to Var_1, Var_2, ... in order of first appearance
        
        This ensures that structurally identical states produce the same canonical string,
        regardless of the original variable numbering.
        """
        # Handle batch dimension if present
        if state.dim() == 3:
            if state.shape[0] != 1:
                raise ValueError("Expected single batch dimension for canonicalization")
            state_2d = state[0]
        else:
            state_2d = state

        # Filter out padding atoms
        valid_mask = state_2d[:, 0] != self.padding_idx
        if not valid_mask.any():
            return ''
        
        valid_atoms = state_2d[valid_mask]
        
        # Step 1: Create sortable keys that don't depend on actual variable indices
        # Key structure: (predicate_str, tuple of (arg_type, arg_value))
        atoms_with_data: List[Tuple[tuple, int, int, int, bool, bool]] = []
        
        for atom in valid_atoms:
            pred, arg1, arg2 = int(atom[0].item()), int(atom[1].item()), int(atom[2].item())
            
            # Track whether args are variables
            arg1_is_var = arg1 != self.padding_idx and arg1 > self.constant_no
            arg2_is_var = arg2 != self.padding_idx and arg2 > self.constant_no
            
            # Get predicate string
            if 0 <= pred < len(self.idx2predicate):
                pred_str = self.idx2predicate[pred]
            else:
                pred_str = f"?p{pred}"
            
            # Create normalized sort key
            # All variables are treated as equal for sorting (only constants have distinguishing values)
            key_parts = [pred_str]
            
            # For arg1
            if arg1 == self.padding_idx:
                key_parts.append(('PAD',))
            elif arg1_is_var:
                # Variables: use ('VAR',) without the actual index
                # This ensures all variables are equivalent for sorting
                key_parts.append(('VAR',))
            else:
                # Constants: include the constant value to distinguish them
                if 0 <= arg1 < len(self.idx2constant):
                    key_parts.append(('CONST', self.idx2constant[arg1]))
                else:
                    key_parts.append(('CONST', f"?c{arg1}"))
            
            # For arg2
            if arg2 == self.padding_idx:
                key_parts.append(('PAD',))
            elif arg2_is_var:
                # Variables: use ('VAR',) without the actual index
                key_parts.append(('VAR',))
            else:
                # Constants: include the constant value
                if 0 <= arg2 < len(self.idx2constant):
                    key_parts.append(('CONST', self.idx2constant[arg2]))
                else:
                    key_parts.append(('CONST', f"?c{arg2}"))
            
            sort_key = tuple(key_parts)
            atoms_with_data.append((sort_key, pred, arg1, arg2, arg1_is_var, arg2_is_var))
        
        # Sort by the normalized key
        atoms_with_data.sort(key=lambda x: x[0])
        
        # Step 2: Rename variables in sorted order
        var_mapping: Dict[int, int] = {}
        next_var_num = 1
        canonical_atoms: List[str] = []
        
        for _, pred, arg1, arg2, arg1_is_var, arg2_is_var in atoms_with_data:
            # Rename variables
            if arg1_is_var:
                if arg1 not in var_mapping:
                    var_mapping[arg1] = next_var_num
                    next_var_num += 1
                arg1 = var_mapping[arg1]
            
            if arg2_is_var:
                if arg2 not in var_mapping:
                    var_mapping[arg2] = next_var_num
                    next_var_num += 1
                arg2 = var_mapping[arg2]
            
            # Format final string with renamed variables
            def format_arg_final(val: int, is_var: bool) -> str:
                if val == self.padding_idx:
                    return "PAD"
                if is_var:
                    return f"Var_{val}"
                else:
                    if 0 <= val < len(self.idx2constant):
                        return self.idx2constant[val]
                    else:
                        return f"?c{val}"
            
            if 0 <= pred < len(self.idx2predicate):
                pred_str = self.idx2predicate[pred]
            else:
                pred_str = f"?p{pred}"
            
            if pred_str in ['True', 'False'] and arg1 == self.padding_idx and arg2 == self.padding_idx:
                canonical_atoms.append(f"{pred_str}()")
            else:
                canonical_atoms.append(f"{pred_str}({format_arg_final(arg1, arg1_is_var)},{format_arg_final(arg2, arg2_is_var)})")
        
        return '|'.join(canonical_atoms)

    def canonical_states_to_str(self, states: Tensor, counts: Optional[Tensor] = None, return_indices: bool = False):
        """
        Canonicalize multiple states (batch) to string representations.
        
        Args:
            states: [B, M, 3] or [B, K, M, 3] tensor of states
            counts: Optional [B] tensor of valid atom counts per state
            return_indices: If True, return (sorted_canonical_strings, original_indices)
                           where original_indices[i] gives the original position of the 
                           state at sorted position i
        
        Returns:
            List[str] if return_indices=False
            Tuple[List[str], List[int]] if return_indices=True
        """
        results = []
        if states.dim() == 3:  # [B, M, 3]
            results = [self.canonical_state_to_str(states[i:i+1]) for i in range(states.shape[0])]
        elif states.dim() == 4:  # [B, K, M, 3]
            for i in range(states.shape[0]):
                count = counts[i].item() if counts is not None else states.shape[1]
                for j in range(count):
                    results.append(self.canonical_state_to_str(states[i, j:j+1]))
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {states.dim()}D")
        
        # Sort for consistent ordering, with index tracking
        results_with_idx = [(s, i) for i, s in enumerate(results)]
        results_with_idx.sort(key=lambda x: x[0])
        
        if return_indices:
            sorted_canonical = [x[0] for x in results_with_idx]
            original_indices = [x[1] for x in results_with_idx]
            return sorted_canonical, original_indices
        else:
            return [x[0] for x in results_with_idx]
        return results


    # -- Sort candidates by canonical string representation ---
    def _sort_candidates_by_str_order(self, states: Tensor,
                                      counts: Tensor,
                                      owners: Tensor,
                                      next_vars: Tensor,
                                      constant_no: int,
                                      padding_idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sort candidates by canonical STRING representation (like test_engines does).
        
        This method:
        1. Canonicalizes each state to its string representation using canonical_state_to_str
        2. Sorts by (owner, canonical_string)
        
        This matches the behavior in test_engines.py where states are sorted by their canonical strings.
        """
        if states.numel() == 0:
            return states, counts, owners, next_vars

        # Generate canonical string for each state
        canonical_strings = [self.canonical_state_to_str(states[i]) for i in range(states.shape[0])]
        
        # Sort by (owner, canonical_string)
        order = sorted(range(len(canonical_strings)), key=lambda idx: (int(owners[idx].item()), canonical_strings[idx]))
        
        # Check if already sorted
        if order == list(range(len(canonical_strings))):
            return states, counts, owners, next_vars

        order_tensor = torch.tensor(order, dtype=torch.long, device=states.device)
        states = states.index_select(0, order_tensor)
        counts = counts.index_select(0, order_tensor)
        owners = owners.index_select(0, order_tensor)
        next_vars = next_vars.index_select(0, order_tensor)
        return states, counts, owners, next_vars




    # --- NOT USED Sort by canonical key ---

    def _tensor_state_canonical_key(self, state: Tensor, constant_no: int, padding_idx: int) -> Tuple[int, ...]:
        """Generate a canonical key for sorting states.
        
        To match canonical_state_to_str (which test_envs uses for comparison):
        1. Renumber variables in original atom order
        2. Sort atoms by (pred, arg1, arg2)
        3. Create tuple key
        
        This ensures states sort the same way whether we use canonical keys or canonical strings.
        """
        valid_mask = state[:, 0] != padding_idx
        valid_atoms = state[valid_mask]
        
        if valid_atoms.numel() == 0:
            return tuple()
        
        # Step 1: Renumber variables in original order
        var_map: Dict[int, int] = {}
        next_var = constant_no + 1
        renumbered_atoms = valid_atoms.clone()
        
        for atom in renumbered_atoms:
            for col in (1, 2):
                val = int(atom[col].item())
                if val == padding_idx or val <= constant_no:
                    continue
                if val not in var_map:
                    var_map[val] = next_var
                    next_var += 1
                atom[col] = var_map[val]
        
        # Step 2: Sort atoms
        atom_tuples = [(int(atom[0].item()), int(atom[1].item()), int(atom[2].item())) for atom in renumbered_atoms]
        sorted_indices = sorted(range(len(atom_tuples)), key=lambda i: atom_tuples[i])
        
        # Step 3: Build key from sorted atoms
        key: List[int] = []
        for idx in sorted_indices:
            atom_tuple = atom_tuples[idx]
            key.extend(atom_tuple)
        
        return tuple(key)

    def _sort_candidates_by_canonical_order(self, states: Tensor,
                                            counts: Tensor,
                                            owners: Tensor,
                                            next_vars: Tensor,
                                            constant_no: int,
                                            padding_idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if states.numel() == 0:
            return states, counts, owners, next_vars

        keys = [self._tensor_state_canonical_key(states[i], constant_no, padding_idx) for i in range(states.shape[0])]
        # sorted() is stable by default (maintains original order for equal keys)
        order = sorted(range(len(keys)), key=lambda idx: (int(owners[idx].item()), keys[idx]))
        if order == list(range(len(keys))):
            return states, counts, owners, next_vars

        order_tensor = torch.tensor(order, dtype=torch.long, device=states.device)
        states = states.index_select(0, order_tensor)
        counts = counts.index_select(0, order_tensor)
        owners = owners.index_select(0, order_tensor)
        next_vars = next_vars.index_select(0, order_tensor)
        return states, counts, owners, next_vars
