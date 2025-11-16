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
                valid_actions = sum(1 for m in mask_for_idx if m)
                self._log(level, f"    Action mask: valid actions: {valid_actions}")

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

    def atom_to_canonical_str(self, atom: Tensor, constant_no: int) -> str:
        """
        Convert an atom tensor to a canonical string representation.
        
        Args:
            atom: [3] tensor representing [predicate, arg1, arg2]
            constant_no: The highest constant index
            
        Returns:
            Canonical string representation of the atom
        """
        p = int(atom[0].item())
        a = int(atom[1].item())
        b = int(atom[2].item())

        if self.idx2predicate and 0 <= p < len(self.idx2predicate):
            pred = self.idx2predicate[p]
        else:
            pred = str(p)

        if pred in ('True', 'False'):
            return f"{pred}()"

        def _format_arg(val: int) -> str:
            if 1 <= val <= constant_no:
                if self.idx2constant and 0 <= val < len(self.idx2constant):
                    return self.idx2constant[val]
                return f"c{val}"
            if val > constant_no:
                return f"Var_{val - constant_no}"
            return f"_{val}"

        return f"{pred}({_format_arg(a)},{_format_arg(b)})"

    def state_to_canonical_str(self, state: Tensor, constant_no: int, padding_idx: int) -> str:
        """
        Convert a state tensor to a canonical string representation.
        
        Args:
            state: [M, 3] tensor representing a state (list of atoms)
            constant_no: The highest constant index
            padding_idx: The padding index value
            
        Returns:
            Canonical string representation of the state (atoms joined by '|')
        """
        valid = state[:, 0] != padding_idx
        atoms = state[valid]
        if atoms.numel() == 0:
            return ''
        return '|'.join(self.atom_to_canonical_str(atom, constant_no) for atom in atoms)

    def _tensor_state_canonical_key(self, state: Tensor, constant_no: int, padding_idx: int) -> Tuple[int, ...]:
        """Generate a canonical key for a tensor state independent of absolute variable ids.
        Atoms are sorted by (pred, arg1, arg2) before generating the key to ensure consistent ordering."""
        # Filter out padding atoms
        valid_mask = state[:, 0] != padding_idx
        valid_atoms = state[valid_mask]
        
        if valid_atoms.numel() == 0:
            return tuple()
        
        # Sort atoms by (pred, arg1, arg2) tuple for canonical order
        # This ensures atoms are in a consistent order regardless of how they were generated
        atom_tuples = [(int(atom[0].item()), int(atom[1].item()), int(atom[2].item())) for atom in valid_atoms]
        sorted_indices = sorted(range(len(atom_tuples)), key=lambda i: atom_tuples[i])
        
        key: List[int] = []
        var_map: Dict[int, int] = {}
        next_var = constant_no + 1
        
        for idx in sorted_indices:
            atom = valid_atoms[idx]
            pred = int(atom[0].item())
            key.append(pred)
            for arg in atom[1:3]:
                val = int(arg.item())
                if val == padding_idx or val <= constant_no:
                    key.append(val)
                else:
                    mapped = var_map.get(val)
                    if mapped is None:
                        mapped = next_var
                        var_map[val] = mapped
                        next_var += 1
                    key.append(mapped)
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
        order = sorted(range(len(keys)), key=lambda idx: (int(owners[idx].item()), keys[idx]))
        if order == list(range(len(keys))):
            return states, counts, owners, next_vars

        order_tensor = torch.tensor(order, dtype=torch.long, device=states.device)
        states = states.index_select(0, order_tensor)
        counts = counts.index_select(0, order_tensor)
        owners = owners.index_select(0, order_tensor)
        next_vars = next_vars.index_select(0, order_tensor)
        return states, counts, owners, next_vars

    def _canonicalize_tensor_state(self, state: Tensor, constant_no: int, padding_idx: int) -> Tensor:
        if state.dim() == 3:
            if state.shape[0] != 1:
                raise ValueError("Expected single batch dimension for canonicalization")
            state_2d = state[0]
        else:
            state_2d = state

        canonical = state_2d.clone()
        valid_rows = canonical[:, 0] != padding_idx
        if not valid_rows.any():
            return canonical

        var_mapping: Dict[int, int] = {}
        next_var = constant_no + 1

        rows = torch.nonzero(valid_rows, as_tuple=False).view(-1)
        for row in rows:
            for col in (1, 2):
                val = int(canonical[row, col].item())
                if val == padding_idx or val <= constant_no:
                    continue
                mapped = var_mapping.get(val)
                if mapped is None:
                    mapped = next_var
                    var_mapping[val] = mapped
                    next_var += 1
                canonical[row, col] = mapped

        return canonical

    def _tensor_atom_to_canonical_str(self, atom: Tensor, constant_no: int) -> str:
        p = int(atom[0].item())
        a = int(atom[1].item())
        b = int(atom[2].item())

        if self.idx2predicate:
            pred = self.idx2predicate[p] if 0 <= p < len(self.idx2predicate) else str(p)
        else:
            pred = str(p)

        if pred in ('True', 'False'):
            return f"{pred}()"

        def _format_arg(val: int) -> str:
            if 1 <= val <= constant_no:
                if self.idx2constant and 0 <= val < len(self.idx2constant):
                    return self.idx2constant[val]
                return f"c{val}"
            if val > constant_no:
                return f"Var_{val - constant_no}"
            return f"_{val}"

        return f"{pred}({_format_arg(a)},{_format_arg(b)})"

    def _tensor_state_to_canonical_str(self, state: Tensor, constant_no: int, padding_idx: int) -> str:
        valid = state[:, 0] != padding_idx
        atoms = state[valid]
        if atoms.numel() == 0:
            return ''
        atoms_list = [self._tensor_atom_to_canonical_str(atom, constant_no) for atom in atoms]
        atoms_list.sort()
        return '|'.join(atoms_list)

    def canonicalize_state(self, state: Tensor, constant_no: int, padding_idx: int) -> Tensor:
        """Canonicalize a single tensor state (drop batch dim if present)."""
        return self._canonicalize_tensor_state(state, constant_no, padding_idx)

    def canonical_state_to_str(self, state: Tensor, constant_no: int, padding_idx: int) -> str:
        """Canonicalize tensor state and convert to string for comparisons."""
        canonical = self.canonicalize_state(state, constant_no, padding_idx)
        return self._tensor_state_to_canonical_str(canonical, constant_no, padding_idx)