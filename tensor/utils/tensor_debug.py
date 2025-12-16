"""
Unified Debug System for RL Pipeline

This module provides a centralized debugging framework that combines configuration,
logging, and state inspection capabilities across all components:
- Environment (state transitions, rewards, action spaces)
- Model (logits, actions, distributions, entropy, values)
- Agent (rollout statistics, training metrics)
- Prover (unification, proof search)
- Rollouts (step-by-step execution)

Usage:
    from utils.debug import Debug, DebugConfig
    
    # Create debug config with specific settings
    debug = Debug.create(
        env=2,        # Environment verbosity level
        model=1,      # Model verbosity level
        agent=2,      # Agent verbosity level
    )
    
    # Use in code
    if debug.is_enabled('env'):
        debug.log_env("State transition", state_before, state_after)
    
    # Print formatted state
    debug.print_state(state_tensor, title="Current State")
    
    # Get canonical string representation for comparison
    canonical = debug.canonical_state_to_str(state_tensor)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Sequence, Union
import torch
from torch import Tensor


@dataclass
class DebugConfig:
    """
    Configuration for debug verbosity and behavior.
    
    Verbosity levels:
        0: Disabled
        1: Basic info
        2+: More detailed (higher = more verbose)
    """
    # Main verbosity levels per category
    env: int = 0
    prover: int = 0
    agent: int = 0
    model: int = 0
    rollouts: int = 0
    
    # Fine-grained flags for specific features
    # Agent
    agent_rollout_stats: bool = True
    agent_train_stats: bool = True
    agent_episode_info: bool = True
    
    # Model
    model_logits: bool = False
    model_actions: bool = False
    model_distribution: bool = False
    model_entropy: bool = False
    model_values: bool = False
    model_action_mask: bool = False
    
    # Rollouts
    rollouts_steps: bool = False
    rollouts_rewards: bool = False
    rollouts_dones: bool = False
    
    # Environment
    env_transitions: bool = False
    env_action_spaces: bool = False
    
    # Prover
    prover_unification: bool = False
    prover_search: bool = False
    
    # Sampling control
    sample_envs: Optional[int] = None  # Only debug first N environments (None = all)
    sample_frequency: int = 1          # Debug every N steps/episodes
    max_actions_display: int = 10      # Max actions to display in detail
    
    # Output control
    prefix: str = "[Debug]"
    use_colors: bool = True
    
    def __post_init__(self):
        """Convert boolean flags to integers for consistency."""
        self.env = int(self.env)
        self.prover = int(self.prover)
        self.agent = int(self.agent)
        self.model = int(self.model)
        self.rollouts = int(self.rollouts)
    
    @classmethod
    def disabled(cls) -> 'DebugConfig':
        """Create config with all debugging disabled."""
        return cls()
    
    @classmethod
    def agent_only(cls) -> 'DebugConfig':
        """Create config for debugging agent only."""
        return cls(
            agent=2,
            agent_rollout_stats=True,
            agent_train_stats=True,
            agent_episode_info=True,
        )
    
    @classmethod
    def model_only(cls) -> 'DebugConfig':
        """Create config for debugging model only."""
        return cls(
            model=2,
            model_logits=True,
            model_actions=True,
            model_distribution=True,
            model_entropy=True,
            model_values=True,
            model_action_mask=True,
            sample_envs=3,
        )
    
    @classmethod
    def full(cls) -> 'DebugConfig':
        """Create config with full debugging."""
        return cls(
            env=2,
            prover=1,
            agent=2,
            model=2,
            rollouts=2,
            agent_rollout_stats=True,
            agent_train_stats=True,
            agent_episode_info=True,
            model_logits=True,
            model_actions=True,
            model_distribution=True,
            model_entropy=True,
            model_values=True,
            model_action_mask=True,
            rollouts_steps=True,
            rollouts_rewards=True,
            rollouts_dones=True,
            env_transitions=True,
            env_action_spaces=True,
            sample_envs=3,
        )
    
    @classmethod
    def entropy_debug(cls) -> 'DebugConfig':
        """Create config for debugging entropy issues."""
        return cls(
            env=1,
            agent=2,
            model=2,
            env_action_spaces=True,
            agent_rollout_stats=True,
            model_logits=True,
            model_actions=True,
            model_distribution=True,
            model_entropy=True,
            model_action_mask=True,
            sample_envs=5,
            max_actions_display=20,
            sample_frequency=4,
        )


class Debug:
    """
    Unified debugging helper with state inspection, logging, and formatting capabilities.
    
    This class combines:
    - Verbosity-based logging
    - Tensor state inspection and formatting
    - Canonical string representations for comparison
    - Action mask and reward visualization
    """

    def __init__(
        self,
        config: Optional[DebugConfig] = None,
        # Atom stringifier parameters (for tensor-to-string conversion)
        n_constants: Optional[int] = None,
        idx2constant: Optional[Sequence[str]] = None,
        idx2predicate: Optional[Sequence[str]] = None,
        idx2template_var: Optional[Sequence[str]] = None,
        padding_idx: int = 0,
        # Context info
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the debug helper.

        Args:
            config: Debug configuration (defaults to disabled)
            n_constants: Number of constants for atom stringification
            idx2constant: Mapping from constant indices to strings
            idx2predicate: Mapping from predicate indices to strings
            idx2template_var: Mapping from template variable indices to strings
            padding_idx: Index used for padding
            batch_size: Batch size for tensor operations
            device: Torch device
        """
        self.config = config or DebugConfig.disabled()
        
        # Atom stringifier parameters
        self.n_constants = n_constants
        self.constant_no = n_constants  # Alias
        self.idx2constant = idx2constant
        self.idx2predicate = idx2predicate
        self.idx2template_var = idx2template_var
        self.padding_idx = padding_idx
        
        # Context
        self.batch_size_int = batch_size
        self._device = device
        
        # ANSI color codes
        self._colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
        }

    # =====================================================================
    # Factory Methods
    # =====================================================================
    
    @classmethod
    def create(
        cls,
        env: int = 0,
        model: int = 0,
        agent: int = 0,
        prover: int = 0,
        rollouts: int = 0,
        **kwargs
    ) -> 'Debug':
        """Create a Debug instance with specific verbosity levels."""
        config = DebugConfig(
            env=env,
            model=model,
            agent=agent,
            prover=prover,
            rollouts=rollouts,
        )
        return cls(config=config, **kwargs)
    
    @classmethod
    def from_index_manager(cls, im, config: Optional[DebugConfig] = None) -> 'Debug':
        """Create Debug instance initialized from IndexManager."""
        return cls(
            config=config,
            n_constants=im.constant_no,
            idx2constant=tuple(im.idx2constant) if hasattr(im, 'idx2constant') else None,
            idx2predicate=tuple(im.idx2predicate) if hasattr(im, 'idx2predicate') else None,
            idx2template_var=tuple(im.idx2template_var) if hasattr(im, 'idx2template_var') else None,
            padding_idx=im.padding_idx if hasattr(im, 'padding_idx') else 0,
            device=im.device if hasattr(im, 'device') else None,
        )

    # =====================================================================
    # Configuration Checks
    # =====================================================================
    
    def is_enabled(self, category: str, level: int = 1) -> bool:
        """Check if debugging is enabled for a category at a given level."""
        return getattr(self.config, category, 0) >= level
    
    def should_debug_env(self, env_idx: Optional[int] = None) -> bool:
        """Check if environment should be debugged based on sampling."""
        if not self.is_enabled('env'):
            return False
        if self.config.sample_envs is None or env_idx is None:
            return True
        return env_idx < self.config.sample_envs
    
    def should_debug_step(self, step: int) -> bool:
        """Check if step should be debugged based on frequency."""
        return step % self.config.sample_frequency == 0

    # =====================================================================
    # Logging
    # =====================================================================
    
    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.config.use_colors:
            return text
        return f"{self._colors.get(color, '')}{text}{self._colors['reset']}"
    
    def _log(self, level: int, category: str, message: str) -> None:
        """Log a message if verbosity level is sufficient."""
        cat_level = getattr(self.config, category, 0)
        if cat_level >= level:
            color = {
                'env': 'green',
                'model': 'cyan',
                'agent': 'yellow',
                'prover': 'magenta',
                'rollouts': 'blue',
            }.get(category, 'reset')
            prefix = self._color(f"[{category.upper()}]", color)
            print(f"{self.config.prefix} {prefix} {message}")
    
    def log_env(self, message: str, level: int = 1) -> None:
        """Log environment message."""
        self._log(level, 'env', message)
    
    def log_model(self, message: str, level: int = 1) -> None:
        """Log model message."""
        self._log(level, 'model', message)
    
    def log_agent(self, message: str, level: int = 1) -> None:
        """Log agent message."""
        self._log(level, 'agent', message)
    
    def log_prover(self, message: str, level: int = 1) -> None:
        """Log prover message."""
        self._log(level, 'prover', message)
    
    def log_rollouts(self, message: str, level: int = 1) -> None:
        """Log rollouts message."""
        self._log(level, 'rollouts', message)

    # =====================================================================
    # Atom/State Stringification
    # =====================================================================
    
    def atom_to_str(self, atom_idx: torch.LongTensor) -> str:
        """Convert an atom index tensor to a string representation."""
        if atom_idx.numel() < 3:
            return "<invalid>"
        p, a, b = [int(x) for x in atom_idx.tolist()[:3]]
        
        # Predicate
        if self.idx2predicate and 0 <= p < len(self.idx2predicate):
            ps = self.idx2predicate[p]
        else:
            ps = str(p)

        def term_str(t: int) -> str:
            if t == self.padding_idx:
                return "_0"
            if self.n_constants is not None and 1 <= t <= self.n_constants:
                if self.idx2constant and 0 <= t < len(self.idx2constant):
                    return self.idx2constant[t]
                return f"c{t}"
            return f"Var_{t}"

        args: List[str] = []
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
            state_idx = state_idx.squeeze(0)
        # Filter out padding atoms
        non_pad_mask = state_idx[:, 0] != self.padding_idx
        non_pad_atoms = state_idx[non_pad_mask]
        if non_pad_atoms.numel() == 0:
            return "<empty>"
        parts = [self.atom_to_str(row) for row in non_pad_atoms]
        return ", ".join(parts)

    def _format_atoms(self, state: torch.Tensor) -> List[str]:
        """Format a state tensor into a list of atom strings or indices."""
        mask = state[:, 0] != self.padding_idx
        atoms = state[mask]
        if atoms.numel() == 0:
            return []
        if self.n_constants is not None and self.idx2constant is not None and self.idx2predicate is not None:
            return [self.atom_to_str(atom) for atom in atoms]
        return atoms.tolist()

    # =====================================================================
    # Canonical String Representation (for comparisons)
    # =====================================================================
    
    def canonical_state_to_str(self, state: Tensor) -> str:
        """
        Canonicalize a tensor state and convert to string for comparisons.
        
        Canonicalization:
        1. Sort atoms alphabetically by (predicate, arg_types)
           - All variables are treated as equivalent for sorting (type 'VAR')
           - Constants are distinguished by their values (type 'CONST')
        2. Rename variables to Var_1, Var_2, ... in order of first appearance
        """
        # Handle batch dimension
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
        
        # Build sortable keys
        atoms_with_data: List[Tuple[tuple, int, int, int, bool, bool]] = []
        
        for atom in valid_atoms:
            pred, arg1, arg2 = int(atom[0].item()), int(atom[1].item()), int(atom[2].item())
            
            # Track variable status
            arg1_is_var = arg1 != self.padding_idx and arg1 > self.constant_no
            arg2_is_var = arg2 != self.padding_idx and arg2 > self.constant_no
            
            # Predicate string
            if self.idx2predicate and 0 <= pred < len(self.idx2predicate):
                pred_str = self.idx2predicate[pred]
            else:
                pred_str = f"?p{pred}"
            
            # Normalized sort key
            key_parts = [pred_str]
            
            for arg, is_var in [(arg1, arg1_is_var), (arg2, arg2_is_var)]:
                if arg == self.padding_idx:
                    key_parts.append(('PAD',))
                elif is_var:
                    key_parts.append(('VAR',))
                else:
                    if self.idx2constant and 0 <= arg < len(self.idx2constant):
                        key_parts.append(('CONST', self.idx2constant[arg]))
                    else:
                        key_parts.append(('CONST', f"?c{arg}"))
            
            sort_key = tuple(key_parts)
            atoms_with_data.append((sort_key, pred, arg1, arg2, arg1_is_var, arg2_is_var))
        
        # Sort by normalized key
        atoms_with_data.sort(key=lambda x: x[0])
        
        # Rename variables in sorted order
        var_mapping: Dict[int, int] = {}
        next_var_num = 1
        canonical_atoms: List[str] = []
        
        for _, pred, arg1, arg2, arg1_is_var, arg2_is_var in atoms_with_data:
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
            
            def format_arg(val: int, is_var: bool) -> str:
                if val == self.padding_idx:
                    return "PAD"
                if is_var:
                    return f"Var_{val}"
                if self.idx2constant and 0 <= val < len(self.idx2constant):
                    return self.idx2constant[val]
                return f"?c{val}"
            
            if self.idx2predicate and 0 <= pred < len(self.idx2predicate):
                pred_str = self.idx2predicate[pred]
            else:
                pred_str = f"?p{pred}"
            
            if pred_str in ['True', 'False'] and arg1 == self.padding_idx and arg2 == self.padding_idx:
                canonical_atoms.append(f"{pred_str}()")
            else:
                canonical_atoms.append(f"{pred_str}({format_arg(arg1, arg1_is_var)},{format_arg(arg2, arg2_is_var)})")
        
        return '|'.join(canonical_atoms)

    def canonical_states_to_str(
        self,
        states: Tensor,
        counts: Optional[Tensor] = None,
        return_indices: bool = False
    ) -> Union[List[str], Tuple[List[str], List[int]]]:
        """
        Canonicalize multiple states to string representations.
        
        Args:
            states: [B, M, 3] or [B, K, M, 3] tensor of states
            counts: Optional [B] tensor of valid atom counts
            return_indices: If True, return (sorted_strings, original_indices)
        """
        results = []
        if states.dim() == 3:
            results = [self.canonical_state_to_str(states[i:i+1]) for i in range(states.shape[0])]
        elif states.dim() == 4:
            for i in range(states.shape[0]):
                count = counts[i].item() if counts is not None else states.shape[1]
                for j in range(count):
                    results.append(self.canonical_state_to_str(states[i, j:j+1]))
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {states.dim()}D")
        
        results_with_idx = [(s, i) for i, s in enumerate(results)]
        results_with_idx.sort(key=lambda x: x[0])
        
        if return_indices:
            return [x[0] for x in results_with_idx], [x[1] for x in results_with_idx]
        return [x[0] for x in results_with_idx]

    # =====================================================================
    # State Inspection / Dumping
    # =====================================================================
    
    def dump_states(
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
        if not self.is_enabled('env', level):
            return
        
        print("\n")
        if rows is None:
            if self.batch_size_int is not None:
                rows = torch.arange(self.batch_size_int, device=self._device)
            else:
                rows = torch.arange(current_queries.shape[0], device=current_queries.device)
        
        rows_list = rows.tolist()
        self.log_env(f"{label}: inspecting rows {rows_list}", level)
        
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
            
            self.log_env(f"  Idx {idx} {depth_info}query={state_atoms}", level)

            if action_mask is not None:
                mask_for_idx = action_mask[idx].tolist()
                valid_indices = [i for i, m in enumerate(mask_for_idx) if m]
                preview = valid_indices[:min(len(valid_indices), 10)]
                self.log_env(
                    f"    Action mask: valid actions: {len(valid_indices)}"
                    f"{' -> ' + str(preview) if preview else ''}",
                    level
                )

            if done is not None:
                self.log_env(f"    Done: {done[idx].item()}", level)
            if rewards is not None:
                self.log_env(f"    Reward: {rewards[idx].item()}", level)

            derived_atoms = []
            for d in range(derived_count):
                derived_atoms.append(self._format_atoms(derived_states_batch[idx, d]))
            self.log_env(f"    Derived: {derived_atoms}\n", level)

    def print_states(
        self,
        title: str,
        states_tensor: torch.Tensor,
        counts: Optional[torch.Tensor] = None,
        padding_idx: Optional[int] = None,
        level: int = 1
    ) -> None:
        """Print states in human-readable format."""
        if not self.is_enabled('env', level):
            return
        
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

    # =====================================================================
    # Sorting for Comparison
    # =====================================================================
    
    def sort_candidates_by_canonical(
        self,
        states: Tensor,
        counts: Tensor,
        owners: Tensor,
        next_vars: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sort candidates by canonical string representation for comparison."""
        if states.numel() == 0:
            return states, counts, owners, next_vars

        canonical_strings = [self.canonical_state_to_str(states[i]) for i in range(states.shape[0])]
        
        order = sorted(
            range(len(canonical_strings)),
            key=lambda idx: (int(owners[idx].item()), canonical_strings[idx])
        )
        
        if order == list(range(len(canonical_strings))):
            return states, counts, owners, next_vars

        order_tensor = torch.tensor(order, dtype=torch.long, device=states.device)
        return (
            states.index_select(0, order_tensor),
            counts.index_select(0, order_tensor),
            owners.index_select(0, order_tensor),
            next_vars.index_select(0, order_tensor),
        )

    # =====================================================================
    # Comparison Utilities
    # =====================================================================
    
    def compare_tensors(
        self,
        name: str,
        tensor1: Tensor,
        tensor2: Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> bool:
        """Compare two tensors and report differences."""
        if tensor1.shape != tensor2.shape:
            self.log_env(f"MISMATCH {name}: shapes differ {tensor1.shape} vs {tensor2.shape}")
            return False
        
        if tensor1.dtype.is_floating_point:
            close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
            if not close:
                max_diff = (tensor1 - tensor2).abs().max().item()
                self.log_env(f"MISMATCH {name}: max diff = {max_diff}")
            return close
        else:
            equal = torch.equal(tensor1, tensor2)
            if not equal:
                diff_mask = tensor1 != tensor2
                num_diff = diff_mask.sum().item()
                self.log_env(f"MISMATCH {name}: {num_diff} elements differ")
            return equal

    def compare_states(
        self,
        name: str,
        states1: Tensor,
        states2: Tensor,
        use_canonical: bool = True,
    ) -> bool:
        """Compare two state tensors, optionally using canonical form."""
        if use_canonical:
            canonical1 = self.canonical_states_to_str(states1)
            canonical2 = self.canonical_states_to_str(states2)
            if canonical1 != canonical2:
                self.log_env(f"MISMATCH {name}: canonical strings differ")
                for i, (c1, c2) in enumerate(zip(canonical1, canonical2)):
                    if c1 != c2:
                        self.log_env(f"  [{i}]: '{c1}' vs '{c2}'")
                return False
            return True
        return self.compare_tensors(name, states1, states2)


# Backward compatibility aliases
DebugHelper = Debug
