"""
Refactored IndexManager - Clean, maintainable, and efficient.

Single source of truth for all index spaces and conversions.
- Constants: [1 .. n]
- Template variables (from rules only): [n+1 .. n+Vt]
- Runtime variables (reserved pool): [n+Vt+1 .. n+Vt+Vr]
- Padding: 0

Key improvements over original:
- Cleaner separation of concerns
- Simpler vocabulary management
- No redundant mappings
- Better performance with unified term maps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import torch


# -----------------------------
# Type aliases (indices only)
# -----------------------------
# Atom layout: [predicate_id, arg1_id, arg2_id]
Tensor = torch.Tensor
LongTensor = torch.LongTensor


@dataclass(frozen=True)
class VocabSizes:
    """Container for vocabulary sizes."""
    num_constants: int
    num_predicates: int
    num_template_vars: int
    num_runtime_vars: int


class IndexManager:
    """
    Single source of truth for all index spaces and conversions.
    
    Index spaces:
    - Constants: [1 .. n]
    - Template variables (from rules only): [n+1 .. n+Vt]
    - Runtime variables (reserved pool): [n+Vt+1 .. n+Vt+Vr]
    - Padding: 0
    
    Strings are *only* used via debug helpers.
    """

    # -----------------------------
    # Construction
    # -----------------------------
    def __init__(
        self,
        constants: Iterable[str],
        predicates: Iterable[str],
        max_total_runtime_vars: int = 4096,
        max_arity: int = 2,
        padding_atoms: int = 10,
        device: Optional[torch.device] = None,
        rules: Optional[List] = None,  # For backward compatibility
        include_kge_predicates: bool = False,
    ) -> None:
        """
        Initialize IndexManager with vocabulary.
        
        Args:
            constants: Iterable of constant strings
            predicates: Iterable of predicate strings
            max_total_runtime_vars: Maximum runtime variables (reserved pool)
            max_arity: Maximum arity of predicates
            padding_atoms: Maximum atoms per state (for padding)
            device: Target device for tensors
            rules: Optional list of Rule objects (for backward compatibility)
        """
        # Freeze sets for reproducible ordering
        const_list = sorted(set(constants))
        pred_list = sorted(set(predicates))
        
        # IMPORTANT: Match str_index_manager.py behavior - add special predicates AFTER regular ones
        # This ensures canonical ordering keys match between str and batched environments
        # First assign indices to regular predicates
        regular_pred_list = pred_list
        special_pred_list = []
        
        # Add special predicates at the end (not in alphabetical order with regular predicates)
        if 'True' not in regular_pred_list:
            special_pred_list.append('True')
        if 'False' not in regular_pred_list:
            special_pred_list.append('False')
        # Align naming with SB3 (Endf)
        if 'Endf' not in regular_pred_list:
            special_pred_list.append('Endf')
            
        # Add KGE predicates to match SB3 behavior
        if include_kge_predicates:
            kge_preds = sorted([f"{p}_kge" for p in regular_pred_list])
            for p in kge_preds:
                if p not in regular_pred_list:
                    special_pred_list.append(p)
        
        # Combine: regular predicates first (sorted), then special predicates
        pred_list = regular_pred_list + special_pred_list

        # String <-> index maps (CPU int32 to save RAM)
        self.constant_str2idx: Dict[str, int] = {s: i + 1 for i, s in enumerate(const_list)}
        self.idx2constant: List[str] = ["<PAD>"] + const_list

        self.predicate_str2idx: Dict[str, int] = {s: i + 1 for i, s in enumerate(pred_list)}
        self.idx2predicate: List[str] = ["<PAD>"] + pred_list

        # Template vars appear only in rules; we'll allocate lazily when rules are materialized
        self.template_var_str2idx: Dict[str, int] = {}
        self.idx2template_var: List[str] = ["<PAD>"]

        # Unified term map used in one-shot conversions (strings -> indices)
        self.unified_term_map: Dict[str, int] = dict(self.constant_str2idx)  # start with constants

        # Sizes
        self.constant_no: int = len(self.constant_str2idx)
        self.predicate_no: int = len(self.predicate_str2idx)
        self.template_variable_no: int = 0
        self.runtime_variable_no: int = max_total_runtime_vars
        self.max_total_vars: int = max_total_runtime_vars  # Alias for compatibility
        self.max_arity: int = max_arity  # Store max_arity

        self.padding_idx: int = 0
        self.padding_atoms: int = padding_atoms
        
        # Special predicate indices
        self.true_pred_idx: Optional[int] = self.predicate_str2idx.get('True')
        self.false_pred_idx: Optional[int] = self.predicate_str2idx.get('False')
        self.end_pred_idx: Optional[int] = self.predicate_str2idx.get('Endf')

        # Runtime var range [start, end]
        # SB3 behavior: variable_start_index = constant_no + 1 (never changes)
        # We match this to ensure parity
        self.runtime_var_start_index: int = self.constant_no + 1
        self.runtime_var_end_index: int = self.runtime_var_start_index + self.runtime_variable_no - 1

        # Total vocabulary size
        # IMPORTANT: Match SB3 behavior exactly - variable_no = max_total_vars
        # SB3 uses: self.variable_no = self.max_total_vars
        # NOT: self.variable_no = constant_no + template_variable_no + runtime_variable_no
        self.variable_no: int = max_total_runtime_vars  # Match SB3 exactly
        self.total_vocab_size: int = self.constant_no + self.variable_no + 1  # constants + vars + padding
        self.pack_base = self.total_vocab_size + 1    # safe 64-bit packing base

        # Tensors for facts and rules
        # Shapes: facts_idx (F, 3), rules_idx (R, M, 3), rule_lens (R,)
        self.facts_idx: Optional[LongTensor] = None
        self.rules_idx: Optional[LongTensor] = None
        self.rule_lens: Optional[LongTensor] = None
        self.rules_heads_idx: Optional[LongTensor] = None

        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.idx_dtype = torch.long  # For compatibility

        # For backward compatibility with old code
        self.constants = set(const_list)
        self.predicates = set(pred_list)
        

        if rules is not None:
            self.rules = rules
            # Pre-index rules by predicate for unification
            self.rules_by_pred = {}
            for r in rules:
                self.rules_by_pred.setdefault(r.head.predicate, []).append(r)
        else:
            self.rules = []
            self.rules_by_pred = {}

        # Build special predicate tensors
        if self.true_pred_idx is not None:
            self.true_tensor = torch.tensor(
                [[self.true_pred_idx, self.padding_idx, self.padding_idx]],
                dtype=torch.long,
                device=self.device
            )
        if self.false_pred_idx is not None:
            self.false_tensor = torch.tensor(
                [[self.false_pred_idx, self.padding_idx, self.padding_idx]],
                dtype=torch.long,
                device=self.device
            )

    # -----------------------------
    # Vocabulary growth for rule variables
    # -----------------------------
    def _ensure_template_var(self, var_name: str) -> int:
        """Ensure template variable exists and return its index.
        
        Note: Template variables are assigned indices in the range 
        [constant_no + 1, constant_no + template_variable_no]. However,
        we do NOT shift the runtime_var_start_index when adding template
        variables - this matches SB3 behavior where variable_start_index
        is always constant_no + 1.
        """
        idx = self.template_var_str2idx.get(var_name)
        if idx is not None:
            return idx
        # allocate next template var
        idx = self.constant_no + self.template_variable_no + 1
        self.template_variable_no += 1
        self.template_var_str2idx[var_name] = idx
        self.idx2template_var.append(var_name)
        # NOTE: Do NOT update runtime_var_start_index here - SB3 keeps it fixed
        # update unified map for one-shot conversions
        self.unified_term_map[var_name] = idx
        # NOTE: Do NOT update variable_no here - SB3 keeps variable_no = max_total_vars
        # Template variables are indexed within the constant_no + 1 to constant_no + template_variable_no range
        # but this doesn't change variable_no (which equals max_total_vars in SB3)
        self.total_vocab_size = self.constant_no + self.variable_no + 1
        return idx

    # -----------------------------
    # Materializers (strings -> indices)
    # -----------------------------
    def term_to_index(self, token: str) -> int:
        """Return index for a constant or a template variable (rules only)."""
        if token in self.unified_term_map:
            return self.unified_term_map[token]
        # treat as template variable if unseen (likely from rules)
        return self._ensure_template_var(token)

    def atom_to_tensor(self, pred_str: str, a_str: str, b_str: str) -> LongTensor:
        """Convert single atom to [3] """
        p = self.predicate_str2idx[pred_str]
        a = self.term_to_index(a_str)
        b = self.term_to_index(b_str)
        return torch.tensor([p, a, b], dtype=torch.long)
    def state_to_tensor(self, atoms: Iterable[Tuple[str, str, str]]) -> LongTensor:
        """Convert an iterable of string atoms into [k, 3] indices with a single allocation."""
        atoms = list(atoms)
        n = len(atoms)
        if n == 0:
            return torch.empty((0, 3), dtype=torch.long)
        p = [self.predicate_str2idx[p] for (p, _, _) in atoms]
        a = [self.term_to_index(x) for (_, x, _) in atoms]
        b = [self.term_to_index(y) for (_, _, y) in atoms]
        out = torch.empty((n, 3), dtype=torch.long)
        out[:, 0] = torch.as_tensor(p, dtype=torch.long)
        out[:, 1] = torch.as_tensor(a, dtype=torch.long)
        out[:, 2] = torch.as_tensor(b, dtype=torch.long)
        return out

    def rules_to_tensor(
        self,
        rules: Iterable[Tuple[Tuple[str, str, str], List[Tuple[str, str, str]]]],
        max_rule_atoms: int,
    ) -> Tuple[LongTensor, LongTensor]:
        """
        Convert rules to padded [R, M, 3] with lens [R].
        Each rule: ((pH, X1, X2), [(p1,A1,B1), (p2,A2,B2), ...])
        All terms are variables by invariant, but we still route via term_to_index.
        """
        heads = []
        bodies = []
        lens = []

        for (hp, hx, hy), body in rules:
            h = self.atom_to_tensor(hp, hx, hy)                # [3]
            b = self.state_to_tensor(body)                     # [k,3]
            k = b.shape[0]
            if k > max_rule_atoms:
                raise ValueError(f"Rule body length {k} > max_rule_atoms={max_rule_atoms}")
            pad = torch.zeros((max_rule_atoms - k, 3), dtype=torch.long)
            rb = torch.cat([b, pad], dim=0) if k > 0 else pad
            heads.append(h)
            bodies.append(rb)
            lens.append(k)

        if not heads:
            raise ValueError("No rules provided for rules_to_tensor")

        rules_idx = torch.stack(bodies, dim=0)
        rule_lens = torch.tensor(lens, dtype=torch.long)       # [R]
        # also keep heads at index 0 of body for convenience if desired
        self.rules_heads_idx = torch.stack(heads, dim=0)       # [R, 3]
        return rules_idx, rule_lens

    # -----------------------------
    # Facts index & storage
    # -----------------------------
    def set_facts(self, facts_idx: LongTensor) -> None:
        """Register facts (index tensor) and build predicate index (CPU)."""
        if facts_idx.numel() == 0:
            self.facts_idx = facts_idx.to(self.device, dtype=torch.long)
            self.predicate_range_map = torch.zeros((self.predicate_no + 1, 2), dtype=torch.int32)
            return

        # Sort by (predicate, head, tail) lexicographically for deterministic ordering
        facts_cpu = facts_idx.detach().to("cpu").to(dtype=torch.long)
        # Create a composite key for lexicographic sorting: pred*V^2 + head*V + tail
        # where V is large enough to avoid collisions (total_vocab_size)
        V = self.total_vocab_size
        sort_keys = facts_cpu[:, 0] * (V ** 2) + facts_cpu[:, 1] * V + facts_cpu[:, 2]
        order = torch.argsort(sort_keys, stable=True)
        facts_sorted = facts_cpu[order]

        # Build ranges
        pr = torch.zeros((self.predicate_no + 1, 2), dtype=torch.int32)  # include padding row 0
        pcol = facts_sorted[:, 0]
        # find boundaries
        # unique with counts
        uniq, counts = torch.unique_consecutive(pcol, return_counts=True)
        starts = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.int64), counts[:-1]]), dim=0)
        ends = torch.cumsum(counts, dim=0)
        for u, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
            pr[u, 0] = int(s)
            pr[u, 1] = int(e)

        self.facts_idx = facts_sorted.to(self.device, dtype=torch.long)
        self.predicate_range_map = pr.to(self.device)

    def set_rules(self, rules_idx: LongTensor, rule_lens: LongTensor) -> None:
        self.rules_idx = rules_idx.to(self.device, dtype=torch.long, non_blocking=True)
        self.rule_lens = rule_lens.to(self.device, dtype=torch.long, non_blocking=True)
        # derive heads for fast predicate filtering
        R, M, _ = self.rules_idx.shape if self.rules_idx.numel() > 0 else (0,0,0)
        if R > 0:
            # first return from rules_to_tensor already saved heads in self.rules_heads_idx;
            # if not, compute here from the original rules (store it alongside rules_idx)
            assert hasattr(self, 'rules_heads_idx') and self.rules_heads_idx is not None
            self.rules_heads_idx = self.rules_heads_idx.to(self.device)
        else:
            self.rules_heads_idx = torch.empty((0,3), dtype=torch.long, device=self.device)

    def adjust_runtime_start_for_head_vars(self, head_var_count: int) -> None:
        """
        Move the runtime variable window so that head variables do not offset the
        starting index. SB3 effectively begins fresh variables after the *body-only*
        variables, whereas this manager initially offsets by all template vars.
        """
        # SB3 starts fresh variables immediately after constants, regardless of
        # how many template variables appear in the rules. Mirroring that behavior
        # prevents body/head variables from shifting the runtime window.
        old_start = self.runtime_var_start_index
        new_start = self.constant_no + 1
        if new_start != self.runtime_var_start_index:
            self.runtime_var_start_index = new_start
            self.runtime_var_end_index = self.runtime_var_start_index + self.runtime_variable_no - 1
            print(f"[DEBUG] Adjusted runtime_var_start: {old_start} -> {new_start} (template_vars={self.template_variable_no})")

    def get_stringifier_params(self):
        """Return the parameters needed for atom stringification."""
        return {
            'n_constants': self.constant_no,
            'idx2constant': tuple(self.idx2constant),
            'idx2predicate': tuple(self.idx2predicate),
            'idx2template_var': tuple(self.idx2template_var),
        }
