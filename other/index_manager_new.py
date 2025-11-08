
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
    num_constants: int
    num_predicates: int
    num_template_vars: int
    num_runtime_vars: int


class IndexManager:
    """
    Single source of truth for all index spaces and conversions.
    - Constants: [1 .. n]
    - Template variables (from rules only): [n+1 .. n+Vt]
    - Runtime variables (reserved pool):    [n+Vt+1 .. n+Vt+Vr]
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
        device: Optional[torch.device] = None,
    ) -> None:
        # Freeze sets for reproducible ordering
        const_list = sorted(set(constants))
        pred_list = sorted(set(predicates))

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

        self.padding_idx: int = 0
        self.true_pred_idx: Optional[int] = None
        self.false_pred_idx: Optional[int] = None

        # Runtime var range [start, end]
        self.runtime_var_start_index: int = self.constant_no + self.template_variable_no + 1
        self.runtime_var_end_index: int = self.runtime_var_start_index + self.runtime_variable_no - 1

        # Tensors (filled later by materializers)
        self.facts_idx: Optional[LongTensor] = None            # [F, 3]
        self.rules_idx: Optional[LongTensor] = None            # [R, M, 3]
        self.rule_lens: Optional[LongTensor] = None            # [R]

        # Fact index (CPU) for quick predicate slices
        self.predicate_range_map: Optional[torch.IntTensor] = None  # [num_predicates, 2]

        self.device: torch.device = device if device is not None else torch.device("cpu")


    # -----------------------------
    # Vocabulary growth for rule variables
    # -----------------------------
    def _ensure_template_var(self, var_name: str) -> int:
        idx = self.template_var_str2idx.get(var_name)
        if idx is not None:
            return idx
        # allocate next template var
        idx = self.constant_no + self.template_variable_no + 1
        self.template_variable_no += 1
        self.template_var_str2idx[var_name] = idx
        self.idx2template_var.append(var_name)
        # update runtime var window
        self.runtime_var_start_index = self.constant_no + self.template_variable_no + 1
        self.runtime_var_end_index = self.runtime_var_start_index + self.runtime_variable_no - 1
        # update unified map for one-shot conversions
        self.unified_term_map[var_name] = idx
        return idx


    # -----------------------------
    # Materializers (strings -> indices)
    # -----------------------------
    def term_to_index(self, token: str) -> int:
        """Return index for a constant or a template variable (rules only)."""
        if token in self.unified_term_map:
            return self.unified_term_map[token]
        # treat as template variable if unseen
        return self._ensure_template_var(token)

    def atom_to_tensor(self, pred_str: str, a_str: str, b_str: str) -> LongTensor:
        p = self.predicate_str2idx[pred_str]
        a = self.term_to_index(a_str)
        b = self.term_to_index(b_str)
        return torch.tensor([p, a, b], dtype=torch.long)

    def state_to_tensor(self, atoms: Iterable[Tuple[str, str, str]]) -> LongTensor:
        """Convert an iterable of string atoms into [k, 3] indices."""
        rows = [self.atom_to_tensor(p, a, b) for (p, a, b) in atoms]
        if not rows:
            return torch.empty((0, 3), dtype=torch.long)
        return torch.stack(rows, dim=0)

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
            rules_idx = torch.empty((0, max_rule_atoms, 3), dtype=torch.long)
            rule_lens = torch.empty((0,), dtype=torch.long)
            return rules_idx, rule_lens

        rules_idx = torch.stack(bodies, dim=0)                 # [R, M, 3]
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
            self.facts_idx = facts_idx.to(self.device)
            self.predicate_range_map = torch.zeros((self.predicate_no + 1, 2), dtype=torch.int32)
            return

        # Sort by predicate for fast slicing
        facts_cpu = facts_idx.detach().to("cpu").to(dtype=torch.long)
        order = torch.argsort(facts_cpu[:, 0], stable=True)
        facts_sorted = facts_cpu.index_select(0, order)

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

        self.facts_idx = facts_sorted.to(self.device)
        self.predicate_range_map = pr  # keep on CPU

    def build_true_false_preds(self, true_pred: Optional[str], false_pred: Optional[str]) -> None:
        """Optionally set special predicate indices for TRUE/FALSE atoms (debug/terminals)."""
        if true_pred is not None:
            self.true_pred_idx = self.predicate_str2idx.get(true_pred)
        if false_pred is not None:
            self.false_pred_idx = self.predicate_str2idx.get(false_pred)


    # -----------------------------
    # Debug helpers (strings only)
    # -----------------------------
    def debug_atom_to_str(self, atom_idx: LongTensor) -> str:
        p, a, b = [int(x) for x in atom_idx.tolist()]
        ps = self.idx2predicate[p] if 0 <= p < len(self.idx2predicate) else str(p)
        def term_str(t: int) -> str:
            if 1 <= t <= self.constant_no:
                return self.idx2constant[t]
            # template var space
            if self.constant_no < t <= self.constant_no + self.template_variable_no:
                tv = t - self.constant_no
                return self.idx2template_var[tv] if 0 <= tv < len(self.idx2template_var) else f"v{t}"
            # runtime range
            return f"_{t}"
        return f"{ps}({term_str(a)},{term_str(b)})"
