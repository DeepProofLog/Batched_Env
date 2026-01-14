"""
Fast integer-based unification engine for depth generation.

Key optimizations over SB3:
- Integer indices instead of strings (no isupper() calls)
- Variable detection via range check: idx > constant_count
- 64-bit packed hashes for O(1) fact membership
- Pre-allocated substitution lists instead of dicts
- Tuple-based terms for fast hashing

Usage:
    from fast_unifier import FastUnifier
    unifier = FastUnifier(data_handler)
    depth = unifier.check_provability(query_str, max_depth=7)
"""
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sb3.sb3_dataset import DataHandler
from sb3.sb3_utils import Term, Rule

# Type aliases for clarity
ITerm = Tuple[int, int, int]  # (pred_idx, arg0_idx, arg1_idx)
IState = Tuple[ITerm, ...]    # Tuple of ITerms
Subs = List[Tuple[int, int]]  # List of (from_var, to_val) substitutions


class FastUnifier:
    """Fast integer-based unification engine."""

    def __init__(self, data_handler: DataHandler):
        """Initialize from DataHandler.

        Args:
            data_handler: Loaded dataset with facts, rules, constants, predicates
        """
        # Build index mappings
        self._build_indices(data_handler)

        # Convert facts to integer representation
        self._build_fact_index(data_handler.facts)

        # Convert rules to integer representation
        self._build_rule_index(data_handler.rules)

    def _build_indices(self, data_handler: DataHandler):
        """Build string-to-index mappings."""
        # Constants: 1..C (0 is padding)
        constants = sorted(data_handler.constants)
        self.const_str2idx: Dict[str, int] = {s: i + 1 for i, s in enumerate(constants)}
        self.const_idx2str: Dict[int, str] = {i + 1: s for i, s in enumerate(constants)}
        self.constant_count = len(constants)

        # Predicates: 1..P
        predicates = sorted(data_handler.predicates)
        # Add True/False if not present
        if 'True' not in predicates:
            predicates.append('True')
        if 'False' not in predicates:
            predicates.append('False')
        predicates = sorted(set(predicates))

        self.pred_str2idx: Dict[str, int] = {s: i + 1 for i, s in enumerate(predicates)}
        self.pred_idx2str: Dict[int, str] = {i + 1: s for i, s in enumerate(predicates)}
        self.true_idx = self.pred_str2idx['True']
        self.false_idx = self.pred_str2idx['False']

        # Template variables from rules: C+1..C+T
        template_vars: Set[str] = set()
        for rule in data_handler.rules:
            for arg in rule.head.args:
                if self._is_var_str(arg):
                    template_vars.add(arg)
            for body_term in rule.body:
                for arg in body_term.args:
                    if self._is_var_str(arg):
                        template_vars.add(arg)

        template_vars_sorted = sorted(template_vars)
        self.template_str2idx: Dict[str, int] = {
            s: self.constant_count + i + 1
            for i, s in enumerate(template_vars_sorted)
        }
        self.template_count = len(template_vars_sorted)

        # Runtime variables start after template vars
        self.runtime_var_start = self.constant_count + self.template_count + 1

        # Combined mapping for any string -> index
        self.str2idx: Dict[str, int] = {}
        self.str2idx.update(self.const_str2idx)
        self.str2idx.update(self.template_str2idx)

    def _is_var_str(self, s: str) -> bool:
        """Check if string is a variable (starts with uppercase or _)."""
        return s[0].isupper() or s[0] == '_'

    def _is_var_idx(self, idx: int) -> bool:
        """Check if index is a variable (faster than string check)."""
        return idx > self.constant_count

    def _term_to_iterm(self, term: Term) -> ITerm:
        """Convert string Term to integer ITerm."""
        pred_idx = self.pred_str2idx[term.predicate]
        arg0_idx = self.str2idx.get(term.args[0], 0)
        arg1_idx = self.str2idx.get(term.args[1], 0) if len(term.args) > 1 else 0
        return (pred_idx, arg0_idx, arg1_idx)

    def _iterm_to_str(self, iterm: ITerm) -> str:
        """Convert ITerm back to string (for debugging/output)."""
        pred = self.pred_idx2str[iterm[0]]
        arg0 = self.const_idx2str.get(iterm[1]) or f"Var_{iterm[1]}"
        arg1 = self.const_idx2str.get(iterm[2]) or f"Var_{iterm[2]}"
        return f"{pred}({arg0},{arg1})"

    def _build_fact_index(self, facts: List[Term]):
        """Build fact index with 64-bit hashing and multi-level indexing."""
        # Convert facts to ITerms
        self.facts: List[ITerm] = [self._term_to_iterm(f) for f in facts]

        # 64-bit packed hashes for O(1) ground fact lookup
        # Hash = pred * base^2 + arg0 * base + arg1
        self.hash_base = self.constant_count + self.template_count + 10000
        self.fact_hashes: FrozenSet[int] = frozenset(
            self._pack_iterm(f) for f in self.facts
        )

        # Index by predicate for candidate retrieval
        self.facts_by_pred: Dict[int, List[ITerm]] = {}
        # Index by (predicate, arg0) for queries with ground arg0
        self.facts_by_pred_arg0: Dict[Tuple[int, int], List[ITerm]] = {}
        # Index by (predicate, arg1) for queries with ground arg1
        self.facts_by_pred_arg1: Dict[Tuple[int, int], List[ITerm]] = {}

        for f in self.facts:
            pred_idx, arg0, arg1 = f

            # By predicate
            if pred_idx not in self.facts_by_pred:
                self.facts_by_pred[pred_idx] = []
            self.facts_by_pred[pred_idx].append(f)

            # By (pred, arg0)
            key0 = (pred_idx, arg0)
            if key0 not in self.facts_by_pred_arg0:
                self.facts_by_pred_arg0[key0] = []
            self.facts_by_pred_arg0[key0].append(f)

            # By (pred, arg1)
            key1 = (pred_idx, arg1)
            if key1 not in self.facts_by_pred_arg1:
                self.facts_by_pred_arg1[key1] = []
            self.facts_by_pred_arg1[key1].append(f)

    def _pack_iterm(self, iterm: ITerm) -> int:
        """Pack ITerm into 64-bit integer for hashing."""
        return (iterm[0] * self.hash_base + iterm[1]) * self.hash_base + iterm[2]

    def _build_rule_index(self, rules: List[Rule]):
        """Build rule index by head predicate."""
        self.rules_by_pred: Dict[int, List[Tuple[ITerm, Tuple[ITerm, ...]]]] = {}

        for rule in rules:
            head = self._term_to_iterm(rule.head)
            body = tuple(self._term_to_iterm(t) for t in rule.body)
            pred_idx = head[0]

            if pred_idx not in self.rules_by_pred:
                self.rules_by_pred[pred_idx] = []
            self.rules_by_pred[pred_idx].append((head, body))

    def _unify(self, query: ITerm, term: ITerm) -> Optional[Subs]:
        """Unify query with term, return substitutions or None.

        Fast integer-based unification:
        - Variable check via idx > constant_count
        - Integer equality instead of string comparison
        - Returns list of (from, to) pairs instead of dict
        """
        # Predicates must match
        if query[0] != term[0]:
            return None

        subs: Subs = []
        const_count = self.constant_count

        # Unify arg0
        q0, t0 = query[1], term[1]
        q0_var = q0 > const_count
        t0_var = t0 > const_count

        if not q0_var and not t0_var:
            # Both constants - must match
            if q0 != t0:
                return None
        elif q0_var and not t0_var:
            # Query var, term constant - bind query var
            subs.append((q0, t0))
        elif not q0_var and t0_var:
            # Query constant, term var - bind term var
            subs.append((t0, q0))
        else:
            # Both vars - bind term var to query var
            if q0 != t0:
                subs.append((t0, q0))

        # Unify arg1
        q1, t1 = query[2], term[2]
        q1_var = q1 > const_count
        t1_var = t1 > const_count

        if not q1_var and not t1_var:
            if q1 != t1:
                return None
        elif q1_var and not t1_var:
            subs.append((q1, t1))
        elif not q1_var and t1_var:
            subs.append((t1, q1))
        else:
            if q1 != t1:
                subs.append((t1, q1))

        return subs

    def _apply_subs(self, iterm: ITerm, subs: Subs) -> ITerm:
        """Apply substitutions to an ITerm."""
        if not subs:
            return iterm

        arg0, arg1 = iterm[1], iterm[2]
        new0, new1 = arg0, arg1

        # Apply substitutions - unrolled for common case of 1-2 subs
        if len(subs) == 1:
            frm, to = subs[0]
            if arg0 == frm: new0 = to
            if arg1 == frm: new1 = to
        elif len(subs) == 2:
            f0, t0 = subs[0]
            f1, t1 = subs[1]
            if arg0 == f0: new0 = t0
            elif arg0 == f1: new0 = t1
            if arg1 == f0: new1 = t0
            elif arg1 == f1: new1 = t1
        else:
            for frm, to in subs:
                if new0 == frm: new0 = to
                if new1 == frm: new1 = to

        # Only create new tuple if changed
        if new0 == arg0 and new1 == arg1:
            return iterm
        return (iterm[0], new0, new1)

    def _unify_with_facts(
        self,
        query: ITerm,
        excluded_hash: Optional[int]
    ) -> List[Subs]:
        """Find all facts that unify with query, return substitutions."""
        results: List[Subs] = []
        const_count = self.constant_count
        hash_base = self.hash_base
        pred_idx = query[0]
        q0, q1 = query[1], query[2]

        # Check if query args are variables
        q0_var = q0 > const_count
        q1_var = q1 > const_count

        if not q0_var and not q1_var:
            # Ground query - O(1) hash lookup (inline hash)
            query_hash = (pred_idx * hash_base + q0) * hash_base + q1
            if excluded_hash is not None and query_hash == excluded_hash:
                return []
            if query_hash in self.fact_hashes:
                return [[]]  # Empty substitution = direct match
            return []

        # Non-ground query - use most specific index available
        results_append = results.append
        if not q0_var:
            # arg0 is ground - use (pred, arg0) index
            candidates = self.facts_by_pred_arg0.get((pred_idx, q0), [])
            if excluded_hash is not None:
                for fact in candidates:
                    if (pred_idx * hash_base + fact[1]) * hash_base + fact[2] == excluded_hash:
                        continue
                    results_append([(q1, fact[2])])
            else:
                for fact in candidates:
                    results_append([(q1, fact[2])])
        elif not q1_var:
            # arg1 is ground - use (pred, arg1) index
            candidates = self.facts_by_pred_arg1.get((pred_idx, q1), [])
            if excluded_hash is not None:
                for fact in candidates:
                    if (pred_idx * hash_base + fact[1]) * hash_base + fact[2] == excluded_hash:
                        continue
                    results_append([(q0, fact[1])])
            else:
                for fact in candidates:
                    results_append([(q0, fact[1])])
        else:
            # Both args are variables - use predicate index
            candidates = self.facts_by_pred.get(pred_idx, [])
            if excluded_hash is not None:
                for fact in candidates:
                    if (pred_idx * hash_base + fact[1]) * hash_base + fact[2] == excluded_hash:
                        continue
                    f1, f2 = fact[1], fact[2]
                    subs = []
                    if q0 != f1: subs.append((q0, f1))
                    if q1 != f2: subs.append((q1, f2))
                    results_append(subs)
            else:
                for fact in candidates:
                    f1, f2 = fact[1], fact[2]
                    subs = []
                    if q0 != f1: subs.append((q0, f1))
                    if q1 != f2: subs.append((q1, f2))
                    results_append(subs)

        return results

    def _unify_with_rules(self, query: ITerm) -> List[Tuple[Subs, Tuple[ITerm, ...]]]:
        """Find all rules whose heads unify with query."""
        results: List[Tuple[Subs, Tuple[ITerm, ...]]] = []

        rules = self.rules_by_pred.get(query[0], [])
        for head, body in rules:
            subs = self._unify(query, head)
            if subs is not None:
                results.append((subs, body))

        return results

    def _rename_vars(
        self,
        atoms: Tuple[ITerm, ...],
        next_var: int
    ) -> Tuple[Tuple[ITerm, ...], int]:
        """Rename TEMPLATE variables to fresh runtime variables.

        Only renames variables in the template range (constant_count < idx < runtime_var_start).
        Runtime variables (idx >= runtime_var_start) are preserved as-is.
        """
        const_count = self.constant_count
        runtime_start = self.runtime_var_start

        # Check if any TEMPLATE variable renaming needed
        needs_rename = False
        for atom in atoms:
            # Template variable: const_count < idx < runtime_start
            if const_count < atom[1] < runtime_start or const_count < atom[2] < runtime_start:
                needs_rename = True
                break

        if not needs_rename:
            return atoms, next_var

        # Build mapping for template variables only
        mapping: Dict[int, int] = {}
        renamed: List[ITerm] = []

        for atom in atoms:
            pred, arg0, arg1 = atom

            # Rename arg0 if TEMPLATE variable
            if const_count < arg0 < runtime_start:
                if arg0 not in mapping:
                    mapping[arg0] = next_var
                    next_var += 1
                arg0 = mapping[arg0]

            # Rename arg1 if TEMPLATE variable
            if const_count < arg1 < runtime_start:
                if arg1 not in mapping:
                    mapping[arg1] = next_var
                    next_var += 1
                arg1 = mapping[arg1]

            renamed.append((pred, arg0, arg1))

        return tuple(renamed), next_var

    def _canonicalize_state(self, state: IState, simple: bool = False) -> FrozenSet[ITerm]:
        """Canonicalize state for deduplication.

        Args:
            state: State to canonicalize
            simple: If True, just use frozenset without variable renaming.
                    This is valid in SB3-compat mode where variables are reused.

        Returns frozenset of atoms with canonicalized variable names.
        Variables are renamed consistently: first var seen = C+1, second = C+2, etc.
        Combined with frozenset, this ensures equivalent states hash the same.
        """
        if simple:
            # Fast path for SB3-compat mode
            return frozenset(state)

        const_count = self.constant_count
        mapping: Dict[int, int] = {}
        next_canonical = const_count + 1
        canonicalized: List[ITerm] = []

        # Sort atoms first for consistent variable ordering
        sorted_state = sorted(state)

        for atom in sorted_state:
            pred, arg0, arg1 = atom

            # Canonicalize arg0 if variable
            if arg0 > const_count:
                if arg0 not in mapping:
                    mapping[arg0] = next_canonical
                    next_canonical += 1
                arg0 = mapping[arg0]

            # Canonicalize arg1 if variable
            if arg1 > const_count:
                if arg1 not in mapping:
                    mapping[arg1] = next_canonical
                    next_canonical += 1
                arg1 = mapping[arg1]

            canonicalized.append((pred, arg0, arg1))

        return frozenset(canonicalized)

    def _get_derived_states(
        self,
        state: IState,
        excluded_hash: Optional[int],
        next_var: int,
    ) -> List[Tuple[IState, int, bool]]:
        """Generate all successor states from current state.

        Returns list of (state, next_var, is_proof) tuples.
        is_proof=True indicates immediate proof (no more processing needed).
        """
        if not state:
            return []

        # Select first goal
        goal = state[0]
        rest = state[1:]

        derived: List[Tuple[IState, int, bool]] = []
        const_count = self.constant_count

        # Check for True/False predicates
        if goal[0] == self.true_idx:
            # True() - remove from state
            if rest:
                derived.append((rest, next_var, False))
            else:
                # All goals are True = proof found!
                derived.append(((), next_var, True))
            return derived

        if goal[0] == self.false_idx:
            # False() - dead end
            return []

        # Unify with facts - with early termination on proof
        fact_subs_list = self._unify_with_facts(goal, excluded_hash)
        for subs in fact_subs_list:
            # Apply substitutions to remaining goals
            new_rest = [self._apply_subs(t, subs) for t in rest]

            if not new_rest:
                # Proved this goal, no more goals = immediate proof
                # Early termination like SB3
                return [((), next_var, True)]

            # Eager fact substitution: check if remaining goals are now facts
            new_rest = self._substitute_ground_facts(new_rest, excluded_hash)
            if new_rest is None:
                # All remaining became facts = proof - early termination
                return [((), next_var, True)]

            derived.append((tuple(new_rest), next_var, False))

        # Unify with rules - with early termination on proof
        rule_matches = self._unify_with_rules(goal)
        for subs, body in rule_matches:
            # Apply substitutions to body and remaining goals
            new_body = [self._apply_subs(t, subs) for t in body]
            new_rest = [self._apply_subs(t, subs) for t in rest]

            # Combine: body goals + remaining goals
            new_state = new_body + new_rest

            # Eager fact substitution
            new_state = self._substitute_ground_facts(new_state, excluded_hash)
            if new_state is None:
                # All became facts = proof - early termination
                return [((), next_var, True)]

            # Rename variables to avoid conflicts
            renamed_state, new_next_var = self._rename_vars(tuple(new_state), next_var)
            derived.append((renamed_state, new_next_var, False))

        return derived

    def _substitute_ground_facts(
        self,
        atoms: List[ITerm],
        excluded_hash: Optional[int]
    ) -> Optional[List[ITerm]]:
        """Replace ground facts with True, filter out True atoms.

        Returns None if all atoms become True (proof found).
        Returns filtered list otherwise.
        """
        # Cache lookups
        const_count = self.constant_count
        true_idx = self.true_idx
        fact_hashes = self.fact_hashes
        hash_base = self.hash_base
        result: List[ITerm] = []
        result_append = result.append  # Cache method

        for atom in atoms:
            pred, a0, a1 = atom

            # Skip if already True
            if pred == true_idx:
                continue

            # Check if ground (no variables)
            if a0 <= const_count and a1 <= const_count:
                # Ground atom - inline hash computation
                atom_hash = (pred * hash_base + a0) * hash_base + a1
                if excluded_hash is not None and atom_hash == excluded_hash:
                    result_append(atom)
                elif atom_hash in fact_hashes:
                    continue  # It's a fact - skip
                else:
                    result_append(atom)
            else:
                result_append(atom)

        # If all atoms became True (result is empty), return None to signal proof
        if not result:
            return None

        return result

    def check_provability(
        self,
        query: Term,
        is_train_data: bool = False,
        max_depth: int = 7,
        max_atoms: int = 20,
        sb3_compat: bool = False,
    ) -> int:
        """Check provability using BFS.

        Args:
            query: Query term to prove
            is_train_data: If True, exclude query from facts
            max_depth: Maximum proof depth
            max_atoms: Maximum atoms per state
            sb3_compat: If True, mimic SB3's variable handling (for speed comparison).
                        SB3 has a bug where it doesn't increment variable indices,
                        causing variables to collapse. This makes it faster but
                        can miss valid proofs.

        Returns:
            Minimum proof depth, or -1 if not provable within max_depth
        """
        # Convert query to ITerm
        iquery = self._term_to_iterm(query)
        excluded_hash = self._pack_iterm(iquery) if is_train_data else None

        # Depth 0: check if query is a direct fact
        if not is_train_data:
            query_hash = self._pack_iterm(iquery)
            if query_hash in self.fact_hashes:
                return 0

        # BFS
        initial_state: IState = (iquery,)
        frontier = deque([(initial_state, 0, self.runtime_var_start)])
        # Use frozenset for deduplication (order-independent like SB3)
        # In SB3-compat mode, use simple frozenset (variables are reused anyway)
        visited: Set[FrozenSet[ITerm]] = {self._canonicalize_state(initial_state, simple=sb3_compat)}

        while frontier:
            state, depth, next_var = frontier.popleft()

            if depth >= max_depth:
                continue

            # Generate successors
            derived = self._get_derived_states(state, excluded_hash, next_var)

            for new_state, new_next_var, is_proof in derived:
                # Check for immediate proof
                if is_proof:
                    return depth + 1

                # Skip empty states (shouldn't happen if not proof)
                if not new_state:
                    continue

                # Skip False states
                if new_state[0][0] == self.false_idx:
                    continue

                # Skip if too many atoms
                if len(new_state) > max_atoms:
                    continue

                # Deduplicate using canonical form
                canonical = self._canonicalize_state(new_state, simple=sb3_compat)
                if canonical in visited:
                    continue
                visited.add(canonical)

                # In SB3-compat mode, always use same next_var (mimics SB3 bug)
                effective_next_var = self.runtime_var_start if sb3_compat else new_next_var
                frontier.append((new_state, depth + 1, effective_next_var))

        return -1

    def query_str_to_term(self, query_str: str) -> Term:
        """Parse query string to Term (e.g., 'owns(alice,book)')."""
        # Simple parser for predicate(arg0,arg1) format
        paren = query_str.index('(')
        pred = query_str[:paren]
        args_str = query_str[paren + 1:-1]
        args = tuple(a.strip() for a in args_str.split(','))
        return Term(pred, args)
