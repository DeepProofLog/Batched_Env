import copy
from itertools import product
import time
from typing import List, Set, Tuple, Dict, Union

from ns_lib.logic.commons import Atom, Domain, Rule, RuleGroundings
from ns_lib.grounding.engine import Engine



class AtomIndex():
    def __init__(self, facts: List[Tuple[str, str, str]]):
        _index = {}
        for f in facts:
            r_idx, s_idx, o_idx = f
            key = (r_idx, None, None)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = (r_idx, s_idx, None)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = (r_idx,  None, o_idx)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = f
            _index[key] = set([f])

        # Store tuple instead of sets.
        self._index = {k: tuple(v) for k, v in _index.items()}

    def get_matching_atoms(self,
                           atom: Tuple[str, str, str]) -> Tuple[Tuple[str, str, str]]:
        return self._index.get(atom, [])


def get_atoms_on_groundings(groundings:Set[Tuple[Tuple, Tuple]]) -> Set[Tuple]:
    atoms = set()
    for rule_atoms in groundings:
        for atom in rule_atoms[0]:  # head
            atoms.add(atom)
        for atom in rule_atoms[1]:  # tail
            atoms.add(atom)
    return atoms



def backward_chaining_grounding_one_rule(
    domains: Dict[str, Domain],
    domain2adaptive_constants: Dict[str, List[str]],
    pure_adaptive: bool,
    rule: Rule,
    queries: List[Tuple],
    fact_index: AtomIndex,
    res: Set[Tuple[Tuple, Tuple]]=None) -> Union[
        None, Set[Tuple[Tuple, Tuple]]]:
    # We have a rule like A(x,y) B(y,z) => C(x,z)
    assert len(rule.head) == 1, (
        'Rule is not a Horn clause %s' % str(rule))
    head = rule.head[0]

    new_ground_atoms = set()

    for q in queries:
      if q[0] != head[0]:  # predicates must match.
        continue

      # Get the variable assignments from the head.
      head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}

      free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
                         if v not in head_ground_vars]
      free_vars = [vd[0] for vd in free_var2domain]
      if pure_adaptive:
          ground_var_groups = [domain2adaptive_constants.get(vd[1], [])
                               for vd in free_var2domain]
      elif domain2adaptive_constants is not None:
          ground_var_groups = [domains[vd[1]].constants +
                               domain2adaptive_constants.get(vd[1], [])
                               for vd in free_var2domain]
      else:
          ground_var_groups = [domains[vd[1]].constants
                               for vd in free_var2domain]

      # Iterate over the groundings of the free vars.
      # If no free vars are available, product returns a single empty
      # tuple, meaning that we still correctly enter in the following
      # for loop for a single round.
      #print('FREE VARS_SPAN', list(product(*ground_var_groups)))
      for ground_vars in product(*ground_var_groups):
          var2ground = dict(zip(free_vars, ground_vars))
          full_ground_vars = {**head_ground_vars, **var2ground}
          body_grounding = []
          for j in range(len(rule.body)):
              body_atom = rule.body[j]
              new_ground_atom = (body_atom[0], ) + tuple(
                  [full_ground_vars.get(body_atom[k+1], None)
                   for k in range(len(body_atom)-1)])
              body_grounding.append(new_ground_atom)
          new_ground_atoms.add(((q,), tuple(body_grounding)))

    if res is None:
        return new_ground_atoms
    else:
        res.update(new_ground_atoms)





class BackwardChainingGrounder(Engine):

    def __init__(self, rules: List[Rule],
                 facts: List[Union[Atom, str, Tuple]],
                 domains: Dict[str, Domain],
                 domain2adaptive_constants: Dict[str, List[str]]=None,
                 pure_adaptive: bool=False,
                 num_steps: int=1,
                 # Whether the groundings should be accumulated across calls.
                 accumulate_groundings: bool=False):
        self.num_steps = num_steps
        self.accumulate_groundings = accumulate_groundings
        self.rules = rules
        self.domains = domains
        self.domain2adaptive_constants = domain2adaptive_constants
        self.pure_adaptive = pure_adaptive
        self.facts = [a if isinstance(a,Tuple) else a.toTuple()
                      if isinstance(a,Atom) else Atom(s=a).toTuple()
                      for a in facts]
        # self.facts = facts
        for rule in self.rules:
            assert len(rule.head) == 1, (
                '%s is not a Horn clause' % str(rule))
        self._fact_index = AtomIndex(self.facts)
        self.relation2queries = {}
        self.rule2groundings = {}
        self.rule2proofs = {}

    def _init_internals(self, queries: List[Tuple], clean: bool):
        self.relation2queries = {}  # reset
        for q in queries:
            if q[0] not in self.relation2queries:
                self.relation2queries[q[0]] = set()
            self.relation2queries[q[0]].add(q)

        # If clean=False, groundings are incrementally added.
        for rule in self.rules:
            if clean or rule.name not in self.rule2groundings:
                self.rule2groundings[rule.name] = set()
                self.rule2proofs[rule.name] = []

    # Ground a batch of queries, the result is cached for speed.
    #@profile
    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:

        if self.rules is None or len(self.rules) == 0:
            return []

        # When accumulating groundings, we keep a single large set of
        # groundings that are reused over all batches.
        self._init_internals(queries, clean=(not self.accumulate_groundings))
        # Keeps track of the queris already processed for this rule.
        self._rule2processed_queries = {rule.name: set() for rule in self.rules}
        for step in range(self.num_steps):
            # print('STEP', step)
            for rule in self.rules:
                # Here we assume to have a Horn clause, fix it.
                queries_per_rule = list(
                    self.relation2queries.get(rule.head[0][0], set()))
                if not queries_per_rule:
                    continue
                backward_chaining_grounding_one_rule(
                    self.domains,
                    self.domain2adaptive_constants,
                    self.pure_adaptive,
                    rule, queries_per_rule, self._fact_index,
                    # Output added here.
                    res=self.rule2groundings[rule.name])
                # Update the list of processed rules.
                self._rule2processed_queries[rule.name].update(queries_per_rule)
                # print(step, 'PROCESSED', len(queries_per_rule), len(self._rule2processed_queries[rule.name]))

            if step == self.num_steps - 1:
                break

            # Get the queries for the next iteration.
            new_queries = set()
            for rule in self.rules:
                groundings = self.rule2groundings[rule.name]
                # Get the new queries left to prove, these facts that are not
                # been processed already and that are not known facts.
                new_queries.update(
                    [a for a in get_atoms_on_groundings(groundings)
                     if a not in self._rule2processed_queries[rule.name] and
                     self._fact_index._index.get(a, None) is None])
            # print(step, 'NEW Q', list(new_queries)[:10], 'FROM', len(groundings))
            self._init_internals(list(new_queries), clean=False)

        #print('R', self.rule2groundings)
        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name: RuleGroundings(
                rule_name, sorted(list(groundings), key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name: RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}

        return ret