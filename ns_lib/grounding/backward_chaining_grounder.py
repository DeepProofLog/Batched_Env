#! /bin/python3
#from memory_profiler import profile
import copy
from itertools import product
import time
from typing import List, Set, Tuple, Dict, Union

from ns_lib.logic.commons import Atom, Domain, Rule, RuleGroundings
from ns_lib.grounding.engine import Engine



class AtomIndexDeterministic():
    def __init__(self, facts: List[Tuple[str, str, str]]):
        _index = {}
        self.is_deterministic = True
        for f in facts:
            r_idx, s_idx, o_idx = f
            key = (r_idx, None, None)
            if key not in _index:
                _index[key] = []  # Use a list
            _index[key].append(f)  # Use append

            key = (r_idx, s_idx, None)
            if key not in _index:
                _index[key] = []  # Use a list
            _index[key].append(f)  # Use append

            key = (r_idx, None, o_idx)
            if key not in _index:
                _index[key] = []  # Use a list
            _index[key].append(f)  # Use append

            key = f
            _index[key] = [f]  # Use a list

        self._index = {k: tuple(v) for k, v in _index.items()}  # Convert to tuple *after* ordering with lists


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

# res is a Set of (Tuple_head_groundings, Tuple_body_groundings)
def approximate_backward_chaining_grounding_one_rule(
    domains: Dict[str, Domain],
    domain2adaptive_constants: Dict[str, List[str]],
    pure_adaptive: bool,
    rule: Rule,
    queries: List[Tuple],
    fact_index: AtomIndex,
    max_unknown_fact_count: int,
    res: Set[Tuple[Tuple, Tuple]]=None,
    proofs: Dict[Tuple[Tuple, Tuple], List[Tuple[Tuple, Tuple]]]=None,
    head_predicates: Set[str]=None,
    prune: bool=True,
    force_determinism: bool=False) -> Union[
        None, Set[Tuple[Tuple, Tuple]]]:
    
    assert len(rule.head) == 1, (
        'Rule is not a Horn clause %s' % str(rule))
    head = rule.head[0]
    build_proofs: bool = (proofs is not None)

    new_ground_atoms = set()
    start = time.time()
    lim = 1000000000
    cont = 0
    for q in queries:
      cont += 1 
    #   print('\n\n***************q', q,'********************') if cont< lim else None
      if q[0] != head[0]:  # predicates must match.
        continue

      # Get the variable assignments from the head.
      head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}

      for i in range(len(rule.body)):
        '''
        For each query to prove, assign constants to variables and substitute them in the body of the rule.
        Iterate through each atom in the body of the rule:
        If not all variables are substituted, retrieve the possible groundings of the atom from the facts.
        Check if the atom is present in the facts. If it is not, add it to the list of atoms to prove.
        '''

        # Ground atom by replacing variables with constants.
        # The result is the partially ground atom A('Antonio',None)
        # with None indicating unground variables.
        body_atom = rule.body[i]
        ground_body_atom = (body_atom[0], ) + tuple(
            [head_ground_vars.get(body_atom[j+1], None)
             for j in range(len(body_atom)-1)])
        # print('\n- i', i,'. grounded body atom:', ground_body_atom) if cont< lim else None
        if all(ground_body_atom[1:]):
            groundings = (ground_body_atom,)
        else:
            # Tuple of atoms matching A(Antonio,None) in the facts.
            # This is the list of ground atoms for the i-th atom in the body.
            groundings = fact_index._index.get(ground_body_atom, [])
            # print('     possible groundings:', groundings) if cont< lim else None

        if len(rule.body) == 1:
            # Shortcut, we are done, the clause has no free variables.
            # print('ADDED', q, '->', (groundings,)) if cont< lim else None
            new_ground_atoms.add(((q,), groundings))
            continue

        for ground_atom in groundings:
            # print('\n     -GROUND ATOM', ground_atom) if cont< lim else None
            # This loop is only needed to ground at least one atom in the body
            # of the formula. Otherwise it would be enough to start with the
            # loop for ground_vars in product(...) but it would often expand
            # into many nodes. The current solution does not allow to fully
            # recover a classical backward_chaining, though.
            head_body_ground_vars = copy.copy(head_ground_vars)
            head_body_ground_vars.update(
                {v: a for v, a in zip(body_atom[1:], ground_atom[1:])})

            free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
                               if v not in head_body_ground_vars]
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
            # print('     FREE VARS_SPAN', list(product(*[domains[vd[1]].constants for vd in free_var2domain]))) if cont< lim else None
            for ground_vars in product(*ground_var_groups):
                var2ground = dict(zip(free_vars, ground_vars))
                full_ground_vars = {**head_body_ground_vars, **var2ground}
                # print('     FULL_VARS', full_ground_vars) if cont< lim else None

                accepted: bool = True
                body_grounding = []
                if build_proofs:
                    body_grounding_to_prove = []
                unknown_fact_count: int = 0
                for j in range(len(rule.body)):
                    if i == j:
                        new_ground_atom = ground_atom
                        # by definition as it is coming from the groundings.
                        is_known_fact = True
                        # print('         -j=',j,'=i, it is a fact by def') if cont< lim else None
                    else:
                        body_atom2 = rule.body[j]
                        new_ground_atom = (body_atom2[0], ) + tuple(
                            [full_ground_vars.get(body_atom2[k+1], None)
                             for k in range(len(body_atom2)-1)])
                        if new_ground_atom == q:
                            # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, ' Same atom as query, discard') if cont< lim else None
                            accepted = False
                            break
                        is_known_fact = (fact_index._index.get(
                            new_ground_atom, None) is not None)
                        # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact) if cont< lim else None

                    assert all(new_ground_atom), (
                        'Unexpected free variables in %s' %
                        str(new_ground_atom))
                    if not is_known_fact and (
                            max_unknown_fact_count < 0 or
                            unknown_fact_count < max_unknown_fact_count):
                        if not prune or (head_predicates is not None and new_ground_atom[0] in head_predicates):
                            body_grounding.append(new_ground_atom)
                        else:
                            accepted = False
                            break
                        # body_grounding.append(new_ground_atom)
                        if build_proofs:
                            body_grounding_to_prove.append(new_ground_atom)
                        unknown_fact_count += 1
                    elif is_known_fact:
                        body_grounding.append(new_ground_atom)
                    else:
                        # print('         Discard',unknown_fact_count,'/', max_unknown_fact_count) if cont< lim and i!=j else None
                        accepted = False
                        break

                if accepted:
                    # print('     ADDED', q, '->', tuple(body_grounding), 'TO_PROVE',          str(body_grounding_to_prove) if build_proofs else '') if cont< lim else None
                    new_ground_atoms.add(((q,), tuple(body_grounding)))
                    if build_proofs:
                        proofs.append((q, body_grounding_to_prove))

    end = time.time()
    # print('NUM GROUNDINGS', len(new_ground_atoms),'. TIME', end - start)
    # print('NEW GROUND ATOMS', new_ground_atoms) if cont< lim else None
    if res is None:
        return new_ground_atoms
    else:
        res.update(new_ground_atoms)


def PruneIncompleteProofs(rule2groundings, rule2proofs, fact_index, num_steps):
    atom2proved = {}

    def update_atom2proved(proofs):
        for query, proof in proofs:
            if query not in atom2proved or not atom2proved[query]:
                atom2proved[query] = all(map(atom2proved.get, proof, [False] * len(proof)))
    
    for _ in range(num_steps):
        list(map(update_atom2proved, rule2proofs.values()))


    fact_index_get = fact_index._index.get
    def is_grounding_proved(g):
        return all(map(lambda a: atom2proved.get(a, False) or fact_index_get(a), g[1]))

    pruned_rule2groundings = {
        rule_name: set(filter(is_grounding_proved, groundings))
        for rule_name, groundings in rule2groundings.items()
    }

    return pruned_rule2groundings





class ApproximateBackwardChainingGrounder(Engine):

    def __init__(self, rules: List[Rule], facts: List[Union[Atom, str, Tuple]],
                 domains: Dict[str, Domain],
                 domain2adaptive_constants: Dict[str, List[str]]=None,
                 pure_adaptive: bool=False,
                 max_unknown_fact_count: int=1,
                 max_unknown_fact_count_last_step: int=1,
                 num_steps: int=1,
                 max_groundings_per_rule: int=-1,  # to speedup the computation.
                 # Whether the groundings should be accumulated across calls.
                 accumulate_groundings: bool=False,
                 prune_incomplete_proofs: bool=True,
                 force_determinism: bool=False):
        self.max_unknown_fact_count = max_unknown_fact_count
        self.max_unknown_fact_count_last_step = max_unknown_fact_count_last_step
        self.num_steps = num_steps
        self.accumulate_groundings = accumulate_groundings
        self.prune_incomplete_proofs = prune_incomplete_proofs
        self.max_groundings_per_rule = max_groundings_per_rule
        self.rules = rules
        self.domains = domains
        self.domain2adaptive_constants = domain2adaptive_constants
        self.pure_adaptive = pure_adaptive
        self.facts = [a if isinstance(a,Tuple) else a.toTuple()
                      if isinstance(a,Atom) else Atom(s=a).toTuple()
                      for a in facts]

        for rule in self.rules:
            assert len(rule.head) == 1, (
                '%s is not a Horn clause' % str(rule))
        self._fact_index = AtomIndex(self.facts)
        self.relation2queries = {}
        self.rule2groundings = {}
        self.rule2proofs = {}
        self.force_determinism = force_determinism
        if force_determinism:
            assert self._fact_index.is_deterministic, ('The fact index must be deterministic for this grounder, choose the proper AtomIndexDeterministic class')

    def _init_internals(self, queries: List[Tuple], clean: bool):
        '''Adds the queries to prove to self.relation2queries
        Resets self.rule2groundings and self.rule2proofs.'''
    
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
        
        # create a tuple with the predicates that are head of the rules
        self.head_predicates = set([rule.head[0][0] for rule in self.rules])

        self._init_internals(queries, clean=(not self.accumulate_groundings))

        self._rule2processed_queries = {rule.name: set() for rule in self.rules}
        # time_start = time.time()
        for step in range(self.num_steps):
            # print('\n\nSTEP NUMBER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', step,'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^','step ',step,'/', self.num_steps-1)
            for j,rule in enumerate(self.rules):
                # print('\nrule ', rule, ' """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" ')
                queries_per_rule = list(self.relation2queries.get(rule.head[0][0], set()))
                if self.force_determinism:
                    queries_per_rule = sorted(queries_per_rule)
                # print('\nqueries to prove\n',len(queries_per_rule), queries_per_rule)
                if not queries_per_rule:
                    continue
                approximate_backward_chaining_grounding_one_rule(
                    self.domains,
                    self.domain2adaptive_constants,
                    self.pure_adaptive,
                    rule, 
                    queries_per_rule, 
                    self._fact_index,
                    (self.max_unknown_fact_count if step < self.num_steps - 1
                     else self.max_unknown_fact_count_last_step),
                    res=self.rule2groundings[rule.name], # Output added here.
                    proofs=(self.rule2proofs[rule.name] if self.prune_incomplete_proofs else None), # Proofs added here.
                    head_predicates=self.head_predicates,
                    prune=self.prune_incomplete_proofs,
                    force_determinism=self.force_determinism)
                # Update the list of processed rules.
                self._rule2processed_queries[rule.name].update(queries_per_rule)
                # print('\nqueries processed (_rule2processed_queries):\n', len(self._rule2processed_queries[rule.name]),self._rule2processed_queries[rule.name])
                # print('\nTotal  groundings in res after rule',j+1,'/',len(self.rules),', step',step,'/',self.num_steps,':',sum([len(v) for k, v in self.rule2groundings.items()])) # careful, here there are duplicates
                # print()
                # [print('\nGroundings:', k, len(v), v) for k, v in self.rule2groundings.items()]
                # print('\nproofs:',self.rule2proofs)
 
            if step == self.num_steps - 1:
                break

            # Get the queries for the next iteration.
            new_queries = set()
            for rule in self.rules:
                groundings = self.rule2groundings[rule.name]
                # New queries left to prove are the body atoms that have not
                # been processed already and that are not known facts.
                new_queries.update(
                    [a for a in get_atoms_on_groundings(groundings)
                     if a not in self._rule2processed_queries[rule.name] and
                     self._fact_index._index.get(a, None) is None])

            # Here we update the queries to process in the next iteration, we only keep the new ones.
            self._init_internals(list(new_queries), clean=False)

        # time_end = time.time()
        # print('Time to ground', time_end - time_start)

        if self.prune_incomplete_proofs:
            # check all the groundings with at least 1 atom missing, to see if they are proved (all atoms present in the facts)
            # time_start = time.time()
            # print('Num groundings before pruning',sum([len(v) for k, v in self.rule2groundings.items()]))
            self.rule2groundings = PruneIncompleteProofs(self.rule2groundings,
                                                         self.rule2proofs,
                                                         self._fact_index,
                                                         self.num_steps)
            # print('Num groundings after pruning',sum([len(v) for k, v in self.rule2groundings.items()]))
            # time_end = time.time()
            # print('Time to prune', time_end - time_start)
        # This should be done after sorting the groundings to ensure the output
        # to be deterministic.
        if self.max_groundings_per_rule > 0:
            self.rule2groundings = {rule_name:set(list(groundings)[:self.max_groundings_per_rule])
                                    for rule_name,groundings in self.rule2groundings.items()}

        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name: RuleGroundings(
                rule_name, sorted(list(groundings), key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name: RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}
 
        return ret