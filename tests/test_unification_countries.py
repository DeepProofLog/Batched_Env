"""
Controlled unification tests for a miniature countries_s3 knowledge base.

The dataset mirrors the public countries_s3 rules (neighbor-based region
propagation) but uses a tiny set of facts so we can predict every reasoning
step and assert exact outcomes.
"""
import torch

from index_manager import IndexManager
from unification_engine import UnificationEngine


def setup_countries_kb(device: torch.device = torch.device('cpu')):
    """Build a compact KB with countries, regions, and neighbor facts."""
    constants = [
        'algeria',
        'andorra',
        'france',
        'iceland',
        'morocco',
        'portugal',
        'tunisia',
        'africa',
        'europe',
    ]
    predicates = ['neighborOf', 'locatedInCR']

    im = IndexManager(
        constants=constants,
        predicates=predicates,
        max_total_runtime_vars=64,
        max_arity=2,
        device=device,
    )

    im.true_pred_idx = im.predicate_str2idx['True']
    im.false_pred_idx = im.predicate_str2idx['False']

    for var in ['X', 'Y', 'Z', 'K']:
        im._ensure_template_var(var)

    c = {name: im.constant_str2idx[name] for name in constants}
    p = {
        name: im.predicate_str2idx[name]
        for name in ['neighborOf', 'locatedInCR', 'True', 'False']
    }
    v = {name: im.template_var_str2idx[name] for name in ['X', 'Y', 'Z', 'K']}

    facts = []

    def add_fact(pred_name, arg1, arg2):
        facts.append([p[pred_name], c[arg1], c[arg2]])

    # Direct region membership facts.
    add_fact('locatedInCR', 'france', 'europe')
    add_fact('locatedInCR', 'morocco', 'africa')

    # Neighbor relationships (bidirectional where needed).
    neighbor_edges = [
        ('andorra', 'france'),
        ('france', 'andorra'),
        ('tunisia', 'algeria'),
        ('algeria', 'morocco'),
    ]
    for a, b in neighbor_edges:
        add_fact('neighborOf', a, b)

    rule_heads = [
        [p['locatedInCR'], v['X'], v['Z']],
        [p['locatedInCR'], v['X'], v['Z']],
    ]
    rule_bodies = [
        [
            [p['neighborOf'], v['X'], v['Y']],
            [p['locatedInCR'], v['Y'], v['Z']],
        ],
        [
            [p['neighborOf'], v['X'], v['Y']],
            [p['neighborOf'], v['Y'], v['K']],
            [p['locatedInCR'], v['K'], v['Z']],
        ],
    ]

    max_body_len = max(len(body) for body in rule_bodies)
    pad_atom = [im.padding_idx, im.padding_idx, im.padding_idx]
    padded_bodies = [
        body + [pad_atom] * (max_body_len - len(body)) for body in rule_bodies
    ]

    im.rules_heads_idx = torch.tensor(rule_heads, dtype=torch.long, device=device)
    im.rules_idx = torch.tensor(padded_bodies, dtype=torch.long, device=device)
    im.rule_lens = torch.tensor([len(body) for body in rule_bodies], dtype=torch.long, device=device)

    facts_tensor = torch.tensor(facts, dtype=torch.long, device=device)
    pred_order = torch.argsort(facts_tensor[:, 0])
    facts_sorted = facts_tensor[pred_order]
    im.facts_idx = facts_sorted

    predicates_in_facts = facts_sorted[:, 0]
    unique_preds, counts = torch.unique_consecutive(
        predicates_in_facts, return_counts=True
    )
    starts = torch.cat(
        [torch.tensor([0], dtype=torch.long, device=device), torch.cumsum(counts[:-1], dim=0)]
    )
    ends = torch.cumsum(counts, dim=0)

    max_pred_idx = max(p.values())
    im.predicate_range_map = torch.zeros((max_pred_idx + 1, 2), dtype=torch.long, device=device)
    im.predicate_range_map[unique_preds, 0] = starts
    im.predicate_range_map[unique_preds, 1] = ends

    return im, c, p, v


def state_atoms(derived, batch_idx, state_idx, pad_idx):
    """Return only the valid atoms for a given state."""
    state = derived[batch_idx, state_idx]
    mask = state[:, 0] != pad_idx
    return state[mask]


def assert_contains_true_state(derived, counts, batch_idx, true_idx, pad_idx):
    assert counts[batch_idx].item() == 1, "Expected exactly one state"
    atoms = state_atoms(derived, batch_idx, 0, pad_idx)
    assert atoms.shape[0] == 1, "True() state should contain a single atom"
    assert atoms[0, 0].item() == true_idx, "State does not contain True()"


def assert_contains_false_state(derived, counts, batch_idx, false_idx, pad_idx):
    assert counts[batch_idx].item() == 1, "Expected exactly one state"
    atoms = state_atoms(derived, batch_idx, 0, pad_idx)
    assert atoms.shape[0] == 1, "False() state should contain a single atom"
    assert atoms[0, 0].item() == false_idx, "State does not contain False()"


def pick_state_by_atom_count(derived, counts, batch_idx, pad_idx, atom_count):
    """Return (state_index, atoms) for the first state with the desired atom count."""
    for idx in range(counts[batch_idx].item()):
        atoms = state_atoms(derived, batch_idx, idx, pad_idx)
        if atoms.shape[0] == atom_count:
            return idx, atoms
    raise AssertionError(f"No state found with {atom_count} atoms (batch {batch_idx})")


def assert_neighbor_stub(state, c, p, im, *,
                         start_const, end_const):
    """Assert state represents neighborOf(start, Var) -> locatedInCR(Var, end)."""
    assert state.shape[0] == 2, f"Expected 2 atoms, got {state.shape[0]}"

    first, second = state[0], state[1]
    assert first[0].item() == p['neighborOf'], "First atom must be neighborOf"
    assert first[1].item() == c[start_const], "neighborOf arg1 mismatch"
    mid_var = first[2].item()
    assert mid_var > im.constant_no, "neighborOf arg2 should be a variable"

    assert second[0].item() == p['locatedInCR'], "Second atom must be locatedInCR"
    assert second[1].item() == mid_var, "Variable binding mismatch"
    assert second[2].item() == c[end_const], "locatedInCR arg2 mismatch"


def assert_two_hop_stub(state, c, p, im, *, start_const, end_const):
    """Assert state represents the two-hop neighbor rule."""
    assert state.shape[0] == 3, f"Expected 3 atoms, got {state.shape[0]}"
    first, second, third = state

    assert first[0].item() == p['neighborOf'], "First atom must be neighborOf"
    assert first[1].item() == c[start_const], "neighborOf arg1 mismatch"
    mid_var = first[2].item()
    assert mid_var > im.constant_no, "neighborOf arg2 should be a variable"

    assert second[0].item() == p['neighborOf'], "Second atom must be neighborOf"
    hop_var = second[2].item()
    assert second[1].item() == mid_var, "Two-hop chain must reuse the first variable"
    assert hop_var > im.constant_no, "Second hop arg2 should stay a variable"

    assert third[0].item() == p['locatedInCR'], "Third atom must be locatedInCR"
    assert third[1].item() == hop_var, "locatedInCR arg1 should be the second hop variable"
    assert third[2].item() == c[end_const], "locatedInCR arg2 mismatch"


def run_fact_test(engine, im, c, p, rv, pad_idx):
    query = torch.tensor(
        [[[p['locatedInCR'], c['france'], c['europe']]]], dtype=torch.long
    )
    next_var = torch.tensor([rv(0)], dtype=torch.long)
    derived, counts, _ = engine.get_derived_states(query, next_var)
    assert_contains_true_state(derived, counts, 0, im.true_pred_idx, pad_idx)


def run_rule_shape_test(engine, im, c, p, rv, pad_idx):
    query = torch.tensor(
        [[[p['locatedInCR'], c['andorra'], c['europe']]]], dtype=torch.long
    )
    next_var = torch.tensor([rv(0)], dtype=torch.long)
    derived, counts, _ = engine.get_derived_states(query, next_var)

    assert counts[0].item() == 2, f"Expected 2 states, got {counts[0].item()}"
    first_state = state_atoms(derived, 0, 0, pad_idx)
    second_state = state_atoms(derived, 0, 1, pad_idx)

    assert_neighbor_stub(first_state, c, p, im, start_const='andorra', end_const='europe')
    assert_two_hop_stub(second_state, c, p, im, start_const='andorra', end_const='europe')


def run_tunisia_chain_test(engine, im, c, p, rv, pad_idx):
    """Walk from tunisia to africa through neighbors, ending in True()."""
    state = torch.tensor(
        [[[p['locatedInCR'], c['tunisia'], c['africa']]]], dtype=torch.long
    )
    next_var = torch.tensor([rv(0)], dtype=torch.long)

    # Step 1: Expand locatedInCR(tunisia, africa)
    derived1, counts1, next_var = engine.get_derived_states(state, next_var)
    assert counts1[0].item() == 2, "Expected both neighbor rules to fire"
    idx_two_atom, step1_state = pick_state_by_atom_count(
        derived1, counts1, 0, pad_idx, atom_count=2
    )
    assert_neighbor_stub(step1_state, c, p, im, start_const='tunisia', end_const='africa')

    # Step 2: Resolve neighborOf(tunisia, _)
    state = derived1[0, idx_two_atom].unsqueeze(0)
    derived2, counts2, next_var = engine.get_derived_states(state, next_var)
    assert counts2[0].item() == 1, "Neighbor fact should produce a single continuation"
    step2_atoms = state_atoms(derived2, 0, 0, pad_idx)
    assert step2_atoms.shape[0] == 1, "Expected a single locatedInCR goal"
    assert step2_atoms[0, 0].item() == p['locatedInCR'], "Remaining goal must be locatedInCR"
    assert step2_atoms[0, 1].item() == c['algeria'], "First argument should bind to algeria"
    assert step2_atoms[0, 2].item() == c['africa'], "Region must stay africa"

    # Step 3: Expand locatedInCR(algeria, africa)
    state = derived2[0, 0].unsqueeze(0)
    derived3, counts3, next_var = engine.get_derived_states(state, next_var)
    assert counts3[0].item() == 2, "Algeria should again trigger both neighbor rules"
    idx_two_atom_step3, step3_state = pick_state_by_atom_count(
        derived3, counts3, 0, pad_idx, atom_count=2
    )
    assert_neighbor_stub(step3_state, c, p, im, start_const='algeria', end_const='africa')

    # Step 4: Resolving neighbor now grounds locatedInCR(morocco, africa) -> True()
    state = derived3[0, idx_two_atom_step3].unsqueeze(0)
    derived4, counts4, _ = engine.get_derived_states(state, next_var)
    assert_contains_true_state(derived4, counts4, 0, im.true_pred_idx, pad_idx)


def run_iceland_false_test(engine, im, c, p, rv, pad_idx):
    """Iceland has no neighbors, so inference should bottom out at False()."""
    query = torch.tensor(
        [[[p['locatedInCR'], c['iceland'], c['europe']]]], dtype=torch.long
    )
    next_var = torch.tensor([rv(0)], dtype=torch.long)

    derived1, counts1, next_var = engine.get_derived_states(query, next_var)
    assert counts1[0].item() == 2, "Rules still fire even without supporting facts"

    state = derived1[0, 0].unsqueeze(0)
    derived2, counts2, _ = engine.get_derived_states(state, next_var)
    assert_contains_false_state(derived2, counts2, 0, im.false_pred_idx, pad_idx)


def main():
    im, c, p, _ = setup_countries_kb()
    engine = UnificationEngine.from_index_manager(im)
    rv = lambda i: im.runtime_var_start_index + i
    pad_idx = im.padding_idx

    tests = [
        ("Fact unification yields True()", lambda: run_fact_test(engine, im, c, p, rv, pad_idx)),
        ("Rule instantiations build predictable states", lambda: run_rule_shape_test(engine, im, c, p, rv, pad_idx)),
        ("Neighbor reasoning proves tunisia locatedInCR africa", lambda: run_tunisia_chain_test(engine, im, c, p, rv, pad_idx)),
        ("Missing neighbors lead to False()", lambda: run_iceland_false_test(engine, im, c, p, rv, pad_idx)),
    ]

    print("#" * 80)
    print("COUNTRIES_S3 MINI UNIFICATION TESTS")
    print("#" * 80)
    for name, fn in tests:
        print(f"\n{name}")
        fn()
        print(f"  ✅ {name}")

    print("\nAll countries_s3 unification tests passed!")


if __name__ == "__main__":
    main()
