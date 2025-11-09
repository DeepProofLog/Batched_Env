"""
Toy test to verify reasoning with rules works correctly.

Setup:
- Facts: parent(a, b), parent(b, c)
- Rule: parent(X, Y), parent(Y, Z) -> grandparent(X, Z)
- Query: grandparent(a, c) [should be provable via rule]
"""
import torch
from index_manager import IndexManager
from unification_engine import UnificationEngine
from env import BatchedEnv

# Manually create a toy dataset
constants = ['a', 'b', 'c']
predicates = ['parent', 'grandparent']

device = torch.device('cpu')  # Use CPU for simplicity

# Create index manager
im = IndexManager(
    constants=constants,
    predicates=predicates,
    max_total_runtime_vars=100,
    padding_atoms=6,
    max_arity=2,
    device=device,
)

print(f"Index Manager created:")
print(f"  Constants: {im.constant_str2idx}")
print(f"  Predicates: {im.predicate_str2idx}")
print(f"  True pred idx: {im.true_pred_idx}")
print(f"  False pred idx: {im.false_pred_idx}")

# Create facts: parent(a, b), parent(b, c)
facts = [
    ('parent', 'a', 'b'),
    ('parent', 'b', 'c'),
]
facts_tensor = im.state_to_tensor(facts)
im.set_facts(facts_tensor)
print(f"\nFacts set: {facts_tensor.shape}")
print(f"  parent(a, b) = {facts_tensor[0]}")
print(f"  parent(b, c) = {facts_tensor[1]}")

# Create rule: parent(X, Y), parent(Y, Z) -> grandparent(X, Z)
# Rule format: (head, body_atoms)
rules = [
    (
        ('grandparent', 'X', 'Z'),  # head
        [('parent', 'X', 'Y'), ('parent', 'Y', 'Z')]  # body
    )
]
rules_tensor, rule_lens = im.rules_to_tensor(rules, max_rule_atoms=2)
im.set_rules(rules_tensor, rule_lens)
print(f"\nRules set: {rules_tensor.shape}")
print(f"  Rule lens: {rule_lens}")
print(f"  Rules tensor:\n{rules_tensor}")
print(f"  Rules heads:\n{im.rules_heads_idx}")

# Create unification engine
engine = UnificationEngine.from_index_manager(im)
print(f"\nUnification engine created")

# Create query: grandparent(a, c)
query = im.state_to_tensor([('grandparent', 'a', 'c')])
print(f"\nQuery: grandparent(a, c)")
print(f"  Tensor: {query}")

# Test manual derivation
print(f"\n{'='*60}")
print("Testing manual derivation")
print(f"{'='*60}")

# Pad query to [1, padding_atoms, 3]
batch_query = torch.zeros((1, 6, 3), dtype=torch.int32, device=device)
batch_query[0, 0] = query[0]
next_var = torch.tensor([im.runtime_var_start_index], dtype=torch.int32, device=device)
labels = torch.tensor([1], dtype=torch.int32, device=device)

print(f"\nStep 1: Initial query grandparent(a, c)")
all_derived, next_var = engine.get_derived_states(
    current_states=batch_query,
    next_var_indices=next_var,
    excluded_queries=query,  # Exclude the original query
    labels=labels,
    verbose=2,
)
print(f"  Derived states: {len(all_derived[0])}")
for i, state in enumerate(all_derived[0]):
    print(f"    {i}: {state.shape} = {state}")
    # Decode each atom
    for j in range(state.shape[0]):
        atom = state[j]
        pred_idx = atom[0].item()
        arg1_idx = atom[1].item()
        arg2_idx = atom[2].item()
        
        # Reverse lookup predicate
        pred_name = None
        for name, idx in im.predicate_str2idx.items():
            if idx == pred_idx:
                pred_name = name
                break
        
        # Reverse lookup args
        arg1_name = None
        arg2_name = None
        for name, idx in im.constant_str2idx.items():
            if idx == arg1_idx:
                arg1_name = name
            if idx == arg2_idx:
                arg2_name = name
        
        # If not in constants, it's a variable
        if arg1_name is None:
            arg1_name = f"VAR_{arg1_idx}"
        if arg2_name is None:
            arg2_name = f"VAR_{arg2_idx}"
        
        print(f"      Atom {j}: {pred_name}({arg1_name}, {arg2_name})")

if len(all_derived[0]) > 0 and not (all_derived[0][0][0, 0] == im.false_pred_idx).item():
    # Apply first derived state
    derived_state = all_derived[0][0]
    batch_query = torch.zeros((1, 6, 3), dtype=torch.int32, device=device)
    batch_query[0, :derived_state.shape[0]] = derived_state
    
    print(f"\nStep 2: After applying rule, state has {derived_state.shape[0]} atoms")
    all_derived, next_var = engine.get_derived_states(
        current_states=batch_query,
        next_var_indices=next_var,
        excluded_queries=query,
        labels=labels,
        verbose=2,
    )
    print(f"  Derived states: {len(all_derived[0])}")
    for i, state in enumerate(all_derived[0]):
        print(f"    {i}: {state.shape} = {state}")

# Now test with environment
print(f"\n{'='*60}")
print("Testing with Environment")
print(f"{'='*60}")

# Build padded query tensor for the environment
padding_atoms = 6
query_state = torch.full(
    (1, padding_atoms, 3),
    im.padding_idx,
    dtype=torch.long,
    device=device,
)
query_state[0, 0] = query[0].to(dtype=torch.long, device=device)
labels_tensor = torch.tensor([1], dtype=torch.long, device=device)
depths_tensor = torch.tensor([0], dtype=torch.long, device=device)
env = BatchedEnv(
    batch_size=1,
    unification_engine=engine,
    queries=query_state,
    labels=labels_tensor,
    query_depths=depths_tensor,
    mode='train',
    max_depth=10,
    memory_pruning=False,
    padding_atoms=6,
    padding_states=20,
    reward_type=1,
    verbose=2,
    prover_verbose=2,
    device=device,
    corruption_mode=False,
    sampler=None,
    end_proof_action=False,
    skip_unary_actions=False,
    true_pred_idx=im.true_pred_idx,
    false_pred_idx=im.false_pred_idx,
    max_arity=2,
    padding_idx=im.padding_idx,
)

print(f"\nEnvironment created, resetting...")
obs = env.reset()
print(f"\nInitial state:")
print(f"  Labels: {env.current_labels}")
print(f"  Action mask: {obs['action_mask']}")

# Run until done
for step in range(10):
    print(f"\n{'='*60}")
    print(f"Step {step + 1}")
    print(f"{'='*60}")
    
    # Select first valid action
    action_mask = obs['action_mask']
    if not action_mask.any():
        print("No valid actions!")
        break
    
    action = action_mask.long().argmax(dim=1)
    print(f"Selected action: {action.item()}")
    
    # Print current state before step
    current = env.current_queries[0]
    non_pad = current[:, 0] != im.padding_idx
    print(f"Current state BEFORE step: {current[non_pad]}")
    
    # Step
    from tensordict import TensorDict
    td = TensorDict({'action': action}, batch_size=env.batch_size, device=device)
    next_obs = env.step(td)
    
    # Print current state after step
    current = env.current_queries[0]
    non_pad = current[:, 0] != im.padding_idx
    print(f"Current state AFTER step: {current[non_pad]}")
    print(f"  Predicates: {current[non_pad, 0]}")
    print(f"  True pred idx: {im.true_pred_idx}")
    print(f"  All true?: {(current[non_pad, 0] == im.true_pred_idx).all() if non_pad.any() else False}")
    
    if 'next' in next_obs.keys():
        reward = next_obs['next']['reward']
        done = next_obs['next']['done']
        obs = TensorDict({
            'action_mask': next_obs['next']['action_mask'],
            'sub_index': next_obs['next']['sub_index'],
            'derived_sub_indices': next_obs['next']['derived_sub_indices'],
        }, batch_size=env.batch_size, device=device)
    else:
        reward = next_obs['reward']
        done = next_obs['done']
        obs = next_obs
    
    print(f"Reward: {reward.item()}")
    print(f"Done: {done.item()}")
    
    if done.item():
        print(f"\n{'='*60}")
        print(f"Episode complete!")
        print(f"  Final reward: {reward.item()}")
        print(f"  Expected: +1.0 (successful proof of positive query)")
        print(f"{'='*60}")
        break

print("\n" + "="*60)
print("Test Complete")
print("="*60)
