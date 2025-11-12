import os
from typing import Optional

import torch
from tensordict import TensorDict

from env import BatchedEnv
from unification_engine import UnificationEngine
from test_unification_countries import setup_countries_kb
from embeddings import EmbedderNonLearnable
from ppo.ppo_model_torchrl import create_torchrl_modules
from ppo.ppo_rollout import RolloutCollector
from model_eval import evaluate_policy

MODEL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENV_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_random_embedder(index_manager, embed_dim: int = 64, device: torch.device = MODEL_DEVICE):
    """Create a random non-learnable embedder compatible with PPO modules."""
    constant_table = torch.randn(index_manager.total_vocab_size + 2, embed_dim, device=device)
    predicate_table = torch.randn(len(index_manager.idx2predicate), embed_dim, device=device)
    return EmbedderNonLearnable(constant_table, predicate_table, device=device)


def build_actor_critic_modules(index_manager, env: BatchedEnv, embed_dim: int = 64, device: torch.device = MODEL_DEVICE):
    """Instantiate TorchRL-compatible actor and value modules."""
    embedder = build_random_embedder(index_manager, embed_dim=embed_dim, device=device)
    actor_module, value_module = create_torchrl_modules(
        embedder=embedder,
        num_actions=env.padding_states,
        embed_dim=embed_dim,
        hidden_dim=64,
        num_layers=2,
        dropout_prob=0.1,
        device=device,
        index_manager=index_manager,
    )
    return actor_module.to(device), value_module.to(device)


def extract_state_atoms(env: BatchedEnv, batch_idx: int, state_idx: int) -> torch.Tensor:
    """Return valid atoms for a specific derived state."""
    state = env.derived_states_batch[batch_idx, state_idx]
    mask = state[:, 0] != env.padding_idx
    return state[mask]


def find_state_with_atom_count(env: BatchedEnv, batch_idx: int, atom_count: int) -> tuple[int, torch.Tensor]:
    """Locate the first derived state that uses the desired number of atoms."""
    count = env.derived_states_counts[batch_idx].item()
    for idx in range(count):
        atoms = extract_state_atoms(env, batch_idx, idx)
        if atoms.shape[0] == atom_count:
            return idx, atoms
    raise AssertionError(f"No state with {atom_count} atoms for batch {batch_idx}")


def make_env(verbose_level: int = 0, device: torch.device = ENV_DEVICE):
    """Instantiate a size-1 BatchedEnv over the miniature countries KB."""
    im, c, p, _ = setup_countries_kb(device=device)
    engine = UnificationEngine.from_index_manager(im, stringifier_params=None)

    queries = torch.tensor(
        [[[p['locatedInCR'], c['tunisia'], c['africa']]]],
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor([1], dtype=torch.long, device=device)
    depths = torch.tensor([3], dtype=torch.long, device=device)

    env = BatchedEnv(
        batch_size=1,
        queries=queries,
        labels=labels,
        query_depths=depths,
        unification_engine=engine,
        sampler=None,
        mode='train',  # deterministic because only one query is available
        max_arity=2,
        padding_idx=im.padding_idx,
        runtime_var_start_index=im.runtime_var_start_index,
        total_vocab_size=im.total_vocab_size,
        padding_atoms=4,
        padding_states=4,
        true_pred_idx=im.true_pred_idx,
        false_pred_idx=im.false_pred_idx,
        atom_stringifier=None,
        corruption_mode=False,
        train_neg_ratio=0.0,
        max_depth=6,
        memory_pruning=False,
        skip_unary_actions=False,
        reward_type=1,
        device=device,
        verbose=verbose_level,
        prover_verbose=0,
    )

    return env, im, c, p


def make_eval_env_with_slots(per_slot_queries, verbose_level: int = 0, device: torch.device = ENV_DEVICE):
    """
    Build an eval-mode BatchedEnv with per-slot schedules.

    per_slot_queries: list where each entry corresponds to a slot (env row) and
    contains tuples of (pred, arg1, arg2) that will be replayed sequentially.
    """
    im, c, p, _ = setup_countries_kb(device=device)
    stringifier_params = im.get_stringifier_params()
    engine = UnificationEngine.from_index_manager(im, stringifier_params=stringifier_params)

    query_rows = []
    labels = []
    depths = []
    per_slot_lengths = []
    for slot in per_slot_queries:
        per_slot_lengths.append(len(slot))
        for pred, arg1, arg2 in slot:
            query_rows.append([p[pred], c[arg1], c[arg2]])
            labels.append(1)
            depths.append(3)

    queries_tensor = torch.tensor(query_rows, dtype=torch.long, device=device).unsqueeze(1)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    depths_tensor = torch.tensor(depths, dtype=torch.long, device=device)

    env = BatchedEnv(
        batch_size=len(per_slot_queries),
        queries=queries_tensor,
        labels=labels_tensor,
        query_depths=depths_tensor,
        unification_engine=engine,
        sampler=None,
        mode='eval',
        max_arity=2,
        padding_idx=im.padding_idx,
        runtime_var_start_index=im.runtime_var_start_index,
        total_vocab_size=im.total_vocab_size,
        padding_atoms=4,
        padding_states=4,
        true_pred_idx=im.true_pred_idx,
        false_pred_idx=im.false_pred_idx,
        stringifier_params=stringifier_params,
        corruption_mode=False,
        train_neg_ratio=0.0,
        max_depth=6,
        memory_pruning=False,
        skip_unary_actions=False,
        reward_type=1,
        device=device,
        verbose=verbose_level,
        prover_verbose=0,
    )

    env.set_eval_dataset(
        queries_tensor,
        labels_tensor,
        depths_tensor,
        per_slot_lengths=torch.tensor(per_slot_lengths, dtype=torch.long, device=device),
    )

    return env, im, c, p, stringifier_params


def build_rollout_env(slot_repeats: int = 3, verbose_level: int = 0, device: torch.device = ENV_DEVICE):
    per_slot = [
        [('locatedInCR', 'tunisia', 'africa')] * slot_repeats,
        [('locatedInCR', 'andorra', 'europe')] * slot_repeats,
    ]
    return make_eval_env_with_slots(per_slot, verbose_level=verbose_level, device=device)


def step_env(env: BatchedEnv, actions) -> TensorDict:
    """
    Feed an action tensor/list/scalar into env._step().

    Accepts:
        - int (for batch_size 1)
        - list/tuple/1D tensor with length batch_size
    """
    if isinstance(actions, int):
        action_tensor = torch.tensor([actions], dtype=torch.long, device=env._device)
    else:
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=env._device)
        if action_tensor.shape[0] != env.batch_size_int:
            raise AssertionError("Action tensor must match env batch size")

    td = TensorDict(
        {'action': action_tensor},
        batch_size=torch.Size([env.batch_size_int]),
        device=env._device,
    )
    return env._step(td)






def test_batched_env_multi_step_and_reset():
    env, im, c, p = make_env()

    # --- Initial reset ---
    obs = env._reset()
    assert env.current_depths[0].item() == 0
    assert env.derived_states_counts[0].item() == 2
    assert torch.equal(
        env.current_queries[0, 0, :3],
        torch.tensor([p['locatedInCR'], c['tunisia'], c['africa']], dtype=torch.long, device=env._device),
    )
    assert set(obs.keys()) >= {'sub_index', 'derived_sub_indices', 'action_mask', 'label', 'done'}

    # --- Step 1: apply the neighbor rule (2-atom state) ---
    rule_idx, rule_state = find_state_with_atom_count(env, 0, atom_count=2)
    assert rule_state[0, 0].item() == p['neighborOf']
    assert rule_state[0, 1].item() == c['tunisia']
    step1 = step_env(env, rule_idx)
    assert not step1['done'][0, 0].item()
    assert env.current_depths[0].item() == 1
    assert torch.equal(env.current_queries[0, :rule_state.shape[0]], rule_state)
    assert env.derived_states_counts[0].item() == 1

    # --- Step 2: satisfy neighbor fact -> locatedInCR(algeria, africa) ---
    fact_state = extract_state_atoms(env, 0, 0)
    assert fact_state.shape[0] == 1
    assert fact_state[0, 0].item() == p['locatedInCR']
    assert fact_state[0, 1].item() == c['algeria']
    assert fact_state[0, 2].item() == c['africa']
    step2 = step_env(env, 0)
    assert not step2['done'][0, 0].item()
    assert env.current_depths[0].item() == 2
    assert torch.equal(
        env.current_queries[0, 0, :3],
        torch.tensor([p['locatedInCR'], c['algeria'], c['africa']], dtype=torch.long, device=env._device),
    )
    assert env.derived_states_counts[0].item() == 2

    # --- Step 3: apply rule again for algeria ---
    rule_idx_2, algeria_rule = find_state_with_atom_count(env, 0, atom_count=2)
    assert algeria_rule[0, 0].item() == p['neighborOf']
    assert algeria_rule[0, 1].item() == c['algeria']
    step3 = step_env(env, rule_idx_2)
    assert not step3['done'][0, 0].item()
    assert env.current_depths[0].item() == 3
    assert torch.equal(env.current_queries[0, :algeria_rule.shape[0]], algeria_rule)
    assert env.derived_states_counts[0].item() == 1

    # --- Step 4: final fact produces True() and reward ---
    final_state = extract_state_atoms(env, 0, 0)
    assert final_state.shape[0] == 1
    assert final_state[0, 0].item() == im.true_pred_idx
    step4 = step_env(env, 0)
    assert step4['done'][0, 0].item()
    assert step4['reward'][0, 0].item() == 1.0
    assert env.current_depths[0].item() == 4
    assert env.derived_states_counts[0].item() == 0
    assert env.current_queries[0, 0, 0].item() == im.true_pred_idx

    # --- Reset again to ensure environment returns to the initial query ---
    env._reset()
    assert env.current_depths[0].item() == 0
    assert env.derived_states_counts[0].item() == 2
    assert torch.equal(
        env.current_queries[0, 0, :3],
        torch.tensor([p['locatedInCR'], c['tunisia'], c['africa']], dtype=torch.long, device=env._device),
    )


def test_batched_env_batch_two_with_partial_resets_and_trace():
    per_slot = [
        [('locatedInCR', 'tunisia', 'africa')] * 2,
        [('locatedInCR', 'andorra', 'europe')] * 2,
    ]
    env, im, c, p, _ = make_eval_env_with_slots(per_slot, verbose_level=0, device=ENV_DEVICE)

    # Full reset: slot 0 -> tunisia, slot 1 -> andorra
    obs = env._reset()
    assert set(obs.keys()) >= {'sub_index', 'derived_sub_indices', 'action_mask'}
    assert env.derived_states_counts.tolist() == [2, 2]

    # Step 1: choose rule expansions (2-atom states) for both envs
    tunisia_rule_idx, _ = find_state_with_atom_count(env, 0, 2)
    andorra_rule_idx, _ = find_state_with_atom_count(env, 1, 2)
    actions_step1 = torch.tensor([tunisia_rule_idx, andorra_rule_idx], device=env._device)
    td1 = step_env(env, actions_step1)
    assert td1['done'].squeeze(-1).tolist() == [False, False]
    assert env.current_depths.tolist() == [1, 1]
    assert env.derived_states_counts.tolist() == [1, 1]

    # Step 2: satisfy neighbor facts simultaneously
    actions_step2 = torch.tensor([0, 0], device=env._device)
    td2 = step_env(env, actions_step2)
    assert td2['done'][0, 0].item() is False
    assert td2['done'][1, 0].item() is True
    assert td2['reward'][1, 0].item() == 1.0
    assert env.current_depths.tolist() == [2, 2]

    # Partial reset for env 1 only to fetch its second scheduled query
    reset_mask = torch.tensor([[False], [True]], dtype=torch.bool, device=env._device)
    obs_partial = env._reset(TensorDict({'_reset': reset_mask}, batch_size=torch.Size([2]), device=env._device))
    assert env.current_depths.tolist() == [2, 0]
    assert env.derived_states_counts.tolist() == [2, 2]
    assert torch.equal(env.current_queries[1, 0, :3], torch.tensor([p['locatedInCR'], c['andorra'], c['europe']], dtype=torch.long, device=env._device))
    assert torch.equal(env.current_queries[0, 0, :3], torch.tensor([p['locatedInCR'], c['algeria'], c['africa']], dtype=torch.long, device=env._device))
    assert set(obs_partial.keys()) >= {'sub_index', 'derived_sub_indices', 'action_mask'}

    # Step 3: continue env 0 (fact) while env 1 applies its rule again
    andorra_rule_idx_2, _ = find_state_with_atom_count(env, 1, 2)
    actions_step3 = torch.tensor([0, andorra_rule_idx_2], device=env._device)
    td3 = step_env(env, actions_step3)
    assert td3['done'].squeeze(-1).tolist() == [False, False]
    assert env.current_depths.tolist() == [3, 1]

    # Step 4: both resolve remaining facts, leading to True() states
    actions_step4 = torch.tensor([0, 0], device=env._device)
    td4 = step_env(env, actions_step4)
    assert td4['done'].squeeze(-1).tolist() == [True, True]
    assert td4['reward'].squeeze(-1).tolist() == [1.0, 1.0]
    assert env.current_queries[0, 0, 0].item() == im.true_pred_idx
    assert env.current_queries[1, 0, 0].item() == im.true_pred_idx

    # Reset env 0 only, ensuring slot scheduling advances without touching env 1
    final_reset_mask = torch.tensor([[True], [False]], dtype=torch.bool, device=env._device)
    env._reset(TensorDict({'_reset': final_reset_mask}, batch_size=torch.Size([2]), device=env._device))
    assert env.current_depths.tolist() == [0, 2]
    assert env.derived_states_counts[0].item() == 2  # env 0 reloaded
    assert env.derived_states_counts[1].item() == 1  # env 1 stays at True()
    assert torch.equal(
        env.current_queries[0, 0, :3],
        torch.tensor([p['locatedInCR'], c['tunisia'], c['africa']], dtype=torch.long, device=env._device),
    )


def test_custom_rollout_collector_with_debug_tracer():
    env, im, c, p, _ = build_rollout_env(slot_repeats=4, verbose_level=3, device=ENV_DEVICE)
    actor, critic = build_actor_critic_modules(im, env, device=MODEL_DEVICE)

    collector = RolloutCollector(
        env=env,
        actor=actor,
        n_envs=env.batch_size_int,
        n_steps=6,
        device=MODEL_DEVICE,
        debug=True,
        debug_action_space=True,
        verbose=2,
    )
    experiences, stats = collector.collect(critic)

    assert len(experiences) == 6
    for step_td in experiences:
        assert 'action' in step_td.keys()
        assert 'next' in step_td.keys()
        nxt = step_td['next']
        assert nxt['reward'].shape[0] == env.batch_size_int
        assert nxt['done'].shape[0] == env.batch_size_int
        assert step_td['action'].device.type == env._device.type

    assert 'episode_info' in stats
    episodes = stats['episode_info']
    assert isinstance(episodes, dict)
    assert 'reward' in episodes and 'length' in episodes
    assert episodes['reward'].numel() > 0
    assert episodes['length'].numel() > 0
    total_lengths = episodes['length'].sum().item()
    assert total_lengths > 0


def test_policy_evaluation_with_custom_rollout_actor():
    env, im, c, p, _ = build_rollout_env(slot_repeats=3, verbose_level=0, device=ENV_DEVICE)
    actor, critic = build_actor_critic_modules(im, env, device=MODEL_DEVICE)

    metrics = evaluate_policy(
        actor,
        env,
        n_eval_episodes=4,
        deterministic=True,
        verbose=0,
    )

    for key in ('rewards', 'success', 'mask', 'lengths'):
        assert key in metrics

    mask = metrics['mask']
    assert mask.any(), "No completed episodes recorded"
    lengths = metrics['lengths'][mask]
    assert torch.all(lengths > 0)
    successes = metrics['success'][mask]
    assert successes.bool().any().item(), "Expected at least one successful episode"
    rewards = metrics['rewards'][mask]
    success_rewards = rewards[successes.bool()]
    assert torch.all(success_rewards >= 1.0)

def main():
    print(f"---------------------Running test_batched_env_multi_step_and_reset()...")
    test_batched_env_multi_step_and_reset()
    print(f"\n\n------------------Running test_batched_env_batch_two_with_partial_resets_and_trace()...")
    test_batched_env_batch_two_with_partial_resets_and_trace()
    print(f"\n\n------------------Running test_custom_rollout_collector_with_debug_tracer()...")
    test_custom_rollout_collector_with_debug_tracer()
    print(f"\n\n------------------Running test_policy_evaluation_with_custom_rollout_actor()...")
    test_policy_evaluation_with_custom_rollout_actor()

if __name__ == "__main__":
    main()
