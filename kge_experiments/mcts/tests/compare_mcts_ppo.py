"""
Compare MCTS vs PPO performance under equal conditions.

Runs both algorithms with identical:
- Batch size, dataset, max_depth
- Warmup methodology (cuda.synchronize)
- Timing approach

Reports speedup ratios for training and evaluation.

Usage:
    python kge_experiments/mcts/tests/compare_mcts_ppo.py
    python kge_experiments/mcts/tests/compare_mcts_ppo.py --n-queries 100 --n-corruptions 100
"""

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'kge_experiments'))

import argparse
from datetime import datetime
from time import time
from types import SimpleNamespace

import torch


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        return any(getattr(f, 'isatty', lambda: False)() for f in self.files)


def setup_mcts(device, config):
    """Setup MCTS trainer."""
    from kge_experiments.mcts.config import MCTSConfig
    from kge_experiments.mcts.trainer import MuZeroTrainer
    from kge_experiments.builder import create_env, create_policy, KGEConfig

    kge_config = KGEConfig(
        dataset=config.dataset,
        data_path=config.data_path,
        device=str(device),
        n_envs=config.batch_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
    )
    env = create_env(kge_config)
    policy = create_policy(kge_config, env)

    mcts_config = MCTSConfig(
        num_simulations=config.num_simulations,
        max_episode_steps=config.max_depth,
        mcts_batch_size=config.batch_size,
        max_actions=env.padding_states,
        use_batched_mcts=True,
        device=str(device),
        learning_rate=config.learning_rate,
        discount=config.gamma,
        compile=True,  # Enable CUDA graph compilation
    )

    trainer = MuZeroTrainer(config=mcts_config, env=env, policy=policy)
    return {'trainer': trainer, 'env': env, 'policy': policy, 'kge_config': kge_config}


def setup_ppo(device, config):
    """Setup PPO trainer."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngineVectorized
    from nn.embeddings import EmbedderLearnable
    from policy import ActorCriticPolicy
    from nn.sampler import Sampler
    from env import EnvVec
    from ppo import PPO

    dh = DataHandler(
        dataset_name=config.dataset, base_path=config.data_path,
        train_file="train.txt", valid_file="valid.txt", test_file="test.txt",
        rules_file="rules.txt", facts_file="train.txt",
    )

    im = IndexManager(
        constants=dh.constants, predicates=dh.predicates,
        max_total_runtime_vars=100, max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms, device=device, rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)

    d2i, e2d = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no, num_relations=im.predicate_no,
        device=device, default_mode='both', seed=42,
        domain2idx=d2i, entity2domain=e2d,
    )

    torch.manual_seed(42)

    embedder = EmbedderLearnable(
        n_constants=im.constant_no, n_predicates=im.predicate_no, n_vars=1000,
        max_arity=dh.max_arity, padding_atoms=config.padding_atoms,
        atom_embedder='transe', state_embedder='mean',
        constant_embedding_size=250, predicate_embedding_size=250, atom_embedding_size=250,
        device=str(device),
    )
    embedder.embed_dim = 250

    vec_engine = UnificationEngineVectorized.from_index_manager(
        im, padding_atoms=config.padding_atoms,
        max_derived_per_state=config.padding_states, end_proof_action=True,
    )

    def convert_queries(queries):
        return torch.stack([im.atom_to_tensor(q.predicate, q.args[0], q.args[1]) for q in queries])

    train_queries = convert_queries(dh.train_queries)
    test_queries = convert_queries(dh.test_queries)

    env = EnvVec(
        vec_engine=vec_engine, batch_size=config.batch_size,
        padding_atoms=config.padding_atoms, padding_states=config.padding_states,
        max_depth=config.max_depth, end_proof_action=True,
        runtime_var_start_index=im.constant_no + 1, device=device,
        memory_pruning=True, sampler=sampler,
        train_queries=train_queries, valid_queries=test_queries,
        skip_unary_actions=True,
    )

    policy = ActorCriticPolicy(
        embedder=embedder, embed_dim=250, action_dim=config.padding_states,
        hidden_dim=256, num_layers=8, dropout_prob=0.0, device=device,
        compile_policy=True, use_amp=True,
    ).to(device)

    # Ensure batch_size divides rollout_buffer_size
    rollout_size = config.batch_size * config.n_steps
    ppo_batch = min(1024, rollout_size)
    while rollout_size % ppo_batch != 0 and ppo_batch > 1:
        ppo_batch -= 1

    ppo_config = SimpleNamespace(
        n_envs=config.batch_size, batch_size_env=config.batch_size,
        n_steps=config.n_steps, n_epochs=5,
        batch_size=ppo_batch,
        learning_rate=config.learning_rate, gamma=config.gamma,
        gae_lambda=0.95, clip_range=0.2, ent_coef=0.2, vf_coef=0.5, max_grad_norm=0.5,
        padding_atoms=config.padding_atoms, padding_states=config.padding_states,
        max_depth=config.max_depth, seed=42, verbose=False, parity=False,
        compile=True, use_amp=True,
    )

    ppo = PPO(policy, env, ppo_config, device=device)
    return {
        'ppo': ppo, 'env': env, 'policy': policy,
        'sampler': sampler, 'test_queries': test_queries, 'dh': dh, 'im': im,
    }


def profile_mcts_train(mcts_components, config):
    """Profile MCTS training."""
    trainer = mcts_components['trainer']
    trainer.env.train()

    # Warmup
    trainer.collect_episodes_batched(num_steps=config.batch_size * 2, add_noise=True)
    torch.cuda.synchronize()

    # Timed run
    torch.cuda.synchronize()
    start = time()

    total_steps = 0
    for _ in range(config.n_iterations):
        stats = trainer.collect_episodes_batched(num_steps=config.n_steps, add_noise=True)
        total_steps += stats.get("steps_collected", config.n_steps)
        if len(trainer.replay_buffer) >= trainer.config.min_buffer_size:
            trainer.train_step()

    torch.cuda.synchronize()
    elapsed = time() - start

    return {
        'runtime': elapsed,
        'steps': total_steps,
        'steps_per_sec': total_steps / elapsed,
    }


def profile_ppo_train(ppo_components, config):
    """Profile PPO training."""
    ppo = ppo_components['ppo']

    # Warmup
    ppo.learn(total_timesteps=config.n_steps)
    torch.cuda.synchronize()

    # Timed run
    torch.cuda.synchronize()
    start = time()

    ppo.learn(total_timesteps=config.n_steps * config.n_iterations, reset_num_timesteps=False)

    torch.cuda.synchronize()
    elapsed = time() - start

    total_steps = config.n_steps * config.n_iterations
    return {
        'runtime': elapsed,
        'steps': total_steps,
        'steps_per_sec': total_steps / elapsed,
    }


def profile_mcts_eval(mcts_components, config, sampler, test_queries):
    """Profile MCTS evaluation."""
    trainer = mcts_components['trainer']
    trainer.env.train()

    # Warmup
    trainer.evaluate_batched(
        queries=test_queries[:5], sampler=sampler,
        n_corruptions=5, corruption_modes=('head',), verbose=False,
    )
    torch.cuda.synchronize()

    # Timed run
    torch.cuda.synchronize()
    start = time()

    results = trainer.evaluate_batched(
        queries=test_queries[:config.n_queries], sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=('head', 'tail'), verbose=False,
    )

    torch.cuda.synchronize()
    elapsed = time() - start

    total_candidates = config.n_queries * (1 + config.n_corruptions) * 2
    return {
        'runtime': elapsed,
        'candidates': total_candidates,
        'ms_per_candidate': (elapsed / total_candidates) * 1000,
        'mrr': results['MRR'],
    }


def profile_ppo_eval(ppo_components, config):
    """Profile PPO evaluation."""
    ppo = ppo_components['ppo']
    sampler = ppo_components['sampler']
    test_queries = ppo_components['test_queries']

    # Warmup
    ppo.evaluate(
        test_queries[:5].to(ppo.device), sampler,
        n_corruptions=5, corruption_modes=('head',),
    )
    torch.cuda.synchronize()

    # Timed run
    torch.cuda.synchronize()
    start = time()

    results = ppo.evaluate(
        test_queries[:config.n_queries].to(ppo.device), sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=('head', 'tail'), verbose=False,
    )

    torch.cuda.synchronize()
    elapsed = time() - start

    total_candidates = config.n_queries * (1 + config.n_corruptions) * 2
    return {
        'runtime': elapsed,
        'candidates': total_candidates,
        'ms_per_candidate': (elapsed / total_candidates) * 1000,
        'mrr': results['MRR'],
    }


def main():
    parser = argparse.ArgumentParser(description='Compare MCTS vs PPO performance')
    parser.add_argument('--dataset', type=str, default='family')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--n-iterations', type=int, default=5)
    parser.add_argument('--n-queries', type=int, default=100)
    parser.add_argument('--n-corruptions', type=int, default=100)
    parser.add_argument('--num-simulations', type=int, default=25)
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare_mcts_ppo.txt')
    log_file = open(output_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_float32_matmul_precision('high')

        config = SimpleNamespace(
            dataset=args.dataset,
            data_path=os.path.join(os.path.dirname(__file__), '..', '..', 'data'),
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_iterations=args.n_iterations,
            n_queries=args.n_queries,
            n_corruptions=args.n_corruptions,
            num_simulations=args.num_simulations,
            learning_rate=5e-5,
            gamma=0.99,
            padding_atoms=6,
            padding_states={'family': 130, 'countries_s3': 20}.get(args.dataset, 130),
            max_depth=20,
        )

        print(f"MCTS vs PPO Performance Comparison")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Dataset: {config.dataset}")
        print(f"\nConfiguration:")
        print(f"  Batch size: {config.batch_size}")
        print(f"  N steps: {config.n_steps}")
        print(f"  N iterations: {config.n_iterations}")
        print(f"  MCTS simulations: {config.num_simulations}")
        print(f"  N queries (eval): {config.n_queries}")
        print(f"  N corruptions: {config.n_corruptions}")

        results = {'mcts': {}, 'ppo': {}}

        # =====================================================================
        # TRAINING COMPARISON
        # =====================================================================
        if not args.eval_only:
            print(f"\n{'='*70}")
            print("TRAINING COMPARISON")
            print(f"{'='*70}")

            print("\nSetting up MCTS...")
            mcts_components = setup_mcts(device, config)

            print("Setting up PPO...")
            ppo_components = setup_ppo(device, config)

            print("\nProfiling MCTS training...")
            results['mcts']['train'] = profile_mcts_train(mcts_components, config)

            print("Profiling PPO training...")
            results['ppo']['train'] = profile_ppo_train(ppo_components, config)

            mcts_train = results['mcts']['train']
            ppo_train = results['ppo']['train']
            slowdown = ppo_train['steps_per_sec'] / mcts_train['steps_per_sec']

            print(f"\n{'Metric':<25} {'MCTS':>15} {'PPO':>15} {'Slowdown':>15}")
            print(f"{'-'*70}")
            print(f"{'Runtime (s)':<25} {mcts_train['runtime']:>15.3f} {ppo_train['runtime']:>15.3f}")
            print(f"{'Steps':<25} {mcts_train['steps']:>15} {ppo_train['steps']:>15}")
            print(f"{'Steps/sec':<25} {mcts_train['steps_per_sec']:>15.1f} {ppo_train['steps_per_sec']:>15.1f} {slowdown:>14.1f}x")

            print(f"\n=> MCTS training is {slowdown:.1f}x slower than PPO")

        # =====================================================================
        # EVALUATION COMPARISON
        # =====================================================================
        if not args.train_only:
            print(f"\n{'='*70}")
            print("EVALUATION COMPARISON")
            print(f"{'='*70}")

            if args.eval_only:
                print("\nSetting up MCTS...")
                mcts_components = setup_mcts(device, config)
                print("Setting up PPO...")
                ppo_components = setup_ppo(device, config)

            # Get sampler for MCTS eval
            from kge_experiments.data_handler import DataHandler
            from kge_experiments.nn.sampler import Sampler
            from kge_experiments.index_manager import IndexManager

            dh = DataHandler(dataset_name=config.dataset, base_path=config.data_path)
            im = IndexManager(
                constants=dh.constants, predicates=dh.predicates,
                max_total_runtime_vars=100, max_arity=dh.max_arity,
                padding_atoms=config.padding_atoms, device=device, rules=dh.rules,
            )
            dh.materialize_indices(im=im, device=device)

            test_queries = torch.stack([
                im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
                for q in dh.test_queries
            ]).to(device)

            d2i, e2d = dh.get_sampler_domain_info()
            sampler = Sampler.from_data(
                all_known_triples_idx=dh.all_known_triples_idx,
                num_entities=im.constant_no, num_relations=im.predicate_no,
                device=device, default_mode='both', seed=42,
                domain2idx=d2i, entity2domain=e2d,
            )

            print("\nProfiling MCTS evaluation...")
            results['mcts']['eval'] = profile_mcts_eval(mcts_components, config, sampler, test_queries)

            print("Profiling PPO evaluation...")
            results['ppo']['eval'] = profile_ppo_eval(ppo_components, config)

            mcts_eval = results['mcts']['eval']
            ppo_eval = results['ppo']['eval']
            slowdown = mcts_eval['ms_per_candidate'] / ppo_eval['ms_per_candidate']

            print(f"\n{'Metric':<25} {'MCTS':>15} {'PPO':>15} {'Slowdown':>15}")
            print(f"{'-'*70}")
            print(f"{'Runtime (s)':<25} {mcts_eval['runtime']:>15.3f} {ppo_eval['runtime']:>15.3f}")
            print(f"{'Candidates':<25} {mcts_eval['candidates']:>15} {ppo_eval['candidates']:>15}")
            print(f"{'ms/candidate':<25} {mcts_eval['ms_per_candidate']:>15.4f} {ppo_eval['ms_per_candidate']:>15.4f} {slowdown:>14.1f}x")
            print(f"{'MRR':<25} {mcts_eval['mrr']:>15.4f} {ppo_eval['mrr']:>15.4f}")

            print(f"\n=> MCTS evaluation is {slowdown:.1f}x slower than PPO")

        # =====================================================================
        # SUMMARY
        # =====================================================================
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        if 'train' in results['mcts'] and 'train' in results['ppo']:
            train_slowdown = results['ppo']['train']['steps_per_sec'] / results['mcts']['train']['steps_per_sec']
            print(f"Training:   MCTS is {train_slowdown:.1f}x slower than PPO")

        if 'eval' in results['mcts'] and 'eval' in results['ppo']:
            eval_slowdown = results['mcts']['eval']['ms_per_candidate'] / results['ppo']['eval']['ms_per_candidate']
            print(f"Evaluation: MCTS is {eval_slowdown:.1f}x slower than PPO")

        print(f"\nResults saved to {output_path}")

    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()
