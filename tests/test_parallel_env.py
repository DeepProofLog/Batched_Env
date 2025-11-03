import sys
import os
# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from types import SimpleNamespace
from tensordict import TensorDict

from dataset import DataHandler
from index_manager import IndexManager
from env_factory import create_environments
from neg_sampling import get_sampler


def test_parallel_env_step_shapes():
    # Minimal args for environment creation
    args = SimpleNamespace(
        n_envs=8,
        n_eval_envs=2,
        seed_run_i=0,
        corruption_mode='dynamic',
        corruption_scheme=['head', 'tail'],
        train_neg_ratio=1,
        max_depth=20,
        memory_pruning=True,
        endt_action=False,
        endf_action=True,
        skip_unary_actions=True,
        padding_atoms=6,
        padding_states=20,
        engine='python',
        engine_strategy='cmp',
        kge_action=False,
        reward_type=4,
        verbose_env=0,
        verbose_prover=0,
        gamma=0.99,
        pbrs_beta=0.0,
        pbrs_gamma=0.99,
    )

    # Build data and index
    dh = DataHandler(
        dataset_name='countries_s3',
        base_path='data',
        janus_file=None,
        train_file='train.txt',
        valid_file='valid.txt',
        test_file='test.txt',
        rules_file='rules.txt',
        facts_file='train.txt',
        n_eval_queries=16,
        corruption_mode='dynamic',  # Enable corruption mode to load domain mappings
    )

    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_vars=100,
        rules=dh.rules,
        padding_atoms=args.padding_atoms,
        max_arity=dh.max_arity,
        device='cpu',
    )
    im.build_fact_index(dh.facts)

    # Create sampler (required by environments)
    dh.sampler = get_sampler(
        data_handler=dh,
        index_manager=im,
        corruption_scheme=args.corruption_scheme,
        device='cpu',
    )

    # Create parallel environments
    print(f"Creating parallel environments with {args.n_envs} workers...")
    train_env, eval_env, cb_env = create_environments(args, dh, im)
    print(f"Parallel environments created successfully!")

    # Reset the parallel env (should return a batched tensordict)
    print("Calling reset on train_env...")
    td = train_env.reset()
    print('reset batch_size:', td.batch_size)

    # Prepare a simple zero action for all envs
    n_envs = int(getattr(args, 'n_envs'))
    action_td = TensorDict({'action': torch.zeros(n_envs, dtype=torch.long)}, batch_size=torch.Size([n_envs]))

    try:
        next_td = train_env.step(action_td)
        print('step returned batch_size:', next_td.batch_size)
    except Exception as e:
        # Surface helpful debugging info
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    test_parallel_env_step_shapes()
    print('\n✓ Test passed: parallel env step shapes are consistent')
    
    # Additional regression tests with different env counts
    print('\n--- Running regression tests with different env counts ---')
    for n in [4, 8]:
        print(f'Testing with n_envs={n}...', end=' ')
        args_test = SimpleNamespace(
            n_envs=n,
            n_eval_envs=2,
            seed_run_i=0,
            corruption_mode='dynamic',
            corruption_scheme=['head', 'tail'],
            train_neg_ratio=1,
            max_depth=20,
            memory_pruning=True,
            endt_action=False,
            endf_action=True,
            skip_unary_actions=True,
            padding_atoms=6,
            padding_states=20,
            engine='python',
            engine_strategy='cmp',
            kge_action=False,
            reward_type=4,
            verbose_env=0,
            verbose_prover=0,
            gamma=0.99,
            pbrs_beta=0.0,
            pbrs_gamma=0.99,
        )
        
        from dataset import DataHandler
        from index_manager import IndexManager
        from env_factory import create_environments
        from neg_sampling import get_sampler
        
        dh_test = DataHandler(
            dataset_name='countries_s3',
            base_path='data',
            janus_file=None,
            train_file='train.txt',
            valid_file='valid.txt',
            test_file='test.txt',
            rules_file='rules.txt',
            facts_file='train.txt',
            n_eval_queries=16,
            corruption_mode='dynamic',
        )
        
        im_test = IndexManager(
            constants=dh_test.constants,
            predicates=dh_test.predicates,
            max_total_vars=100,
            rules=dh_test.rules,
            padding_atoms=args_test.padding_atoms,
            max_arity=dh_test.max_arity,
            device='cpu',
        )
        im_test.build_fact_index(dh_test.facts)
        
        dh_test.sampler = get_sampler(
            data_handler=dh_test,
            index_manager=im_test,
            corruption_scheme=args_test.corruption_scheme,
            device='cpu',
        )
        
        train_env_test, _, _ = create_environments(args_test, dh_test, im_test)
        td_test = train_env_test.reset()
        action_test = TensorDict({'action': torch.zeros(n, dtype=torch.long)}, batch_size=torch.Size([n]))
        next_td_test = train_env_test.step(action_test)
        
        assert td_test.batch_size == torch.Size([n]), f"Expected batch_size {torch.Size([n])}, got {td_test.batch_size}"
        assert next_td_test.batch_size == torch.Size([n]), f"Expected batch_size {torch.Size([n])}, got {next_td_test.batch_size}"
        print('✓')
    
    print('\n✅ All regression tests passed!')
