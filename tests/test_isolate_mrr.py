"""
Test to isolate MRR mismatch causes.

This script tests different combinations to find root causes:
1. Original with memory_pruning=False
2. Original with skip_unary_actions=False
3. Original with both disabled (should match optimized behavior more closely)
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from types import SimpleNamespace

def setup_components(device: torch.device, config: SimpleNamespace):
    """Initialize all components needed for evaluation."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngine, set_compile_mode
    from unification_vectorized import UnificationEngineVectorized
    from env import BatchedEnv
    from embeddings import EmbedderLearnable as TensorEmbedder
    from model import ActorCriticPolicy as TensorPolicy
    from sampler import Sampler
    
    set_compile_mode(False)
    
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        corruption_mode='dynamic',
    )
    
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        device=device,
        rules=dh.rules,
    )
    
    dh.materialize_indices(im=im, device=device)
    
    default_mode = 'tail' if 'countries' in config.dataset else 'both'
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode=default_mode,
        seed=config.seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    torch.manual_seed(config.seed)
    
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=1000,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder='transe',
        state_embedder='mean',
        constant_embedding_size=config.atom_embedding_size,
        predicate_embedding_size=config.atom_embedding_size,
        atom_embedding_size=config.atom_embedding_size,
        device=str(device),
    )
    embedder.embed_dim = config.atom_embedding_size
    
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    base_engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.padding_states,
    )
    base_engine.index_manager = im
    
    def convert_queries_unpadded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensors.append(atom)
        return torch.stack(tensors, dim=0)
    
    def convert_queries_padded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            padded = torch.full((config.padding_atoms, 3), im.padding_idx, dtype=torch.long, device=device)
            padded[0] = atom
            tensors.append(padded)
        return torch.stack(tensors, dim=0)
    
    test_queries_unpadded = convert_queries_unpadded(dh.test_queries)
    test_queries_padded = convert_queries_padded(dh.test_queries)
    
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=config.padding_states,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        compile_policy=False,
    ).to(device)
    
    return {
        'policy': policy,
        'sampler': sampler,
        'dh': dh,
        'im': im,
        'base_engine': base_engine,
        'test_queries_unpadded': test_queries_unpadded,
        'test_queries_padded': test_queries_padded,
        'stringifier_params': stringifier_params,
    }


def create_env(components, config, device, memory_pruning=True, skip_unary_actions=False):
    """Create an evaluation environment with specific settings."""
    from env import BatchedEnv
    
    im = components['im']
    base_engine = components['base_engine']
    test_queries_padded = components['test_queries_padded']
    
    return BatchedEnv(
        batch_size=config.batch_size_env,
        queries=test_queries_padded[:config.batch_size_env],
        labels=torch.ones(config.batch_size_env, dtype=torch.long, device=device),
        query_depths=torch.ones(config.batch_size_env, dtype=torch.long, device=device),
        unification_engine=base_engine,
        mode='eval',
        max_depth=config.max_depth,
        memory_pruning=memory_pruning,
        use_exact_memory=False,
        skip_unary_actions=skip_unary_actions,
        end_proof_action=config.end_proof_action,
        reward_type=config.reward_type,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=0,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + config.max_total_vars,
        sample_deterministic_per_env=False,
    )


def run_original_eval_with_env(components, config, env, seed=None):
    """Run original eval_corruptions with a specific env."""
    from model_eval import eval_corruptions
    
    actor = components['policy']
    sampler = components['sampler']
    queries = components['test_queries_unpadded'][:config.n_test_queries]
    
    if seed is not None:
        sampler.rng = np.random.RandomState(seed)
    
    return eval_corruptions(
        actor=actor,
        env=env,
        queries=queries,
        sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        verbose=False,
    )


def main():
    config = SimpleNamespace(
        dataset='family',
        data_path='./data/',
        n_test_queries=20,
        n_corruptions=50,
        batch_size_env=100,
        corruption_modes=['both'],
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        end_proof_action=True,
        reward_type=0,
        max_total_vars=100,
        atom_embedding_size=250,
        seed=0,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    components = setup_components(device, config)
    
    seeds = [0, 1, 2]
    results = {}
    
    # Test configurations
    configs_to_test = [
        ("Original (memory=True, skip_unary=False)", True, False),  # Default original
        ("Original (memory=False, skip_unary=False)", False, False),  # Disable memory
        ("Original (memory=True, skip_unary=True)", True, True),   # Enable skip_unary
        ("Original (memory=False, skip_unary=True)", False, True),  # Both disabled
    ]
    
    for name, memory_pruning, skip_unary_actions in configs_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        mrrs = []
        for seed in seeds:
            torch.cuda.empty_cache()
            env = create_env(components, config, device, 
                           memory_pruning=memory_pruning, 
                           skip_unary_actions=skip_unary_actions)
            res = run_original_eval_with_env(components, config, env, seed=seed)
            mrrs.append(res['MRR'])
            print(f"  Seed {seed}: MRR = {res['MRR']:.4f}")
        
        avg_mrr = np.mean(mrrs)
        std_mrr = np.std(mrrs)
        results[name] = (avg_mrr, std_mrr)
        print(f"  Average: {avg_mrr:.4f} ± {std_mrr:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Configuration':<50} {'MRR (avg ± std)':<20}")
    print("-"*70)
    for name, (avg, std) in results.items():
        print(f"{name:<50} {avg:.4f} ± {std:.4f}")
    
    print("\n" + "="*60)
    print("OPTIMIZED BASELINE (from previous test)")
    print("="*60)
    print("Optimized (no memory, no skip_unary): 0.4641 ± 0.0099")
    
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    baseline = results["Original (memory=True, skip_unary=False)"][0]
    no_memory = results["Original (memory=False, skip_unary=False)"][0]
    
    memory_impact = baseline - no_memory
    print(f"Memory pruning impact: {memory_impact:+.4f} MRR")
    
    remaining_gap = no_memory - 0.4641  # vs optimized
    print(f"Remaining gap after disabling memory: {remaining_gap:+.4f} MRR")
    print(f"This remaining gap must be due to other implementation differences.")


if __name__ == '__main__':
    main()
