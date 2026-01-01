"""
Profiler for Fused Multi-Step Evaluation.

Tests architecture with ALL steps in single graph invocation.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cProfile
import pstats
import io
from datetime import datetime
from time import time
from types import SimpleNamespace

import torch


def setup(device, config):
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngineVectorized
    from nn.embeddings import EmbedderLearnable
    from policy import ActorCriticPolicy
    from nn.sampler import Sampler
    from eval_fused import EnvEvalFused, PPOEvalFused
    
    import unification
    unification.COMPILE_MODE = config.compile
    torch.set_float32_matmul_precision('high')
    
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file="train.txt", valid_file="valid.txt", test_file="test.txt",
        rules_file="rules.txt", facts_file="train.txt", corruption_mode='dynamic',
    )
    
    im = IndexManager(
        constants=dh.constants, predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity, padding_atoms=config.padding_atoms,
        device=device, rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)
    
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no, num_relations=im.predicate_no,
        device=device, default_mode='both', seed=config.seed,
        domain2idx=domain2idx, entity2domain=entity2domain,
    )
    
    torch.manual_seed(config.seed)
    
    embedder = EmbedderLearnable(
        n_constants=im.constant_no, n_predicates=im.predicate_no, n_vars=1000,
        max_arity=dh.max_arity, padding_atoms=config.padding_atoms,
        atom_embedder='transe', state_embedder='mean',
        constant_embedding_size=config.atom_embedding_size,
        predicate_embedding_size=config.atom_embedding_size,
        atom_embedding_size=config.atom_embedding_size,
        device=str(device),
    )
    embedder.embed_dim = config.atom_embedding_size
    
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im, padding_atoms=config.padding_atoms,
        max_derived_per_state=config.padding_states,
        end_proof_action=config.end_proof_action,
    )
    
    im.facts_idx = im.rules_idx = im.rule_lens = im.rules_heads_idx = None
    
    test_queries = torch.stack([
        im.atom_to_tensor(q.predicate, q.args[0], q.args[1]) for q in dh.test_queries
    ], dim=0)
    
    env = EnvEvalFused(
        vec_engine=vec_engine, batch_size=config.batch_size,
        padding_atoms=config.padding_atoms, padding_states=config.padding_states,
        max_depth=config.max_depth, device=device,
        runtime_var_start_index=im.constant_no + 1,
        end_proof_action=config.end_proof_action,
    )
    
    policy = ActorCriticPolicy(
        embedder=embedder, embed_dim=config.atom_embedding_size,
        action_dim=config.padding_states, hidden_dim=256, num_layers=8,
        dropout_prob=0.0, device=device, compile_policy=config.compile,
    ).to(device)
    
    ppo = PPOEvalFused(policy, env, device)
    
    return {'ppo': ppo, 'sampler': sampler, 'queries': test_queries}


def warmup(c, config):
    c['ppo'].compile(mode=config.compile_mode)
    c['ppo'].evaluate(c['queries'][:5].to(c['ppo'].device), c['sampler'], n_corruptions=5, corruption_modes=('head',))
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(c, config):
    return c['ppo'].evaluate(
        c['queries'][:config.n_queries].to(c['ppo'].device),
        c['sampler'], n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='family')
    parser.add_argument('--n-queries', type=int, default=100)
    parser.add_argument('--n-corruptions', type=int, default=100)
    parser.add_argument('--corruption-modes', nargs='+', default=['head', 'tail'])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false')
    parser.add_argument('--compile-mode', default='reduce-overhead')
    args = parser.parse_args()
    
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
        n_queries=args.n_queries, n_corruptions=args.n_corruptions,
        corruption_modes=args.corruption_modes, batch_size=args.batch_size,
        padding_atoms=6, padding_states=120, max_depth=20,
        end_proof_action=True, max_total_vars=100, atom_embedding_size=250,
        seed=42, compile=args.compile, compile_mode=args.compile_mode,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_modes = len(config.corruption_modes)
    total = config.n_queries * (1 + config.n_corruptions) * n_modes
    
    print(f"Fused Eval Profile | {datetime.now()}")
    print(f"Candidates: {total}, Batch: {config.batch_size}")
    
    print("\nSetup...")
    t0 = time()
    c = setup(device, config)
    print(f"Init: {time()-t0:.2f}s")
    
    print("\nWarmup...")
    t0 = time()
    warmup(c, config)
    warmup_t = time() - t0
    print(f"Warmup: {warmup_t:.2f}s")
    
    print("\nProfiling...")
    prof = cProfile.Profile()
    prof.enable()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time()
    results = run(c, config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime = time() - t0
    prof.disable()
    
    ms = (runtime / total) * 1000
    print(f"\n{'='*50}")
    print(f"Runtime:      {runtime:.4f}s")
    print(f"ms/candidate: {ms:.3f}  (target: 0.3)")
    print(f"MRR:          {results.get('MRR', 0):.4f}")
    print(f"Hits@10:      {results.get('Hits@10', 0):.4f}")
    
    print(f"\n{'='*50}")
    print("Top 20 by cumulative time:")
    s = io.StringIO()
    pstats.Stats(prof, stream=s).strip_dirs().sort_stats('cumulative').print_stats(20)
    print(s.getvalue())


if __name__ == '__main__':
    main()
