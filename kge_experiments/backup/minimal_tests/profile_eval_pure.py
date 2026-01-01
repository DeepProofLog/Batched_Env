"""
Simple Evaluation Profiler - Pure Functional Version.

Uses pure functional step (returns new tensors) + copy-outside pattern.
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


def setup_components(device: torch.device, config: SimpleNamespace):
    """Initialize components."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngineVectorized
    from nn.embeddings import EmbedderLearnable as TensorEmbedder
    from policy import ActorCriticPolicy as TensorPolicy
    from nn.sampler import Sampler
    from env_evaluate_pure import EnvEvalPure
    from ppo_evaluate_pure import PPOEvalPure
    
    import unification
    unification.COMPILE_MODE = config.compile
    torch.set_float32_matmul_precision('high')
    
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
    
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=config.padding_atoms,
        max_derived_per_state=config.padding_states,
        end_proof_action=config.end_proof_action,
    )
    
    im.facts_idx = None
    im.rules_idx = None
    im.rule_lens = None
    im.rules_heads_idx = None
    
    def convert_queries(queries):
        return torch.stack([im.atom_to_tensor(q.predicate, q.args[0], q.args[1]) for q in queries], dim=0)
    
    test_queries = convert_queries(dh.test_queries)
    
    env = EnvEvalPure(
        vec_engine=vec_engine,
        batch_size=config.batch_size,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        end_proof_action=config.end_proof_action,
    )
    
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=config.padding_states,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        compile_policy=config.compile,
    ).to(device)
    
    ppo = PPOEvalPure(policy, env, device)
    
    return {'ppo': ppo, 'sampler': sampler, 'test_queries': test_queries}


def warmup(components, config):
    ppo = components['ppo']
    queries = components['test_queries'][:5].to(ppo.device)
    sampler = components['sampler']
    
    start = time()
    ppo.compile(mode=config.compile_mode)
    _ = ppo.evaluate(queries=queries, sampler=sampler, n_corruptions=5, corruption_modes=('head',))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time() - start


def run_eval(components, config):
    ppo = components['ppo']
    queries = components['test_queries'][:config.n_queries].to(ppo.device)
    return ppo.evaluate(
        queries=queries,
        sampler=components['sampler'],
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
    )


def profile(config):
    output_path = os.path.join(os.path.dirname(__file__), 'profile_eval_pure_results.txt')
    
    with open(output_path, 'w') as log:
        def p(msg):
            print(msg)
            log.write(msg + '\n')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        p(f"Pure Functional Eval Profile")
        p(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        p(f"Device: {device}")
        p(f"Queries: {config.n_queries}, Corruptions: {config.n_corruptions}")
        
        n_modes = len(config.corruption_modes)
        total_candidates = config.n_queries * (1 + config.n_corruptions) * n_modes
        p(f"Total candidates: {total_candidates}")
        
        p("\nSetting up...")
        init_start = time()
        components = setup_components(device, config)
        p(f"Init: {time() - init_start:.2f}s")
        
        p("\nWarmup...")
        warmup_time = warmup(components, config)
        p(f"Warmup: {warmup_time:.2f}s")
        
        p("\nProfiling...")
        profiler = cProfile.Profile()
        profiler.enable()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time()
        results = run_eval(components, config)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime = time() - start
        
        profiler.disable()
        
        ms_per_candidate = (runtime / total_candidates) * 1000
        
        p(f"\n{'='*60}")
        p("RESULTS")
        p(f"{'='*60}")
        p(f"Warmup:       {warmup_time:.4f}s")
        p(f"Runtime:      {runtime:.4f}s")
        p(f"ms/candidate: {ms_per_candidate:.3f}  (target: 0.3)")
        p(f"MRR:          {results.get('MRR', 0):.4f}")
        p(f"Hits@10:      {results.get('Hits@10', 0):.4f}")
        
        p(f"\n{'='*60}")
        p("PROFILE - Cumulative")
        p(f"{'='*60}")
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).strip_dirs().sort_stats('cumulative').print_stats(25)
        p(s.getvalue())
        
        p(f"\n{'='*60}")
        p("PROFILE - Total Time")
        p(f"{'='*60}")
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).strip_dirs().sort_stats('tottime').print_stats(25)
        p(s.getvalue())
        
        p(f"\nSaved to {output_path}")


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
        n_queries=args.n_queries,
        n_corruptions=args.n_corruptions,
        corruption_modes=args.corruption_modes,
        batch_size=args.batch_size,
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        end_proof_action=True,
        max_total_vars=100,
        atom_embedding_size=250,
        seed=42,
        compile=args.compile,
        compile_mode=args.compile_mode,
    )
    
    profile(config)


if __name__ == '__main__':
    main()
