"""Profile streaming evaluation V2."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cProfile
import pstats
import io
from time import time
from types import SimpleNamespace

import torch


def setup(device, config):
    """Setup using existing PPO/EnvVec infrastructure."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngineVectorized
    from nn.embeddings import EmbedderLearnable
    from policy import ActorCriticPolicy
    from nn.sampler import Sampler
    from env import EnvVec
    from ppo import PPO
    from eval_streaming_v2 import EvalStreamingV2
    
    import unification
    unification.COMPILE_MODE = config.compile
    torch.set_float32_matmul_precision('high')
    
    dh = DataHandler(
        dataset_name=config.dataset, base_path=config.data_path,
        train_file="train.txt", valid_file="valid.txt", test_file="test.txt",
        rules_file="rules.txt", facts_file="train.txt", corruption_mode='dynamic',
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
        device=device, default_mode='both', seed=42, domain2idx=d2i, entity2domain=e2d,
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
    
    queries = torch.stack([im.atom_to_tensor(q.predicate, q.args[0], q.args[1]) for q in dh.test_queries], 0)
    
    policy = ActorCriticPolicy(
        embedder=embedder, embed_dim=250, action_dim=config.padding_states,
        hidden_dim=256, num_layers=8, dropout_prob=0.0, device=device, compile_policy=config.compile,
    ).to(device)
    
    env = EnvVec(
        vec_engine=vec_engine, batch_size=config.batch_size,
        padding_atoms=config.padding_atoms, padding_states=config.padding_states,
        max_depth=config.max_depth, device=device,
        runtime_var_start_index=im.constant_no + 1,
        sampler=sampler, negative_ratio=1.0,
        end_proof_action=True, memory_pruning=True,
    )
    
    ppo = PPO(
        policy=policy, env=env, sampler=sampler,
        steps_per_rollout=100, num_rollouts=1,
        lr=1e-4, gamma=0.99, clip_eps=0.2,
        entropy_coef=0.01, value_coef=0.5,
        device=device, compile=config.compile,
    )
    
    eval_stream = EvalStreamingV2(ppo=ppo, device=device)
    
    return {'eval': eval_stream, 'sampler': sampler, 'queries': queries, 'env': env}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-queries', type=int, default=100)
    parser.add_argument('--n-corruptions', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false')
    args = parser.parse_args()
    
    config = SimpleNamespace(
        dataset='family',
        data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
        padding_atoms=6, padding_states=120, max_depth=20,
        batch_size=args.batch_size, compile=args.compile,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = args.n_queries * (1 + args.n_corruptions) * 2
    
    print(f"Streaming Eval V2 | Candidates: {total}, Batch: {config.batch_size}")
    
    print("Setup...")
    c = setup(device, config)
    
    print("Compile + warmup...")
    t0 = time()
    c['eval'].compile(mode='reduce-overhead')
    # Quick warmup
    c['eval'].evaluate(c['queries'][:5].to(device), c['sampler'], n_corruptions=5, corruption_modes=('head',), verbose=False)
    torch.cuda.synchronize()
    warmup = time() - t0
    print(f"Warmup: {warmup:.2f}s")
    
    print("Profiling...")
    prof = cProfile.Profile()
    prof.enable()
    torch.cuda.synchronize()
    t0 = time()
    results = c['eval'].evaluate(
        c['queries'][:args.n_queries].to(device),
        c['sampler'], n_corruptions=args.n_corruptions,
        corruption_modes=('head', 'tail'), verbose=True,
    )
    torch.cuda.synchronize()
    runtime = time() - t0
    prof.disable()
    
    ms = (runtime / total) * 1000
    print(f"\n{'='*50}")
    print(f"Runtime:      {runtime:.4f}s")
    print(f"ms/candidate: {ms:.4f}  (target: 0.3)")
    print(f"MRR:          {results['MRR']:.4f}")
    
    print(f"\nTop 20:")
    s = io.StringIO()
    pstats.Stats(prof, stream=s).strip_dirs().sort_stats('tottime').print_stats(20)
    print(s.getvalue())


if __name__ == '__main__':
    main()
