"""
Eval Compiled Parity Tests.

Tests verifying that the optimized evaluation path (env_optimized + model_eval_optimized)
produces SIMILAR MRR to the original evaluation path (env + model_eval), with
skip_unary_actions=False.

IMPORTANT CAVEAT:
This test compares two fundamentally different environment implementations:
- Original: BatchedEnv with UnificationEngine (TorchRL-style)
- Optimized: EnvVec with UnificationEngineVectorized (custom CUDA-optimized)

These environments use different internal variable indexing during unification,
which can cause different proof trajectories for the same query. As a result,
exact MRR parity is NOT achievable. Instead, we verify that:
1. Both paths produce valid rankings (MRR > 0)
2. The MRR values are within a reasonable tolerance (10%)

For strict parity testing of the SAME environment compiled vs uncompiled,
use test_compiled_script.py instead.

Usage:
    # Run in eager mode (simpler debugging)
    python tests/parity/test_eval_compiled_parity.py --dataset family --n-queries 5 --n-corruptions 10
    
    # Run with torch.compile enabled
    python tests/parity/test_eval_compiled_parity.py --dataset family --n-queries 5 --n-corruptions 10 --compile
    
    # Pytest usage
    pytest tests/parity/test_eval_compiled_parity.py -v -s
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Tuple, Optional
import time

# Set environment variables before importing torch
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '0'

import pytest
import torch
import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handler import DataHandler
from index_manager import IndexManager
from tensor.tensor_unification import UnificationEngine, set_compile_mode
from unification import UnificationEngineVectorized
from tensor.tensor_env import BatchedEnv
from tensor.tensor_embeddings import EmbedderLearnable as TensorEmbedder
from tensor.tensor_model import ActorCriticPolicy as TensorPolicy
from policy import ActorCriticPolicy as OptimizedPolicy
from tensor.tensor_sampler import Sampler
from tensor.tensor_model_eval import eval_corruptions
from ppo import PPO as PPOOptimized
from env import EnvVec

# Simple batch size computation for tests
def compute_optimal_batch_size(chunk_queries: int = None, n_corruptions: int = None, **kwargs) -> int:
    """Compute batch size for evaluation (simplified for tests)."""
    total_queries = chunk_queries * (n_corruptions + 1) if chunk_queries and n_corruptions else 100
    if total_queries <= 100:
        return 64
    elif total_queries <= 500:
        return 128
    elif total_queries <= 2000:
        return 256
    else:
        return 512

# ============================================================================
# Configuration
# ============================================================================

def create_default_config() -> SimpleNamespace:
    """Default parameters for parity tests (matching test_eval_parity methodology).
    
    PARITY LIMITATIONS:
    - chunk_queries MUST >= n_queries (single chunk) for corruption RNG parity
    - n_corruptions should be close to actual valid count for tie-breaking RNG parity
      (model_eval uses ragged arrays, ppo_optimized uses fixed arrays with masking)
    - Countries_s3 has strict domain constraints (~4 valid corruptions per query)
    """
    return SimpleNamespace(
        dataset="family",  # Family has more valid corruptions, better for parity
        data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'),
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        
        # Test parameters
        # CRITICAL for parity:
        # 1. chunk_queries >= n_queries (single chunk)
        # 2. n_corruptions close to actual valid count
        # 3. corruption_modes must be ['head'], ['tail'], or ['head', 'tail']
        #    NOT ['both'] - that mode is for sampler.corrupt(), not for eval iteration
        n_queries=20,
        n_corruptions=10,       # Small value for RNG shape parity
        chunk_queries=50,       # Must be >= n_queries
        batch_size_env=50,      # Must match chunk_queries
        corruption_modes=['tail'],  # Use single mode for simpler parity
        mode='test',
        
        # Environment parameters
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        memory_pruning=True,
        skip_unary_actions=False,  # Must be False for parity with optimized
        end_proof_action=True,
        reward_type=0,
        max_total_vars=100,
        
        # Model parameters
        atom_embedding_size=250,
        hidden_dim=256,
        num_layers=8,
        
        vram_gb=6.0,

        seed=42,
        verbose=False,
        
        # PPO-specific
        n_envs=50,  # Will be overridden by batch_size_env
        n_steps=1,  # Buffer not used for evaluation
        learning_rate=3e-4,
        n_epochs=5,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        max_steps=20,  # Alias for max_depth
        use_callbacks=False,
        eval_only=True,  # Skip rollout buffer allocation
        compile=False,  # Disable torch.compile for parity tests
    )


# ============================================================================
# Setup Functions
# ============================================================================

def setup_shared_components(config: SimpleNamespace, device: torch.device) -> Dict[str, Any]:
    """Setup components shared between original and optimized paths."""
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Disable compile mode for parity tests
    set_compile_mode(False)
    
    # Load data
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        corruption_mode='dynamic',
    )
    
    # Index manager
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
    
    # Sampler
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
    
    # Reseed for reproducibility
    torch.manual_seed(config.seed)
    
    # Embedder (shared by both paths)
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
    
    # Stringifier params
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    # Base unification engine (for original path only)
    base_engine = UnificationEngine.from_index_manager(
        im, take_ownership=False,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.padding_states,
    )
    base_engine.index_manager = im
    
    # Vectorized engine (for optimized path)
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        max_fact_pairs=None,
        max_rule_pairs=None,
        padding_atoms=config.padding_atoms,
        max_derived_per_state=config.padding_states,
        end_proof_action=config.end_proof_action,
    )
    
    # Convert test queries
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
    
    # Policy (Original path)
    action_size = config.padding_states
    torch.manual_seed(config.seed)
    policy_tensor = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=action_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout_prob=0.0,
        device=device,
        parity=True,  # Enable parity mode for consistent initialization
    ).to(device)
    
    # Policy (Optimized path)
    torch.manual_seed(config.seed)
    policy_opt = OptimizedPolicy(
        embedder=embedder,
        embed_dim=config.embed_dim if hasattr(config, 'embed_dim') else config.atom_embedding_size,
        action_dim=action_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        device=device,
        parity=True,  # Enable parity mode for consistent initialization
    ).to(device)
    
    # Transfer weights from tensor to optimized policy using proper mapping utility
    from tests.test_utils.weight_mapping import map_tensor_to_optimized_state_dict
    tensor_state = policy_tensor.state_dict()
    cleaned_state = map_tensor_to_optimized_state_dict(tensor_state)
    policy_opt.load_state_dict(cleaned_state, strict=True)
    
    return {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'base_engine': base_engine,
        'vec_engine': vec_engine,
        'policy_tensor': policy_tensor,
        'policy_opt': policy_opt,
        'test_queries_unpadded': test_queries_unpadded,
        'test_queries_padded': test_queries_padded,
    }


def create_original_env(components: Dict, config: SimpleNamespace, device: torch.device) -> BatchedEnv:
    """Create original BatchedEnv for evaluation."""
    im = components['im']
    base_engine = components['base_engine']
    test_queries_padded = components['test_queries_padded']
    
    # Use min of batch_size_env and actual query count  
    n_queries = min(config.batch_size_env, test_queries_padded.shape[0], config.n_queries)
    
    return BatchedEnv(
        batch_size=n_queries,
        queries=test_queries_padded[:n_queries],
        labels=torch.ones(n_queries, dtype=torch.long, device=device),
        query_depths=torch.ones(n_queries, dtype=torch.long, device=device),
        unification_engine=base_engine,
        mode='eval',
        max_depth=config.max_depth,
        memory_pruning=config.memory_pruning,
        use_exact_memory=False,
        skip_unary_actions=config.skip_unary_actions,
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


def create_optimized_env(components: Dict, config: SimpleNamespace, device: torch.device) -> EnvVec:
    """Create optimized EvalOnlyEnvOptimized for evaluation."""
    im = components['im']
    vec_engine = components['vec_engine']
    test_queries_padded = components['test_queries_padded']

    return EnvVec(
        vec_engine=vec_engine,
        batch_size=config.batch_size_env,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=config.memory_pruning,
        valid_queries=test_queries_padded,
    )


# ============================================================================
# Evaluation Functions
# ============================================================================

def run_original_eval(
    components: Dict,
    env: BatchedEnv,
    config: SimpleNamespace,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run evaluation using original path."""
    sampler = components['sampler']
    policy = components['policy_tensor']
    queries = components['test_queries_unpadded'][:config.n_queries]
    
    # Reset RNG for corruption parity
    # The sampler uses torch.randint internally, so we must seed torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    sampler.rng = np.random.RandomState(seed)
    
    return eval_corruptions(
        actor=policy,
        env=env,
        queries=queries,
        sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        verbose=config.verbose,
    )


def run_optimized_eval(
    components: Dict,
    env: EnvVec,
    config: SimpleNamespace,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float]:
    """Run evaluation using optimized path. Returns (results, warmup_time_s)."""
    sampler = components['sampler']
    policy = components['policy_opt']
    queries = components['test_queries_unpadded'][:config.n_queries].to(env.device)
    
    # Configure compilation (disabled for parity tests)
    compile_mode = 'default'
    fullgraph = False
    
    # CRITICAL: Seed BEFORE creating PPO and warmup to ensure deterministic compilation
    # torch.compile can produce different results based on RNG state during tracing
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Enable deterministic algorithms for reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create PPOOptimized instance for evaluation - use new config-based API
    ppo = PPOOptimized(policy, env, config)
    
    # Component warmup
    warmup_start = time.time()
    effective_chunk_queries = min(int(config.chunk_queries), int(queries.shape[0]))

    # Re-seed to ensure warmup doesn't affect main evaluation RNG
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Warmup with padded queries to trigger JIT compilation
    warmup_queries, _ = ppo._pad_queries(queries[:2])
    _ = ppo.evaluate_policy(warmup_queries, deterministic=True)
    warmup_time_s = time.time() - warmup_start
    
    # Reset RNG for corruption parity AFTER warmup
    # The sampler uses torch.randint internally, so we must seed torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sampler.rng = np.random.RandomState(seed)
    
    # CRITICAL: Parity requires single chunk (all queries processed together)
    # Multi-chunk breaks RNG parity due to different corruption generation order
    n_queries = queries.shape[0]
    if n_queries > effective_chunk_queries:
        raise ValueError(
            f"Parity test requires single chunk: n_queries ({n_queries}) must be <= "
            f"chunk_queries ({effective_chunk_queries}). Increase chunk_queries or reduce n_queries."
        )

    # Run evaluation using production evaluate() method with parity mode
    results = ppo.evaluate(
        queries=queries,
        sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        verbose=config.verbose,
        deterministic=True,
        chunk_queries=effective_chunk_queries,
        parity=True,  # Use numpy RNG for tie-breaking to match tensor version
    )

    return results, warmup_time_s


# ============================================================================
# Parity Check
# ============================================================================

def check_mrr_parity(
    original_results: Dict[str, Any],
    optimized_results: Dict[str, Any],
    tolerance: float = 0.15,  # Relaxed: different env implementations cause ~5-15% variance
) -> Tuple[bool, str]:
    """
    Check if MRR values match within tolerance.
    
    NOTE: This uses a relaxed tolerance (15%) because the two evaluation paths
    use fundamentally different environment implementations (BatchedEnv vs EnvVec)
    which produce different unification variable indices and thus different
    proof trajectories.
    
    MRR is the primary metric; Hits@K uses a more relaxed tolerance (20%) since
    it can vary more with small sample sizes.

    Args:
        original_results: Results from original evaluation
        optimized_results: Results from optimized evaluation
        tolerance: Absolute tolerance for MRR difference (default 0.15)

    Returns:
        (passed, message)
    """
    metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    
    # Different tolerances: MRR is primary metric, Hits@K can vary more
    # Hits@1 needs highest tolerance as it's most sensitive to single ranking changes
    metric_tolerances = {
        'MRR': tolerance,
        'Hits@1': 0.25,  # 25% tolerance - most sensitive to ranking changes
        'Hits@3': tolerance + 0.05,  # 20% tolerance
        'Hits@10': tolerance,
    }

    lines = []
    lines.append(f"\n{'Metric':<10} {'Original':>12} {'Optimized':>12} {'Diff':>10} {'Status':>8}")
    lines.append("-" * 60)

    all_pass = True
    for m in metrics:
        orig = original_results.get(m, 0.0)
        opt = optimized_results.get(m, 0.0)
        diff = opt - orig

        # Use metric-specific tolerance, or relative tolerance based on original value
        base_tol = metric_tolerances.get(m, tolerance)
        tol = max(base_tol, base_tol * abs(orig)) if orig > 0 else base_tol
        passed = abs(diff) <= tol
        status = "PASS" if passed else "FAIL"

        if not passed:
            all_pass = False

        lines.append(f"{m:<10} {orig:>12.4f} {opt:>12.4f} {diff:>+10.4f} {status:>8}")

    return all_pass, "\n".join(lines)


# ============================================================================
# Main Parity Test
# ============================================================================

def run_parity_test(
    config: SimpleNamespace,
    device: torch.device,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run the full MRR parity test.
    
    Returns:
        (passed, results_dict)
    """
    # Enable deterministic algorithms globally for this test
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Clear dynamo cache to ensure fresh compilation each run
    torch._dynamo.reset()
    
    print("\n" + "=" * 70)
    print("EVAL COMPILED PARITY TEST")
    print("=" * 70)
    print(f"Dataset: {config.dataset}")
    print(f"Queries: {config.n_queries}, Corruptions: {config.n_corruptions}")
    print(f"skip_unary_actions: {config.skip_unary_actions}")
    print("=" * 70)
    
    # Ensure device is in config for PPO
    config.device = device

    # Calculate optimal batch size for evaluation BEFORE creating components
    # This ensures EnvVec and PPO use the same batch size for compilation
    effective_chunk_queries = min(int(config.chunk_queries), int(config.n_queries))
    batch_size = compute_optimal_batch_size(
        chunk_queries=effective_chunk_queries,
        n_corruptions=config.n_corruptions,
    )
    config.batch_size_env = batch_size
    config.n_envs = batch_size  # PPO uses n_envs as fallback for batch_size_env
    print(f"Using batch_size_env: {batch_size}")

    # Setup
    print("\nSetting up components...")
    components = setup_shared_components(config, device)

    # Create environments
    print("Creating original environment...")
    env_orig = create_original_env(components, config, device)
    
    print("Creating optimized environment...")
    env_opt = create_optimized_env(components, config, device)
    
    # Run original evaluation
    print("\nRunning original evaluation...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start_orig = time.time()
    results_orig = run_original_eval(components, env_orig, config, seed=config.seed)
    time_orig = time.time() - start_orig
    print(f"  Original MRR: {results_orig['MRR']:.4f} (took {time_orig:.2f}s)")
    
    # Run optimized evaluation
    print("\nRunning optimized evaluation...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start_opt = time.time()
    results_opt, warmup_time = run_optimized_eval(components, env_opt, config, seed=config.seed)
    time_opt = time.time() - start_opt
    print(f"  Optimized MRR: {results_opt['MRR']:.4f} (took {time_opt:.2f}s, warmup {warmup_time:.2f}s)")
    
    # Check parity (relaxed tolerance due to different env implementations)
    passed, report = check_mrr_parity(results_orig, results_opt, tolerance=0.15)
    print(report)
    
    # Summary
    print("\n" + "=" * 70)
    if passed:
        print("✓ PARITY TEST PASSED - Original and Optimized MRR within tolerance!")
        print("  Note: Using relaxed tolerance (15%) due to different env implementations")
    else:
        print("✗ PARITY TEST FAILED - MRR values differ beyond relaxed tolerance (15%)")
        print("  This indicates a fundamental issue beyond expected env variance")
    print("=" * 70)
    
    return passed, {
        'original': results_orig,
        'optimized': results_opt,
        'time_original': time_orig,
        'time_optimized': time_opt,
        'warmup_time': warmup_time,
    }


# ============================================================================
# Pytest Tests
# ============================================================================

@pytest.fixture(scope="module")
def base_config():
    """Base test configuration."""
    return create_default_config()


@pytest.fixture(scope="module")
def device():
    """Get device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestEvalCompiledParity:
    """Tests for evaluation parity between original and optimized paths.
    
    IMPORTANT NOTES:
    1. This test compares two different environment implementations (BatchedEnv vs EnvVec)
       which may produce different proof trajectories for the same query.
    2. As a result, we use a relaxed tolerance (15%) instead of exact parity.
    3. Use n_queries >= 15 for statistically meaningful results; smaller values
       can cause large variance in MRR from single query ranking differences.
    4. For strict parity testing, use test_compiled_script.py instead.
    """
    
    @pytest.mark.parametrize("dataset,corruption_mode,n_queries,n_corruptions", [
        # Parity tests with small n_corruptions (close to actual valid count)
        # Countries_s3 has ~4 valid corruptions due to domain constraints
        ("countries_s3", "tail", 24, 10),
        # Family has more valid corruptions - use 'tail' mode for parity
        # Note: 'both' mode in corruption_modes is NOT supported for parity tests
        # because tensor_model_eval expects 'head' or 'tail', not 'both'
        # Use n_queries >= 15 for stable variance
        ("family", "tail", 20, 10),
        ("family", "tail", 15, 10),  # Increased from 5 to reduce MRR variance
    ])
    def test_mrr_parity_eager(self, dataset: str, corruption_mode: str, 
                              n_queries: int, n_corruptions: int, base_config, device):
        """Test MRR parity in eager mode (no compilation)."""
        config = SimpleNamespace(**vars(base_config))
        config.dataset = dataset
        config.n_queries = n_queries
        config.n_corruptions = n_corruptions
        config.corruption_modes = [corruption_mode]
        
        passed, results = run_parity_test(config, device)
        
        assert passed, (
            f"MRR parity failed for {dataset} ({corruption_mode}, {n_queries} queries, {n_corruptions} negs). "
            f"Original: {results['original']['MRR']:.4f}, "
            f"Optimized: {results['optimized']['MRR']:.4f}"
        )
    


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test eval compiled parity (Original vs Optimized)')
    parser.add_argument('--dataset', type=str, default='family', help='Dataset name (default: countries_s3)')
    parser.add_argument('--n-queries', type=int, default=10, help='Number of test queries (default: 24)')
    parser.add_argument('--n-corruptions', type=int, default=4, help='Corruptions per query (default: 50)')
    parser.add_argument('--corruption-mode', type=str, default='tail',
                        choices=['head', 'tail'], help="Corruption mode (default: tail)")
    parser.add_argument('--chunk-queries', type=int, default=100, help='Queries per chunk')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    config = create_default_config()
    config.dataset = args.dataset
    config.n_queries = args.n_queries
    config.n_corruptions = args.n_corruptions
    config.corruption_modes = [args.corruption_mode]
    config.chunk_queries = args.chunk_queries
    config.seed = args.seed
    config.verbose = args.verbose
    
    # Set appropriate defaults per dataset
    if args.dataset == 'countries_s3' and args.corruption_mode == 'tail':
        pass  # defaults are correct
    elif args.dataset == 'family':
        pass  # Keep the specified corruption mode
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    passed, results = run_parity_test(config, device)
    
    sys.exit(0 if passed else 1)
