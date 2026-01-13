"""
KGE Experiment Builder - Factory functions for KGE components.

This module hides internal complexity (DataHandler, IndexManager, 
UnificationEngine, Sampler) from the user. The main entry point is
through runner.py which calls these factories.

Example:
    from kge_experiments.builder import get_default_config, create_env, create_policy
    
    config = get_default_config(dataset='countries_s3')
    env = create_env(config)
    policy = create_policy(config)
"""
import torch
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from .config import TrainConfig
from .data_handler import DataHandler
from .index_manager import IndexManager
from .unification import UnificationEngineVectorized
from .env import EnvVec
from .nn.sampler import Sampler
from .nn.embeddings import EmbedderLearnable as Embedder
from .policy import ActorCriticPolicy as Policy
from utils import seed_all


@dataclass 
class KGEConfig(TrainConfig):
    """KGE experiment configuration with experiment_type marker."""
    experiment_type: str = field(default='kge', init=False, repr=False)
    
    # Cached internal components (populated by create_env)
    _components: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def get_default_config(**overrides) -> KGEConfig:
    """
    Get default KGE config with optional overrides.
    
    Args:
        **overrides: Any TrainConfig field to override (e.g., dataset='fb15k')
        
    Returns:
        KGEConfig with defaults and overrides applied.
    """
    config = KGEConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def create_env(config: KGEConfig) -> EnvVec:
    """
    Create KGE environment with all internal components.
    
    Internally initializes: DataHandler, IndexManager, UnificationEngine, Sampler.
    These are cached in config._components for use by create_policy().
    
    Args:
        config: KGEConfig with experiment parameters.
        
    Returns:
        EnvVec environment ready for training/evaluation.
    """
    device = torch.device(config.device)
    seed_all(config.seed, deterministic=config.parity)
    
    # =========================================================================
    # DataHandler - loads facts, rules, queries from disk
    # =========================================================================
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
        corruption_mode="dynamic",
    )
    
    # =========================================================================
    # IndexManager - converts strings to indices, manages vocabulary
    # =========================================================================
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
    
    # =========================================================================
    # Sampler - generates negative samples for training/evaluation
    # =========================================================================
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode=config.sampler_default_mode,
        seed=config.seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    # =========================================================================
    # UnificationEngine - performs logical unification for reasoning
    # =========================================================================
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=config.padding_atoms,
        parity_mode=config.parity,
        max_derived_per_state=config.padding_states,
        end_proof_action=config.end_proof_action,
    )
    
    # Clean up index manager tensors (already transferred to engine)
    im.facts_idx = None
    im.rules_idx = None
    im.rule_lens = None
    im.rules_heads_idx = None
    
    # =========================================================================
    # Convert queries to tensor format
    # =========================================================================
    train_queries = torch.stack([
        im.atom_to_tensor(q.predicate, q.args[0], q.args[1]) 
        for q in dh.train_queries
    ], dim=0)
    
    test_queries = torch.stack([
        im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        for q in dh.test_queries
    ], dim=0)
    
    # =========================================================================
    # EnvVec - the actual environment
    # =========================================================================
    env = EnvVec(
        vec_engine=vec_engine,
        batch_size=config.n_envs,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_steps,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=config.memory_pruning,
        train_queries=train_queries,
        valid_queries=test_queries,
        sample_deterministic_per_env=config.parity,  # Only use parity path when testing
        sampler=sampler,
        order=True if config.parity else False,
        negative_ratio=config.negative_ratio,
    )
    
    # Cache components for policy creation and evaluation
    config._components = {
        'dh': dh,
        'im': im,
        'sampler': sampler,
        'vec_engine': vec_engine,
        'train_queries': train_queries,
        'test_queries': test_queries,
        'device': device,
    }
    
    return env


def create_policy(config: KGEConfig, env: Optional[EnvVec] = None) -> Policy:
    """
    Create KGE policy network.
    
    Must be called after create_env() to access cached components.
    
    Args:
        config: KGEConfig (must have _components populated by create_env).
        env: Optional EnvVec (unused, for API compatibility).
        
    Returns:
        ActorCriticPolicy ready for training.
    """
    if not config._components:
        raise RuntimeError("create_env() must be called before create_policy()")
    
    device = config._components['device']
    im = config._components['im']
    dh = config._components['dh']
    
    # Create embedder
    torch.manual_seed(config.seed)
    embedder = Embedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder=config.atom_embedder,
        state_embedder=config.state_embedder,
        constant_embedding_size=config.atom_embedding_size,
        predicate_embedding_size=config.atom_embedding_size,
        atom_embedding_size=config.atom_embedding_size,
        device=str(device),
    )
    embedder.embed_dim = config.atom_embedding_size
    
    # Create policy
    torch.manual_seed(config.seed)
    policy = Policy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=config.padding_states,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        device=device,
        parity=True,  # Use SB3-identical initialization
    ).to(device)
    
    return policy


# =============================================================================
# Helper accessors for evaluation and callbacks
# =============================================================================

def get_sampler(config: KGEConfig) -> Sampler:
    """Get the sampler (for evaluation callbacks)."""
    if not config._components:
        raise RuntimeError("create_env() must be called first")
    return config._components['sampler']


def get_data_handler(config: KGEConfig) -> DataHandler:
    """Get the data handler (for callbacks that need query info)."""
    if not config._components:
        raise RuntimeError("create_env() must be called first")
    return config._components['dh']


def get_index_manager(config: KGEConfig) -> IndexManager:
    """Get the index manager (for advanced usage)."""
    if not config._components:
        raise RuntimeError("create_env() must be called first")
    return config._components['im']


def get_test_queries(config: KGEConfig) -> torch.Tensor:
    """Get test queries tensor (for evaluation)."""
    if not config._components:
        raise RuntimeError("create_env() must be called first")
    return config._components['test_queries']


def get_train_queries(config: KGEConfig) -> torch.Tensor:
    """Get training queries tensor."""
    if not config._components:
        raise RuntimeError("create_env() must be called first")
    return config._components['train_queries']
