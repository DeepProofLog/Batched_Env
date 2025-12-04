"""
Model Parity Tests.

Tests verifying that the tensor-based ActorCriticPolicy produces EXACTLY the
same results as the SB3 CustomActorCriticPolicy using real EmbedderLearnable.

This module tests:
- Embedder output parity (observation and action embeddings)
- Forward pass parity (actions, values, log probabilities)
- evaluate_actions parity (values, log probs, entropy)
- predict_values parity

Usage:
    pytest tests/parity/test_model_parity.py -v
    pytest tests/parity/test_model_parity.py -v -k "embedder"
    pytest tests/parity/test_model_parity.py -v -k "forward"
    
    # Run from command line:
    python tests/parity/test_model_parity.py
"""
import sys
from pathlib import Path
from typing import Dict
from types import SimpleNamespace

import gymnasium as gym
import torch
import pytest
from tensordict import TensorDict


# ============================================================================
# Path Setup
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]
SB3_ROOT = ROOT / "sb3"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SB3_ROOT) not in sys.path:
    sys.path.insert(1, str(SB3_ROOT))

# Import tensor-based components
from embeddings import EmbedderLearnable
from model import ActorCriticPolicy

# Try to import SB3 components for parity testing
try:
    from sb3.sb3_model import CustomActorCriticPolicy as SB3CustomActorCriticPolicy
    from sb3.sb3_model import CustomCombinedExtractor as SB3CustomCombinedExtractor
    from sb3.sb3_embeddings import EmbedderLearnable as SB3EmbedderLearnable
    SB3_AVAILABLE = True
except ImportError as e:
    SB3_AVAILABLE = False
    _SB3_IMPORT_ERROR = e


# ============================================================================
# Default Configuration
# ============================================================================

def create_default_config() -> SimpleNamespace:
    """
    Create default configuration for model parity tests.
    
    Returns:
        Configuration namespace with all test parameters
    """
    return SimpleNamespace(
        # Model dimensions
        n_constants=100,
        n_predicates=20,
        n_vars=10,
        embed_dim=64,
        max_arity=2,
        padding_atoms=3,
        
        # Network architecture
        hidden_dim=128,
        num_layers=4,
        dropout_prob=0.0,
        
        # Test settings
        batch_size=4,
        n_actions=16,
        seed=42,
        device="cpu",
        
        # Embedder settings
        atom_embedder='transe',
        state_embedder='sum',
        kge_regularization=0.0,
        kge_dropout_rate=0.0,
    )


def clone_config(config: SimpleNamespace) -> SimpleNamespace:
    """Create a shallow clone of a configuration namespace."""
    return SimpleNamespace(**vars(config))


# ============================================================================
# Component Creation Helpers
# ============================================================================

def create_embedder(config: SimpleNamespace) -> EmbedderLearnable:
    """Create a tensor-based EmbedderLearnable."""
    torch.manual_seed(config.seed)
    return EmbedderLearnable(
        n_constants=config.n_constants,
        n_predicates=config.n_predicates,
        n_vars=config.n_vars,
        max_arity=config.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder=config.atom_embedder,
        state_embedder=config.state_embedder,
        constant_embedding_size=config.embed_dim,
        predicate_embedding_size=config.embed_dim,
        atom_embedding_size=config.embed_dim,
        kge_regularization=config.kge_regularization,
        kge_dropout_rate=config.kge_dropout_rate,
        device=config.device,
    )


def create_sb3_embedder(config: SimpleNamespace):
    """Create a SB3 EmbedderLearnable."""
    if not SB3_AVAILABLE:
        pytest.skip(f"SB3 dependencies unavailable: {_SB3_IMPORT_ERROR}")
    
    torch.manual_seed(config.seed)
    return SB3EmbedderLearnable(
        n_constants=config.n_constants,
        n_predicates=config.n_predicates,
        n_vars=config.n_vars,
        max_arity=config.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder=config.atom_embedder,
        state_embedder=config.state_embedder,
        constant_embedding_size=config.embed_dim,
        predicate_embedding_size=config.embed_dim,
        atom_embedding_size=config.embed_dim,
        kge_regularization=config.kge_regularization,
        kge_dropout_rate=config.kge_dropout_rate,
        device=config.device,
    )


def create_random_observation(config: SimpleNamespace, obs_seed: int = 123) -> TensorDict:
    """
    Create a random observation TensorDict with valid indices.
    
    Args:
        config: Test configuration
        obs_seed: Seed for observation generation
        
    Returns:
        TensorDict with sub_index, derived_sub_indices, and action_mask
    """
    torch.manual_seed(obs_seed)
    
    # Predicate indices in [1, n_predicates]
    pred_indices = torch.randint(
        1, config.n_predicates + 1,
        (config.batch_size, 1, config.padding_atoms, 1),
        dtype=torch.int32, device=config.device
    )
    
    # Constant indices in [1, n_constants + n_vars]
    const_indices = torch.randint(
        1, config.n_constants + config.n_vars + 1,
        (config.batch_size, 1, config.padding_atoms, config.max_arity),
        dtype=torch.int32, device=config.device
    )
    
    sub_index = torch.cat([pred_indices, const_indices], dim=-1)
    
    # Action observations
    action_pred_indices = torch.randint(
        1, config.n_predicates + 1,
        (config.batch_size, config.n_actions, config.padding_atoms, 1),
        dtype=torch.int32, device=config.device
    )
    action_const_indices = torch.randint(
        1, config.n_constants + config.n_vars + 1,
        (config.batch_size, config.n_actions, config.padding_atoms, config.max_arity),
        dtype=torch.int32, device=config.device
    )
    derived_sub_indices = torch.cat([action_pred_indices, action_const_indices], dim=-1)
    
    # All actions valid
    action_mask = torch.ones(config.batch_size, config.n_actions, dtype=torch.bool, device=config.device)
    
    return TensorDict({
        "sub_index": sub_index,
        "derived_sub_indices": derived_sub_indices,
        "action_mask": action_mask,
    }, batch_size=torch.Size([config.batch_size]))


def create_sb3_observation(obs_td: TensorDict) -> Dict[str, torch.Tensor]:
    """Convert TensorDict observation to SB3 dict format."""
    return {
        "sub_index": obs_td["sub_index"],
        "derived_sub_indices": obs_td["derived_sub_indices"],
        "action_mask": obs_td["action_mask"].to(torch.uint8),
    }


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def base_config():
    """Base configuration for all tests in this module."""
    return create_default_config()


# ============================================================================
# Embedder Parity Tests
# ============================================================================

@pytest.mark.skipif(not SB3_AVAILABLE, reason="SB3 dependencies unavailable")
def test_embedder_parity():
    """
    Test that tensor and SB3 embedders produce identical outputs.
    
    Verifies that get_embeddings_batch produces the same embeddings
    for both observation states and action states.
    """
    config = create_default_config()
    
    tensor_emb = create_embedder(config)
    sb3_emb = create_sb3_embedder(config)
    
    obs = create_random_observation(config, obs_seed=123)
    
    with torch.no_grad():
        tensor_obs_emb = tensor_emb.get_embeddings_batch(obs["sub_index"])
        sb3_obs_emb = sb3_emb.get_embeddings_batch(obs["sub_index"])
        tensor_action_emb = tensor_emb.get_embeddings_batch(obs["derived_sub_indices"])
        sb3_action_emb = sb3_emb.get_embeddings_batch(obs["derived_sub_indices"])
    
    assert torch.allclose(tensor_obs_emb, sb3_obs_emb, atol=1e-5), \
        f"Obs embeddings differ: max diff = {(tensor_obs_emb - sb3_obs_emb).abs().max().item()}"
    assert torch.allclose(tensor_action_emb, sb3_action_emb, atol=1e-5), \
        f"Action embeddings differ: max diff = {(tensor_action_emb - sb3_action_emb).abs().max().item()}"


# ============================================================================
# Forward Pass Parity Tests
# ============================================================================

@pytest.mark.skipif(not SB3_AVAILABLE, reason="SB3 dependencies unavailable")
def test_forward_parity():
    """
    Test that tensor and SB3 policies produce identical forward outputs.
    
    Verifies that forward pass produces same:
    - Actions (in deterministic mode)
    - Values
    - Log probabilities
    """
    config = create_default_config()
    
    # Create embedders
    tensor_emb = create_embedder(config)
    sb3_emb = create_sb3_embedder(config)
    
    # Create tensor policy
    torch.manual_seed(config.seed)
    tensor_policy = ActorCriticPolicy(
        embedder=tensor_emb,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout_prob=config.dropout_prob,
        device=torch.device(config.device),
        action_dim=config.n_actions,
    )
    tensor_policy.eval()
    
    # Create SB3 policy
    obs_space = gym.spaces.Dict({
        "sub_index": gym.spaces.Box(
            low=0, high=config.n_constants + config.n_vars,
            shape=(1, config.padding_atoms, config.max_arity + 1), dtype=int
        ),
        "derived_sub_indices": gym.spaces.Box(
            low=0, high=config.n_constants + config.n_vars,
            shape=(config.n_actions, config.padding_atoms, config.max_arity + 1), dtype=int
        ),
        "action_mask": gym.spaces.Box(
            low=0, high=1, shape=(config.n_actions,), dtype=int
        ),
    })
    action_space = gym.spaces.Discrete(config.n_actions)
    
    torch.manual_seed(config.seed)
    sb3_policy = SB3CustomActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 0.0,
        features_extractor_class=SB3CustomCombinedExtractor,
        features_extractor_kwargs={"embedder": sb3_emb, "features_dim": config.embed_dim},
        share_features_extractor=True,
    ).to(config.device)
    sb3_policy.eval()
    
    # Create test observations
    obs_td = create_random_observation(config, obs_seed=123)
    obs_sb3 = create_sb3_observation(obs_td)
    
    # Forward pass
    with torch.no_grad():
        tensor_actions, tensor_values, tensor_log_probs = tensor_policy(obs_td, deterministic=True)
        sb3_actions, sb3_values, sb3_log_probs = sb3_policy(obs_sb3, deterministic=True)
    
    sb3_values = sb3_values.flatten()
    
    assert torch.equal(tensor_actions, sb3_actions), \
        f"Actions differ: tensor={tensor_actions}, sb3={sb3_actions}"
    assert torch.allclose(tensor_values, sb3_values, atol=1e-5), \
        f"Values differ: max diff = {(tensor_values - sb3_values).abs().max().item()}"
    assert torch.allclose(tensor_log_probs, sb3_log_probs, atol=1e-5), \
        f"Log probs differ: max diff = {(tensor_log_probs - sb3_log_probs).abs().max().item()}"


# ============================================================================
# Evaluate Actions Parity Tests
# ============================================================================

@pytest.mark.skipif(not SB3_AVAILABLE, reason="SB3 dependencies unavailable")
def test_evaluate_actions_parity():
    """
    Test that tensor and SB3 policies produce identical evaluate_actions outputs.
    
    Verifies that evaluate_actions produces same:
    - Values
    - Log probabilities
    - Entropy
    """
    config = create_default_config()
    
    # Create embedders
    tensor_emb = create_embedder(config)
    sb3_emb = create_sb3_embedder(config)
    
    # Create tensor policy
    torch.manual_seed(config.seed)
    tensor_policy = ActorCriticPolicy(
        embedder=tensor_emb,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout_prob=config.dropout_prob,
        device=torch.device(config.device),
        action_dim=config.n_actions,
    )
    tensor_policy.eval()
    
    # Create SB3 policy
    obs_space = gym.spaces.Dict({
        "sub_index": gym.spaces.Box(
            low=0, high=config.n_constants + config.n_vars,
            shape=(1, config.padding_atoms, config.max_arity + 1), dtype=int
        ),
        "derived_sub_indices": gym.spaces.Box(
            low=0, high=config.n_constants + config.n_vars,
            shape=(config.n_actions, config.padding_atoms, config.max_arity + 1), dtype=int
        ),
        "action_mask": gym.spaces.Box(
            low=0, high=1, shape=(config.n_actions,), dtype=int
        ),
    })
    action_space = gym.spaces.Discrete(config.n_actions)
    
    torch.manual_seed(config.seed)
    sb3_policy = SB3CustomActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 0.0,
        features_extractor_class=SB3CustomCombinedExtractor,
        features_extractor_kwargs={"embedder": sb3_emb, "features_dim": config.embed_dim},
        share_features_extractor=True,
    ).to(config.device)
    sb3_policy.eval()
    
    # Create test observations and actions
    obs_td = create_random_observation(config, obs_seed=123)
    obs_sb3 = create_sb3_observation(obs_td)
    actions = torch.arange(config.batch_size, device=config.device) % config.n_actions
    
    # Evaluate actions
    with torch.no_grad():
        tensor_values, tensor_log_probs, tensor_entropy = tensor_policy.evaluate_actions(obs_td, actions)
        sb3_values, sb3_log_probs, sb3_entropy = sb3_policy.evaluate_actions(obs_sb3, actions)
    
    sb3_values = sb3_values.flatten()
    sb3_log_probs = sb3_log_probs.flatten()
    sb3_entropy = sb3_entropy.flatten() if sb3_entropy is not None else None
    
    assert torch.allclose(tensor_values, sb3_values, atol=1e-5), \
        f"Values differ: max diff = {(tensor_values - sb3_values).abs().max().item()}"
    assert torch.allclose(tensor_log_probs, sb3_log_probs, atol=1e-5), \
        f"Log probs differ: max diff = {(tensor_log_probs - sb3_log_probs).abs().max().item()}"
    if sb3_entropy is not None:
        assert torch.allclose(tensor_entropy, sb3_entropy, atol=1e-5), \
            f"Entropy differs: max diff = {(tensor_entropy - sb3_entropy).abs().max().item()}"


# ============================================================================
# Predict Values Parity Tests
# ============================================================================

@pytest.mark.skipif(not SB3_AVAILABLE, reason="SB3 dependencies unavailable")
def test_predict_values_parity():
    """
    Test that tensor and SB3 policies produce identical predict_values outputs.
    
    Verifies that predict_values produces the same value estimates.
    """
    config = create_default_config()
    
    # Create embedders
    tensor_emb = create_embedder(config)
    sb3_emb = create_sb3_embedder(config)
    
    # Create tensor policy
    torch.manual_seed(config.seed)
    tensor_policy = ActorCriticPolicy(
        embedder=tensor_emb,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout_prob=config.dropout_prob,
        device=torch.device(config.device),
        action_dim=config.n_actions,
    )
    tensor_policy.eval()
    
    # Create SB3 policy
    obs_space = gym.spaces.Dict({
        "sub_index": gym.spaces.Box(
            low=0, high=config.n_constants + config.n_vars,
            shape=(1, config.padding_atoms, config.max_arity + 1), dtype=int
        ),
        "derived_sub_indices": gym.spaces.Box(
            low=0, high=config.n_constants + config.n_vars,
            shape=(config.n_actions, config.padding_atoms, config.max_arity + 1), dtype=int
        ),
        "action_mask": gym.spaces.Box(
            low=0, high=1, shape=(config.n_actions,), dtype=int
        ),
    })
    action_space = gym.spaces.Discrete(config.n_actions)
    
    torch.manual_seed(config.seed)
    sb3_policy = SB3CustomActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 0.0,
        features_extractor_class=SB3CustomCombinedExtractor,
        features_extractor_kwargs={"embedder": sb3_emb, "features_dim": config.embed_dim},
        share_features_extractor=True,
    ).to(config.device)
    sb3_policy.eval()
    
    # Create test observations
    obs_td = create_random_observation(config, obs_seed=123)
    obs_sb3 = create_sb3_observation(obs_td)
    
    # Predict values
    with torch.no_grad():
        tensor_values = tensor_policy.predict_values(obs_td)
        sb3_values = sb3_policy.predict_values(obs_sb3)
    
    sb3_values = sb3_values.flatten()
    
    assert torch.allclose(tensor_values, sb3_values, atol=1e-5), \
        f"Predicted values differ: max diff = {(tensor_values - sb3_values).abs().max().item()}"


# ============================================================================
# CLI Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
