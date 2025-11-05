"""
Quick test to verify env_factory integration works with runner.py workflow.
This test creates a minimal training run to ensure all components work together.
"""

import sys
import torch
import argparse
from pathlib import Path

# Import the main training function
from train import main

def test_env_factory_integration():
    """Test that env_factory works with the training pipeline."""
    
    print("\n" + "="*60)
    print("Testing env_factory Integration")
    print("="*60 + "\n")
    
    # Create minimal test configuration
    test_config = argparse.Namespace(
        # Dataset params
        dataset_name='countries_s3',
        eval_neg_samples=3,
        test_neg_samples=None,  # Will be auto-configured
        train_depth=None,
        valid_depth=None,
        test_depth=None,
        n_train_queries=10,  # Small number for quick test
        n_eval_queries=5,
        n_test_queries=5,
        prob_facts=False,
        topk_facts=None,
        topk_facts_threshold=None,
        
        # Model params
        model_name='PPO',
        ent_coef=0.5,
        clip_range=0.2,
        n_epochs=2,  # Small for quick test
        lr=3e-4,
        gamma=0.99,
        
        # Training params
        seed=[0],
        seed_run_i=0,
        timesteps_train=512,  # Very small for quick test
        restore_best_val_model=False,  # Disable to avoid checkpoint issues
        load_model=False,
        save_model=False,  # Disable saving for test
        n_envs=2,  # Single environment for simplicity
        n_steps=128,
        n_eval_envs=2,
        batch_size=128,
        
        # Env params
        reward_type=4,
        train_neg_ratio=1,
        engine='python',
        engine_strategy='cmp',
        endf_action=True,
        endt_action=False,
        skip_unary_actions=True,
        max_depth=20,
        memory_pruning=True,
        corruption_mode='dynamic',
        corruption_scheme=['head', 'tail'],
        false_rules=False,
        
        # KGE integration params (disabled)
        kge_action=False,
        logit_fusion=False,
        inference_fusion=False,
        inference_success_only=False,
        pbrs=False,
        pbrs_beta=0.0,
        pbrs_gamma=None,
        enable_top_k=False,
        kge_engine='tf',
        kge_checkpoint_dir='./../checkpoints/',
        kge_run_signature=None,
        kge_scores_file=None,
        
        # Evaluation params
        eval_hybrid_success_only=True,
        eval_hybrid_kge_weight=2.0,
        eval_hybrid_rl_weight=1.0,
        
        # Embedding params
        atom_embedder='transe',
        state_embedder='mean',
        atom_embedding_size=256,
        constant_embedding_size=256,
        predicate_embedding_size=256,
        state_embedding_size=256,
        learn_embeddings=True,
        padding_atoms=6,
        padding_states=20,
        max_total_vars=100,
        
        # Other params
        device='cpu',  # Use CPU for test to avoid GPU requirements
        min_gpu_memory_gb=2.0,
        extended_eval_info=True,
        eval_best_metric='mrr_mean',
        plot_trajectories=False,
        plot=False,
        depth_info=False,
        verbose_cb=False,
        verbose_env=0,
        verbose_prover=0,
        data_path='./data/',
        models_path='models/',
        rules_file='rules.txt',
        facts_file='train.txt',
        janus_file=None,  # No prolog file for python engine
        train_file='train_depths.txt',
        valid_file='valid_depths.txt',
        test_file='test_depths.txt',
        use_logger=False,  # Disable logging for test
        logger_path='./runs/',
        use_wb=False,
        wb_path='./../wandb/',
        eval_freq=128,
        run_signature='test-env-factory',
        
        # env_factory specific params
        use_parallel_envs=True,
        parallel_env_start_method='fork',
    )
    
    print("Configuration:")
    print(f"  Dataset: {test_config.dataset_name}")
    print(f"  Training timesteps: {test_config.timesteps_train}")
    print(f"  Device: {test_config.device}")
    print(f"  Using env_factory: Yes")
    print(f"  Parallel envs: {test_config.use_parallel_envs}")
    print()
    
    try:
        # Run the training
        print("Starting training with env_factory...\n")
        train_metrics, valid_metrics, test_metrics = main(
            args=test_config,
            log_filename='test_log.csv',
            use_logger=False,
            use_WB=False,  # Capital WB
            WB_path=None,  # Capital WB
            date='test',
        )
        
        print("\n" + "="*60)
        print("✓ Test PASSED: env_factory integration works!")
        print("="*60)
        print("\nMetrics summary:")
        print(f"  Training completed: {train_metrics is not None}")
        print(f"  Validation completed: {valid_metrics is not None}")
        print(f"  Test completed: {test_metrics is not None}")
        
        if valid_metrics:
            mrr = valid_metrics.get('mrr_mean', 0.0)
            # Handle both list and scalar formats
            if isinstance(mrr, (list, tuple)) and len(mrr) > 0:
                mrr = mrr[0]
            print(f"  Validation MRR: {mrr:.4f}")
        
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ Test FAILED: env_factory integration error")
        print("="*60)
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if data exists
    if not Path("./data/countries_s3").exists():
        print("ERROR: Dataset 'countries_s3' not found in ./data/")
        print("Please ensure the dataset is available before running this test.")
        sys.exit(1)
    
    success = test_env_factory_integration()
    sys.exit(0 if success else 1)
