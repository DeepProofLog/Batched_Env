"""
countries: 
    - (lr, entropy) decay true, else 5e-5 and 0.2, train_neg_ratio 4, envsxsteps=16x128. 
    - adviced before: clip_range=0.2, lr=5e-5, ent_coef=0.2, n_epochs=10, batch_size_env=64, 
    batch_size_env_eval=64, vf_coef=1.0, max_grad_norm=0.5, target_kl=0.02, gae_lambda=0.95, gamma=0.99, 
    - 111(train queries)x5(pos+4 neg)x4(avg_len)=2,220 steps. set 16(envs)x128(steps)=2048. Recomended to cover 10%
        BUT i have seen that by using 128(envs)x256(steps) i get better and more stable results

    countries_s3 (Solved/Optimal Config): 
    - Strategy: Balanced training + L2 Normalization + Early Stopping.
    - Architecture: use_l2_norm=True, temperature=0.1, hidden_dim=256, dropout_prob=0.0.
    - PPO Args: lr=3e-5, ent_coef=0.01 (low), clip_range_vf=None (unclipped critic).
    - Data: train_neg_ratio=1 (Balanced), 16 envs x 256 steps.
    - Result: consistently achieves MRR > 0.95.
"""

countries_config = {
    # Dataset params
    'dataset_name': 'countries_s3',

    'eval_neg_samples': None,
    'test_neg_samples': None,

    'train_depth': {1,2,3,4,5,6},
    'valid_depth': None,
    'test_depth': None,

    'load_depth_info': True,

    'n_train_queries': None,
    'n_eval_queries': None,
    'n_test_queries': None,


    # Model params
    'model_name': 'PPO',
    'clip_range': 0.2,
    'n_epochs': 5,  # REDUCED from 10 - prevents KL spikes without reducing LR
    'gamma': 0.99,
    'gae_lambda': 0.95,  # GAE lambda for advantage estimation
    'clip_range_vf': None,# 0.5,  # Enable value function clipping for stability
    'vf_coef': 2.0,  # INCREASED from 1.0 - fixes negative explained variance
    'max_grad_norm': 0.5,  # Gradient clipping for stability
    'target_kl': 0.07,  # INCREASED from 0.05 - allow more epochs before early stopping
    'hidden_dim': 128,
    'num_layers': 8,
    'dropout_prob': 0.1,

    # Training params
    'seed': [0],
    'timesteps_train': 1000000,
    'restore_best_val_model': True,
    'load_best_metric': 'eval', # 'eval' (best MRR) or 'train' (best reward)
    'load_model': False,
    'save_model': True,
    'use_amp': True,  # Enable AMP for performance
    'use_compile': True,  # Enable torch.compile for performance
    'batch_size_env': 128,  # Number of parallel environments
    'batch_size_env_eval': 128,
    'n_steps': 256,  # INCREASED from 128 - more samples = better return estimates
    'batch_size': 8192,  # INCREASED from 4096 - larger batches = more stable gradients
    'eval_freq': 1,  # In multiples of n_steps (matches SB3)

    # Env params
    'reward_type': 4,  # Aligned with SB3 (was 4)
    'train_neg_ratio': 1,
    'end_proof_action': True,
    'skip_unary_actions': True,
    'max_depth': 20,
    'memory_pruning': True,
    'corruption_mode': 'dynamic',  # Aligned with SB3 (was True)
    'use_exact_memory': False,


    # Entropy coefficient decay params
    'ent_coef': 0.1,  # Base entropy coefficient (will be scheduled from higher value)

    'ent_coef_decay': True,  # Enable entropy coefficient decay
    'ent_coef_init_value': 0.01,  # Start with moderate exploration
    'ent_coef_final_value': 0.01,  # End with low exploration
    'ent_coef_start': 0.0,  # Fraction of training when decay starts
    'ent_coef_end': 1.0,  # Fraction of training when decay ends
    'ent_coef_transform': 'linear',  # Decay schedule: 'linear', 'exp', 'cos', 'log'
    
    # Learning rate decay params
    'lr': 3e-5,  # Keep learning rate - use other methods to control KL

    'lr_decay': True,  # Enable learning rate decay
    'lr_init_value': 3e-5,  # Reduced from 5e-5 for more stable updates
    'lr_final_value': 1e-6,  # Final value
    'lr_start': 0.0,  # Fraction of training when decay starts
    'lr_end': 1.0,  # Fraction of training when decay ends
    'lr_transform': 'linear',  # Decay schedule: 'linear', 'exp', 'cos', 'log'
    

    # Embedding params
    'atom_embedder': 'transe',
    'state_embedder': 'mean',
    'atom_embedding_size': 250,  # Aligned with SB3 (was 256)
    'learn_embeddings': True,
    'padding_atoms': 6,
    'padding_states': -1,  # Aligned with SB3 (was 20), auto-computed from dataset
    'max_total_vars': 100,  # Aligned with SB3 (was 1000000)
    
    # Policy logit scaling options (can be combined)
    # use_l2_norm: L2 normalize embeddings before dot product (cosine similarity ∈ [-1, +1])
    # sqrt_scale: Divide logits by sqrt(embed_dim) (attention-style scaling)
    # temperature: Lower = sharper distribution, Higher = more uniform

    'use_l2_norm': True,  # DISABLED - was causing policy to be too uniform
    'sqrt_scale': False,    # Enable sqrt(embed_dim) scaling for stable logits
    'temperature': 0.1,    # Standard temperature


    # Other params
    'device': 'cuda',  # Aligned with SB3 (was 'cuda:1')
    'rollout_device': None,  # Device for rollout collection, None means same as device
    'min_gpu_memory_gb': 2.0,
    'extended_eval_info': True,
    'eval_best_metric': 'mrr',
    'plot_trajectories': False,
    'plot': False,
    'depth_info': True,  # Enable depth info by default to see metrics by depth
    'verbose': False,  # Added for SB3 parity
    'verbose_cb': False,  # Verbose callback debugging
    'verbose_env': 0,  # Environment verbosity level (0=quiet, 1=verbose)
    'verbose_prover': 0,  # Prover verbosity level (0=quiet, 1=verbose)
    'prover_verbose': False,  # Added for SB3 parity
    'data_path': './data/',
    'models_path': 'models/',
    'rules_file': 'rules.txt',
    'facts_file': 'train.txt',
    'use_logger': True,
    'logger_path': './runs/',
    'use_wb': False,
    'wb_path': './../wandb/',
    'debug_ppo': False,
    
    # Determinism settings
    'deterministic': False,  # Enable strict reproducibility (slower, set False for production)
    'sample_deterministic_per_env': False,  # Sample deterministic per environment (slower, set False for production)
    'canonical_action_order': False,
}
