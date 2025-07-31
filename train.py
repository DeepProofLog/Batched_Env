import numpy as np
import os
import random
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from index_manager import IndexManager
from utils import get_device, print_eval_info, profile_code
from callbacks import SB3TrainCheckpoint, CustomEvalCallbackMRR, CustomEvalCallback
from custom_dummy_env import create_environments
from dataset import DataHandler
from model import CustomActorCriticPolicy, CustomCombinedExtractor, PPO_custom as PPO
from embeddings import get_embedder
from neg_sampling import get_sampler
from model_eval import eval_corruptions
from kge_inference import KGEInference
from stable_baselines3.common.callbacks import (
    StopTrainingOnRewardThreshold,
    CallbackList,
    StopTrainingOnNoModelImprovement,
)


def main(args, log_filename, use_logger, use_WB, WB_path, date):

    if args.restore_best_val_model==False:
        print("Warning: This setting is not reproducible when creating 2 models "\
            "from scratch, but yes when loading pretrained models. You can use" \
            "export CUBLAS_WORKSPACE_CONFIG=:16:8;export PYTHONHASHSEED=0; "\
            "to make it reproducible.")
    print(args.run_signature)

    torch.manual_seed(args.seed_run_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed_run_i)
    random.seed(args.seed_run_i)
    np.random.seed(args.seed_run_i)

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")

    # ---- KGE INFERENCE ENGINE ----
    kge_inference_engine = None
    if args.use_kge_action or args.kge_integration_strategy== 'sum_eval':
        print("\nInitializing KGE Inference Engine...", flush=True)
        kge_inference_engine = KGEInference(
            dataset_name=args.dataset_name,
            base_path=args.data_path,
            checkpoint_dir=args.kge_checkpoint_dir,
            run_signature=args.kge_run_signature,
            seed=0,
            scores_file_path=args.kge_scores_file
        )
        print("KGE Inference Engine Initialized.\n")

    # ---- DATASET, INDEX MANAGER ----
    data_handler = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        rules_file= args.rules_file,
        facts_file= args.facts_file,
        n_eval_queries = args.n_eval_queries,
        n_test_queries = args.n_test_queries,
        corruption_mode=args.corruption_mode,
        train_depth=args.train_depth,
        valid_depth=args.valid_depth,
        test_depth=args.test_depth,)

    args.n_eval_queries = len(data_handler.valid_queries) if args.n_eval_queries is None else min(args.n_eval_queries, len(data_handler.valid_queries))
    assert args.n_eval_queries > 1, "Number of evaluation queries must be greater than 1. Otherwise there are problems in callback"
    args.n_test_queries = len(data_handler.test_queries) if args.n_test_queries is None else min(args.n_test_queries, len(data_handler.test_queries))

    index_manager = IndexManager(data_handler.constants,
                                data_handler.predicates,
                                args.max_total_vars,
                                constants_images=data_handler.constants_images if args.dataset_name == 'mnist_addition' else set(),
                                constant_images_no=data_handler.constant_images_no if args.dataset_name == 'mnist_addition' else 0,
                                rules=data_handler.rules,
                                max_arity=data_handler.max_arity,
                                device='cpu',
                                padding_atoms=args.padding_atoms)
    index_manager.build_fact_index(data_handler.facts)
    
    data_handler.sampler = get_sampler(data_handler=data_handler, 
                                        index_manager=index_manager, 
                                        corruption_scheme=args.corruption_scheme,
                                        device=device)
    sampler = data_handler.sampler

    embedder_getter = get_embedder(args,
                         data_handler,
                         index_manager,
                         device,
                         )
    embedder = embedder_getter.embedder

    args.atom_embedding_size = args.atom_embedding_size if args.atom_embedder != "concat" else (1+data_handler.max_arity)*args.atom_embedding_size
    args.state_embedding_size = args.atom_embedding_size if args.state_embedder != "concat" else args.atom_embedding_size*args.padding_atoms
    embedder.embed_dim = args.state_embedding_size


    # --- ENVIRONMENT ---
    env, eval_env, callback_env = create_environments(args, data_handler, index_manager, detailed_eval_env=False)

    # --- INIT MODEL ---
    policy_kwargs = {
        'features_extractor_class': CustomCombinedExtractor,
        'features_extractor_kwargs': {'features_dim': embedder.embed_dim, 'embedder': embedder}
    }
    # if args.kge_integration_strategy and args.kge_integration_strategy is not 'sum_eval':
    #     policy_kwargs.update({
    #         'kge_inference_engine': kge_inference_engine,
    #         'index_manager': index_manager,
    #         'kge_integration_strategy': args.kge_integration_strategy,
    #     })

    model = PPO(
        CustomActorCriticPolicy,
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        verbose=1,
        device=device,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        policy_kwargs=policy_kwargs
    )
    
    if args.use_kge_action:
        # model.policy._setup_kge_integration()
        model.policy.kge_inference_engine = kge_inference_engine
        model.policy.index_manager = index_manager
        kge_indices = [idx for pred, idx in index_manager.predicate_str2idx.items() if pred.endswith('_kge')]
        model.policy.kge_indices_tensor = torch.tensor(kge_indices, device=device, dtype=torch.long)

    # --- TRAIN ---
    model_path = os.path.join(args.models_path, args.run_signature, f"seed_{args.seed_run_i}")
    if args.load_model:
        try:
            # Determine which model to load based on the restore flag
            if args.restore_best_val_model:
                load_keyword = "best_eval"
            else:
                load_keyword = "last_epoch"  # Use the new, clear filename

            # Ensure the model directory exists before trying to list its contents
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"Model directory does not exist: {model_path}")
            # Find all model files matching the keyword and seed
            models = sorted([
                m for m in os.listdir(model_path)
                if load_keyword in m and m.endswith('.zip')
            ])
            if models:
                # Load the most recent model (lexicographically, due to date in name)
                model_to_load = models[-1]
                load_path = os.path.join(model_path, model_to_load)
                print(f"Loading model from {load_path}")
                model = PPO.load(load_path, env=eval_env, device=device)#, policy_kwargs=policy_kwargs)
            else:
                raise FileNotFoundError(f"No suitable '{load_keyword}' model found in {model_path}")

        except (FileNotFoundError, NotADirectoryError) as e:
            # If training is scheduled, we can create a new model. Otherwise, it's an error.
            if args.timesteps_train > 0:
                print(f"Warning: Could not load pre-existing model ({e}). A new model will be trained from scratch.")
                args.load_model = False
            else:
                # If in evaluation-only mode (no training), a model is required.
                raise ValueError(f"Error: In evaluation-only mode but could not load model. Reason: {e}")

    if args.timesteps_train > 0 and not args.load_model:
        model_name = f"{date}"
        # timing_callback = EpochTimingCallback(verbose=1)
        callbacks = []
        # no_improvement_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=7, verbose=1)
        reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
        if hasattr(callback_env, 'type_') and callback_env.type_ == "custom_dummy":
            eval_callback = CustomEvalCallbackMRR(eval_env=callback_env,
                                                sampler=sampler,
                                                eval_data=data_handler.valid_queries[:args.n_eval_queries],
                                                n_corruptions=args.eval_neg_samples,
                                                model_path=model_path if args.save_model else None,
                                                log_path=log_filename if use_logger else None,
                                                eval_freq=max(int(args.eval_freq//args.n_envs),1),
                                                n_eval_episodes=args.n_eval_queries-1,
                                                deterministic=True,
                                                render=False,
                                                name=model_name,
                                                callback_on_new_best=reward_threshold_callback if args.restore_best_val_model and not args.use_kge_action else None,
                                                # callback_after_eval=no_improvement_callback,
                                                verbose=0,
                                                )
        else:
            eval_callback = CustomEvalCallback(eval_env=callback_env, 
                                        model_path=model_path if args.save_model else None,
                                        log_path=log_filename if use_logger else None,
                                        eval_freq=max(int(args.eval_freq//args.n_envs),1),
                                        n_eval_episodes=args.n_eval_queries-1,
                                        deterministic=True,
                                        render=False,
                                        name=model_name,
                                        callback_on_new_best=reward_threshold_callback if args.restore_best_val_model and not args.use_kge_action else None,
                                        # callback_after_eval=no_improvement_callback,
                                        )

        training_callback = SB3TrainCheckpoint(
            model, monitor="rollout/ep_rew_mean", frequency=5,
            model_path=model_path if args.save_model else None, name=model_name
        )
        callbacks.append(eval_callback)
        callbacks.append(training_callback)

        if use_WB:
            run = wandb.init(
                project="RL-NeSy",
                group=args.run_signature,
                name=model_name,
                dir=WB_path,
                sync_tensorboard=True,
                config=vars(args)
            )
            callbacks.append(WandbCallback())

        training_function = model.learn
        training_args = {'total_timesteps': args.timesteps_train, 'callback': CallbackList(callbacks)}

        profile_code(False, training_function, **training_args) #cProfile
        # raise SystemExit("Profiling complete. Exiting...")
        if args.restore_best_val_model: 
            model = eval_callback.restore_best_ckpt(env)
        else:
            model = training_callback.restore_last_ckpt(env)

        if use_WB: run.finish()

    # --- TEST ---
    import torch.nn as nn
    def strip_eval_modules(m: nn.Module):
        if isinstance(m, (nn.Dropout, nn.LayerNorm)):
            m.eval()          # freeze statistics
            m.training = False

    # --- freeze Dropout & LayerNorm ---
    model.policy.apply(strip_eval_modules)

    if args.use_kge_action:
        # model.policy._setup_kge_integration()
        model.policy.kge_inference_engine = kge_inference_engine
        model.policy.index_manager = index_manager
        kge_indices = [idx for pred, idx in index_manager.predicate_str2idx.items() if pred.endswith('_kge')]
        model.policy.kge_indices_tensor = torch.tensor(kge_indices, device=device, dtype=torch.long)

    model.policy = torch.compile(
        model.policy, mode="reduce-overhead", fullgraph=False
    )

    model.policy.set_training_mode(False)
        
    print('\nTest set evaluation...')
    eval_function = eval_corruptions
    if args.test_neg_samples is not None:
        print(f"Warning: Not using full samples will give different negatives for different env size")
    # Determine evaluation mode for eval_corruptions
    if args.kge_integration_strategy == 'sum_eval':
        eval_mode = 'hybrid'
    else:
        # For sum_logprob and learned_fusion, the model itself handles the fusion.
        eval_mode = 'rl_only'
    # eval_mode = 'kge_only'
    eval_args = {
        'model': model,
        'env': eval_env,
        'data': data_handler.test_queries,
        'sampler': sampler,
        'n_corruptions': args.test_neg_samples,
        'verbose': 2,
        'kge_inference_engine': kge_inference_engine,
        'evaluation_mode': eval_mode,
        'plot': False,
        'corruption_scheme': args.corruption_scheme,
    }
    
    # metrics_test = profile_code('cProfile', eval_function, **eval_args)
    metrics_test = profile_code(False, eval_function, **eval_args)
    print('results for:',args.run_signature)
    print_eval_info('Test', metrics_test)

    eval_only_test = True
    if not eval_only_test:
        print('Val set eval...')
        metrics_valid = eval_corruptions(model,
                                        eval_env,
                                        data_handler.valid_queries,
                                        sampler,
                                        n_corruptions=args.eval_neg_samples,
                                        evaluation_mode=eval_mode,
                                        kge_inference_engine=kge_inference_engine,
                                        corruption_scheme=args.corruption_scheme,
                                        )
        print_eval_info('Validation', metrics_valid)

        print('Train set eval...')
        metrics_train = eval_corruptions(model,
                                        eval_env,
                                        data_handler.train_queries,
                                        sampler,
                                        n_corruptions=args.train_neg_ratio,
                                        evaluation_mode=eval_mode,
                                        kge_inference_engine=kge_inference_engine,
                                        corruption_scheme=args.corruption_scheme,
                                        )
        print_eval_info('Train', metrics_train)
    else:
        metrics_train = {k: 0 for k in metrics_test.keys()}
        metrics_valid = {k: 0 for k in metrics_test.keys()}


    return metrics_train, metrics_valid, metrics_test