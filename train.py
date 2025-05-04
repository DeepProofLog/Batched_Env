import numpy as np
import os
import random

import torch
import torch.profiler
from torch.profiler import ProfilerActivity, schedule

from env import LogicEnv_gym
from index_manager import IndexManager
from utils import get_device, print_eval_info
from my_callbacks import SB3ModelCheckpoint, CustomEvalCallback, EpochTimingCallback
from dataset import DataHandler
from model_SB3 import CustomActorCriticPolicy, CustomCombinedExtractor
from embeddings import get_embedder
from neg_sampling import get_sampler
from model_eval import eval_corruptions

# from stable_baselines3 import PPO
# from stable_baselines3 import DQN
from model_SB3 import PPO_custom as PPO
from stable_baselines3.common.callbacks import (
    StopTrainingOnMaxEpisodes,
    StopTrainingOnRewardThreshold,
    CallbackList,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb
from pykeen.triples import TriplesFactory



def main(args,log_filename,use_logger,use_WB,WB_path,date):

    print(args.run_signature)

    torch.manual_seed(args.seed_run_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed_run_i)
    random.seed(args.seed_run_i)
    np.random.seed(args.seed_run_i)

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")

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

    args.n_eval_queries = len(data_handler.valid_queries) if args.n_eval_queries == None else min(args.n_eval_queries, len(data_handler.valid_queries))
    args.n_test_queries = len(data_handler.test_queries) if args.n_test_queries == None else min(args.n_test_queries, len(data_handler.valid_queries))

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
    
    if args.corruption_mode == 'dynamic':
        np_facts = np.array([[f.args[0], f.predicate, f.args[1]] for f in data_handler.facts],dtype=str)
        triples_factory = TriplesFactory.from_labeled_triples(triples=np_facts,
                                                            entity_to_id=index_manager.constant_str2idx,
                                                            relation_to_id=index_manager.predicate_str2idx,
                                                            compact_id=False,
                                                            create_inverse_triples=False)
        
        data_handler.sampler = get_sampler(data_handler=data_handler, index_manager=index_manager, triples_factory=triples_factory,
                                           corruption_scheme=args.corruption_scheme)
        data_handler.triples_factory = triples_factory

    embedder_getter = get_embedder(args, 
                         data_handler, 
                         index_manager, 
                         device, 
                        #  n_body_constants=data_handler.n_digits if args.dataset_name == 'mnist_addition' else None, # to concat the body atoms
                         )
    embedder = embedder_getter.embedder

    # Define the state_embedding_size for the model
    args.atom_embedding_size = args.atom_embedding_size if args.atom_embedder != "concat" else (1+data_handler.max_arity)*args.atom_embedding_size
    args.state_embedding_size = args.atom_embedding_size if args.state_embedder != "concat" else args.atom_embedding_size*args.padding_atoms
    embedder.embed_dim = args.state_embedding_size


    # --- ENVIRONMENT ---

    def make_env(mode='train', seed=0, queries=None, labels=None):
        def _init():
            env = LogicEnv_gym(
                        index_manager=index_manager,
                        data_handler=data_handler,
                        queries=queries,
                        labels=labels,
                        mode=mode,
                        corruption_mode=args.corruption_mode,
                        corruption_scheme=args.corruption_scheme,
                        train_neg_pos_ratio=args.train_neg_pos_ratio,
                        seed=seed,
                        dynamic_consult=args.dynamic_consult,
                        max_depth=args.max_depth,
                        memory_pruning=args.memory_pruning,
                        end_proof_action=args.end_proof_action,
                        skip_unary_actions=args.skip_unary_actions,
                        padding_atoms=args.padding_atoms,
                        padding_states=args.padding_states,
                        device='cpu', 
                        engine=args.engine,
                        )
            env = Monitor(env)
            return env
        return _init

    # Create vectorized environments for training
    env_type = 'dummy'
    env_seeds = np.random.randint(0, 2**10, size=args.n_envs)
    eval_env_seeds = np.random.randint(0, 2**10, size=args.n_eval_envs)
    callback_env_seeds = np.random.randint(0, 2**10, size=1)
    if env_type == 'dummy':
        env = DummyVecEnv([make_env(
                                    mode='train', 
                                    seed=int(env_seeds[i]), 
                                    queries=data_handler.train_queries, 
                                    labels=[1]*len(data_handler.train_queries)
                                    ) 
                                    for i in range(args.n_envs)])
        
        eval_env = DummyVecEnv([make_env(
                                        mode='eval', 
                                        seed=int(eval_env_seeds[i]), 
                                        queries=data_handler.valid_queries,
                                        labels=[1]*len(data_handler.valid_queries),
                                        ) 
                                        for i in range(args.n_eval_envs)])
        
        callback_env = DummyVecEnv([make_env(
                                        mode='eval_with_restart', 
                                        seed=int(callback_env_seeds[i]), 
                                        queries=data_handler.valid_queries,
                                        labels=[1]*len(data_handler.valid_queries),
                                        ) 
                                        for i in range(1)])
    else:
        env = SubprocVecEnv([make_env(
                                    mode='train', 
                                    seed=int(env_seeds[i]), 
                                    queries=data_handler.train_queries, 
                                    labels=[1]*len(data_handler.train_queries)
                                    ) 
                                    for i in range(args.n_envs)])

        eval_env = SubprocVecEnv([make_env(
                                        mode='eval', 
                                        seed=int(eval_env_seeds[i]), 
                                        queries=data_handler.valid_queries,
                                        labels=[1]*len(data_handler.valid_queries),
                                        ) 
                                        for i in range(args.n_eval_envs)])

        callback_env = SubprocVecEnv([make_env(
                                        mode='eval',
                                        seed=int(callback_env_seeds[i]),
                                        queries=data_handler.valid_queries,
                                        labels=[1]*len(data_handler.valid_queries),
                                        )
                                        for i in range(1)])

    # --- INIT MODEL ---
    if args.model_name == "PPO":
        model = PPO(CustomActorCriticPolicy, 
                    env,
                    learning_rate=args.lr,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    verbose=1, 
                    device=device,
                    ent_coef=args.ent_coef,
                    clip_range=args.clip_range,
                    policy_kwargs={'features_extractor_class':CustomCombinedExtractor,
                                    'features_extractor_kwargs':{'features_dim':embedder.embed_dim,
                                                                    'embedder': embedder}})
    else:
        raise ValueError("Model not supported")
    

    # --- TRAIN ---
    model_path = os.path.join(args.models_path,args.run_signature, args.run_signature + f'-seed_{args.seed_run_i}')
    model_name = args.run_signature+'-'+date+'-seed_{}'.format(args.seed_run_i)
    if args.load_model:
        try:
            models = sorted(
                [m for m in os.listdir(model_path) if 'zip' in m and str(args.load_model) in m and f'seed_{args.seed_run_i}' in m])
            if args.restore_best_val_model:
                models = sorted([m for m in models if 'best_eval' in m])
            else:
                models = sorted([m for m in models if 'last_train' in m])
            if models:
                print(f"Loading model from {os.path.join(model_path, models[-1])}")
                model = PPO.load(os.path.join(model_path, models[-1]), env=eval_env, device=device)
            else:
                raise FileNotFoundError(f"No suitable model found in {model_path}")
        except (FileNotFoundError, ValueError) as e:
            print(e)
            args.load_model = False
        
    if not args.load_model and args.timesteps_train > 0:

        timing_callback = EpochTimingCallback(verbose=1)

        callbacks = [timing_callback]

        no_improvement_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, verbose=1)
        reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)

        eval_callback = CustomEvalCallback(eval_env=callback_env, 
                                    model_path=model_path if args.save_model else None,
                                    log_path=log_filename if use_logger else None,
                                    eval_freq=max(int(args.eval_freq//args.n_envs),1),
                                    n_eval_episodes=args.n_eval_queries-1,
                                    deterministic=True,
                                    render=False,
                                    name=model_name,
                                    callback_on_new_best=reward_threshold_callback if args.restore_best_val_model else None,
                                    # callback_after_eval=no_improvement_callback,
                                    )

        callbacks.append(eval_callback)
    
        checkpoint_callback = SB3ModelCheckpoint(model,monitor="rollout/ep_rew_mean", frequency=5000, 
                                                    total_steps=args.timesteps_train, 
                                                    model_path=model_path if args.save_model else None,
                                                    log_path = log_filename if use_logger else None,
                                                    name=model_name,)
        callbacks.append(checkpoint_callback)

        # Initialize a W&B run
        if use_WB:
            run = wandb.init(project = "RL-NeSy", 
                            group= args.run_signature,
                            name=model_name,
                            dir=WB_path,  
                            sync_tensorboard=True,
                            config = dict(
                                seed = args.seed_run_i,
                                shuffle_buffer = 1024,
                                batch_size = args.batch_size,
                                learning_rate = args.lr,
                                n_envs = args.n_envs,
                                epochs = args.n_epochs)) 
            
            callbacks.append(WandbCallback(
                                            # gradient_save_freq=100,
                                            # model_save_path=WB_path,
                                            # verbose=2,
                                            ))


        callbacks = CallbackList(callbacks)

        # --- Run the profiler ---
        profile = False
        if profile:
            import cProfile
            import pstats
            import io
            profiler = cProfile.Profile()
            profiler.enable()

        enable_torch_profiler = False
        if enable_torch_profiler:
            prof_activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available() and device.type == 'cuda':
                prof_activities.append(ProfilerActivity.CUDA)

            trace_dir = "./profiler_traces"
            os.makedirs(trace_dir, exist_ok=True)
            trace_file = f"{trace_dir}/full_training_trace.json"

            with torch.profiler.profile(
                activities=prof_activities,
                # schedule=schedule(wait=1, warmup=2, active=3, repeat=2),
                record_shapes=True,      # Optional: Enable for more detail if memory permits
                # profile_memory=True,     # Optional: Enable for memory usage if memory permits
                # with_stack=True          # Optional: Enable for stack traces if memory permits
            ) as prof:
                model.learn(total_timesteps=args.timesteps_train, callback=callbacks)

            # Export AFTER the 'with' block is finished
            print(f"\n--- Exporting Full PyTorch Profiling Trace to {trace_file} ---")
            try:
                prof.export_chrome_trace(trace_file)
                print(f"--- Trace exported to {trace_file}. Use chrome://tracing or Perfetto UI to view. ---")
            except Exception as e:
                print(f"--- Failed to export trace: {e} ---")

        else:
            model.learn(total_timesteps=args.timesteps_train, callback=callbacks)

        if profile:
            profiler.disable()
            # --- Analyze the results ---
            print("\n\n--- Profiling results ---")
            s = io.StringIO()
            # Sort by cumulative time spent in function and its callees
            stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            stats.print_stats(30) # Print top 30 functions
            print(s.getvalue())

            # Optional: Sort by time spent within the function itself (excluding sub-calls)
            print("\n\n--- Time spent in function itself (excluding sub-calls) ---")
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats('tottime') # or 'time'
            stats.print_stats(30)
            print(s.getvalue())

            # # --- Save the results to a file ---
            # profile_output_file = "profile_results.prof"
            # profiler.dump_stats(profile_output_file)
            # print(f"\nProfiling results saved to {profile_output_file}")

        if args.restore_best_val_model:
            eval_callback.restore_best_ckpt()

        if use_WB:
            run.finish()   

    # --- TEST ---   

    print('\nTest set eval...')

    eval_profiling = False
    if eval_profiling:
        import cProfile
        import pstats
        import io
        profiler = cProfile.Profile()
        profiler.enable()
    
    torch_eval_profiling = False
    if torch_eval_profiling:
        print("\n--- Starting PyTorch Profiler for Evaluation ---")
        prof_activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available() and device.type == 'cuda':
             prof_activities.append(ProfilerActivity.CUDA)
        
        with torch.profiler.profile(
            activities=prof_activities, record_shapes=True, profile_memory=True, with_stack=True
        ) as prof_eval:
            
            metrics_test = eval_corruptions(model,
                                            eval_env,
                                            data_handler.test_queries,
                                            corruption_mode=args.corruption_mode,
                                            corruptions=data_handler.test_corruptions if args.corruption_mode == 'static' else None,
                                            n_corruptions=args.test_negatives,
                                            consult_janus=False,
                                            verbose=1,
                                            )
            
        print("\n--- PyTorch Profiling Results (Evaluation) ---")
        sort_key = "cuda_time_total" if ProfilerActivity.CUDA in prof_activities else "cpu_time_total"
        print(prof_eval.key_averages().table(sort_by=sort_key, row_limit=30))
        # Optional: Export eval trace
        # prof_eval.export_chrome_trace(f"eval_trace_{args.run_signature}_seed{args.seed_run_i}.json")
    else:
        metrics_test = eval_corruptions(model,
                                        eval_env,
                                        data_handler.test_queries,
                                        n_corruptions=args.test_negatives,
                                        consult_janus=False,
                                        verbose=1,
                                        )
    if eval_profiling:
        profiler.disable()
        # --- Analyze the results ---
        print("\n\n--- Profiling results ---")
        s = io.StringIO()
        # Sort by cumulative time spent in function and its callees
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(30) # Print top 30 functions
        print(s.getvalue())

        # Optional: Sort by time spent within the function itself (excluding sub-calls)
        print("\n\n--- Time spent in function itself (excluding sub-calls) ---")
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('tottime') # or 'time'
        stats.print_stats(30)
        print(s.getvalue())

    print_eval_info('Test', metrics_test)

    eval_only_test = False
    if eval_only_test:
        print('Val set eval...')
        metrics_valid = eval_corruptions(model,
                                        eval_env,
                                        data_handler.valid_queries,
                                        n_corruptions=args.valid_negatives,
                                        consult_janus=False,
                                        )
        print_eval_info('Validation', metrics_valid)

        print('Train set eval...')
        metrics_train = eval_corruptions(model,
                                        eval_env,
                                        data_handler.train_queries,
                                        n_corruptions=args.train_neg_pos_ratio,
                                        consult_janus=True,
                                        )
        print_eval_info('Train', metrics_train)
    else:
        metrics_train = {k: 0 for k in metrics_test.keys()}
        metrics_valid = {k: 0 for k in metrics_test.keys()}


    return metrics_train, metrics_valid, metrics_test

