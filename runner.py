import argparse
from registry import (
    load_config,
    build_env,
    build_policy,
    get_algorithm,
)
import torch
from kge_experiments.utils import print_results


def parse_args():
    parser = argparse.ArgumentParser(description="KGE Experiment Runner")
    parser.add_argument("--dataset", type=str, default="countries_s3", help="Dataset name")
    parser.add_argument("--n_envs", type=int, default=128, help="Number of environments")
    parser.add_argument("--n_steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--total_timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument("--eval_freq", type=int, default=2, help="Evaluation frequency (iterations)")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda")
    parser.add_argument("--experiment", type=str, default="kge", help="Experiment type (e.g., kge)")
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    
    print("=" * 70)
    print("Experiment Runner")
    print("=" * 70)
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config(args.experiment, 
        dataset=args.dataset,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        device=device,
    )
    print(f"\n[1] Config: {config.dataset}, device={config.device}, signature={config.run_signature}")
    
    env = build_env(config)
    print(f"[2] Environment built")
    
    policy = build_policy(config)
    print(f"[3] Policy built")
    
    algorithm = get_algorithm(policy, env, config)
    print(f"[4] Algorithm built")
    
    print(f"\n[5] Training...")
    algorithm.learn(total_timesteps=config.total_timesteps)
    
    print(f"\n[6] Evaluating...")
    results = algorithm.evaluate()
    
    print_results(results)
