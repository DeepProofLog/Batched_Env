# DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs

Authors: Ying Jiao*, Rodrigo Castellano Ontiveros*, Luc De Raedt, Marco Gori, Francesco Giannini, Michelangelo Diligenti, and Giuseppe Marra.

Paper: [arXiv:2511.08581](https://arxiv.org/abs/2511.08581)

## Description
DeepProofLog (DPrL) is a neurosymbolic (NeSy) system that bridges probabilistic logic programming (under distribution semantics) with deep learning. It addresses the scalability challenges of traditional neurosymbolic models by parameterizing the derivation steps of stochastic logic programs with neural networks. 

By establishing a formal connection between the resolution process and Markov Decision Processes (MDPs), DPrL enables the use of efficient Reinforcement Learning techniques (like PPO) and dynamic programming for inference and learning. This allows the system to scale to significantly larger knowledge bases and more complex proof spaces than previously possible, as demonstrated on standard NeSy benchmarks and knowledge graph reasoning tasks.

## Repository Structure
- `runner.py`: High-level entry point for running experiments with unified configuration.
- `registry.py`: Experiment registry that manages different types of neurosymbolic tasks.
- `kge_experiments/`: Core implementation for Knowledge Graph Embedding and reasoning tasks.
    - `ppo.py`: Optimized PPO algorithm implementation.
    - `env.py`: Vectorized environment representing the logical resolution MDP.
    - `unification.py`: High-performance logic unification engine.
    - `policy.py`: Actor-Critic policy architectures for logical guidance.
    - `builder.py`: Factory for assembling environments, policies, and algorithms.
    - `config.py`: Configuration management for experiments.
    - `data/`: Dataset management scripts and processed benchmarks (e.g., Countries, Family, FB15k-237).

## Installation

### Prerequisites
- Python 3.12+
- CUDA-enabled GPU (for optimized vectorized execution)

### Setup
Create the conda environment using the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate rl
```

## Usage

The easiest way to run an experiment is using the `runner.py` script:

```bash
python runner.py --dataset countries_s3 --experiment kge --total_timesteps 50000
```

### Main Arguments
- `--dataset`: The dataset to use (e.g., `countries_s1`, `countries_s2`, `countries_s3`, `family`, `fb15k237`, `wn18rr`).
- `--experiment`: The experiment type (currently `kge`).
- `--n_envs`: Number of parallel environments for vectorized rollouts (default: 128).
- `--n_steps`: Number of steps per rollout (default: 128).
- `--batch_size`: PPO mini-batch size (default: 512).
- `--total_timesteps`: Total training timesteps.
- `--device`: Target device (`cuda` or `cpu`).
