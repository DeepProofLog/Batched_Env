"""
MCTS Configuration.

Hyperparameters for MuZero-style Monte Carlo Tree Search.

Reference values from MuZero paper (Schrittwieser et al., 2020):
- num_simulations: 50 for Atari, 800 for board games
- pb_c_base: 19652, pb_c_init: 1.25 (PUCT exploration constants)
- root_dirichlet_alpha: 0.3 for chess, 0.25 for Go, 0.15 for shogi
- discount: 0.997 for Atari, 1.0 for board games
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class MCTSConfig:
    """Configuration for MuZero-style MCTS.

    Attributes:
        num_simulations: Number of MCTS simulations per action selection.
            Higher values give better action quality but slower inference.
            Typical: 25-50 for fast inference, 200-800 for best quality.

        pb_c_base: Base constant for PUCT exploration formula.
            Affects how quickly exploration bonus decays with visit count.

        pb_c_init: Initial constant for PUCT exploration formula.
            Higher values encourage more exploration early on.

        discount: Discount factor for future rewards (gamma).
            0.99 for continuing tasks, 1.0 for episodic without discounting.

        root_dirichlet_alpha: Alpha parameter for Dirichlet noise at root.
            Lower values (0.1-0.3) give more peaked distributions.
            Should be tuned based on action space size.

        root_exploration_fraction: Fraction of prior to replace with noise.
            0.25 is standard; higher values increase root exploration.

        value_nstep: Number of steps for n-step return bootstrapping.
            5 is standard; longer horizons for sparse reward settings.

        temperature_init: Initial temperature for action selection.
            1.0 samples proportionally to visit counts.

        temperature_final: Final temperature after decay.
            0.0 or 0.1 for near-greedy selection.

        temperature_decay_steps: Steps over which to decay temperature.

        hidden_dim: Dimension of latent state representation.
            Should match or exceed policy embedding dimension.

        unroll_steps: Number of steps to unroll for training.
            5 is standard; matches value_nstep typically.

        replay_buffer_size: Maximum trajectories in replay buffer.
            Larger buffers improve sample diversity.

        td_steps: Number of steps for TD target computation.
            Same as value_nstep in most implementations.

        batch_size: Batch size for training updates.

        learning_rate: Learning rate for optimizer.

        weight_decay: L2 regularization weight.

        max_grad_norm: Maximum gradient norm for clipping.
    """

    # =========================================================================
    # MCTS Search Parameters
    # =========================================================================
    num_simulations: int = 50
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    discount: float = 0.99

    # Root exploration noise (Dirichlet)
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    add_exploration_noise: bool = True

    # =========================================================================
    # Value Estimation
    # =========================================================================
    value_nstep: int = 5
    td_steps: int = 5  # Alias for value_nstep in some implementations

    # =========================================================================
    # Temperature Schedule for Action Selection
    # =========================================================================
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_decay_steps: int = 50000
    temperature_schedule: str = "linear"  # "linear", "constant", or "exponential"

    # =========================================================================
    # Network Architecture
    # =========================================================================
    hidden_dim: int = 256
    num_layers: int = 4

    # =========================================================================
    # Training Parameters
    # =========================================================================
    unroll_steps: int = 5
    replay_buffer_size: int = 100000
    min_buffer_size: int = 1000  # Minimum samples before training starts

    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0

    # Loss weights
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.5

    # =========================================================================
    # Data Collection
    # =========================================================================
    episodes_per_iteration: int = 100
    max_episode_steps: int = 20  # Max steps per episode

    # =========================================================================
    # Device and Compilation
    # =========================================================================
    device: str = "cuda"
    compile: bool = True
    compile_mode: str = "reduce-overhead"

    # =========================================================================
    # Batched MCTS Parameters (CUDA Graph Compatible)
    # =========================================================================
    # Batch size for batched MCTS operations (number of parallel environments)
    # Should match n_envs from training config
    mcts_batch_size: int = 100

    # Fixed batch size for evaluation slot recycling
    # If None, uses mcts_batch_size
    fixed_batch_size: Optional[int] = None

    # Maximum tree depth for tensor allocation (typically max_episode_steps + 1)
    max_tree_depth: Optional[int] = None

    # Maximum actions per state (padding_states from env, typically 120)
    max_actions: int = 120

    # Whether to use batched MCTS implementation
    use_batched_mcts: bool = True

    # =========================================================================
    # Logging and Checkpointing
    # =========================================================================
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 1000
    verbose: bool = True

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure td_steps matches value_nstep
        self.td_steps = self.value_nstep

        # Validate temperature schedule
        valid_schedules = {"linear", "constant", "exponential"}
        if self.temperature_schedule not in valid_schedules:
            raise ValueError(
                f"Invalid temperature_schedule: {self.temperature_schedule}. "
                f"Must be one of {valid_schedules}"
            )

        # Validate discount
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError(f"discount must be in [0, 1], got {self.discount}")

        # Validate exploration fraction
        if not 0.0 <= self.root_exploration_fraction <= 1.0:
            raise ValueError(
                f"root_exploration_fraction must be in [0, 1], "
                f"got {self.root_exploration_fraction}"
            )

        # Set defaults for batched MCTS parameters
        if self.max_tree_depth is None:
            self.max_tree_depth = self.max_episode_steps + 1

        if self.fixed_batch_size is None:
            self.fixed_batch_size = self.mcts_batch_size

        # Validate batched parameters
        if self.mcts_batch_size <= 0:
            raise ValueError(f"mcts_batch_size must be positive, got {self.mcts_batch_size}")

        if self.max_actions <= 0:
            raise ValueError(f"max_actions must be positive, got {self.max_actions}")

    def get_temperature(self, timestep: int) -> float:
        """Get temperature for action selection at given timestep.

        Args:
            timestep: Current training timestep.

        Returns:
            Temperature value for visit count -> action probability conversion.
        """
        if self.temperature_schedule == "constant":
            return self.temperature_init

        progress = min(1.0, timestep / max(1, self.temperature_decay_steps))

        if self.temperature_schedule == "linear":
            return self.temperature_init + progress * (
                self.temperature_final - self.temperature_init
            )
        elif self.temperature_schedule == "exponential":
            import math
            # Exponential decay from init to final
            log_init = math.log(max(self.temperature_init, 1e-8))
            log_final = math.log(max(self.temperature_final, 1e-8))
            return math.exp(log_init + progress * (log_final - log_init))

        return self.temperature_init

    def get_puct_constant(self, parent_visit_count: int) -> float:
        """Compute PUCT exploration constant.

        The PUCT formula uses a visit-count-dependent exploration term:
            c(s) = log((1 + N(s) + pb_c_base) / pb_c_base) + pb_c_init

        This makes exploration bonus decrease as parent is visited more.

        Args:
            parent_visit_count: Total visits to parent node.

        Returns:
            PUCT constant c(s) for UCB computation.
        """
        import math
        return (
            math.log((1 + parent_visit_count + self.pb_c_base) / self.pb_c_base)
            + self.pb_c_init
        )
