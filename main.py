import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import random
import os
import copy
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from joblib import Parallel, delayed
import highway_env

# At the top with other constants
ARTIFACTS_DIR = os.path.join("artifacts", "highway-ppo")


# Then ensure_artifacts_dir
def ensure_artifacts_dir(custom_path=None):
    """Create the artifacts directory if it doesn't exist."""
    artifacts_dir = custom_path or ARTIFACTS_DIR
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir


# Directory to store log files (under artifacts)
LOGS_DIR = os.path.join(ARTIFACTS_DIR, "logs")


# Configure logging functionality
def setup_master_logger(log_level=logging.INFO):
    """
    Create and configure the master logger.
    This logger writes to:
      - A single 'master.log' file in LOGS_DIR
      - The console (stdout) with INFO-level
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log_path = os.path.join(LOGS_DIR, f"{timestamp}_master.log")

    # Configure master logger
    logger = logging.getLogger("master_logger")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear any existing handlers

    # File handler for master.log
    fh = logging.FileHandler(master_log_path)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    # Console handler at INFO-level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(ch)

    logger.info(f"Master logger initialized. Log file: {master_log_path}")
    return logger


def setup_experiment_logger(
    experiment_id, log_level=logging.INFO, console_level=logging.WARNING
):
    """
    Create and configure a per-experiment logger.
    Writes detailed logs to 'experiment_<id>.log'
    Logs only warnings or higher to the console to avoid spamming.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger_name = f"experiment_{experiment_id}"
    exp_log_path = os.path.join(LOGS_DIR, f"{timestamp}_{logger_name}.log")

    # Configure experiment logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.handlers = []  # Clear any existing handlers

    # File handler for experiment logs - capture all logs
    fh = logging.FileHandler(exp_log_path)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    # Console handler - only show important messages
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)  # Higher level to reduce console noise
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%H:%M:%S"
        )
    )
    logger.addHandler(ch)

    logger.info(
        f"Experiment logger initialized for experiment_{experiment_id}. Log file: {exp_log_path}"
    )
    return logger


# For backward compatibility with existing code
def setup_logger(experiment_name="", log_level=logging.INFO):
    """Legacy function for backward compatibility"""
    if experiment_name:
        return setup_experiment_logger(experiment_name, log_level)
    else:
        return setup_master_logger(log_level)


# if highway_env is not registered, register it
if "highway-v0" not in gym.envs.registry:
    highway_env._register_highway_envs()

# This implementation uses highway-env's built-in observation normalization
# By setting "normalize": True in the environment config, observations are
# automatically scaled to the range [-1, 1] based on features_range

# Constants for reproducibility
SEED = 42

# PPO Hyperparameters
LEARNING_RATE = 1e-4  # Learning rate for optimizer (typical: 1e-4 to 3e-4)
GAMMA = 0.99  # Discount factor (typical: 0.99, sometimes 0.999)
LAMBDA = 0.95  # GAE parameter (typical: 0.95-0.98)
EPSILON_CLIP = 0.2  # PPO clip range (typical: 0.2 - default in PPO paper)
ENTROPY_COEF = 0.005  # Entropy coefficient (typical: 0.001-0.01)
VALUE_COEF = 0.5  # Value function coefficient (typical: 0.5)
MAX_GRAD_NORM = 0.5  # Maximum gradient norm for clipping
EPOCHS = 6  # Number of epochs when optimizing (typical: 4-10 epochs per batch)
BATCH_SIZE = 64  # Mini-batch size (typical: 64)
HIDDEN_DIM = 128  # Size of hidden layers in network (typical: 64-256)
STEPS_PER_UPDATE = (
    2048  # Number of steps to collect before updating policy (typical: 1024-2048)
)

# Training parameters
MAX_EPISODES = 1500  # Maximum number of episodes to train
TARGET_REWARD = (
    20.0  # Target reward to consider environment solved (adjusted for highway-env)
)
LOG_INTERVAL = 10  # How often to log training progress
EVAL_INTERVAL = 50  # How often to run evaluation

# Highway environment configuration
HIGHWAY_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,  # Number of vehicles to observe
        "features": ["x", "y", "vx", "vy"],  # Features to include
        "normalize": True,  # Use built-in highway-env normalization
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-30, 30],
            "vy": [-30, 30],
            "presence": [0, 1],
            "cos_h": [-1, 1],
            "sin_h": [-1, 1],
        },
        "absolute": False,
        "order": "random",
    },
    "action": {
        "type": "ContinuousAction",  # Continuous steering + throttle
        "longitudinal": True,  # Enable acceleration control
        "lateral": True,  # Enable steering control
    },
    "simulation_frequency": 15,  # Simulation steps per second
    "policy_frequency": 5,  # Decision frequency
    "duration": 40,  # Episode duration in seconds
    "lanes_count": 3,  # Number of lanes
    "vehicles_count": 50,  # Total vehicles in the environment
    "vehicles_density": 2,  # Initial density of vehicles
    "collision_reward": -1,  # Reward for colliding with a vehicle
    "right_lane_reward": 0.1,  # Reward for driving on the right lane
    "high_speed_reward": 0.4,  # Reward for driving at full speed
    "lane_change_reward": -0.05,  # Reward for changing lanes
    "reward_speed_range": [20, 30],  # Speed range for positive reward
}


# Setup reproducibility across all libraries
def set_random_seeds(seed=SEED, exact_reproducibility=False):
    """Set random seeds for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Only enforce deterministic behavior if exact reproducibility needed
    if exact_reproducibility:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow cuDNN to benchmark and select fastest algorithms
        torch.backends.cudnn.benchmark = True


# Determine the optimal compute device
def get_device():
    """Determine and return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda"), f"GPU: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon GPU"
    else:
        return torch.device("cpu"), "CPU"


# Set seeds for reproducibility
set_random_seeds()

# Set device for training
device, device_name = get_device()
print(f"Using {device_name} for training.")


# Neural Network Architecture
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # Shared feature extractor with ReLU activations
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor mean (for continuous actions)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # Remove Tanh here as we'll apply it after sampling
        )

        # Log standard deviation (learnable parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic (Value Function) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Convert numpy arrays to tensors
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)

        shared_features = self.shared(x)

        # Actor: action mean and std
        action_mean = self.actor_mean(shared_features)
        action_std = self.log_std.exp()

        # Critic: state value
        state_value = self.critic(shared_features)

        return action_mean, action_std, state_value

    def get_action(self, state, deterministic=False):
        # Get mean and std
        action_mean, action_std, value = self.forward(state)

        # Create normal distribution
        normal_dist = Normal(action_mean, action_std)

        if deterministic:
            # For deterministic, just use the mean (no sampling)
            z = action_mean
            action = torch.tanh(z)
            log_prob = None  # Not needed for deterministic actions
        else:
            # Sample from normal distribution (pre-tanh)
            z = normal_dist.sample()
            # Apply tanh to bound actions to [-1, 1]
            action = torch.tanh(z)

            # Compute log_prob with change of variables formula
            # log π(a) = log π(z) - log(1 - tanh²(z))
            # = log π(z) - sum(log(1 - a²))
            log_prob = normal_dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)

        return (
            action.cpu().numpy(),
            z.cpu().numpy(),
            log_prob.item() if log_prob is not None else None,
            value.cpu().numpy()[0],
        )

    def evaluate(self, states, actions, pre_tanh_actions):
        # Get action distribution parameters and state values
        action_means, action_stds, state_values = self.forward(states)

        # Create normal distributions
        dist = Normal(action_means, action_stds)

        # Get log probabilities using pre-tanh actions
        log_probs = dist.log_prob(pre_tanh_actions)

        # Apply change of variables formula for tanh transformation
        tanh_actions = torch.tanh(pre_tanh_actions)
        log_probs = log_probs - torch.log(1 - tanh_actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)

        # Get entropy (sum across action dimensions)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, state_values, entropy


# Memory Buffer
class PPOMemory:
    def __init__(self, batch_size=64):
        self.states = []
        self.actions = []  # Store post-Tanh actions for environment interaction
        self.pre_tanh_actions = []  # Store pre-Tanh actions for correct log_prob calculation
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.dones = []
        self.values = []
        self.batch_size = batch_size

    def store(
        self, state, action, pre_tanh_action, reward, next_state, log_prob, done, value
    ):
        self.states.append(state)
        self.actions.append(action)
        self.pre_tanh_actions.append(pre_tanh_action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.pre_tanh_actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.dones = []
        self.values = []

    def compute_advantages(self, gamma=0.99, lam=0.95, last_value=0):
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0

        # Calculate GAE (Generalized Advantage Estimation)
        for t in reversed(range(len(rewards))):
            # If it's the terminal state, there's no next value, so we set the delta to reward - value
            # Otherwise, we calculate delta as reward + gamma * next_value * (1 - done) - value
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        # Calculate returns
        returns = advantages + np.array(self.values)

        return advantages, returns

    def get_batches(self):
        n_states = len(self.states)
        batch_start_indices = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start_indices]

        return batches

    def get_tensors(self):
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        pre_tanh_actions = torch.FloatTensor(np.array(self.pre_tanh_actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)

        return states, actions, pre_tanh_actions, old_log_probs


# PPO Agent
class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        lam=LAMBDA,
        eps_clip=EPSILON_CLIP,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        logger=None,
    ):
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.logger = logger or logging.getLogger("ppo_highway")

        self.memory = PPOMemory(batch_size)

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            # Get action, pre_tanh_action, log probability, and value
            action, pre_tanh_action, log_prob, value = self.actor_critic.get_action(
                state, deterministic
            )

        return action, pre_tanh_action, log_prob, value

    def update(self, last_value=0.0):
        # Get data from memory
        states, actions, pre_tanh_actions, old_log_probs = self.memory.get_tensors()
        advantages, returns = self.memory.compute_advantages(
            self.gamma, self.lam, last_value
        )

        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)

        # Normalize advantages (helpful for stable training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get batches
        batches = self.memory.get_batches()

        # Metrics to track
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_loss = 0
        clip_fraction = 0
        approx_kl_div = 0
        explained_var = 0

        # Optimization loop
        for epoch in range(self.epochs):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            epoch_total_loss = 0
            epoch_clip_count = 0
            epoch_kl_sum = 0

            for batch_indices in batches:
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_pre_tanh_actions = pre_tanh_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions using pre-tanh actions for correct log probs
                new_log_probs, state_values, entropy = self.actor_critic.evaluate(
                    batch_states, batch_actions, batch_pre_tanh_actions
                )

                # Calculate ratios
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # Calculate KL divergence
                with torch.no_grad():
                    log_ratio = new_log_probs - batch_old_log_probs
                    batch_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    epoch_kl_sum += batch_kl

                # Calculate surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                    * batch_advantages
                )

                # Calculate actor loss (negative because we perform gradient ascent)
                actor_loss = -torch.min(surr1, surr2).mean()

                # Calculate critic loss - ensure shapes are compatible
                state_values = state_values.squeeze(-1)  # Fix potential shape issues
                critic_loss = F.mse_loss(state_values, batch_returns)

                # Calculate entropy bonus
                entropy_bonus = entropy.mean()
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy_bonus
                )

                # Update weights
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Count clipped samples
                with torch.no_grad():
                    clip_count = (
                        (torch.abs(ratios - 1.0) > self.eps_clip).float().sum().item()
                    )
                    epoch_clip_count += clip_count / len(batch_indices)

                # Accumulate losses for this epoch
                epoch_policy_loss += actor_loss.item()
                epoch_value_loss += critic_loss.item()
                epoch_entropy += entropy_bonus.item()
                epoch_total_loss += loss.item()

            # Average over batches
            num_batches = len(batches)
            epoch_policy_loss /= num_batches
            epoch_value_loss /= num_batches
            epoch_entropy /= num_batches
            epoch_total_loss /= num_batches
            epoch_clip_fraction = epoch_clip_count / num_batches
            epoch_approx_kl = epoch_kl_sum / num_batches

            # Accumulate for overall metrics
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            total_entropy += epoch_entropy
            total_loss += epoch_total_loss
            clip_fraction += epoch_clip_fraction
            approx_kl_div += epoch_approx_kl

            # Log epoch details in compact key=value format at DEBUG level
            self.logger.debug(
                "epoch=%d/%d loss=%.4f policy_loss=%.4f value_loss=%.4f entropy=%.4f clip_frac=%.3f kl=%.5f",
                epoch + 1,
                self.epochs,
                epoch_total_loss,
                epoch_policy_loss,
                epoch_value_loss,
                epoch_entropy,
                epoch_clip_fraction,
                epoch_approx_kl,
            )

        # Calculate explained variance
        with torch.no_grad():
            y_pred = torch.FloatTensor(self.memory.values).to(device)
            y_true = returns[:-1] if len(returns) > len(y_pred) else returns
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / var_y if var_y > 0 else 0
            explained_var = explained_var.item()

        # Average over epochs
        avg_policy_loss = total_policy_loss / self.epochs
        avg_value_loss = total_value_loss / self.epochs
        avg_entropy = total_entropy / self.epochs
        avg_total_loss = total_loss / self.epochs
        avg_clip_fraction = clip_fraction / self.epochs
        avg_approx_kl = approx_kl_div / self.epochs

        # Log update summary in more concise key=value format at INFO level
        self.logger.info(
            "update_complete loss=%.4f policy_loss=%.4f value_loss=%.4f entropy=%.4f clip_frac=%.3f kl=%.5f explained_var=%.3f",
            avg_total_loss,
            avg_policy_loss,
            avg_value_loss,
            avg_entropy,
            avg_clip_fraction,
            avg_approx_kl,
            explained_var,
        )

        # Clear memory
        self.memory.clear()

        # Return metrics for potential further use
        return {
            "loss": avg_total_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "clip_fraction": avg_clip_fraction,
            "approx_kl": avg_approx_kl,
            "explained_variance": explained_var,
        }

    def save(self, path):
        artifacts_dir = ensure_artifacts_dir()
        full_path = os.path.join(artifacts_dir, path)

        # Save both model and optimizer state for full resumability
        checkpoint = {
            "model": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer_stats": None,  # If you ever add normalizer back
            "config": {
                "state_dim": self.actor_critic.shared[0].in_features,
                "action_dim": len(self.actor_critic.log_std),
                "hidden_dim": self.actor_critic.shared[0].out_features,
                "lr": self.optimizer.param_groups[0]["lr"],
                "gamma": self.gamma,
                "lam": self.lam,
                "eps_clip": self.eps_clip,
            },
        }

        torch.save(checkpoint, full_path)
        self.logger.info("model_saved path=%s", full_path)

    def load(self, path, load_optimizer=True):
        if os.path.dirname(path) == "":
            artifacts_dir = ensure_artifacts_dir()
            path = os.path.join(artifacts_dir, path)

        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint["model"])

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("model_loaded path=%s", path)
        return checkpoint.get("config", {})  # Return config for inspection


# Training function
def train(
    env,
    agent,
    max_episodes=500,
    target_reward=0.0,
    log_interval=20,
    eval_interval=50,
    steps_per_update=STEPS_PER_UPDATE,
    logger=None,
    experiment_name="",
):
    # Initialize logger if not provided
    if logger is None:
        logger = setup_logger()

    logger.info("Starting training...")
    logger.info(f"Device: {device_name}")
    logger.info(f"Max episodes: {max_episodes}, Target reward: {target_reward}")
    logger.info(f"Environment: {env.spec.id}")
    logger.info(f"Steps per update: {steps_per_update}, PPO epochs: {agent.epochs}")
    logger.info(
        f"Learning rate: {agent.optimizer.param_groups[0]['lr']}, Gamma: {agent.gamma}, Lambda: {agent.lam}"
    )
    logger.info(
        f"Clip epsilon: {agent.eps_clip}, Value coef: {agent.value_coef}, Entropy coef: {agent.entropy_coef}"
    )

    # For tracking progress
    rewards = []  # Evaluation rewards
    episode_rewards = []  # Individual episode rewards during training
    avg_rewards = []  # Moving average of evaluation rewards
    training_episodes = []  # To track episode numbers for plotting
    eval_episodes = [0]  # To track episode numbers for evaluations
    best_avg_reward = -float("inf")

    # For storing metrics
    metrics_history = {
        "episode_rewards": [],
        "eval_rewards": [],
        "avg_eval_rewards": [],
        "policy_updates": [],
        "episode_numbers": [],
        "eval_episode_numbers": [],
        "timestamps": [],
    }

    # For early stopping
    solved = False

    start_time = time.time()
    total_steps = 0
    episode_num = 0

    # Ensure artifacts directory exists
    artifacts_dir = ensure_artifacts_dir()

    # Do initial evaluation
    logger.info("Performing initial evaluation...")
    eval_reward = evaluate(env, agent, num_episodes=5)
    rewards.append(eval_reward)
    avg_rewards.append(eval_reward)
    metrics_history["eval_rewards"].append(eval_reward)
    metrics_history["avg_eval_rewards"].append(eval_reward)
    metrics_history["eval_episode_numbers"].append(0)
    metrics_history["timestamps"].append(0)
    logger.info(f"initial_eval reward={eval_reward:.2f}")

    while episode_num < max_episodes:
        # Collect a batch of transitions
        steps_collected = 0
        update_start_time = time.time()

        while steps_collected < steps_per_update and episode_num < max_episodes:
            episode_num += 1
            state, _ = env.reset(seed=SEED + episode_num)

            # Flatten observation from (N, F) to (N*F,)
            state = state.reshape(-1)

            episode_reward = 0
            done = False
            episode_steps = 0

            while not done and steps_collected < steps_per_update:
                # Select action with normalized state
                action, pre_tanh_action, log_prob, value = agent.select_action(state)

                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Flatten next_state from (N, F) to (N*F,)
                next_state = next_state.reshape(-1)

                # Store in memory (using normalized state and both action forms)
                agent.memory.store(
                    state,
                    action,
                    pre_tanh_action,
                    reward,
                    next_state,
                    log_prob,
                    done,
                    value,
                )

                # Update state and reward
                state = next_state
                episode_reward += reward
                steps_collected += 1
                total_steps += 1
                episode_steps += 1

                # If we've collected enough steps or episode is done, we can stop
                if steps_collected >= steps_per_update:
                    break

            # Record the completed episode's reward and episode number
            episode_rewards.append(episode_reward)
            training_episodes.append(episode_num)
            metrics_history["episode_rewards"].append(episode_reward)
            metrics_history["episode_numbers"].append(episode_num)

            # Log episode info based on log_interval
            if episode_num % log_interval == 0:
                avg_ep_reward = np.mean(episode_rewards[-log_interval:])
                elapsed_time = time.time() - start_time
                logger.info(
                    "episode=%d reward=%.2f avg_reward=%.2f steps=%d episode_steps=%d time=%.2fs",
                    episode_num,
                    episode_reward,
                    avg_ep_reward,
                    total_steps,
                    episode_steps,
                    elapsed_time,
                )

            # Check for evaluation based on eval_interval - do it here to ensure evaluations happen exactly every eval_interval episodes
            if episode_num % eval_interval == 0:
                logger.info(f"Evaluating at episode {episode_num}...")
                eval_reward = evaluate(env, agent, num_episodes=5)
                rewards.append(eval_reward)
                eval_episodes.append(episode_num)
                eval_time = time.time() - start_time

                # Calculate average reward from last 10 evaluations
                avg_reward = (
                    np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                )
                avg_rewards.append(avg_reward)

                # Store in metrics
                metrics_history["eval_rewards"].append(eval_reward)
                metrics_history["avg_eval_rewards"].append(avg_reward)
                metrics_history["eval_episode_numbers"].append(episode_num)
                metrics_history["timestamps"].append(eval_time)

                logger.info(
                    "eval episode=%d reward=%.2f avg_reward=%.2f time=%.2fs",
                    episode_num,
                    eval_reward,
                    avg_reward,
                    eval_time,
                )

                # Check if environment is solved
                if avg_reward >= target_reward and not solved and len(rewards) >= 10:
                    logger.info(
                        f"Environment solved in {episode_num} episodes! Average reward: {avg_reward:.2f}"
                    )
                    # Save the model
                    agent.save("ppo_highway_solved.pth")
                    solved = True

                # Save the best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    agent.save("ppo_highway_best.pth")
                    logger.info(
                        f"New best model saved with average reward: {best_avg_reward:.2f}"
                    )

            # Only break outer loop if we've collected enough steps
            if steps_collected >= steps_per_update:
                break

        # Calculate the final state value for bootstrapping
        final_value = 0.0
        if not done:  # If we stopped collection mid-episode
            with torch.no_grad():
                # Fix: Properly unpack three values (mean, std, value) from forward method
                _, _, final_value = agent.actor_critic(state)
                final_value = final_value.cpu().item()

        # Update policy with proper bootstrapping after collecting full batch
        logger.debug(f"Updating policy after collecting {steps_collected} steps...")
        update_metrics = agent.update(last_value=final_value)
        update_time = time.time() - update_start_time

        # Store policy update metrics
        metrics_history["policy_updates"].append(
            {
                "episode": episode_num,
                "steps_collected": steps_collected,
                "time": update_time,
                **update_metrics,
            }
        )

        logger.debug(f"Policy update completed in {update_time:.2f}s")

    # Save training metrics to JSON file
    metrics_path = os.path.join(artifacts_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")

    # Plot rewards
    plt.figure(figsize=(12, 8))

    # Plot training rewards - fixing the dimension issue by ensuring both arrays have the same length
    plt.plot(
        training_episodes,
        episode_rewards,
        alpha=0.3,
        color="gray",
        label="Training Episode Reward",
    )

    # Plot smoothed training rewards
    window_size = 20
    if len(episode_rewards) > window_size:
        smoothed_rewards = np.convolve(
            episode_rewards, np.ones(window_size) / window_size, mode="valid"
        )
        plt.plot(
            training_episodes[window_size - 1 :],
            smoothed_rewards,
            color="blue",
            label=f"Training Reward (Moving Avg {window_size})",
        )

    # Plot evaluation rewards - using eval_episodes which tracks when evaluations happen
    plt.plot(
        eval_episodes,
        rewards,
        "ro-",
        label="Evaluation Reward (5 episodes)",
        markersize=8,
    )

    # Plot average evaluation rewards
    plt.plot(
        eval_episodes,
        avg_rewards,
        "go-",
        label="Evaluation Reward (Moving Avg)",
        markersize=6,
    )

    # Plot target line
    plt.axhline(y=target_reward, color="r", linestyle="--", label="Target Reward")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Progress on Highway-v0")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot to artifacts directory
    plot_path = os.path.join(artifacts_dir, "ppo_highway_rewards.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Training plot saved to {plot_path}")

    # Create CSV summary for this run
    csv_path = os.path.join(artifacts_dir, f"summary_{experiment_name}.csv")
    with open(csv_path, "w") as f:
        f.write(
            "experiment,final_reward,max_reward,training_steps,best_model_path,plot_path\n"
        )
        f.write(
            f"{experiment_name},{avg_rewards[-1]:.4f},{max(avg_rewards):.4f},{total_steps},"
        )
        f.write(
            f"{os.path.join(artifacts_dir, f'ppo_highway_best_{experiment_name}.pth')},"
        )
        f.write(f"{plot_path}\n")

    # Also update metrics_history to include artifact paths
    metrics_history["best_model_path"] = os.path.join(
        artifacts_dir, f"ppo_highway_best_{experiment_name}.pth"
    )
    metrics_history["plot_path"] = plot_path

    logger.info("Training completed!")
    return rewards, avg_rewards, metrics_history


# Evaluation function
def evaluate(env, agent, num_episodes=10, render=False):
    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=SEED + 1000 + episode)

        # Flatten observation
        state = state.reshape(-1)

        episode_reward = 0
        done = False

        while not done:
            # Select action (deterministic) with normalized state
            action, _, _, _ = agent.select_action(state, deterministic=True)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Flatten next_state
            next_state = next_state.reshape(-1)

            # Update state and reward
            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    return avg_reward


# Function to visualize a trained agent
def visualize_agent(env, agent, num_episodes=3, logger=None):
    if logger is None:
        logger = logging.getLogger("ppo_highway")

    logger.info("Visualizing agent...")

    # Register highway environment
    if "highway-v0" not in gym.envs.registry:
        highway_env._register_highway_envs()
    logger.info("Registered highway-env in visualization process")

    # Create visualization environment once with render mode
    env_viz = gym.make("highway-v0", render_mode="human", config=HIGHWAY_CONFIG)

    try:
        for episode in range(num_episodes):
            state, _ = env_viz.reset(seed=SEED + 2000 + episode)

            # Flatten observation
            state = state.reshape(-1)

            episode_reward = 0
            done = False

            while not done:
                # Select action (deterministic)
                action, _, _, _ = agent.select_action(state, deterministic=True)

                # Take action
                next_state, reward, terminated, truncated, _ = env_viz.step(action)
                done = terminated or truncated

                # Flatten next_state
                next_state = next_state.reshape(-1)

                # Update state and reward
                state = next_state
                episode_reward += reward

            logger.info("viz_episode=%d reward=%.2f", episode + 1, episode_reward)

    finally:
        # Ensure environment is closed even if an exception occurs
        env_viz.close()


# Main function
def main():
    # Create environment with continuous actions
    env = gym.make("highway-v0", config=HIGHWAY_CONFIG)

    # Calculate state dimension by flattening observation space
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]

    # Setup master logger
    master_logger = setup_master_logger()
    master_logger.info(f"Starting Highway-Env PPO Training")
    master_logger.info(f"Using {device_name} for training")
    master_logger.info(f"State dimension: {state_dim}")
    master_logger.info(f"Action dimension: {action_dim}")

    # Flag to run a single training session instead of hyperparameter sweep
    run_single = False

    if run_single:
        master_logger.info("\n=== Running Single Training Session ===")

        # Create experiment logger for the single run
        experiment_logger = setup_experiment_logger(
            "single_run", console_level=logging.INFO
        )

        # Create agent with default parameters
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=LEARNING_RATE,
            gamma=GAMMA,
            lam=LAMBDA,
            eps_clip=EPSILON_CLIP,
            value_coef=VALUE_COEF,
            entropy_coef=ENTROPY_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            logger=experiment_logger,
        )

        # Train the agent
        rewards, avg_rewards, metrics_history = train(
            env=env,
            agent=agent,
            max_episodes=300,  # Shorter run for testing
            target_reward=TARGET_REWARD,
            log_interval=LOG_INTERVAL,
            eval_interval=EVAL_INTERVAL,
            steps_per_update=STEPS_PER_UPDATE,
            logger=experiment_logger,
            experiment_name="single_run",
        )

        # Log final results to master logger
        master_logger.info(
            f"Single run completed with final reward: {avg_rewards[-1]:.2f}"
        )

        # Visualize the trained agent
        experiment_logger.info("Visualizing trained agent...")
        # Register highway environment
        if "highway-v0" not in gym.envs.registry:
            highway_env._register_highway_envs()
            experiment_logger.info("Registered highway-env for visualization")

        viz_env = gym.make("highway-v0", config=HIGHWAY_CONFIG, render_mode="human")
        visualize_agent(viz_env, agent, num_episodes=3, logger=experiment_logger)
        viz_env.close()

        # Close the training environment
        env.close()
    else:
        # Run experiments with different hyperparameter values in parallel
        master_logger.info("Running hyperparameter experiments in parallel")
        # Set n_jobs to control parallelism: -1 means all cores, 2 means 2 workers, etc.
        run_experiments(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            hyperparams_to_vary={
                "epochs": [4, 6, 8],
                "lr": [1e-4, 3e-4],
                "hidden_dim": [64, 128, 256],
                "features": [
                    ["x", "y", "vx", "vy"],  # Basic features
                    ["presence", "x", "y", "vx", "vy"],  # With vehicle presence
                    [
                        "x",
                        "y",
                        "vx",
                        "vy",
                        "cos_h",
                        "sin_h",
                    ],  # With heading information
                ],
                "batch_size": [32, 64, 128],
            },
            n_jobs=42,  # Adjust based on your CPU cores and memory
            logger=master_logger,
        )

    master_logger.info("All experiments completed successfully!")
    # No need to close environment as it's already closed in run_experiments


# Function to run experiments with different hyperparameters
def run_experiments(
    env, state_dim, action_dim, hyperparams_to_vary, n_jobs=-1, logger=None
):
    """
    Run multiple experiments with different hyperparameter values in parallel.

    Args:
        env: The environment to train on (will be closed and recreated in each worker)
        state_dim: State dimension
        action_dim: Action dimension
        hyperparams_to_vary: Dict mapping hyperparameter names to lists of values to try
        n_jobs: Number of parallel jobs to run (-1 for all available cores)
        logger: Master logger instance for logging (should be created in the main process)
    """
    # Initialize master logger if not provided
    master_logger = logger or setup_master_logger()

    # Default hyperparameters
    default_hyperparams = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "lr": LEARNING_RATE,
        "gamma": GAMMA,
        "lam": LAMBDA,
        "eps_clip": EPSILON_CLIP,
        "value_coef": VALUE_COEF,
        "entropy_coef": ENTROPY_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
    }

    # Generate all hyperparameter combinations
    import itertools

    param_names = list(hyperparams_to_vary.keys())
    param_values = list(
        itertools.product(*(hyperparams_to_vary[name] for name in param_names))
    )

    total_experiments = len(param_values)
    master_logger.info(
        f"\nRunning {total_experiments} experiments with varying {', '.join(param_names)} using {n_jobs} parallel workers.\n"
    )

    # We need to close the environment as each worker will create its own
    env.close()

    # Define the function to run a single experiment (to be executed in parallel)
    def run_single_experiment(experiment_idx, values):
        # Create a deep copy of the highway config to avoid cross-contamination
        local_config = copy.deepcopy(HIGHWAY_CONFIG)

        param_desc = []
        feature_set = None

        # Create unique experiment ID
        experiment_id = f"exp_{experiment_idx}"

        # Create experiment logger - only warnings and errors go to console
        experiment_logger = setup_experiment_logger(experiment_id)

        # First, extract any special hyperparameters like "features"
        for name, value in zip(param_names, values):
            if name == "features":
                # Store the feature set for environment creation
                feature_set = value
                # Create a shorter representation for the experiment name
                feature_str = ",".join(value)
                param_desc.append(f"feat={feature_str}")

        # Update the config with the feature set if provided
        if feature_set is not None:
            local_config["observation"]["features"] = feature_set

        # Register highway environment
        if "highway-v0" not in gym.envs.registry:
            highway_env._register_highway_envs()
            experiment_logger.info("Registered highway-env in worker process")

        # Create a new environment for this worker with the potentially modified config
        worker_env = gym.make("highway-v0", config=local_config)

        # Calculate state dimension by flattening observation space
        worker_state_dim = np.prod(worker_env.observation_space.shape)
        worker_action_dim = worker_env.action_space.shape[0]

        # Set a unique seed for this experiment
        worker_seed = SEED + experiment_idx * 1000
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

        # Create hyperparameter set for this experiment
        experiment_hyperparams = default_hyperparams.copy()
        # Update state_dim to worker's calculated dimension
        experiment_hyperparams["state_dim"] = worker_state_dim
        experiment_hyperparams["action_dim"] = worker_action_dim

        # Now add the remaining scalar hyperparameters
        for name, value in zip(param_names, values):
            if name != "features":  # Skip features as they're handled differently
                experiment_hyperparams[name] = value
                param_desc.append(f"{name}={value}")

        experiment_name = "_".join(param_desc)
        experiment_logger.info(
            f"\n=== Starting Experiment {experiment_idx + 1}/{total_experiments}: {experiment_name} ===\n"
        )
        experiment_logger.info(
            f"State dimension: {worker_state_dim}, Action dimension: {worker_action_dim}"
        )

        if feature_set is not None:
            experiment_logger.info(f"Using features: {feature_set}")

        # Create agent with these hyperparameters
        agent = PPOAgent(**experiment_hyperparams, logger=experiment_logger)

        # Train agent with modified model saving to include hyperparameter values
        rewards, avg_rewards, metrics_history = train_with_experiment_name(
            env=worker_env,
            agent=agent,
            max_episodes=MAX_EPISODES,
            target_reward=TARGET_REWARD,
            log_interval=LOG_INTERVAL,
            eval_interval=EVAL_INTERVAL,
            steps_per_update=STEPS_PER_UPDATE,
            experiment_name=experiment_name,
            logger=experiment_logger,
        )

        # Log completion to experiment logger
        experiment_logger.info(
            f"Experiment {experiment_idx + 1}/{total_experiments} completed. "
            f"Final avg reward: {avg_rewards[-1]:.2f}, Max avg reward: {max(avg_rewards):.2f}"
        )

        # Log a warning-level message that will appear in the console
        experiment_logger.warning(
            f"Experiment {experiment_name} completed! Final reward: {avg_rewards[-1]:.2f}"
        )

        # Close the worker environment
        worker_env.close()

        # Return the results for this experiment
        return {
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
            "hyperparams": experiment_hyperparams.copy(),
            "features": feature_set,  # Store the feature set used
            "final_avg_reward": avg_rewards[-1],
            "max_avg_reward": max(avg_rewards),
            "rewards": rewards,
            "avg_rewards": avg_rewards,
            "metrics_history": metrics_history,
        }

    # Run experiments in parallel
    master_logger.info("Starting parallel experiments...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_experiment)(i, values)
        for i, values in enumerate(param_values)
    )

    # Convert results list to dictionary
    experiment_results = {result["experiment_name"]: result for result in results}

    # Summarize results in master logger
    master_logger.info("\n=== Experiment Results Summary ===")
    master_logger.info("experiment_name final_reward max_reward")
    master_logger.info("-" * 70)

    for exp_name, results in experiment_results.items():
        master_logger.info(
            "exp=%s final_reward=%.2f max_reward=%.2f",
            exp_name,
            results["final_avg_reward"],
            results["max_avg_reward"],
        )

    # Visualize the best models after all experiments are done (optional)
    visualize_best_models = True
    if visualize_best_models:
        master_logger.info("\n=== Visualizing Best Models ===")

        for exp_name, result in experiment_results.items():
            master_logger.info(f"\nVisualizing agent for experiment: {exp_name}")

            # Create a visualization environment with the correct feature set
            viz_config = copy.deepcopy(HIGHWAY_CONFIG)
            if result.get("features") is not None:
                viz_config["observation"]["features"] = result["features"]
                master_logger.info(f"Using features: {result['features']}")

            # Register highway environment
            if "highway-v0" not in gym.envs.registry:
                highway_env._register_highway_envs()

            viz_env = gym.make("highway-v0", config=viz_config, render_mode="human")

            artifacts_dir = ensure_artifacts_dir()
            best_model_path = os.path.join(
                artifacts_dir, f"ppo_highway_best_{exp_name}.pth"
            )

            # Create a new agent and load the model
            agent = PPOAgent(
                state_dim=result["hyperparams"]["state_dim"],
                action_dim=result["hyperparams"]["action_dim"],
                **{
                    k: v
                    for k, v in result["hyperparams"].items()
                    if k not in ["state_dim", "action_dim"]
                },
                logger=master_logger,
            )
            agent.load(best_model_path)

            # Visualize
            visualize_agent(viz_env, agent, num_episodes=1, logger=master_logger)

            # Close visualization environment after each experiment
            viz_env.close()

    # Plot comparison of learning curves
    plt.figure(figsize=(14, 10))

    for exp_name, results in experiment_results.items():
        eval_episodes = list(
            range(0, len(results["avg_rewards"]) * EVAL_INTERVAL, EVAL_INTERVAL)
        )
        if len(eval_episodes) != len(results["avg_rewards"]):
            eval_episodes = [0] + eval_episodes
        plt.plot(eval_episodes, results["avg_rewards"], "-o", label=exp_name)

    plt.axhline(y=TARGET_REWARD, color="r", linestyle="--", label="Target Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Comparison of Hyperparameter Settings")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save comparison plot
    artifacts_dir = ensure_artifacts_dir()
    comparison_plot_path = os.path.join(artifacts_dir, "hyperparameter_comparison.png")
    plt.savefig(comparison_plot_path)
    plt.close()
    master_logger.info(f"\nComparison plot saved to {comparison_plot_path}")

    # Save combined metrics to JSON
    metrics_path = os.path.join(artifacts_dir, "all_experiments_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "experiments": {
                    exp_name: {
                        "hyperparams": results["hyperparams"],
                        "features": results["features"],
                        "final_avg_reward": results["final_avg_reward"],
                        "max_avg_reward": results["max_avg_reward"],
                    }
                    for exp_name, results in experiment_results.items()
                }
            },
            f,
            indent=2,
        )
    master_logger.info(f"Combined experiment metrics saved to {metrics_path}")

    # Create a master CSV with results from all experiments
    master_csv_path = os.path.join(artifacts_dir, "all_experiments_summary.csv")
    with open(master_csv_path, "w") as f:
        # Write header with all possible hyperparameters
        f.write("experiment_name,final_reward,max_reward,")
        f.write(",".join(default_hyperparams.keys()))
        f.write(",plot_path,best_model_path\n")

        # Write one row per experiment
        for exp_name, results in experiment_results.items():
            f.write(
                f"{exp_name},{results['final_avg_reward']:.4f},{results['max_avg_reward']:.4f},"
            )
            # Write hyperparameter values
            for param, default in default_hyperparams.items():
                if param in results["hyperparams"]:
                    f.write(f"{results['hyperparams'][param]},")
                else:
                    f.write(f"{default},")
            # Write paths
            f.write(
                f"{os.path.join(artifacts_dir, f'ppo_highway_rewards_{exp_name}.png')},"
            )
            f.write(
                f"{os.path.join(artifacts_dir, f'ppo_highway_best_{exp_name}.pth')}\n"
            )

    return experiment_results


# Modified training function for experiments
def train_with_experiment_name(
    env,
    agent,
    max_episodes=500,
    target_reward=0.0,
    log_interval=20,
    eval_interval=50,
    steps_per_update=STEPS_PER_UPDATE,
    experiment_name="",
    logger=None,
):
    """Modified version of train() that includes experiment_name in saved artifacts"""
    # Initialize logger with experiment name if not provided
    if logger is None:
        logger = setup_experiment_logger(experiment_name)

    # Use a consistent prefix for experiment logs
    exp_prefix = f"[{experiment_name}]" if experiment_name else ""

    logger.info(f"{exp_prefix} Starting training for experiment: {experiment_name}")
    logger.info(f"{exp_prefix} Device: {device_name}")
    logger.info(
        f"{exp_prefix} Max episodes: {max_episodes}, Target reward: {target_reward}"
    )
    logger.info(f"{exp_prefix} Environment: {env.spec.id}")
    logger.info(
        f"{exp_prefix} Steps per update: {steps_per_update}, PPO epochs: {agent.epochs}"
    )
    logger.info(
        f"{exp_prefix} Learning rate: {agent.optimizer.param_groups[0]['lr']}, Gamma: {agent.gamma}, Lambda: {agent.lam}"
    )
    logger.info(
        f"{exp_prefix} Clip epsilon: {agent.eps_clip}, Value coef: {agent.value_coef}, Entropy coef: {agent.entropy_coef}"
    )

    # For tracking progress
    rewards = []  # Evaluation rewards
    episode_rewards = []  # Individual episode rewards during training
    avg_rewards = []  # Moving average of evaluation rewards
    training_episodes = []  # To track episode numbers for plotting
    eval_episodes = [0]  # To track episode numbers for evaluations
    best_avg_reward = -float("inf")

    # For storing metrics
    metrics_history = {
        "experiment_name": experiment_name,
        "episode_rewards": [],
        "eval_rewards": [],
        "avg_eval_rewards": [],
        "policy_updates": [],
        "episode_numbers": [],
        "eval_episode_numbers": [],
        "timestamps": [],
    }

    # For early stopping
    solved = False

    start_time = time.time()
    total_steps = 0
    episode_num = 0

    # Ensure artifacts directory exists
    artifacts_dir = ensure_artifacts_dir()

    # Do initial evaluation
    logger.info(f"{exp_prefix} Performing initial evaluation...")
    eval_reward = evaluate(env, agent, num_episodes=5)
    rewards.append(eval_reward)
    avg_rewards.append(eval_reward)
    metrics_history["eval_rewards"].append(eval_reward)
    metrics_history["avg_eval_rewards"].append(eval_reward)
    metrics_history["eval_episode_numbers"].append(0)
    metrics_history["timestamps"].append(0)
    logger.info(f"{exp_prefix} initial_eval reward={eval_reward:.2f}")

    while episode_num < max_episodes:
        # Collect a batch of transitions
        steps_collected = 0
        update_start_time = time.time()

        while steps_collected < steps_per_update and episode_num < max_episodes:
            episode_num += 1
            state, _ = env.reset(seed=SEED + episode_num)

            # Flatten observation
            state = state.reshape(-1)

            episode_reward = 0
            done = False
            episode_steps = 0

            while not done and steps_collected < steps_per_update:
                # Select action with state
                action, pre_tanh_action, log_prob, value = agent.select_action(state)

                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Flatten next_state
                next_state = next_state.reshape(-1)

                # Store in memory (both action forms)
                agent.memory.store(
                    state,
                    action,
                    pre_tanh_action,
                    reward,
                    next_state,
                    log_prob,
                    done,
                    value,
                )

                # Update state and reward
                state = next_state
                episode_reward += reward
                steps_collected += 1
                total_steps += 1
                episode_steps += 1

                # If we've collected enough steps or episode is done, we can stop
                if steps_collected >= steps_per_update:
                    break

            # Record the completed episode's reward and episode number
            episode_rewards.append(episode_reward)
            training_episodes.append(episode_num)
            metrics_history["episode_rewards"].append(episode_reward)
            metrics_history["episode_numbers"].append(episode_num)

            # Log episode info based on log_interval
            if episode_num % log_interval == 0:
                avg_ep_reward = np.mean(episode_rewards[-log_interval:])
                elapsed_time = time.time() - start_time
                logger.info(
                    "%s episode=%d reward=%.2f avg_reward=%.2f steps=%d episode_steps=%d time=%.2fs",
                    exp_prefix,
                    episode_num,
                    episode_reward,
                    avg_ep_reward,
                    total_steps,
                    episode_steps,
                    elapsed_time,
                )

            # Check for evaluation based on eval_interval
            if episode_num % eval_interval == 0:
                logger.info(f"{exp_prefix} Evaluating at episode {episode_num}...")
                eval_reward = evaluate(env, agent, num_episodes=5)
                rewards.append(eval_reward)
                eval_episodes.append(episode_num)
                eval_time = time.time() - start_time

                # Calculate average reward from last 10 evaluations
                avg_reward = (
                    np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                )
                avg_rewards.append(avg_reward)

                # Store in metrics
                metrics_history["eval_rewards"].append(eval_reward)
                metrics_history["avg_eval_rewards"].append(avg_reward)
                metrics_history["eval_episode_numbers"].append(episode_num)
                metrics_history["timestamps"].append(eval_time)

                logger.info(
                    "%s eval episode=%d reward=%.2f avg_reward=%.2f time=%.2fs",
                    exp_prefix,
                    episode_num,
                    eval_reward,
                    avg_reward,
                    eval_time,
                )

                # Check if environment is solved
                if avg_reward >= target_reward and not solved and len(rewards) >= 10:
                    logger.info(
                        f"{exp_prefix} Environment solved in {episode_num} episodes! Average reward: {avg_reward:.2f}"
                    )
                    # Save the model with experiment name
                    agent.save(f"ppo_highway_solved_{experiment_name}.pth")
                    solved = True

                # Save the best model with experiment name
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    agent.save(f"ppo_highway_best_{experiment_name}.pth")
                    logger.info(
                        f"{exp_prefix} New best model saved with average reward: {best_avg_reward:.2f}"
                    )

            # Only break outer loop if we've collected enough steps
            if steps_collected >= steps_per_update:
                break

        # Calculate the final state value for bootstrapping
        final_value = 0.0
        if not done:  # If we stopped collection mid-episode
            with torch.no_grad():
                # Get final value directly from state
                _, _, final_value = agent.actor_critic(state)
                final_value = final_value.cpu().item()

        # Update policy with proper bootstrapping after collecting full batch
        logger.debug(f"Updating policy after collecting {steps_collected} steps...")
        update_metrics = agent.update(last_value=final_value)
        update_time = time.time() - update_start_time

        # Store policy update metrics
        metrics_history["policy_updates"].append(
            {
                "episode": episode_num,
                "steps_collected": steps_collected,
                "time": update_time,
                **update_metrics,
            }
        )

        logger.debug(f"Policy update completed in {update_time:.2f}s")

    # Save training metrics to JSON file
    metrics_path = os.path.join(
        artifacts_dir, f"training_metrics_{experiment_name}.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    logger.info(f"{exp_prefix} Training metrics saved to {metrics_path}")

    # Plot rewards
    plt.figure(figsize=(12, 8))

    # Plot training rewards
    plt.plot(
        training_episodes,
        episode_rewards,
        alpha=0.3,
        color="gray",
        label="Training Episode Reward",
    )

    # Plot smoothed training rewards
    window_size = 20
    if len(episode_rewards) > window_size:
        smoothed_rewards = np.convolve(
            episode_rewards, np.ones(window_size) / window_size, mode="valid"
        )
        plt.plot(
            training_episodes[window_size - 1 :],
            smoothed_rewards,
            color="blue",
            label=f"Training Reward (Moving Avg {window_size})",
        )

    # Plot evaluation rewards
    plt.plot(
        eval_episodes,
        rewards,
        "ro-",
        label="Evaluation Reward (5 episodes)",
        markersize=8,
    )

    # Plot average evaluation rewards
    plt.plot(
        eval_episodes,
        avg_rewards,
        "go-",
        label="Evaluation Reward (Moving Avg)",
        markersize=6,
    )

    # Plot target line
    plt.axhline(y=target_reward, color="r", linestyle="--", label="Target Reward")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"PPO Training Progress on Highway-v0 ({experiment_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot to artifacts directory with experiment name
    plot_path = os.path.join(
        artifacts_dir, f"ppo_highway_rewards_{experiment_name}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"{exp_prefix} Training plot saved to {plot_path}")

    # Create CSV summary for this run
    csv_path = os.path.join(artifacts_dir, f"summary_{experiment_name}.csv")
    with open(csv_path, "w") as f:
        f.write(
            "experiment,final_reward,max_reward,training_steps,best_model_path,plot_path\n"
        )
        f.write(
            f"{experiment_name},{avg_rewards[-1]:.4f},{max(avg_rewards):.4f},{total_steps},"
        )
        f.write(
            f"{os.path.join(artifacts_dir, f'ppo_highway_best_{experiment_name}.pth')},"
        )
        f.write(f"{plot_path}\n")

    # Also update metrics_history to include artifact paths
    metrics_history["best_model_path"] = os.path.join(
        artifacts_dir, f"ppo_highway_best_{experiment_name}.pth"
    )
    metrics_history["plot_path"] = plot_path

    logger.info(f"{exp_prefix} Training completed!")
    return rewards, avg_rewards, metrics_history


def safe_log_prob(pre_tanh_action, action):
    # More robust version with clipping
    correction = torch.log(1 - action.pow(2) + 1e-6)
    # Add clipping to avoid extreme values
    correction = torch.clamp(correction, min=-20.0)
    return correction


if __name__ == "__main__":
    main()
