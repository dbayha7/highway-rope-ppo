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
from joblib import Parallel, delayed
from highway_env import _register_highway_envs

# if highway_env is not registered, register it
if "highway-v0" not in gym.envs.registry:
    _register_highway_envs()

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
MAX_EPISODES = 1050  # Maximum number of episodes to train
TARGET_REWARD = (
    0.0  # Target reward to consider environment solved (adjusted for highway-env)
)
LOG_INTERVAL = 10  # How often to log training progress
EVAL_INTERVAL = 50  # How often to run evaluation

# Highway environment configuration
HIGHWAY_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,  # Number of vehicles to observe
        "features": ["x", "y", "vx", "vy"],  # Features to include
        "normalize": False,  # We'll do our own normalization
        "absolute": False,
        "order": "sorted",
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
def set_random_seeds(seed=SEED):
    """Set random seeds for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # more consistent


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


# Create the artifacts directory if it doesn't exist
def ensure_artifacts_dir():
    """Create the artifacts/highway-ppo directory if it doesn't exist."""
    artifacts_dir = os.path.join("artifacts", "highway-ppo")
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir


# State Normalizer for better learning
class RunningStatNormalizer:
    """
    Maintains running statistics (mean, variance) to normalize state inputs.

    State normalization is crucial in RL to ensure consistent input scales,
    which helps stabilize training and improve convergence.
    """

    def __init__(self, shape, eps=1e-8):
        """
        Initialize running statistics trackers.

        Args:
            shape: Dimensions of the state space
            eps: Small constant to avoid numerical instability and division by zero
        """
        self.mean = np.zeros(
            shape, dtype=np.float32
        )  # Running mean initialized to zeros
        self.var = np.ones(
            shape, dtype=np.float32
        )  # Running variance initialized to ones
        self.count = eps  # Counter for number of samples seen (initialized with eps for stability)

    def update(self, x):
        """
        Update running statistics with new observations using Welford's online algorithm.

        This method efficiently computes running mean and variance without storing all data.

        Args:
            x: New state observation(s) to incorporate into statistics
        """
        # Handle single state case by reshaping to a batch of size 1
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Calculate statistics of the new batch
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        # Calculate the difference between current mean and batch mean
        delta = batch_mean - self.mean

        # Compute the updated count of all samples seen so far
        total_count = self.count + batch_count

        # Update running mean using the weighted average formula
        # new_mean = old_mean + (batch_mean - old_mean) * (batch_count / total_count)
        new_mean = self.mean + delta * (batch_count / total_count)

        # Update running variance using the parallel combination formula
        # This is mathematically equivalent to recalculating variance with all samples
        m_a = self.var * self.count  # Sum of squares for previous samples
        m_b = batch_var * batch_count  # Sum of squares for new batch
        # Additional term accounts for the shift in mean
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        # Store the updated statistics
        self.mean, self.var, self.count = new_mean, new_var, total_count

    def normalize(self, x):
        """
        Normalize input states using current running statistics.

        Standardizes data to have zero mean and unit variance, which helps
        neural networks train more effectively.

        Args:
            x: State(s) to normalize - can be numpy array or torch tensor

        Returns:
            Normalized state(s) in the same format as input
        """
        # Handle torch tensors by converting to numpy, normalizing, then back to tensor
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
            normalized = (x_np - self.mean) / np.sqrt(self.var + 1e-8)
            return torch.FloatTensor(normalized).to(x.device)

        # For numpy arrays, directly apply normalization
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


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
            nn.Tanh(),  # Bound outputs to [-1, 1]
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
        # Get action distribution parameters
        action_mean, action_std, value = self.forward(state)

        # Create a normal distribution
        dist = Normal(action_mean, action_std)

        # Sample from the distribution or take the mean
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()

        # Clamp actions to valid range [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)

        # Get log probability of the action (sum across action dimensions)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.cpu().numpy(), log_prob.item(), value.cpu().numpy()[0]

    def evaluate(self, states, actions):
        # Get action distribution parameters and state values
        action_means, action_stds, state_values = self.forward(states)

        # Create normal distributions
        dist = Normal(action_means, action_stds)

        # Get log probabilities (sum across action dimensions)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Get entropy (sum across action dimensions)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, state_values, entropy


# Memory Buffer
class PPOMemory:
    def __init__(self, batch_size=64):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.dones = []
        self.values = []
        self.batch_size = batch_size

    def store(self, state, action, reward, next_state, log_prob, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
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
        # Change to FloatTensor for continuous actions
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)

        return states, actions, old_log_probs


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

        self.memory = PPOMemory(batch_size)
        self.normalizer = RunningStatNormalizer(shape=(state_dim,))

    def select_action(self, state, deterministic=False):
        # Update normalizer with the state
        self.normalizer.update(state)

        # Normalize state
        normalized_state = self.normalizer.normalize(state)

        with torch.no_grad():
            # Get action, log probability, and value
            action, log_prob, value = self.actor_critic.get_action(
                normalized_state, deterministic
            )

        return action, log_prob, value, normalized_state

    def update(self, last_value=0.0):
        # Get data from memory
        states, actions, old_log_probs = self.memory.get_tensors()
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

        # Optimization loop
        for _ in range(self.epochs):
            for batch_indices in batches:
                # Get batch data
                batch_states = states[batch_indices]
                # No longer need to normalize states here as we're already using normalized states
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                new_log_probs, state_values, entropy = self.actor_critic.evaluate(
                    batch_states, batch_actions
                )

                # Calculate ratios
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

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

                # Calculate entropy
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                )

                # Update weights
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

        # Clear memory
        self.memory.clear()

    def save(self, path):
        # Ensure the artifacts directory exists
        artifacts_dir = ensure_artifacts_dir()
        # Create full path including artifacts directory
        full_path = os.path.join(artifacts_dir, path)
        torch.save(self.actor_critic.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load(self, path):
        # Check if path includes directory, if not, assume it's in artifacts
        if os.path.dirname(path) == "":
            artifacts_dir = ensure_artifacts_dir()
            path = os.path.join(artifacts_dir, path)
        self.actor_critic.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


# Training function
def train(
    env,
    agent,
    max_episodes=500,
    target_reward=0.0,
    log_interval=20,
    eval_interval=50,
    steps_per_update=STEPS_PER_UPDATE,
):
    print("Starting training...")

    # For tracking progress
    rewards = []  # Evaluation rewards
    episode_rewards = []  # Individual episode rewards during training
    avg_rewards = []  # Moving average of evaluation rewards
    training_episodes = []  # To track episode numbers for plotting
    eval_episodes = [0]  # To track episode numbers for evaluations
    best_avg_reward = -float("inf")

    # For early stopping
    solved = False

    start_time = time.time()
    total_steps = 0
    episode_num = 0

    # Ensure artifacts directory exists
    artifacts_dir = ensure_artifacts_dir()

    # Do initial evaluation
    eval_reward = evaluate(env, agent, num_episodes=5)
    rewards.append(eval_reward)
    avg_rewards.append(eval_reward)
    print(f"Initial evaluation: Average Reward over 5 episodes: {eval_reward:.2f}")

    while episode_num < max_episodes:
        # Collect a batch of transitions
        steps_collected = 0

        while steps_collected < steps_per_update and episode_num < max_episodes:
            episode_num += 1
            state, _ = env.reset(seed=SEED + episode_num)

            # Flatten observation from (N, F) to (N*F,)
            state = state.reshape(-1)

            episode_reward = 0
            done = False

            while not done and steps_collected < steps_per_update:
                # Select action with normalized state
                action, log_prob, value, normalized_state = agent.select_action(state)

                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Flatten next_state from (N, F) to (N*F,)
                next_state = next_state.reshape(-1)

                # Store in memory (using normalized state instead of raw state)
                agent.memory.store(
                    normalized_state, action, reward, next_state, log_prob, done, value
                )

                # Update state and reward
                state = next_state
                episode_reward += reward
                steps_collected += 1
                total_steps += 1

                # If we've collected enough steps or episode is done, we can stop
                if steps_collected >= steps_per_update:
                    break

            # Record the completed episode's reward and episode number
            episode_rewards.append(episode_reward)
            training_episodes.append(episode_num)

            # Log episode info based on log_interval
            if episode_num % log_interval == 0:
                avg_ep_reward = np.mean(episode_rewards[-log_interval:])
                elapsed_time = time.time() - start_time
                print(
                    f"Episode {episode_num} | Avg Episode Reward: {avg_ep_reward:.2f} | Steps: {total_steps} | Time: {elapsed_time:.2f}s"
                )

            # Check for evaluation based on eval_interval - do it here to ensure evaluations happen exactly every eval_interval episodes
            if episode_num % eval_interval == 0:
                eval_reward = evaluate(env, agent, num_episodes=5)
                rewards.append(eval_reward)
                eval_episodes.append(episode_num)

                # Calculate average reward from last 10 evaluations
                avg_reward = (
                    np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                )
                avg_rewards.append(avg_reward)

                print(
                    f"\nEvaluation at episode {episode_num}: Average Reward over 5 episodes: {eval_reward:.2f}\n"
                )

                # Check if environment is solved
                if avg_reward >= target_reward and not solved and len(rewards) >= 10:
                    print(
                        f"Environment solved in {episode_num} episodes! Average reward: {avg_reward:.2f}"
                    )
                    # Save the model
                    agent.save("ppo_highway_solved.pth")
                    solved = True

                # Save the best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    agent.save("ppo_highway_best.pth")

            # Only break outer loop if we've collected enough steps
            if steps_collected >= steps_per_update:
                break

        # Calculate the final state value for bootstrapping
        final_value = 0.0
        if not done:  # If we stopped collection mid-episode
            with torch.no_grad():
                normalized_final_state = agent.normalizer.normalize(state)
                # Fix: Properly unpack three values (mean, std, value) from forward method
                _, _, final_value = agent.actor_critic(normalized_final_state)
                final_value = final_value.cpu().item()

        # Update policy with proper bootstrapping after collecting full batch
        agent.update(last_value=final_value)

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
    print(f"Training plot saved to {plot_path}")

    print("Training completed!")
    return rewards, avg_rewards


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
def visualize_agent(env, agent, num_episodes=3):
    print("Visualizing agent...")

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

            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

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

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Flag to run a single training session instead of hyperparameter sweep
    run_single = True

    if run_single:
        print("\n=== Running Single Training Session ===")
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
        )

        # Train the agent
        rewards, avg_rewards = train(
            env=env,
            agent=agent,
            max_episodes=300,  # Shorter run for testing
            target_reward=TARGET_REWARD,
            log_interval=LOG_INTERVAL,
            eval_interval=EVAL_INTERVAL,
            steps_per_update=STEPS_PER_UPDATE,
        )

        # Visualize the trained agent
        viz_env = gym.make("highway-v0", config=HIGHWAY_CONFIG, render_mode="human")
        visualize_agent(viz_env, agent, num_episodes=3)
        viz_env.close()

        # Close the training environment
        env.close()
    else:
        # Run experiments with different hyperparameter values in parallel
        # Set n_jobs to control parallelism: -1 means all cores, 2 means 2 workers, etc.
        run_experiments(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            hyperparams_to_vary={
                "epochs": [4, 6],
                "lr": [1e-4, 3e-4],
                "hidden_dim": [64, 128],
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
            },
            n_jobs=4,  # Adjust based on your CPU cores and memory
        )

    # No need to close environment as it's already closed in run_experiments


# Function to run experiments with different hyperparameters
def run_experiments(env, state_dim, action_dim, hyperparams_to_vary, n_jobs=-1):
    """
    Run multiple experiments with different hyperparameter values in parallel.

    Args:
        env: The environment to train on (will be closed and recreated in each worker)
        state_dim: State dimension
        action_dim: Action dimension
        hyperparams_to_vary: Dict mapping hyperparameter names to lists of values to try
        n_jobs: Number of parallel jobs to run (-1 for all available cores)
    """
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
    print(
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
        print(
            f"\n=== Starting Experiment {experiment_idx + 1}/{total_experiments}: {experiment_name} ===\n"
        )
        print(
            f"State dimension: {worker_state_dim}, Action dimension: {worker_action_dim}"
        )

        if feature_set is not None:
            print(f"Using features: {feature_set}")

        # Create agent with these hyperparameters
        agent = PPOAgent(**experiment_hyperparams)

        # Train agent with modified model saving to include hyperparameter values
        rewards, avg_rewards = train_with_experiment_name(
            env=worker_env,
            agent=agent,
            max_episodes=MAX_EPISODES,
            target_reward=TARGET_REWARD,
            log_interval=LOG_INTERVAL,
            eval_interval=EVAL_INTERVAL,
            steps_per_update=STEPS_PER_UPDATE,
            experiment_name=experiment_name,
        )

        # Disable visualization during parallel training as it can cause issues
        # If visualization is needed, it can be done after all experiments complete

        # Close the worker environment
        worker_env.close()

        # Return the results for this experiment
        return {
            "experiment_name": experiment_name,
            "hyperparams": experiment_hyperparams.copy(),
            "features": feature_set,  # Store the feature set used
            "final_avg_reward": avg_rewards[-1],
            "max_avg_reward": max(avg_rewards),
            "rewards": rewards,
            "avg_rewards": avg_rewards,
        }

    # Run experiments in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_experiment)(i, values)
        for i, values in enumerate(param_values)
    )

    # Convert results list to dictionary
    experiment_results = {result["experiment_name"]: result for result in results}

    # Visualize the best models after all experiments are done (optional)
    visualize_best_models = True
    if visualize_best_models:
        print("\n=== Visualizing Best Models ===")

        for exp_name, result in experiment_results.items():
            print(f"\nVisualizing agent for experiment: {exp_name}")

            # Create a visualization environment with the correct feature set
            viz_config = copy.deepcopy(HIGHWAY_CONFIG)
            if result.get("features") is not None:
                viz_config["observation"]["features"] = result["features"]
                print(f"Using features: {result['features']}")

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
            )
            agent.load(best_model_path)

            # Visualize
            visualize_agent(viz_env, agent, num_episodes=1)

            # Close visualization environment after each experiment
            viz_env.close()

    # Compare and display results
    print("\n=== Experiment Results ===")
    print(f"{'Experiment':<40} {'Final Reward':<15} {'Max Reward':<15}")
    print("-" * 70)

    for exp_name, results in experiment_results.items():
        print(
            f"{exp_name:<40} {results['final_avg_reward']:<15.2f} {results['max_avg_reward']:<15.2f}"
        )

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
    print(f"\nComparison plot saved to {comparison_plot_path}")

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
):
    """Modified version of train() that includes experiment_name in saved artifacts"""
    print(f"Starting training for experiment: {experiment_name}...")

    # For tracking progress
    rewards = []  # Evaluation rewards
    episode_rewards = []  # Individual episode rewards during training
    avg_rewards = []  # Moving average of evaluation rewards
    training_episodes = []  # To track episode numbers for plotting
    eval_episodes = [0]  # To track episode numbers for evaluations
    best_avg_reward = -float("inf")

    # For early stopping
    solved = False

    start_time = time.time()
    total_steps = 0
    episode_num = 0

    # Ensure artifacts directory exists
    artifacts_dir = ensure_artifacts_dir()

    # Do initial evaluation
    eval_reward = evaluate(env, agent, num_episodes=5)
    rewards.append(eval_reward)
    avg_rewards.append(eval_reward)
    print(f"Initial evaluation: Average Reward over 5 episodes: {eval_reward:.2f}")

    while episode_num < max_episodes:
        # Collect a batch of transitions
        steps_collected = 0

        while steps_collected < steps_per_update and episode_num < max_episodes:
            episode_num += 1
            state, _ = env.reset(seed=SEED + episode_num)

            # Flatten observation
            state = state.reshape(-1)

            episode_reward = 0
            done = False

            while not done and steps_collected < steps_per_update:
                # Select action with normalized state
                action, log_prob, value, normalized_state = agent.select_action(state)

                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Flatten next_state
                next_state = next_state.reshape(-1)

                # Store in memory (using normalized state instead of raw state)
                agent.memory.store(
                    normalized_state, action, reward, next_state, log_prob, done, value
                )

                # Update state and reward
                state = next_state
                episode_reward += reward
                steps_collected += 1
                total_steps += 1

                # If we've collected enough steps or episode is done, we can stop
                if steps_collected >= steps_per_update:
                    break

            # Record the completed episode's reward and episode number
            episode_rewards.append(episode_reward)
            training_episodes.append(episode_num)

            # Log episode info based on log_interval
            if episode_num % log_interval == 0:
                avg_ep_reward = np.mean(episode_rewards[-log_interval:])
                elapsed_time = time.time() - start_time
                print(
                    f"[{experiment_name}] Episode {episode_num} | Avg Episode Reward: {avg_ep_reward:.2f} | Steps: {total_steps} | Time: {elapsed_time:.2f}s"
                )

            # Check for evaluation based on eval_interval
            if episode_num % eval_interval == 0:
                eval_reward = evaluate(env, agent, num_episodes=5)
                rewards.append(eval_reward)
                eval_episodes.append(episode_num)

                # Calculate average reward from last 10 evaluations
                avg_reward = (
                    np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                )
                avg_rewards.append(avg_reward)

                print(
                    f"\n[{experiment_name}] Evaluation at episode {episode_num}: Average Reward over 5 episodes: {eval_reward:.2f}\n"
                )

                # Check if environment is solved
                if avg_reward >= target_reward and not solved and len(rewards) >= 10:
                    print(
                        f"[{experiment_name}] Environment solved in {episode_num} episodes! Average reward: {avg_reward:.2f}"
                    )
                    # Save the model with experiment name
                    agent.save(f"ppo_highway_solved_{experiment_name}.pth")
                    solved = True

                # Save the best model with experiment name
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    agent.save(f"ppo_highway_best_{experiment_name}.pth")

            # Only break outer loop if we've collected enough steps
            if steps_collected >= steps_per_update:
                break

        # Calculate the final state value for bootstrapping
        final_value = 0.0
        if not done:  # If we stopped collection mid-episode
            with torch.no_grad():
                normalized_final_state = agent.normalizer.normalize(state)
                # Fix: Properly unpack three values (mean, std, value) from forward method
                _, _, final_value = agent.actor_critic(normalized_final_state)
                final_value = final_value.cpu().item()

        # Update policy with proper bootstrapping after collecting full batch
        agent.update(last_value=final_value)

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
    print(f"Training plot saved to {plot_path}")

    print(f"Training completed for experiment: {experiment_name}!")
    return rewards, avg_rewards


if __name__ == "__main__":
    main()
