# Install necessary libraries
!pip install ray[rllib] gymnasium[atari] autorom[accept-rom-license] matplotlib

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig
import gymnasium as gym
import matplotlib.pyplot as plt

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Register the environment
def env_creator(env_config):
    return gym.make('ALE/Breakout-v5')

register_env("Breakout-v5", env_creator)

# Configuration for Double DQN
config = DQNConfig().environment(
    env="Breakout-v5"
).rollouts(
    num_rollout_workers=1
).training(
    lr=1e-4,
    train_batch_size=32,
    gamma=0.99,
    double_q=True,  # Enable Double Q-learning
    dueling=False,  # Disable Dueling Network
    target_network_update_freq=500,
    replay_buffer_config={
        "type": "ReplayBuffer",
        "capacity": 50000,
    }
).resources(
    num_gpus=0  # Set based on your hardware
).framework(
    "torch"
)

# Set exploration configuration separately
config.exploration(
    exploration_config={
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.02,
        "epsilon_timesteps": 10000,
    }
)

# Convert config to dictionary
config_dict = config.to_dict()

# Initialize the trainer
from ray.rllib.algorithms.dqn import DQN
trainer = DQN(config=config_dict)

# Training loop
episode_rewards = []
for i in range(1000):
    result = trainer.train()
    episode_rewards.append(result['episode_reward_mean'])
    print(f"Episode {i}: reward: {result['episode_reward_mean']}")
    if i % 100 == 0:
        checkpoint = trainer.save()
        print(f"Checkpoint saved at {checkpoint}"

# Shutdown Ray
ray.shutdown()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(episode_rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Double DQN Training on Breakout-v5')
plt.legend()

# Plot rolling average of rewards
rolling_avg = pd.Series(episode_rewards).rolling(window=10).mean()
plt.plot(rolling_avg, label='Rolling Average Reward')
plt.legend()
plt.show()
