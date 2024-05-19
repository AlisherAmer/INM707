import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import typing

# Action Selection and Evaluation Functions
def select_greedy_actions(states: torch.Tensor, q_network: nn.Module) -> torch.Tensor:
    _, actions = q_network(states).max(dim=1, keepdim=True)
    return actions

def evaluate_selected_actions(states: torch.Tensor,
                              actions: torch.Tensor,
                              rewards: torch.Tensor,
                              dones: torch.Tensor,
                              gamma: float,
                              q_network: nn.Module) -> torch.Tensor:
    next_q_values = q_network(states).gather(dim=1, index=actions)
    q_values = rewards + (gamma * next_q_values * (1 - dones))
    return q_values

def double_q_learning_update(states: torch.Tensor,
                             rewards: torch.Tensor,
                             dones: torch.Tensor,
                             gamma: float,
                             q_network_1: nn.Module,
                             q_network_2: nn.Module) -> torch.Tensor:
    actions = select_greedy_actions(states, q_network_1)
    q_values = evaluate_selected_actions(states, actions, rewards, dones, gamma, q_network_2)
    return q_values

# Experience Replay Buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer:
    def __init__(self, batch_size: int, buffer_size: int = None, random_state: np.random.RandomState = None) -> None:
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer = deque(maxlen=buffer_size)
        self._random_state = np.random.RandomState() if random_state is None else random_state

    def __len__(self) -> int:
        return len(self._buffer)

    def append(self, experience: Experience) -> None:
        self._buffer.append(experience)

    def sample(self) -> typing.List[Experience]:
        idxs = self._random_state.randint(len(self._buffer), size=self._batch_size)
        experiences = [self._buffer[idx] for idx in idxs]
        return experiences

    @property
    def batch_size(self) -> int:
        return self._batch_size

# DQN Network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# DQN Agent with Double DQN and Experience Replay
class DeepQAgent:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 number_hidden_units: int,
                 optimizer_fn: typing.Callable[[typing.Iterable[nn.Parameter]], optim.Optimizer],
                 batch_size: int,
                 buffer_size: int,
                 epsilon_decay_schedule: typing.Callable[[int], float],
                 alpha: float,
                 gamma: float,
                 update_frequency: int,
                 double_dqn: bool = False,
                 seed: int = None) -> None:

        self._state_size = state_size
        self._action_size = action_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        _replay_buffer_kwargs = {
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "random_state": self._random_state
        }
        self._memory = ExperienceReplayBuffer(**_replay_buffer_kwargs)
        self._epsilon_decay_schedule = epsilon_decay_schedule
        self._alpha = alpha
        self._gamma = gamma
        self._double_dqn = double_dqn
        self._update_frequency = update_frequency

        self._online_q_network = self._initialize_q_network(number_hidden_units)
        self._target_q_network = self._initialize_q_network(number_hidden_units)
        self._synchronize_q_networks(self._target_q_network, self._online_q_network)
        self._online_q_network.to(self._device)
        self._target_q_network.to(self._device)
        self._optimizer = optimizer_fn(self._online_q_network.parameters())

        self._number_episodes = 0
        self._number_timesteps = 0

    def _initialize_q_network(self, number_hidden_units: int) -> nn.Module:
        q_network = nn.Sequential(
            nn.Linear(in_features=self._state_size, out_features=number_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=number_hidden_units, out_features=number_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=number_hidden_units, out_features=self._action_size)
        )
        return q_network

    @staticmethod
    def _soft_update_q_network_parameters(q_network_1: nn.Module,
                                          q_network_2: nn.Module,
                                          alpha: float) -> None:
        for p1, p2 in zip(q_network_1.parameters(), q_network_2.parameters()):
            p1.data.copy_(alpha * p2.data + (1 - alpha) * p1.data)

    @staticmethod
    def _synchronize_q_networks(q_network_1: nn.Module, q_network_2: nn.Module) -> None:
        _ = q_network_1.load_state_dict(q_network_2.state_dict())

    def _uniform_random_policy(self, state: torch.Tensor) -> int:
        return self._random_state.randint(self._action_size)

    def _greedy_policy(self, state: torch.Tensor) -> int:
        action = (self._online_q_network(state)
                      .argmax()
                      .cpu()
                      .item())
        return action

    def _epsilon_greedy_policy(self, state: torch.Tensor, epsilon: float) -> int:
        if self._random_state.random() < epsilon:
            action = self._uniform_random_policy(state)
        else:
            action = self._greedy_policy(state)
        return action

    def choose_action(self, state: np.array) -> int:
        state_tensor = (torch.from_numpy(state)
                             .unsqueeze(dim=0)
                             .to(self._device))
        if not self.has_sufficient_experience():
            action = self._uniform_random_policy(state_tensor)
        else:
            epsilon = self._epsilon_decay_schedule(self._number_episodes)
            action = self._epsilon_greedy_policy(state_tensor, epsilon)
        return action

    def learn(self, experiences: typing.List[Experience]) -> None:
        states, actions, rewards, next_states, dones = zip(*experiences)
    
        states = torch.tensor(states, dtype=torch.float32).to(self._device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(dim=1).to(self._device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(dim=1).to(self._device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self._device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(dim=1).to(self._device)

        if self._double_dqn:
            target_q_values = double_q_learning_update(next_states, rewards, dones, self._gamma, self._online_q_network, self._target_q_network)
        else:
            target_q_values = evaluate_selected_actions(next_states, select_greedy_actions(next_states, self._target_q_network), rewards, dones, self._gamma, self._target_q_network)

        online_q_values = self._online_q_network(states).gather(dim=1, index=actions)
        loss = F.mse_loss(online_q_values, target_q_values)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._soft_update_q_network_parameters(self._target_q_network, self._online_q_network, self._alpha)

    def has_sufficient_experience(self) -> bool:
        return len(self._memory) >= self._memory.batch_size

    def save(self, filepath: str) -> None:
        checkpoint = {
            "q-network-state": self._online_q_network.state_dict(),
            "optimizer-state": self._optimizer.state_dict(),
            "agent-hyperparameters": {
                "alpha": self._alpha,
                "batch_size": self._memory.batch_size,
                "buffer_size": self._memory.buffer_size,
                "gamma": self._gamma,
                "update_frequency": self._update_frequency
            }
        }
        torch.save(checkpoint, filepath)

    def step(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool) -> None:
        experience = Experience(state, action, reward, next_state, done)
        self._memory.append(experience)
        if done:
            self._number_episodes += 1
        else:
            self._number_timesteps += 1
            if self._number_timesteps % self._update_frequency == 0 and self.has_sufficient_experience():
                experiences = self._memory.sample()
                self.learn(experiences)

# Environment setup
env = gym.make('Pendulum-v1')
n_observations = env.observation_space.shape[0]
action_space = np.linspace(env.action_space.low[0], env.action_space.high[0], 5)
n_actions = len(action_space)

# Hyperparameters
BATCH_SIZE = 64
BUFFER_SIZE = 10000
LR = 1e-3
GAMMA = 0.99
ALPHA = 0.005
EPSILON_DECAY = lambda episode: max(0.01, min(1.0, 1.0 - 0.001 * episode))
UPDATE_FREQUENCY = 4
HIDDEN_UNITS = 128
DOUBLE_DQN = True

# Agent and optimizer initialization
optimizer_fn = lambda params: optim.Adam(params, lr=LR)

# Training loop
num_episodes = 1000
episode_rewards = []
dqn_rewards = []

# DQN Agent
dqn_agent = DeepQAgent(
    state_size=n_observations,
    action_size=n_actions,
    number_hidden_units=HIDDEN_UNITS,
    optimizer_fn=optimizer_fn,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    epsilon_decay_schedule=EPSILON_DECAY,
    alpha=ALPHA,
    gamma=GAMMA,
    update_frequency=UPDATE_FREQUENCY,
    double_dqn=False
)

# Double DQN Agent
double_dqn_agent = DeepQAgent(
    state_size=n_observations,
    action_size=n_actions,
    number_hidden_units=HIDDEN_UNITS,
    optimizer_fn=optimizer_fn,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    epsilon_decay_schedule=EPSILON_DECAY,
    alpha=ALPHA,
    gamma=GAMMA,
    update_frequency=UPDATE_FREQUENCY,
    double_dqn=DOUBLE_DQN
)

# Training both agents
for agent, rewards in [(dqn_agent, dqn_rewards), (double_dqn_agent, episode_rewards)]:
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for t in count():
            action_index = agent.choose_action(state)
            action_continuous = action_space[action_index]
            next_state, reward, terminated, truncated, _ = env.step([action_continuous])
            reward = torch.tensor([reward], dtype=torch.float32)
            total_reward += reward.item()
            done = terminated or truncated

            agent.step(state, action_index, reward.item(), next_state, done)

            state = next_state

            if done:
                rewards.append(total_reward)
                break

        if i_episode % 10 == 0:
            print(f'Episode {i_episode}, Total Reward: {total_reward}')

print('Training complete.')

# Visualization of scores
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

dqn_scores = pd.Series(dqn_rewards, name="scores")
double_dqn_scores = pd.Series(episode_rewards, name="scores")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
_ = dqn_scores.plot(ax=axes[0], label="DQN Scores")
_ = (dqn_scores.rolling(window=100)
               .mean()
               .rename("Rolling Average")
               .plot(ax=axes[0]))
_ = axes[0].legend()
_ = axes[0].set_ylabel("Score")

_ = double_dqn_scores.plot(ax=axes[1], label="Double DQN Scores")
_ = (double_dqn_scores.rolling(window=100)
                      .mean()
                      .rename("Rolling Average")
                      .plot(ax=axes[1]))
_ = axes[1].legend()
_ = axes[1].set_ylabel("Score")
_ = axes[1].set_xlabel("Episode Number")

plt.show()
