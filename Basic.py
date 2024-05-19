import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# Taxi Environment Definition
class TaxiEnv:
    def __init__(self, size=6):
        self.size = size
        self.grid = np.zeros((size, size))
        self.taxi_position = (0, 0)
        self.obstacles = []
        self._place_obstacles()
        self.passenger_position = self._place_valid_passenger()
        self.destination = self._place_random(exclude_positions=[self.taxi_position] + [obs[0] for obs in self.obstacles] + [self.passenger_position])
        self.has_passenger = False
        self.actions = ["up", "down", "left", "right", "pick_up", "drop_off"]
        
    def _place_random(self, exclude_positions):
        while True:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos not in exclude_positions:
                return pos

    def _place_obstacles(self):
        obstacle_types = ['car', 'car', 'child', 'school']
        for obstacle in obstacle_types:
            pos = self._place_random(exclude_positions=[self.taxi_position] + [obs[0] for obs in self.obstacles])
            self.obstacles.append((pos, obstacle))
    
    def _is_valid_passenger_position(self, pos):
        row, col = pos
        adjacent_positions = [
            (row - 1, col), (row + 1, col),
            (row, col - 1), (row, col + 1)
        ]
        for adj_pos in adjacent_positions:
            if 0 <= adj_pos[0] < self.size and 0 <= adj_pos[1] < self.size:
                if adj_pos not in [self.taxi_position] + [obs[0] for obs in self.obstacles]:
                    return True
        return False
    
    def _place_valid_passenger(self):
        while True:
            pos = self._place_random(exclude_positions=[self.taxi_position] + [obs[0] for obs in self.obstacles])
            if self._is_valid_passenger_position(pos):
                return pos

    def reset(self):
        self.taxi_position = (0, 0)
        self.obstacles = []
        self._place_obstacles()
        self.passenger_position = self._place_valid_passenger()
        self.destination = self._place_random(exclude_positions=[self.taxi_position] + [obs[0] for obs in self.obstacles] + [self.passenger_position])
        self.has_passenger = False
        return (self.taxi_position, self.passenger_position, self.has_passenger)
    
    def step(self, action, episode, steps):
        row, col = self.taxi_position
        reward = -1  # default step reward
        done = False

        if action == "up" and row > 0:
            row -= 1
        elif action == "down" and row < self.size - 1:
            row += 1
        elif action == "left" and col > 0:
            col -= 1
        elif action == "right" and col < self.size - 1:
            col += 1
        elif action == "pick_up":
            if self.taxi_position == self.passenger_position:
                self.has_passenger = True
                self.passenger_position = None
                reward = 10  # successfully picked up passenger
            else:
                reward = -10  # tried to pick up where there's no passenger
            return (self.taxi_position, self.passenger_position, self.has_passenger), reward, done
        elif action == "drop_off":
            if self.taxi_position == self.destination and self.has_passenger:
                reward = 20  # successfully dropped off passenger
                done = True  # end the episode
                if episode % 100 == 0:
                    print(f"Successful drop off at step {steps}: State: {self.taxi_position}, Reward: {reward}, Done: {done}")
            else:
                reward = -10  # tried to drop off at wrong location
                done = False  # do not end the episode
            return (self.taxi_position, self.passenger_position, self.has_passenger), reward, done

        new_position = (row, col)
        for obs_pos, obs_type in self.obstacles:
            if new_position == obs_pos:
                if obs_type == 'car':
                    reward = -10
                elif obs_type == 'child':
                    reward = -20
                elif obs_type == 'school':
                    reward = -40
                new_position = self.taxi_position  # revert to previous position
                break

        self.taxi_position = new_position
        return (self.taxi_position, self.passenger_position, self.has_passenger), reward, done

    def render(self):
        grid_copy = np.zeros((self.size, self.size))
        row, col = self.taxi_position
        grid_copy[row, col] = 2  # Taxi
        
        if self.passenger_position:
            row, col = self.passenger_position
            grid_copy[row, col] = 3  # Passenger
            
        row, col = self.destination
        grid_copy[row, col] = 4  # Destination
        
        for (row, col), obs_type in self.obstacles:
            grid_copy[row, col] = 1  # Obstacles are all marked as 1 for simplicity
            
        print(grid_copy)

# Q-learning Agent with PyTorch
class QLearningAgent:
    def __init__(self, env, alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_min=0.001, epsilon_decay=0.9999, episodes=2000, max_steps_per_episode=5000, device='cpu'):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.device = device
        self.q_table = torch.zeros((env.size, env.size, env.size, env.size, 2, len(env.actions)), device=device)

    def state_to_index(self, state):
        taxi_row, taxi_col = state[0]
        passenger_row, passenger_col = (state[1] if state[1] is not None else (-1, -1))
        has_passenger = int(state[2])
        return (taxi_row, taxi_col, passenger_row, passenger_col, has_passenger)

    def choose_action(self, state):
        state_index = self.state_to_index(state)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(len(self.env.actions))  # Explore: Choose a random action
        else:
            action = torch.argmax(self.q_table[state_index]).item()  # Exploit: Choose the action with the highest Q-value
        return action

    def learn(self, state, action, reward, next_state):
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        predict = self.q_table[state_index + (action,)]
        target = reward + self.gamma * torch.max(self.q_table[next_state_index]).item()
        self.q_table[state_index + (action,)] = predict + self.alpha * (target - predict)

    def train(self):
        rewards = []
        for episode in range(self.episodes):  # Train for the specified number of episodes
            state = self.env.reset()
            total_rewards = 0
            done = False
            steps = 0
            
            while not done and steps < self.max_steps_per_episode:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(self.env.actions[action], episode, steps)
                self.learn(state, action, reward, next_state)
                state = next_state
                total_rewards += reward
                steps += 1

            if episode % 100 == 0:
                print(f"Episode {episode} completed in {steps} steps: Final State: {state}, Final Reward: {reward}, Done: {done}")
                print(f"Episode {episode}: Total Reward: {total_rewards}")

            rewards.append(total_rewards)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        return rewards

    def run(self):
        state = self.env.reset()
        done = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(self.env.actions[action], 0, 0)
            state = next_state
            self.env.render()

# Define the Taxi Environment
env = TaxiEnv()

# Define the ranges for alpha and gamma
alpha_values = [0.2, 0.5, 0.9]
gamma_values = [0.2, 0.5, 0.9]

# Store results for each combination
results = []

# Loop through each combination 
for alpha in alpha_values:
    for gamma in gamma_values:
        print(f"Training with alpha={alpha} and gamma={gamma}")
        # Define the Q-learning Agent with GPU acceleration
        agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=1.0, epsilon_min=0.001, epsilon_decay=0.9999, episodes=2000, max_steps_per_episode=5000, device='cpu')
        
        # Train the Q-learning Agent
        rewards = agent.train()
        
        # Store the results
        results.append((alpha, gamma, rewards))

# Colour code
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple']

# Plot for each constant gamma with different alpha values
for i, gamma in enumerate(gamma_values):
    plt.figure()
    for j, alpha in enumerate(alpha_values):
        idx = i * len(alpha_values) + j
        _, _, rewards = results[idx]
        plt.plot(rewards, label=f'alpha={alpha}', color=colors[j])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Q-learning Training Performance (gamma={gamma})')
    plt.legend()
    plt.show()

# Plot for each constant alpha with different gamma values
for i, alpha in enumerate(alpha_values):
    plt.figure()
    for j, gamma in enumerate(gamma_values):
        idx = j * len(alpha_values) + i
        _, _, rewards = results[idx]
        plt.plot(rewards, label=f'gamma={gamma}', color=colors[j])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Q-learning Training Performance (alpha={alpha})')
    plt.legend()
    plt.show()

# Plot all combinations together
plt.figure()
for idx, (alpha, gamma, rewards) in enumerate(results):
    plt.plot(rewards, label=f'alpha={alpha}, gamma={gamma}', color=colors[idx % len(colors)])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning Training Performance (All Combinations)')
plt.legend()
plt.show()


#With epsilon decay 0.4
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# Taxi Environment Definition
class TaxiEnv:
    def __init__(self, size=6):
        self.size = size
        self.grid = np.zeros((size, size))
        self.taxi_position = (0, 0)
        self.obstacles = []
        self._place_obstacles()
        self.passenger_position = self._place_valid_passenger()
        self.destination = self._place_random(exclude_positions=[self.taxi_position] + [obs[0] for obs in self.obstacles] + [self.passenger_position])
        self.has_passenger = False
        self.actions = ["up", "down", "left", "right", "pick_up", "drop_off"]
        
    def _place_random(self, exclude_positions):
        while True:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos not in exclude_positions:
                return pos

    def _place_obstacles(self):
        obstacle_types = ['car', 'car', 'child', 'school']
        for obstacle in obstacle_types:
            pos = self._place_random(exclude_positions=[self.taxi_position] + [obs[0] for obs in self.obstacles])
            self.obstacles.append((pos, obstacle))
    
    def _is_valid_passenger_position(self, pos):
        row, col = pos
        adjacent_positions = [
            (row - 1, col), (row + 1, col),
            (row, col - 1), (row, col + 1)
        ]
        for adj_pos in adjacent_positions:
            if 0 <= adj_pos[0] < self.size and 0 <= adj_pos[1] < self.size:
                if adj_pos not in [self.taxi_position] + [obs[0] for obs in self.obstacles]:
                    return True
        return False
    
    def _place_valid_passenger(self):
        while True:
            pos = self._place_random(exclude_positions=[self.taxi_position] + [obs[0] for obs in self.obstacles])
            if self._is_valid_passenger_position(pos):
                return pos

    def reset(self):
        self.taxi_position = (0, 0)
        self.obstacles = []
        self._place_obstacles()
        self.passenger_position = self._place_valid_passenger()
        self.destination = self._place_random(exclude_positions=[self.taxi_position] + [obs[0] for obs in self.obstacles] + [self.passenger_position])
        self.has_passenger = False
        return (self.taxi_position, self.passenger_position, self.has_passenger)
    
    def step(self, action, episode, steps):
        row, col = self.taxi_position
        reward = -1  # default step reward
        done = False

        if action == "up" and row > 0:
            row -= 1
        elif action == "down" and row < self.size - 1:
            row += 1
        elif action == "left" and col > 0:
            col -= 1
        elif action == "right" and col < self.size - 1:
            col += 1
        elif action == "pick_up":
            if self.taxi_position == self.passenger_position:
                self.has_passenger = True
                self.passenger_position = None
                reward = 10  # successfully picked up passenger
            else:
                reward = -10  # tried to pick up where there's no passenger
            return (self.taxi_position, self.passenger_position, self.has_passenger), reward, done
        elif action == "drop_off":
            if self.taxi_position == self.destination and self.has_passenger:
                reward = 20  # successfully dropped off passenger
                done = True  # end the episode
                if episode % 100 == 0:
                    print(f"Successful drop off at step {steps}: State: {self.taxi_position}, Reward: {reward}, Done: {done}")
            else:
                reward = -10  # tried to drop off at wrong location
                done = False  # do not end the episode
            return (self.taxi_position, self.passenger_position, self.has_passenger), reward, done

        new_position = (row, col)
        for obs_pos, obs_type in self.obstacles:
            if new_position == obs_pos:
                if obs_type == 'car':
                    reward = -10
                elif obs_type == 'child':
                    reward = -20
                elif obs_type == 'school':
                    reward = -40
                new_position = self.taxi_position  # revert to previous position
                break

        self.taxi_position = new_position
        return (self.taxi_position, self.passenger_position, self.has_passenger), reward, done

    def render(self):
        grid_copy = np.zeros((self.size, self.size))
        row, col = self.taxi_position
        grid_copy[row, col] = 2  # Taxi
        
        if self.passenger_position:
            row, col = self.passenger_position
            grid_copy[row, col] = 3  # Passenger
            
        row, col = self.destination
        grid_copy[row, col] = 4  # Destination
        
        for (row, col), obs_type in self.obstacles:
            grid_copy[row, col] = 1  # Obstacles are all marked as 1 for simplicity
            
        print(grid_copy)

# Q-learning Agent Definition with PyTorch
class QLearningAgent:
    def __init__(self, env, alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_min=0.001, epsilon_decay=0.4, episodes=2000, max_steps_per_episode=10000, device='cpu'):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.device = device
        self.q_table = torch.zeros((env.size, env.size, env.size, env.size, 2, len(env.actions)), device=device)

    def state_to_index(self, state):
        taxi_row, taxi_col = state[0]
        passenger_row, passenger_col = (state[1] if state[1] is not None else (-1, -1))
        has_passenger = int(state[2])
        return (taxi_row, taxi_col, passenger_row, passenger_col, has_passenger)

    def choose_action(self, state):
        state_index = self.state_to_index(state)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(len(self.env.actions))  # Explore: Choose a random action
        else:
            action = torch.argmax(self.q_table[state_index]).item()  # Exploit: Choose the action with the highest Q-value
        return action

    def learn(self, state, action, reward, next_state):
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        predict = self.q_table[state_index + (action,)]
        target = reward + self.gamma * torch.max(self.q_table[next_state_index]).item()
        self.q_table[state_index + (action,)] = predict + self.alpha * (target - predict)

    def train(self):
        rewards = []
        for episode in range(self.episodes):  # Train for the specified number of episodes
            state = self.env.reset()
            total_rewards = 0
            done = False
            steps = 0
            
            while not done and steps < self.max_steps_per_episode:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(self.env.actions[action], episode, steps)
                self.learn(state, action, reward, next_state)
                state = next_state
                total_rewards += reward
                steps += 1

            if episode % 100 == 0:
                print(f"Episode {episode} completed in {steps} steps: Final State: {state}, Final Reward: {reward}, Done: {done}")
                print(f"Episode {episode}: Total Reward: {total_rewards}")

            rewards.append(total_rewards)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        return rewards

    def run(self):
        state = self.env.reset()
        done = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(self.env.actions[action], 0, 0)
            state = next_state
            self.env.render()

# Define the Taxi Environment
env = TaxiEnv()

# Define the ranges for alpha and gamma
alpha_values = [0.9]
gamma_values = [0.9]

# Store results for each combination 
results = []

# Loop through each combination of alpha and gamma
for alpha in alpha_values:
    for gamma in gamma_values:
        print(f"Training with alpha={alpha} and gamma={gamma}")
        # Define the Q-learning Agent with GPU acceleration
        agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=1.0, epsilon_min=0.001, epsilon_decay=0.4, episodes=2000, max_steps_per_episode=10000, device='cpu')
        
        # Train the Q-learning Agent
        rewards = agent.train()
        
        # Store the results
        results.append((alpha, gamma, rewards))

# Plotting
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple']

# Plot for each constant gamma with different alpha values
for i, gamma in enumerate(gamma_values):
    plt.figure()
    for j, alpha in enumerate(alpha_values):
        idx = i * len(alpha_values) + j
        _, _, rewards = results[idx]
        plt.plot(rewards, label=f'alpha={alpha}', color=colors[j])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Q-learning Training Performance (gamma={gamma})')
    plt.legend()
    plt.show()

# Plot for each constant alpha with different gamma values
for i, alpha in enumerate(alpha_values):
    plt.figure()
    for j, gamma in enumerate(gamma_values):
        idx = j * len(alpha_values) + i
        _, _, rewards = results[idx]
        plt.plot(rewards, label=f'gamma={gamma}', color=colors[j])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Q-learning Training Performance (alpha={alpha})')
    plt.legend()
    plt.show()

# Plot all combinations together
plt.figure()
for idx, (alpha, gamma, rewards) in enumerate(results):
    plt.plot(rewards, label=f'alpha={alpha}, gamma={gamma}', color=colors[idx % len(colors)])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning Training Performance (All Combinations)')
plt.legend()
plt.show()
