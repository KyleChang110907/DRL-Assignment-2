import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
import pickle

from student_agent import Game2048Env

# -------------------------------
# Value Network using PyTorch
# -------------------------------
class ValueNetwork(nn.Module):
    def __init__(self, board_size=4):
        super(ValueNetwork, self).__init__()
        # Input shape: (batch, 1, board_size, board_size)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2),  # output: (32, 3, 3)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2), # output: (64, 2, 2)
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output: single scalar value
        )
        
    def forward(self, x):
        # x is expected to be (batch, 1, board_size, board_size)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        value = self.fc(x)
        return value

# -------------------------------
# Helper functions
# -------------------------------
def board_to_tensor(board):
    with np.errstate(divide='ignore', invalid='ignore'):
        transformed = np.where(board == 0, 0, np.log2(board))
    tensor = torch.tensor(transformed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


# -------------------------------
# TD-Learning Training Loop for Value Network
# -------------------------------
def train_value_network(env, value_net, num_episodes=10000, gamma=0.99, lr=0.001, 
                        epsilon_start=1.0, epsilon_end=0.1):
    optimizer = optim.Adam(value_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    final_scores = []
    losses = []
    # Calculate the decay rate per episode (linear decay)
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / num_episodes

    for episode in range(num_episodes):
        state = env.reset()  # state is a numpy array (board)
        done = False
        previous_score = 0
        
        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            # Epsilon-greedy: choose random move with probability epsilon,
            # otherwise choose best move via one-step lookahead using the value network.
            if random.random() < epsilon:
                action = random.choice(legal_moves)
            else:
                best_value = -float('inf')
                best_action = None
                for a in legal_moves:
                    temp_env = copy.deepcopy(env)
                    next_state, temp_score, temp_done, _ = temp_env.step(a)
                    incremental_reward = temp_score - previous_score
                    state_tensor = board_to_tensor(next_state)
                    with torch.no_grad():
                        value_est = incremental_reward + value_net(state_tensor).item()
                    if value_est > best_value:
                        best_value = value_est
                        best_action = a
                action = best_action if best_action is not None else random.choice(legal_moves)
            
            original_state = copy.deepcopy(state)
            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score

            # Convert current and next states to tensors.
            state_tensor = board_to_tensor(original_state)
            next_state_tensor = board_to_tensor(next_state)
            
            # Compute current and next value estimates.
            current_value = value_net(state_tensor)
            with torch.no_grad():
                next_value = value_net(next_state_tensor) if not done else torch.tensor([[0.0]])
            
            # TD target and error.
            td_target = incremental_reward + gamma * next_value
            loss = loss_fn(current_value, td_target)
            
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
        
        final_scores.append(env.score)
        # Decay epsilon linearly.
        epsilon = max(epsilon_end, epsilon - epsilon_decay)
        
        if (episode+1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            print(f"Episode {episode+1}/{num_episodes}, Average Score (last 100): {avg_score:.2f}, Epsilon: {epsilon:.2f}, Losses: {np.mean(losses[-100:]):.4f}")
    
    return final_scores


# -------------------------------
# Main Training Execution
# -------------------------------
# Assume the Game2048Env is defined as provided.
env = Game2048Env()
value_net = ValueNetwork(board_size=env.size)

final_scores = train_value_network(env, value_net, num_episodes=10000, gamma=0.99, lr=0.001)

# Plot training progress.
plt.figure(figsize=(10, 5))
plt.plot(final_scores)
plt.xlabel("Episode")
plt.ylabel("Final Score")
plt.title("Training Progress of the Value Network for 2048")
plt.show()

# Optionally, save the trained model.
torch.save(value_net.state_dict(), "./Q3/checkpoints/value/CNN.pth")

# test it 
from Modules2048.value_approximation.test import evaluate_value_network
evaluate_value_network(env, value_net, num_episodes=10, gamma=0.99)