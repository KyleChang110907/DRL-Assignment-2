import copy
import random
import math
import numpy as np
import dill as pickle
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from student_agent import Game2048Env

folder_path = './Modules2048/checkpoints/value_approximation/N_tuple_TD5_6Tuples_8pattern_lr10-1_OI25/'
os.makedirs(folder_path, exist_ok=True)

CHECKPOINT_FILE = folder_path + 'value_net.pkl'
final_score_sav_path = folder_path + 'final_scores.npy'
save_fig_path = folder_path + 'scores.png'
save_fig_100_avg_path = folder_path + 'scores_100_avg.png'

# -------------------------------
# Transformation Functions
# -------------------------------
def rot90(coord, board_size):
    r, c = coord
    return (c, board_size - 1 - r)

def rot180(coord, board_size):
    r, c = coord
    return (board_size - 1 - r, board_size - 1 - c)

def rot270(coord, board_size):
    r, c = coord
    return (board_size - 1 - c, r)

def flip_horizontal(coord, board_size):
    r, c = coord
    return (r, board_size - 1 - c)

class OptimisticDefault:
    def __init__(self, opt_init):
        self.opt_init = opt_init
    def __call__(self):
        return self.opt_init

# -------------------------------
# NTupleApproximator (8 symmetric variants for each pattern)
# -------------------------------
class NTupleApproximator:
    def __init__(self, board_size, patterns, optimistic_init=25):
        """
        Initializes the N-Tuple approximator for value estimation.
        We use optimistic initialization to encourage exploration.
        """
        self.board_size = board_size
        self.patterns = patterns
        # Each pattern's weight table is optimistically initialized.
        self.weights = [defaultdict(OptimisticDefault(optimistic_init)) for _ in patterns]
        # Generate 8 symmetrical transformations for each pattern.
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

    def generate_symmetries(self, pattern):
        """
        Generate 8 symmetrical transformations of the given pattern.
        Returns a list of 8 patterns.
        """
        p0 = pattern
        p90  = [rot90(coord, self.board_size) for coord in pattern]
        p180 = [rot180(coord, self.board_size) for coord in pattern]
        p270 = [rot270(coord, self.board_size) for coord in pattern]

        p0f   = [flip_horizontal(coord, self.board_size) for coord in p0]
        p90f  = [flip_horizontal(coord, self.board_size) for coord in p90]
        p180f = [flip_horizontal(coord, self.board_size) for coord in p180]
        p270f = [flip_horizontal(coord, self.board_size) for coord in p270]

        return [p0, p90, p180, p270, p0f, p90f, p180f, p270f]

    def tile_to_index(self, tile):
        """Convert tile value to an index (0 if empty, else log2(tile))."""
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        """Extract tile values from the board at given coordinates into a feature tuple."""
        return tuple(self.tile_to_index(board[r][c]) for (r, c) in coords)

    def value(self, board):
        """
        Estimate the board's value as the sum of contributions from all 8 symmetric variants
        of each pattern.
        """
        total_val = 0.0
        for i, syms in enumerate(self.symmetry_patterns):
            for coords in syms:
                feat = self.get_feature(board, coords)
                total_val += self.weights[i][feat]
        return total_val

    def update(self, board, delta, alpha):
        """
        Update the weights based on the TD error delta.
        The update is spread equally among all symmetric variants.
        """
        total_features = sum(len(syms) for syms in self.symmetry_patterns)  # = num_patterns * 8
        update_amount = (alpha * delta) / total_features
        for i, syms in enumerate(self.symmetry_patterns):
            for coords in syms:
                feat = self.get_feature(board, coords)
                self.weights[i][feat] += update_amount

# -------------------------------
# Helper Function: Simulate Move Without Random Tile
# -------------------------------
def simulate_move_without_tile(env, action):
    """
    Returns a copy of the environment after applying the move, without adding a random tile.
    """
    sim_env = copy.deepcopy(env)
    if action == 0:
        sim_env.move_up()
    elif action == 1:
        sim_env.move_down()
    elif action == 2:
        sim_env.move_left()
    elif action == 3:
        sim_env.move_right()
    return sim_env

# -------------------------------
# Multi-Step TD-Learning Training Function (Using Pure States)
# -------------------------------
def td_learning(env, approximator, num_episodes=30000, alpha=0.1, gamma=0.99, epsilon=0.1, n_step=5):
    """
    Train the 2048 agent using multi-step TD learning with an NTuple approximator.
    
    For each move, the best action is selected by a one-step lookahead on the state 
    before random tile addition. The agent collects a trajectory of pure states and 
    incremental rewards. Then, for each time step in the trajectory, an n-step return is 
    computed and used to update the approximator.
    """
    final_scores = []
    success_flags = []
    avg_scores_list = []
    time_start = time.time()

    for episode in range(num_episodes):
        state = env.reset()  # environment state includes initial random tiles
        done = False
        previous_score = env.score  # record starting score
        max_tile = np.max(state)
        # Use lists to store the sequence of pure states and rewards.
        states = []
        rewards = []
        # Record the initial pure state.
        pure_state = copy.deepcopy(state)
        states.append(pure_state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            best_value = -float('inf')
            best_action = None
            best_pure_state = None
            
            for a in legal_moves:
                temp_env = simulate_move_without_tile(env, a)
                pure_state_candidate = temp_env.board.copy()
                inc_reward_candidate = temp_env.score - previous_score
                future_value_candidate = approximator.value(pure_state_candidate)
                value_est = inc_reward_candidate + gamma * future_value_candidate
                if value_est > best_value:
                    best_value = value_est
                    best_action = a
                    best_pure_state = pure_state_candidate
            
            action = best_action
            # Use the "pure" state before random tile as the state for TD update.
            pure_after_state = copy.deepcopy(best_pure_state)
            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            
            # Append the reward and next state to the trajectory lists.
            rewards.append(incremental_reward)
            states.append(pure_after_state)
            
            state = next_state
            pure_state = copy.deepcopy(pure_after_state)

        # Multi-step TD update: for each time step in the episode, compute the n-step return.
        T = len(rewards)  # number of transitions
        for t in range(T):
            G = 0.0
            # Sum up rewards for n steps (or until episode termination)
            for k in range(n_step):
                if t + k < T:
                    G += (gamma ** k) * rewards[t + k]
                    # If a terminal state is encountered, break out.
                    if t + k < T and states[t + k + 1] is None:
                        break
                else:
                    break
            # If the n-th step exists (and the episode didn't terminate before that), bootstrap.
            if t + n_step < len(states):
                G += (gamma ** n_step) * approximator.value(states[t + n_step])
            # TD error and update for state at time t.
            current_value = approximator.value(states[t])
            td_error = G - current_value
            approximator.update(states[t], td_error, alpha)
        
        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)
        
        if (episode+1) % 100 == 0:
            time_interval = time.time() - time_start
            avg_score_block = np.mean(final_scores[-100:])
            avg_scores_list.append(avg_score_block)
            with open(CHECKPOINT_FILE, "wb") as f:
                pickle.dump(approximator.weights, f)
            np.save(final_score_sav_path, np.array(final_scores), allow_pickle=True)
            overall_avg = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Block Avg Score: {overall_avg:.2f} | Success Rate: {success_rate:.2f} | Time Spent: {time_interval:.2f} sec")
            time_start = time.time()
            
    return final_scores

# -------------------------------
# Define 4 Patterns of 6 Tuples
# -------------------------------
# Pattern 1: Top-left 3x2 block (rows 0-2, cols 0-1)
pattern1 = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
# Pattern 2: Second column and top two of third column
pattern2 = [(1,0), (1,1), (1,2), (1,3), (2,0), (2,1)]
# Pattern 3: Top row and first two cells of second row 
pattern3 = [(0,0), (1,0), (2,0), (3,0), (1,0), (1,1)]
# Pattern 4:
pattern4 = [(0,0), (1,0), (1,1), (2,1), (2,2), (3,1)]
# Pattern 5:
pattern5 = [(0,0), (1,0), (2,0), (1,1), (1,2), (2,2)]
# Pattern 6:
pattern6 = [(0,0), (1,0), (1,1), (1,2), (1,3), (2,3)]
# Pattern 7:
pattern7 = [(0,0), (1,0), (1,1), (1,2), (0,2), (1,3)]
# Pattern 8:
pattern8 = [(0,0), (1,0), (0,1), (2,0), (2,1), (2,2)]

patterns = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8]

# -------------------------------
# Main Training Loop
# -------------------------------
if __name__ == "__main__":
    optimistic_init = 25
    approximator = NTupleApproximator(board_size=4, patterns=patterns, optimistic_init=optimistic_init)
    env = Game2048Env()

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            approximator.weights = pickle.load(f)
        for d in approximator.weights:
            d.default_factory = OptimisticDefault(optimistic_init)

    # Train for 100,000 episodes with multi-step TD updates (default n_step=5)
    final_scores = td_learning(env, approximator, num_episodes=100000, alpha=0.1, gamma=0.99, epsilon=0.1, n_step=5)

    plt.figure(figsize=(8, 4))
    plt.plot(final_scores)
    plt.xlabel("Episode")
    plt.ylabel("Final Score")
    plt.title("2048 Multi-Step TD Learning with N-Tuple Approximation (Optimistic Initialization)")
    plt.savefig(save_fig_path)
    plt.close()

    plt.figure(figsize=(8, 4))
    avg_100_list = [np.mean(final_scores[i:i+100]) for i in range(0, len(final_scores), 100)]
    plt.plot(avg_100_list)
    plt.xlabel("Episode (x100)")
    plt.ylabel("Average Score (Last 100)")
    plt.title("Average Score over Last 100 Episodes")
    plt.savefig(save_fig_100_avg_path)
    plt.close()
