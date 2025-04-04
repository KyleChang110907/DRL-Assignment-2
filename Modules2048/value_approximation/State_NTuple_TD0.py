import copy
import random
import math
import numpy as np
import dill as pickle
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from student_agent import Game2048Env

import os 
folder_path = './Modules2048/checkpoints/value_approximation/State_as_key_N_tuple_TD0_6Tuples_4pattern_lr10-1/'
os.makedirs(folder_path, exist_ok=True)

CHECKPOINT_FILE = folder_path + 'value_net.pkl'
final_score_sav_path = folder_path+'final_scores.npy'
save_fig_path = folder_path+'scores.png'
save_fig_100_avg_path = folder_path+'scores_100_avg.png'

# -------------------------------
# Transformation Functions for Symmetries
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
# NTuple Approximator with Symmetries
# -------------------------------
class NTupleApproximator:
    def __init__(self, board_size, patterns, optimistic_init=500):
        """
        Initializes the N-Tuple approximator with symmetry transformations.
        """
        self.board_size = board_size
        self.patterns = patterns  # list of lists of (row, col) tuples
        # One weight dictionary per pattern.
        self.weights = [defaultdict(OptimisticDefault(optimistic_init)) for _ in patterns]
        # Generate 8 symmetrical transformations per pattern.
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

    def generate_symmetries(self, pattern):
        p0    = pattern
        p90   = [rot90(coord, self.board_size) for coord in pattern]
        p180  = [rot180(coord, self.board_size) for coord in pattern]
        p270  = [rot270(coord, self.board_size) for coord in pattern]
        p0f   = [flip_horizontal(coord, self.board_size) for coord in p0]
        p90f  = [flip_horizontal(coord, self.board_size) for coord in p90]
        p180f = [flip_horizontal(coord, self.board_size) for coord in p180]
        p270f = [flip_horizontal(coord, self.board_size) for coord in p270]
        return [p0, p90, p180, p270, p0f, p90f, p180f, p270f]

    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[r][c]) for (r, c) in coords)

    def value(self, board):
        total_val = 0.0
        for i, pattern_syms in enumerate(self.symmetry_patterns):
            for coords in pattern_syms:
                feat = self.get_feature(board, coords)
                total_val += self.weights[i][feat]
        return total_val

    def update(self, board, delta, alpha):
        total_features = sum(len(pattern_syms) for pattern_syms in self.symmetry_patterns)
        update_amount = (alpha * delta) / total_features
        for i, pattern_syms in enumerate(self.symmetry_patterns):
            for coords in pattern_syms:
                feat = self.get_feature(board, coords)
                self.weights[i][feat] += update_amount

# -------------------------------
# Expectimax Action Evaluation
# -------------------------------
def expectimax_action_value(env, approximator, action, previous_score):
    """
    Simulate the move for a given action. If legal, for every empty cell in the resulting board,
    simulate the random insertion (2 with probability 0.9, 4 with probability 0.1) and compute:
      probability * (incremental reward + approximator.value(next_state_with_tile)).
    Sum over all empty cells.
    Returns -infinity if move is illegal.
    """
    temp_env = copy.deepcopy(env)
    # Capture score before move.
    score_before = temp_env.score
    # Apply action.
    if action == 0:
        valid = temp_env.move_up()
    elif action == 1:
        valid = temp_env.move_down()
    elif action == 2:
        valid = temp_env.move_left()
    elif action == 3:
        valid = temp_env.move_right()
    else:
        valid = False

    if not valid:
        return -float('inf')

    # Immediate incremental reward.
    incremental_reward = temp_env.score - score_before

    board_after = temp_env.board
    empties = list(zip(*np.where(board_after == 0)))
    if not empties:
        # Terminal state.
        return incremental_reward

    expected_value = 0.0
    for pos in empties:
        for tile, tile_prob in [(2, 0.9), (4, 0.1)]:
            board_next = board_after.copy()
            board_next[pos] = tile
            p = tile_prob / len(empties)
            expected_value += p * (incremental_reward + approximator.value(board_next))
    return expected_value

# -------------------------------
# TD-Learning Training Function (Episode-wise Updates)
# -------------------------------

def td_learning(env, approximator, num_episodes=30000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Train the agent with episodic TD(0) updates.
    Uses ε-greedy exploration and one-step expectimax lookahead for action selection.
    """
    final_scores = []
    success_flags = []
    avg_scores_list = []
    time_start = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        previous_score = 0
        done = False
        max_tile = np.max(state)
        trajectory = []  # will store tuples: (state, reward, next_state, done)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            best_value = -float('inf')
            best_action = None
            for a in legal_moves:
                value_est = expectimax_action_value(env, approximator, a, previous_score)
                if value_est > best_value:
                    best_value = value_est
                    best_action = a

            action = best_action if best_action is not None else random.choice(legal_moves)

            # Save current state.
            original_state = copy.deepcopy(state)
            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))
            trajectory.append((original_state, incremental_reward, copy.deepcopy(next_state), done))
            state = next_state

        # Episode finished: update approximator for each transition.
        for (s, r, s_next, done_flag) in trajectory:
            current_value = approximator.value(s)
            next_value = approximator.value(s_next) if not done_flag else 0
            td_target = r + gamma * next_value
            td_error = td_target - current_value
            approximator.update(s, td_error, alpha)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
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
# Define N-Tuple Patterns (Your Design)
# -------------------------------
pattern1 = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]          # Top-left 3x2 block.
pattern2 = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]          # Top-left 2x3 block.
pattern3 = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)]          # Example pattern.
pattern4 = [(1,0), (1,1), (1,2), (1,3), (2,1), (2,2)]          # Example pattern.
patterns = [pattern1, pattern2, pattern3, pattern4]

# -------------------------------
# Main Training Loop
# -------------------------------
if __name__ == "__main__":
    approximator = NTupleApproximator(board_size=4, patterns=patterns, optimistic_init=500)
    env = Game2048Env()

    # Adjust num_episodes, alpha, gamma, epsilon as needed.
    final_scores = td_learning(env, approximator, num_episodes=50000, alpha=0.1, gamma=0.99, epsilon=0.1)

    plt.figure(figsize=(8, 4))
    plt.plot(final_scores)
    plt.xlabel("Episode")
    plt.ylabel("Final Score")
    plt.title("2048 TD Learning with N-Tuple Approximation and Expectimax Lookahead")
    plt.savefig(save_fig_path)
    plt.close()

    # Calculate and plot the average score over the last 100 episodes.
    avg_scores = np.convolve(final_scores, np.ones(100)/100, mode='valid')
    plt.figure(figsize=(8, 4))
    plt.plot(avg_scores)
    plt.xlabel("Episode")
    plt.ylabel("Average Score (Last 100 Episodes)")
    plt.title("Average Score Over Last 100 Episodes")
    plt.savefig(save_fig_100_avg_path)
    plt.close()

