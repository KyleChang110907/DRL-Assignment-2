import copy
import random
import math
import numpy as np
# import pickle
import dill as pickle
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from student_agent import Game2048Env

folder_path = './Modules2048/checkpoints/value_approximation/N_tuple_TD0_6Tuples_4pattern_lr10-1/'
os.makedirs(folder_path, exist_ok=True)

CHECKPOINT_FILE = folder_path + 'value_net.pkl'
final_score_sav_path = folder_path+'final_scores.npy'
save_fig_path = folder_path+'scores.png'
save_fig_100_avg_path = folder_path+'scores_100_avg.png'

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
# NTupleApproximator (4 patterns of 6 tuples)
# -------------------------------
class NTupleApproximator:
    def __init__(self, board_size, patterns, optimistic_init=2000):
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
    # Directly call the move method corresponding to the action.
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
# TD-Learning Training Function (Using Pure States)
# -------------------------------
def td_learning(env, approximator, num_episodes=30000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Train the 2048 agent using TD-learning with an NTuple approximator.
    
    For each move, the best action is selected by a one-step lookahead on the state before random tile addition.
    The incremental reward and the value of the "pure" (non-randomized) state are used.
    The trajectory is recorded using these pure states, and then TD updates are applied.
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
        trajectory = []
        pure_state = copy.deepcopy(state)  # pure state before random tile addition
        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            # Use one-step lookahead on the pure state (simulate move without random tile)
            best_value = -float('inf')
            best_action = None
            best_pure_state = None
            best_inc_reward = 0
            
            original_state = copy.deepcopy(state)
            # print('='*20)
            # print("original_state\n", original_state)
            for a in legal_moves:
                # print("env.board\n", env.board)
                temp_env = simulate_move_without_tile(env, a)
                # print("temp_env.board\n", temp_env.board)
                pure_state_candidate = temp_env.board.copy()
                # Compute incremental reward from simulation: difference in score before random tile.
                # print("temp_env.score", temp_env.score, "env.score", env.score, "previous_score", previous_score)          
                inc_reward_candidate = temp_env.score - previous_score
                future_value_candidate = approximator.value(pure_state_candidate)
                value_est = inc_reward_candidate + gamma * future_value_candidate
                if value_est > best_value:
                    best_value = value_est
                    best_action = a
                    best_pure_state = pure_state_candidate
                    best_inc_reward = inc_reward_candidate
            
            action = best_action
            # Use the "pure" state before random tile as the state for TD update.
            pure_after_state = copy.deepcopy(best_pure_state)
            # print("pure_after_state\n", pure_after_state)
            # Execute the chosen action in the real environment (which adds a random tile).
            next_state, new_score, done, _ = env.step(action)
            # print("next_state\n", next_state)
            # For update, we use the incremental reward and the pure state.
            # Here, we assume that the incremental reward computed from simulation approximates
            # the immediate reward due to the move, before random tile addition.
            incremental_reward = new_score - previous_score
            previous_score = new_score
            
            trajectory.append((pure_state, incremental_reward, pure_after_state, done))
            # Note: We use the same pure state as both the "after move" state and for bootstrapping,
            # because the real state (next_state) includes the random tile.
            state = next_state
            pure_state = copy.deepcopy(pure_after_state)

        # Perform TD(0) updates using the trajectory.
        for (s, r, s_next, done_flag) in trajectory:
            current_value = approximator.value(s)
            next_value = approximator.value(s_next) if not done_flag else 0
            td_target = r + gamma * next_value
            td_error = td_target - current_value
            approximator.update(s, td_error, alpha)
        
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
# Pattern 2: Top-left 2x3 block (rows 0-1, cols 0-2)
pattern2 = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
# Pattern 3: First row (columns 1,2,3) and first three cells of second row (columns 0,1,2)
pattern3 = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)]
# Pattern 4: Second row (columns 1,2,3) and first three cells of third row (columns 0,1,2)
pattern4 = [(1,0), (1,1), (1,2), (1,3), (2,1), (2,2)]

patterns = [pattern1, pattern2, pattern3, pattern4]

# -------------------------------
# Main Training Loop
# -------------------------------
if __name__ == "__main__":
    approximator = NTupleApproximator(board_size=4, patterns=patterns, optimistic_init=2000)
    env = Game2048Env()

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            approximator.weights = pickle.load(f)
    
    # Train for 30,000 episodes.
    final_scores = td_learning(env, approximator, num_episodes=30000, alpha=0.01, gamma=0.99, epsilon=0.1)

    plt.figure(figsize=(8, 4))
    plt.plot(final_scores)
    plt.xlabel("Episode")
    plt.ylabel("Final Score")
    plt.title("2048 TD Learning with N-Tuple Approximation (Optimistic Initialization)")
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


