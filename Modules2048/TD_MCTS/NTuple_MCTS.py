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


import dotenv
dotenv.load_dotenv()

from Modules2048.value_approximation.NTuple_TD0 import NTupleApproximator, simulate_move_without_tile
from Modules2048.value_approximation.NTuple_TD0 import patterns, OptimisticDefault
    
# Folder and checkpoint paths.
folder_path = './Modules2048/checkpoints/value_approximation/N_tuple_TD0_6Tuples_4pattern_lr10-1_ori/'

MODE = os.getenv('MODE', 'test')
if MODE == 'LOCAL':
    CHECKPOINT_FILE = os.path.join(folder_path, 'value_net.pkl')
    # CHECKPOINT_FILE = './value_net.pkl'
else:
    CHECKPOINT_FILE = 'value_net.pkl'

# final_score_sav_path = os.path.join(folder_path, 'final_scores.npy')
# save_fig_path = os.path.join(folder_path, 'scores.png')
# save_fig_100_avg_path = os.path.join(folder_path, 'scores_100_avg.png')

# -------------------------------
# MCTS-PUCT Integration with the Trained Value Approximator
# -------------------------------

def simulate_row_move(row, board_size=4):
    # Compress the row: move non-zero elements to the left.
    new_row = row[row != 0]
    new_row = np.pad(new_row, (0, board_size - len(new_row)), mode='constant')
    # Merge adjacent equal numbers.
    for i in range(len(new_row) - 1):
        if new_row[i] == new_row[i + 1] and new_row[i] != 0:
            new_row[i] *= 2
            new_row[i + 1] = 0
    # Compress again after merging.
    new_row = new_row[new_row != 0]
    new_row = np.pad(new_row, (0, board_size - len(new_row)), mode='constant')
    return new_row

def get_legal_moves_from_state(state, board_size=4):
    legal_moves = []
    for action in range(4):
        temp_board = state.copy()
        if action == 0:  # move up: process columns
            for j in range(board_size):
                col = temp_board[:, j]
                new_col = simulate_row_move(col, board_size)
                temp_board[:, j] = new_col
        elif action == 1:  # move down: process reversed columns
            for j in range(board_size):
                col = temp_board[:, j][::-1]
                new_col = simulate_row_move(col, board_size)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # move left: process rows
            for i in range(board_size):
                row = temp_board[i]
                temp_board[i] = simulate_row_move(row, board_size)
        elif action == 3:  # move right: process reversed rows
            for i in range(board_size):
                row = temp_board[i][::-1]
                new_row = simulate_row_move(row, board_size)
                temp_board[i] = new_row[::-1]
        # If the simulated board changed, the move is legal.
        if not np.array_equal(temp_board, state):
            legal_moves.append(action)
    return legal_moves

class PUCTNode:
    def __init__(self, state, score, parent=None, action=None):
        """
        A node in the MCTS tree.
        state: The "pure" board state after a move (before random tile addition).
        score: The score corresponding to that pure state.
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # Only consider legal moves from the pure state.
        # Only consider legal moves directly from the state.
        self.untried_actions = get_legal_moves_from_state(self.state, 4)
    
    def fully_expanded(self):
        return len(self.untried_actions) == 0

class MCTS_PUCT:
    def __init__(self, env, value_approximator, iterations=50, c_puct=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.value_approximator = value_approximator
        self.iterations = iterations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.gamma = gamma

        # get the maximum value of the weights in approximator
        self.max_weight = 0
        # print('approximator.weights:', value_approximator.weights)
        for pattern in value_approximator.weights:
            for weight in pattern.values():
                self.max_weight = max(self.max_weight, abs(weight))
        # print(f"Max weight in approximator: {self.max_weight}")
    
    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env
    
    def select_child(self, node):
        best_value = -float('inf')
        best_child = None
        for action, child in node.children.items():
            # Q value: average reward.
            Q = child.total_reward / self.max_weight / child.visits if child.visits > 0 else 0
            # U term for exploration.
            U = self.c_puct * math.sqrt(math.log(node.visits) / child.visits) if child.visits > 0 else float('inf')
            value = Q + U
            # print(f"Action: {action}, Value: {value}, Q: {Q}, U: {U}")
            if value > best_value:
                best_value = value
                best_child = child
        return best_child
    
    def rollout(self, sim_env, depth):
        """
        Perform a random rollout for a fixed depth from the current simulated environment,
        but simulate moves without adding a random tile so that we evaluate the "pure" state.
        At the end of the rollout, use the value approximator to evaluate that pure state.
        """
        for _ in range(depth):
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            # Simulate the move without adding a random tile.
            sim_env = simulate_move_without_tile(sim_env, action)
        # Evaluate the pure state (without the random tile).
        pure_value = self.value_approximator.value(sim_env.board)
        # Optionally print if the pure_value exceeds a threshold.
        
        return pure_value
    
    def backpropagate(self, node, rollout_reward):
        discount = 1.0
        while node is not None:
            node.visits += 1
            # For non-root nodes, compute the incremental reward (reward due to the move that led to this node)
            if node.parent is not None:
                inc_reward = node.score - node.parent.score
            else:
                inc_reward = 0
            # The reward to propagate at this node is the incremental reward plus the rollout reward.
            total_reward = inc_reward + rollout_reward
            node.total_reward += total_reward * discount
            discount *= self.gamma
            node = node.parent

    
    def run_simulation(self, root):
        node = root
        # Create a simulation environment starting from the node's pure state.
        sim_env = self.create_env_from_state(node.state, node.score)
        
        # Selection: traverse the tree until reaching an expandable node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            sim_env.step(node.action)
        
        # Expansion: if there is an untried action, expand it.
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            # Simulate the move without adding a random tile.
            temp_env = simulate_move_without_tile(sim_env, action)
            new_state = temp_env.board.copy()  # pure state after the move
            new_score = temp_env.score
            new_node = PUCTNode(new_state, new_score, parent=node, action=action)
            node.children[action] = new_node
            node = new_node
            # Update simulation environment to reflect the pure state.
            sim_env = self.create_env_from_state(new_state, new_score)
        
        # Rollout phase: perform a random rollout.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the reward.
        self.backpropagate(node, rollout_reward)

    
    def best_action(self, root):
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action


# Instantiate the NTupleApproximator with optimistic initialization.
optimistic_init = 0.1  # Adjust this value as needed.
approximator = NTupleApproximator(board_size=4, patterns=patterns, optimistic_init=optimistic_init)
# Optionally load trained weights:
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "rb") as f:
        approximator.weights = pickle.load(f)
    print(f"Loaded weights from {CHECKPOINT_FILE}")

    for d in approximator.weights:
            d.default_factory = OptimisticDefault(optimistic_init)

else:
    print(f"Checkpoint file {CHECKPOINT_FILE} not found. Using default weights.")

from student_agent import Game2048Env  # Provided environment
env = Game2048Env()
mcts_puct = MCTS_PUCT(env, approximator, iterations=100, c_puct=1.41, rollout_depth=1 , gamma=0.99)
    

def get_action(state, score):
    """
    Given a state (after random tile assignment) and the current score,
    returns an action as determined by TD-MCTS.

    Args:
        state: A numpy array representing the 2048 board.
        score: Current score.
    
    Returns:
        action: An integer in {0,1,2,3} representing the selected move.
    """
    # Create a new environment and override its board and score.
    
    # Create the root node for MCTS using the given state and score.
    root = PUCTNode(state, score)
    
    # print out illegal moves
    # print('legal moves:', root.untried_actions)
    
    # Run MCTS simulations from the root.
    for _ in range(mcts_puct.iterations):
        mcts_puct.run_simulation(root)
    
    # Select the best action from the MCTS tree.
    action = mcts_puct.best_action(root)
    return action


# -------------------------------
# Main Integration and Testing
# -------------------------------
if __name__ == "__main__":

    # Run ten episode using MCTS_PUCT for action selection.
    scores = []
    for _ in range(10):
        state = env.reset()
        done = False
        while not done:
            best_act = get_action(state, env.score)
            state, score, done, _ = env.step(best_act)
            # env.render(action=best_act)
        
        print("Final score:", env.score)
        scores.append(env.score)
    
    # avg and std of scores
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print("Average score:", avg_score)
    print("Standard deviation:", std_score)

