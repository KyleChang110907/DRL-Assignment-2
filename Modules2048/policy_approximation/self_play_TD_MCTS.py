import copy
import random
import math
import numpy as np
import dill as pickle  # using dill for serialization
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from student_agent import Game2048Env  # Provided environment

# Import the NTuple approximator and helper functions from your value approximation module.
from Modules2048.value_approximation.NTuple_TD0 import NTupleApproximator, simulate_move_without_tile
from Modules2048.value_approximation.NTuple_TD0 import patterns
from Modules2048.value_approximation.NTuple_TD0 import rot90, rot180, rot270, flip_horizontal

from Modules2048.TD_MCTS.NTuple_MCTS import PUCTNode, MCTS_PUCT
value_folder_path = './Modules2048/checkpoints/value_approximation/N_tuple_TD0_6Tuples_4pattern_lr10-1_/'

VALUE_CHECKPOINT_FILE = os.path.join(value_folder_path, 'value_net.pkl')

folder_path = './Modules2048/checkpoints/policy_approximation/self_play_TD_MCTS/'
os.makedirs(folder_path, exist_ok=True)
CHECKPOINT_FILE = os.path.join(folder_path, 'policy_net.pkl')
# -----------------------------------------
# Action Transformation Functions
# -----------------------------------------
def identity_action(action):
    return action

def rot90_action(action):
    mapping = {0: 3, 3: 1, 1: 2, 2: 0}
    return mapping[action]

def rot180_action(action):
    mapping = {0: 1, 1: 0, 2: 3, 3: 2}
    return mapping[action]

def rot270_action(action):
    mapping = {0: 2, 2: 1, 1: 3, 3: 0}
    return mapping[action]

def flip_horizontal_action(action):
    mapping = {0: 0, 1: 1, 2: 3, 3: 2}
    return mapping[action]

def inverse_from_type(sym_type):
    """
    Given a symmetry type string, return a function that maps a transformed action
    back to the original action.
    """
    if sym_type == "rot0":
        return identity_action
    elif sym_type == "rot90":
        return rot270_action
    elif sym_type == "rot180":
        return rot180_action
    elif sym_type == "rot270":
        return rot90_action
    elif sym_type == "flip_rot0":
        return flip_horizontal_action
    elif sym_type == "flip_rot90":
        def inv(a): return rot270_action(flip_horizontal_action(a))
        return inv
    elif sym_type == "flip_rot180":
        def inv(a): return rot180_action(flip_horizontal_action(a))
        return inv
    elif sym_type == "flip_rot270":
        def inv(a): return rot90_action(flip_horizontal_action(a))
        return inv
    else:
        return identity_action

# -----------------------------------------
# Policy Approximator Class (with Action Transformations)
# -----------------------------------------
class PolicyApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the policy approximator.
        This table-based approximator maps features (extracted from predefined board patterns)
        to action preferences. It is trained using MCTS visit distributions as targets.
        We incorporate symmetry by generating eight symmetric variants for each pattern,
        and for each variant we also store a corresponding action transformation function.
        """
        self.board_size = board_size
        self.patterns = patterns
        self.actions = [0, 1, 2, 3]  # 0: up, 1: down, 2: left, 3: right
        # Weight structure: one dictionary per pattern mapping feature tuples to dictionaries of action weights.
        self.weights = [defaultdict(lambda: defaultdict(float)) for _ in range(len(patterns))]
        # Generate symmetric variants and corresponding action transformation functions.
        self.symmetry_patterns = []
        self.symmetry_types = []  # List of lists, one per pattern.
        self.action_transforms = []  # List of lists, one per pattern.
        for pattern in self.patterns:
            syms, types = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)
            self.symmetry_types.append(types)
            transforms = []
            for t in types:
                if t == "rot0":
                    transforms.append(identity_action)
                elif t == "rot90":
                    transforms.append(rot90_action)
                elif t == "rot180":
                    transforms.append(rot180_action)
                elif t == "rot270":
                    transforms.append(rot270_action)
                elif t == "flip_rot0":
                    transforms.append(flip_horizontal_action)
                elif t == "flip_rot90":
                    def flip_rot90(a, f=rot90_action):
                        return flip_horizontal_action(f(a))
                    transforms.append(flip_rot90)
                elif t == "flip_rot180":
                    def flip_rot180(a, f=rot180_action):
                        return flip_horizontal_action(f(a))
                    transforms.append(flip_rot180)
                elif t == "flip_rot270":
                    def flip_rot270(a, f=rot270_action):
                        return flip_horizontal_action(f(a))
                    transforms.append(flip_rot270)
                else:
                    transforms.append(identity_action)
            self.action_transforms.append(transforms)
    
    def generate_symmetries(self, pattern):
        """
        Generate 8 symmetrical transformations of the given pattern.
        Returns a tuple (syms, types) where syms is a list of 8 patterns and
        types is a list of corresponding symmetry type strings.
        The types are: "rot0", "rot90", "rot180", "rot270", "flip_rot0", "flip_rot90", "flip_rot180", "flip_rot270".
        """
        p0 = pattern
        p90  = [rot90(coord, self.board_size) for coord in pattern]
        p180 = [rot180(coord, self.board_size) for coord in pattern]
        p270 = [rot270(coord, self.board_size) for coord in pattern]
        p0f  = [flip_horizontal(coord, self.board_size) for coord in p0]
        p90f  = [flip_horizontal(coord, self.board_size) for coord in p90]
        p180f = [flip_horizontal(coord, self.board_size) for coord in p180]
        p270f = [flip_horizontal(coord, self.board_size) for coord in p270]
        syms = [p0, p90, p180, p270, p0f, p90f, p180f, p270f]
        types = ["rot0", "rot90", "rot180", "rot270", "flip_rot0", "flip_rot90", "flip_rot180", "flip_rot270"]
        return syms, types
    
    def tile_to_index(self, tile):
        return 0 if tile == 0 else int(math.log(tile, 2))
    
    def get_feature(self, board, coords):
        """
        Extracts tile values from the board at the given coordinates and returns a feature tuple.
        """
        return tuple(self.tile_to_index(board[r][c]) for (r, c) in coords)
    
    def predict(self, board):
        """
        Predicts a probability distribution over actions for the given board state.
        For each pattern, we extract the feature from each of its 8 symmetric variants,
        retrieve the stored weights for the transformed actions, and then use the inverse
        transformation to map these contributions back to the original action space.
        The aggregated scores are then averaged over all patterns and passed through softmax.
        """
        scores = np.zeros(len(self.actions))
        for i in range(len(self.patterns)):
            for j in range(len(self.symmetry_patterns[i])):
                feature = self.get_feature(board, self.symmetry_patterns[i][j])
                weight_dict = self.weights[i][feature]
                inv_transform = inverse_from_type(self.symmetry_types[i][j])
                for sym_action, weight in weight_dict.items():
                    orig_action = inv_transform(sym_action)
                    scores[orig_action] += weight
        scores /= (len(self.patterns) * 8)
        exp_scores = np.exp(scores - np.max(scores))
        prob = exp_scores / np.sum(exp_scores)
        return prob
    
    def update(self, board, target_distribution, alpha=0.1):
        """
        Updates the policy approximator based on the target distribution from MCTS.
        For each pattern, the feature is extracted from the board and for each original action,
        the corresponding symmetric action is computed (using the action transform), and its weight
        is updated via gradient descent to move closer to the target.
        """
        for i in range(len(self.patterns)):
            for j in range(len(self.symmetry_patterns[i])):
                feature = self.get_feature(board, self.symmetry_patterns[i][j])
                for a in self.actions:
                    sym_a = self.action_transforms[i][j](a)
                    current_val = self.weights[i][feature].get(sym_a, 0.0)
                    error = target_distribution[a] - current_val
                    self.weights[i][feature][sym_a] = current_val + alpha * error

# -----------------------------------------
# Evaluation Function for Policy Approximator
# -----------------------------------------
def evaluate_policy(env, policy_approximator, num_episodes=10):
    """
    Evaluate the trained policy approximator by choosing the action with the largest probability.
    For each episode, the function resets the environment and then, at each step:
      - Computes the action probability distribution using policy_approximator.predict(state)
      - Selects the action with the highest probability (greedy)
      - Executes that action in the environment and renders the board
    Finally, returns the scores for all episodes along with average and standard deviation.
    """
    scores = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Get probability distribution over actions.
            action_probs = policy_approximator.predict(state)
            best_action = int(np.argmax(action_probs))
            state, score, done, _ = env.step(best_action)
            # env.render(action=best_action)
        scores.append(env.score)
        print(f"Episode {ep+1}/{num_episodes} finished, final score: {env.score}")
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print("Average score:", avg_score)
    print("Standard deviation:", std_score)
    return scores, avg_score, std_score

# Run self-play training for the policy approximator using TD-MCTS.
def self_play_training_policy_with_td_mcts(env, mcts_puct, policy_approximator, num_episodes=50, alpha=0.1):
    """
    Self-play training for the policy approximator.
    For each move in an episode:
        1. Build an MCTS-PUCT tree from the current state.
        2. Obtain the normalized visit count distribution from MCTS.
        3. Use that distribution as the target label to update the policy approximator.
        4. Execute the best action in the real environment.
    """
    for episode in range(num_episodes):
        state = env.reset()
        temp_env = copy.deepcopy(env)
        done = False
        while not done:
            root = PUCTNode(temp_env.board, temp_env.score)
            for _ in range(mcts_puct.iterations):
                mcts_puct.run_simulation(root)
            
            # Get visit distribution as target.
            # First filter legal moves
            legal_actions = [a for a in range(4) if env.is_move_legal(a)]
            # Get the best action based on visit counts within legal moves.
            best_act = max(legal_actions, key=lambda a: root.children[a].visits)
            total_visits = sum(child.visits for child in root.children.values())
            
            target_distribution = np.zeros(4)
            for action, child in root.children.items():
                target_distribution[action] = child.visits / total_visits if total_visits > 0 else 0
                
            # Update the policy approximator.
            policy_approximator.update(temp_env.board, target_distribution, alpha)
            copy_env = copy.deepcopy(env)
            # execute the action without tile addition
            temp_env = simulate_move_without_tile(copy_env, best_act)
            # get the new state
            state, score, done, _ = env.step(best_act)
            print(f'State: {state}, Action: {best_act}, Score: {env.score}, Target distribution: {target_distribution}')
            legal_actions = [a for a in range(4) if env.is_move_legal(a)]
            print(f'legal actions: {legal_actions}')
            # print(f"Action taken: {best_act}, Score: {env.score}, Target distribution: {target_distribution}")
            # env.render(action=best_act)
        print(f"Episode {episode+1}/{num_episodes} finished, final score: {env.score}")
        # Save the weights after each episode.
        with open(CHECKPOINT_FILE, "wb") as f:  
            pickle.dump(policy_approximator.weights, f)  
    return
    

# -----------------------------------------
# Main Integration and Testing
# -----------------------------------------
if __name__ == "__main__":
    
    
    # Instantiate the NTupleApproximator with optimistic initialization.
    value_approximator = NTupleApproximator(board_size=4, patterns=patterns, optimistic_init=50)
    if os.path.exists(VALUE_CHECKPOINT_FILE):
        with open(VALUE_CHECKPOINT_FILE, "rb") as f:
            value_approximator.weights = pickle.load(f)
    
    env = Game2048Env()
    # Instantiate MCTS-PUCT with the trained value approximator.
    mcts_puct = MCTS_PUCT(env, value_approximator, iterations=100, c_puct=1.41, rollout_depth=1, gamma=0.99)
    
    # Instantiate the Policy Approximator.
    policy_approximator = PolicyApproximator(board_size=4, patterns=patterns)
    
    self_play_training_policy_with_td_mcts(env, mcts_puct, policy_approximator, num_episodes=50, alpha=0.1)
    
    # Evaluate the policy approximator: choose the action with the largest probability.
    evaluate_policy(env, policy_approximator, num_episodes=10)
