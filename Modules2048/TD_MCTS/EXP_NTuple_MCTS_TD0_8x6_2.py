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

from Modules2048.value_approximation.NTuple_TD5_8x6 import NTupleApproximator, simulate_move_without_tile
from Modules2048.value_approximation.NTuple_TD5_8x6 import patterns, OptimisticDefault
    
# Folder and checkpoint paths.
folder_path = './Modules2048/checkpoints/value_approximation/Exp_N_tuple_TD5_6Tuples_8pattern_lr10-1_OI25_test/'
# folder_path = './Modules2048/checkpoints/value_approximation/Exp_N_tuple_TD5_6Tuples_8pattern_lr10-1_OI25_test_19566/'
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

def insert_random_tile(pure_state):
    """
    Insert a random tile (2 or 4) into the board at a random empty position.
    """
    empty_positions = np.argwhere(pure_state == 0)
    if len(empty_positions) == 0:
        return pure_state  # No empty positions to fill.
    
    row, col = random.choice(empty_positions)
    # the probability of 2 and 4 is 0.9 and 0.1 respectively
    new_tile_value = 2 if random.random() < 0.9 else 4
    new_state = pure_state.copy()
    new_state[row, col] = new_tile_value
    return new_state

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
        print(f"Max weight in approximator: {self.max_weight}")
    
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
            # print(f"Action: {action}, Q: {Q}, U: {U}, Visits: {child.visits}")
            value = Q + U
            # print(f"Action: {action}, Value: {value}, Q: {Q}, U: {U}")
            if value > best_value:
                best_value = value
                best_child = child
        return best_child

    
    # def rollout(self, sim_env, depth):
    #     """
    #     Perform a random rollout for a fixed depth from the current simulated environment,
    #     but simulate moves without adding a random tile so that we evaluate the "pure" state.
    #     At the end of the rollout, use the value approximator to evaluate that pure state.
    #     """
    #     for i in range(depth):
    #         new_state = insert_random_tile(sim_env.board)
    #         sim_env = self.create_env_from_state(new_state, sim_env.score)

    #         legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
    #         if not legal_moves:
    #             break
    #         action = random.choice(legal_moves)
    #         # Simulate the move without adding a random tile.
    #         sim_env = simulate_move_without_tile(sim_env, action)
            
    #     # Evaluate the pure state (without the random tile).
    #     pure_value = self.value_approximator.value(sim_env.board)
    #     # Optionally print if the pure_value exceeds a threshold.
        
    #     return pure_value

    # def rollout(self, sim_env, depth):
    #     """
    #     從當前 sim_env (full state) 以遞迴方式進行 rollout，
    #     每一步首先對當前 board 執行 insert_random_tile，
    #     然後在該步隨機產生 5 個 outcome 分支，分別：
    #     1. 建立新的環境（full state）；
    #     2. 取得所有合法 move，隨機選一個，利用 simulate_move_without_tile 模擬得到 pure state；
    #     3. 遞迴進入 rollout(depth-1) 得到該分支的 reward。
    #     最終將 5 條分支的 reward 平均作為本層的未來 reward，並加上該步的 immediate reward後返回。
        
    #     當 depth = 0 或當前環境無合法 move 時，直接利用 value_approximator 對當前 board（full state）進行評估。

    #     返回值：一個 float，代表從當前 sim_env 開始、深度為 depth 時的 rollout reward。
    #     """
    #     # 若達到 rollout 深度上限，直接用 value approximator 評估。
    #     legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
    #     if depth <= 0 or not legal_moves:
    #         return self.value_approximator.value(sim_env.board)
        
    #     # Decision phase: 先選擇一個隨機合法動作並模擬玩家動作(不加入 random tile) 
    #     # ※這裡可視需求先做 insert_random_tile，再 simulate_move_without_tile。
    #     # 按照需求：必須先 insert_random_tile，再 simulate_move_without_tile。
    #     # 於是我們在每層 rollout 中，先針對 sim_env.board 進行 random tile 插入，
    #     # 但不是只採樣一個 outcome，而是採樣 K = 5 條 outcome，再對每條 outcome後續遞迴 rollout。
    #     K = 10
    #     branch_rewards = []
    #     for _ in range(K):
    #         # 先執行 random tile 插入，產生一個 outcome (full state)
    #         new_state = insert_random_tile(sim_env.board)
    #         # 建立對應的新環境，延用原先的 score
    #         branch_env = self.create_env_from_state(new_state, sim_env.score)
            
    #         # 接著 decision phase：從 branch_env 中取得合法動作
    #         branch_legal_moves = [a for a in range(4) if branch_env.is_move_legal(a)]
    #         if not branch_legal_moves:
    #             # 若無合法動作，直接評估此 outcome 狀態
    #             branch_reward = self.value_approximator.value(branch_env.board)
    #         else:
    #             action = random.choice(branch_legal_moves)
    #             # 模擬玩家動作（不加入 random tile）
    #             branch_env = simulate_move_without_tile(branch_env, action)
    #             # immediate reward 為模擬後分數增量
    #             # 這裡我們用遞迴繼續 rollout，深度減 1
    #             branch_reward = self.rollout(branch_env, depth - 1)
    #         branch_rewards.append(branch_reward)
            
    #     # 計算所有分支的平均 reward
    #     avg_future_reward = sum(branch_rewards) / K
    #     # print(f'Num    of branches length: {len(branch_rewards)}, avg_future_reward: {avg_future_reward}')
    #     # 此層沒有額外的 immediate reward（或可以視作該層隨機 tile 的操作本身不帶獎勵）
    #     # 返回折扣後的 future reward
    #     return avg_future_reward

    def rollout(self, sim_env, depth):
        """
        從當前 sim_env (full state) 以遞迴方式進行 rollout：
        1. 每一層首先對當前 board 執行 insert_random_tile，產生一個 outcome (full state)。
        2. 建立對應的新環境 branch_env，延用原先的 score。
        3. 在 branch_env 上進入決策 phase：取得所有合法 move，若有則隨機選一個，
            並利用 simulate_move_without_tile 模擬執行該 move，計算 immediate reward 為
            (branch_env.score_after_move - branch_env.score_before_move)。
        4. 對該 outcome 後續以 rollout(depth-1) 遞迴得到後續 reward。
        5. 最終，每一條 outcome 的 branch reward = immediate reward + (recursive rollout reward)。
        6. 對同一層採樣固定 K 條 outcome，取這 K 條分支的平均作為本層未來 reward返回。
        
        如果 depth <= 0 或當前 sim_env 沒有合法 move，直接以 value_approximator 評估目前 full state。
        
        返回值為一個 float，代表從當前 sim_env 開始、深度為 depth 時的 rollout reward。
        """
        # Termination condition: depth 到限或無合法 move。
        legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
        if depth <= 0 or not legal_moves:
            return self.value_approximator.value(sim_env.board)
        
        K = 5  # 固定每層採樣分支數，調整 K 以平衡運算量與準確度
        branch_rewards = []
        # 在每一層的 rollout 先對當前 board 進行 insert_random_tile，
        # 並從這個 outcome 分別隨機採樣 K 次後續分支
        for _ in range(K):
            # 先執行 random tile 插入，產生 outcome full state
            new_state = insert_random_tile(sim_env.board)
            # 產生新的環境並延用原始 score
            branch_env = self.create_env_from_state(new_state, sim_env.score)
            
            # Decision phase：從 branch_env 中取得合法 move
            branch_legal_moves = [a for a in range(4) if branch_env.is_move_legal(a)]
            if not branch_legal_moves:
                # 若無合法動作，直接評估該 outcome 狀態
                branch_reward = self.value_approximator.value(branch_env.board)
            else:
                # 記錄執行玩家動作前的分數
                score_before = branch_env.score
                action = random.choice(branch_legal_moves)
                # 模擬玩家的動作，不加入 random tile，
                # 此時 branch_env.board 轉為純狀態
                branch_env = simulate_move_without_tile(branch_env, action)
                # immediate reward 為模擬後分數減去動作前的分數
                immediate_reward = branch_env.score - score_before
                # 遞迴 rollout：進入下一層深度 (depth - 1)
                recursive_reward = self.rollout(branch_env, depth - 1)
                branch_reward = immediate_reward + recursive_reward
            branch_rewards.append(branch_reward)
        
        # 取 K 條分支 reward 的平均作為本層未來 reward
        avg_future_reward = sum(branch_rewards) / K
        return avg_future_reward




    def backpropagate(self, node, rollout_reward):
        """
        沿著選擇路徑回傳更新，每次更新時把該步 immediate reward 與以後的 reward 累加起來，
        使得前面的 Q 值包含後來的 reward。
        
        公式： R = inc_reward + gamma * R，其中 inc_reward 為當前節點由父節點帶來的 reward，
        rollout_reward 為從 leaf 模擬得到的最終評估值。
        """
        R = rollout_reward
        while node is not None:
            # 計算當前節點的 incremental reward：非 root node 時，為 node.score 與父節點 score 的差
            if node.parent is not None:
                inc_reward = node.score - node.parent.score
            else:
                inc_reward = 0
            # 累計回報：把當前步的 incremental reward 與後續累積 reward 相加（再乘以折扣）
            R = inc_reward + self.gamma * R
            node.visits += 1
            node.total_reward += R
            node = node.parent

    
    def run_simulation(self, root):
        node = root
        # Create a simulation environment starting from the node's pure state.
        sim_env = self.create_env_from_state(node.state, node.score)
        
        # Selection: traverse the tree until reaching an expandable node.
        # while node.fully_expanded() and node.children:
        #     node = self.select_child(node)
        #     sim_env.step(node.action)
        
        while node.fully_expanded() and node.children:
            next_node = self.select_child(node)
            # 執行 move 並捕捉返回值
            _, new_score, _, _ = sim_env.step(next_node.action)
            # 如果需要，可以將 next_node.score 更新為 new_score
            next_node.score = new_score
            node = next_node

        
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
optimistic_init = 25  # Adjust this value as needed.
approximator = NTupleApproximator(board_size=4, patterns=patterns, optimistic_init=optimistic_init)
# Optionally load trained weights:
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "rb") as f:
        approximator.weights = pickle.load(f)
    print(f"Loaded weights from {CHECKPOINT_FILE}")

    # for d in approximator.weights:
    #     d.default_factory = OptimisticDefault(optimistic_init)

else:
    print(f"Checkpoint file {CHECKPOINT_FILE} not found. Using default weights.")

from student_agent import Game2048Env  # Provided environment
env = Game2048Env()
mcts_puct = MCTS_PUCT(env, approximator, iterations=100, c_puct=1.5, rollout_depth=2 , gamma=0.99)
    

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
        start_time = time.time()
        state = env.reset()
        done = False
        while not done:
            best_act = get_action(state, env.score)
            state, score, done, _ = env.step(best_act)
            # env.render(action=best_act)
        time_taken = time.time() - start_time
        print("Final score:", env.score, "Max tile:", np.max(state), "Time taken:", time_taken)
        scores.append(env.score)
    
    # avg and std of scores
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print("Average score:", avg_score)
    print("Standard deviation:", std_score)

