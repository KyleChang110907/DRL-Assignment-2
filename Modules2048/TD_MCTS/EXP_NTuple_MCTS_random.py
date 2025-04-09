import copy
import random
import math
import numpy as np
import dill as pickle
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict

import dotenv
dotenv.load_dotenv()

# 引入你的 N-Tuple approximator 與相關函式
from Modules2048.value_approximation.EXP_NTuple_TD0_8x6_2 import NTupleApproximator, simulate_move_without_tile
from Modules2048.value_approximation.EXP_NTuple_TD0_8x6_2 import patterns, OptimisticDefault

# Folder 與 checkpoint 路徑
folder_path = './Modules2048/checkpoints/value_approximation/Exp_N_tuple_TD5_6Tuples_8pattern_lr10-1_OI25_test/'
MODE = os.getenv('MODE', 'test')
if MODE == 'LOCAL':
    CHECKPOINT_FILE = os.path.join(folder_path, 'value_net.pkl')
else:
    CHECKPOINT_FILE = 'value_net.pkl'

# -------------------------------
# 輔助函式：將盤面轉為可 hash 的格式（用作字典 key）
# -------------------------------
def state_to_key(state):
    """將 NumPy 陣列轉換為 tuple-of-tuples"""
    return tuple(map(tuple, state))

def make_node_key(state, action):
    """以 (state, action) 作為節點 key，其中 state 已轉換為可 hash 格式"""
    return (state_to_key(state), action)

# -------------------------------
# 盤面移動相關函式（保持原樣）
# -------------------------------
def simulate_row_move(row, board_size=4):
    new_row = row[row != 0]
    new_row = np.pad(new_row, (0, board_size - len(new_row)), mode='constant')
    for i in range(len(new_row) - 1):
        if new_row[i] == new_row[i + 1] and new_row[i] != 0:
            new_row[i] *= 2
            new_row[i + 1] = 0
    new_row = new_row[new_row != 0]
    new_row = np.pad(new_row, (0, board_size - len(new_row)), mode='constant')
    return new_row

def get_legal_moves_from_state(state, board_size=4):
    """
    根據 full state（已含隨機 tile）判斷合法動作
    """
    legal_moves = []
    for action in range(4):
        temp_board = state.copy()
        if action == 0:  # 向上
            for j in range(board_size):
                col = temp_board[:, j]
                new_col = simulate_row_move(col, board_size)
                temp_board[:, j] = new_col
        elif action == 1:  # 向下
            for j in range(board_size):
                col = temp_board[:, j][::-1]
                new_col = simulate_row_move(col, board_size)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # 向左
            for i in range(board_size):
                row = temp_board[i]
                temp_board[i] = simulate_row_move(row, board_size)
        elif action == 3:  # 向右
            for i in range(board_size):
                row = temp_board[i][::-1]
                new_row = simulate_row_move(row, board_size)
                temp_board[i] = new_row[::-1]
        if not np.array_equal(temp_board, state):
            legal_moves.append(action)
    return legal_moves

# -------------------------------
# 隨機 tile 插入函式：將 pure state 轉換為 full state
# -------------------------------
def insert_random_tile(env):
    """
    以純狀態（未加隨機 tile）為基礎，
    從盤面空格中依機率插入 tile（2:0.9, 4:0.1），
    回傳新的 full state
    """
    new_env = copy.deepcopy(env)
    empty_cells = list(zip(*np.where(new_env.board == 0)))
    if empty_cells:
        pos = random.choice(empty_cells)
        tile = 2 if random.random() < 0.9 else 4
        new_env.board[pos[0], pos[1]] = tile
    return new_env

# -------------------------------
# 環境建立輔助函式
# -------------------------------
def create_env_from_state(state, score):
    new_env = copy.deepcopy(env)
    new_env.board = state.copy()
    new_env.score = score
    return new_env

# -------------------------------
# 節點定義：DecisionNode，只存 full state
# 並在邊上記錄從父節點執行 action 得到的 pure state（用於 value 評估）
# -------------------------------
class DecisionNode:
    def __init__(self, state, score, parent=None, action=None, pure_state_edge=None):
        """
        state: full state（已含隨機 tile），即玩家觀察到的盤面。
        score: 該 full state 的分數。
        action: 從父節點執行的玩家動作（若根節點則為 None）。
        pure_state_edge: 執行 action 後、但尚未加入 random tile 的 pure state，
                         此值用來向 value approximator 輸入，計算該邊的價值。
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}    # key 為 (state, action)
        self.untried_actions = get_legal_moves_from_state(self.state, board_size=4)
        self.visits = 0
        self.total_reward = 0.0
        self.pure_state_edge = pure_state_edge  # 只有非根節點會有此值

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

# -------------------------------
# MCTS-PUCT 實作
# -------------------------------
class MCTS_PUCT:
    def __init__(self, env, value_approximator, iterations=50, c_puct=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.value_approximator = value_approximator
        self.iterations = iterations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.max_weight = 0
        for pattern in value_approximator.weights:
            for weight in pattern.values():
                self.max_weight = max(self.max_weight, abs(weight))
        print(f"Max weight in approximator: {self.max_weight}")

    def select_child(self, node):
        # 使用 UCT 公式選擇子節點（這裡以父節點的 children 為基礎）
        best_value = -float('inf')
        best_child = None
        # Get the max total reward of the children
        max_total_reward = max(child.total_reward for child in node.children.values()) if node.children else 1
        # print(f"Max total reward of children: {max_total_reward}")
        for key, child in node.children.items():
            if child.visits > 0:
                Q = (child.total_reward / child.visits) / self.max_weight
                Q = (child.total_reward / child.visits) / max_total_reward

                U = self.c_puct * math.sqrt(math.log(node.visits) / child.visits)
            else:
                Q = 0
                U = float('inf')
            # print(f"Child {key}: visits: {child.visits}, Q: {Q}, U: {U}")
            value = Q + U
            if value > best_value:
                best_value = value
                best_child = child
        return best_child

    def expand_node(self, node):
        """
        擴展節點：
         從當前 node（full state）中選一未嘗試的合法動作，
         使用 simulate_move_without_tile 產生 pure state，
         再依該 pure state 隨機加入 tile（insert_random_tile），
         得到新 full state作為子節點。
         並將產生的 pure state 存於子節點，用於後續值評估。
        """
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            # 以父節點 full state 模擬玩家動作，不加入 tile，此為 pure state
            env_temp = create_env_from_state(node.state, node.score)
            sim_env = simulate_move_without_tile(env_temp, action)
            pure_state = sim_env.board.copy()
            new_score_pure = sim_env.score
            # 將 pure state 轉換為 full state：隨機插入 tile
            env_after_tile = insert_random_tile(create_env_from_state(pure_state, new_score_pure))
            full_state = env_after_tile.board.copy()
            new_score = env_after_tile.score
            # 建立子節點：其 state 為 full state，新節點的 pure_state_edge 為上步得到的 pure state
            child_node = DecisionNode(full_state, new_score, parent=node, action=action, pure_state_edge=pure_state)
            key = make_node_key(child_node.state, action)
            node.children[key] = child_node
            return child_node
        return None

    def rollout(self, state, current_score, depth):
        """
        Rollout 過程：
         從給定 full state（含 random tile）開始，
         每回合先隨機選一合法動作，
         執行玩家動作（不加入 tile）得到 pure state，
         再隨機加入 tile得到下一個 full state，
         直到達到 rollout depth 或無合法動作。
         最後，使用 value approximator 評估最後一次動作後所得到的 pure state（未加入 tile）。
        """
        current_state = state.copy()
        score = current_score
        last_pure_state = None
        for _ in range(depth):
            legal_moves = get_legal_moves_from_state(current_state, board_size=4)
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            env_temp = create_env_from_state(current_state, score)
            sim_env = simulate_move_without_tile(env_temp, action)
            last_pure_state = sim_env.board.copy()   # pure state
            new_score_pure = sim_env.score
            env_after_tile = insert_random_tile(create_env_from_state(last_pure_state, new_score_pure))
            current_state = env_after_tile.board.copy()
            score = env_after_tile.score
        # 若 rollout 至少執行過一次動作，則使用最後的 pure state做估值；否則評估當前 full state轉換為 pure state
        if last_pure_state is None:
            # 如果沒有合法動作，就用 simulate_move_without_tile 隨機選一個動作試試
            legal_moves = get_legal_moves_from_state(state, board_size=4)
            if legal_moves:
                action = random.choice(legal_moves)
                env_temp = create_env_from_state(state, score)
                sim_env = simulate_move_without_tile(env_temp, action)
                last_pure_state = sim_env.board.copy()
            else:
                last_pure_state = state.copy()
        return self.value_approximator.value(last_pure_state)

    def backpropagate(self, node, rollout_reward):
        discount = 1.0
        while node is not None:
            node.visits += 1
            inc_reward = 0
            if node.parent is not None:
                inc_reward = node.score - node.parent.score
            # print(f"Backpropagation: node visits: {node.visits}, inc_reward: {inc_reward}, rollout_reward: {rollout_reward}, total_reward: {node.total_reward}")
            node.total_reward += (inc_reward + rollout_reward) * discount
            discount *= self.gamma
            node = node.parent

    def run_simulation(self, root):
        node = root
        # 選擇與擴展階段：沿著樹從根節點向下移動
        while True:
            if not node.is_fully_expanded():
                node = self.expand_node(node)
                break
            else:
                next_node = self.select_child(node)
                if next_node is None:
                    break
                node = next_node
        # Rollout 階段：從擴展節點所在的 full state進行 rollout
        rollout_reward = self.rollout(node.state, node.score, self.rollout_depth)
        self.backpropagate(node, rollout_reward)

    def best_action(self, root):
        best_visits = -1
        best_action = None
        # 從根節點的子節點中選擇訪問次數最多的邊，回傳其 action
        for key, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = child.action
        return best_action

# -------------------------------
# 載入並初始化 Approximator 與 Environment
# -------------------------------
optimistic_init = 25
approximator = NTupleApproximator(board_size=4, patterns=patterns, optimistic_init=optimistic_init)
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "rb") as f:
        approximator.weights = pickle.load(f)
    print(f"Loaded weights from {CHECKPOINT_FILE}")
else:
    print(f"Checkpoint file {CHECKPOINT_FILE} not found. Using default weights.")

from student_agent import Game2048Env  # 載入 2048 環境
env = Game2048Env()

# 初始化 MCTS_PUCT（參數可調）
mcts_puct = MCTS_PUCT(env, approximator, iterations=200, c_puct=1.41, rollout_depth=1, gamma=0.99)

def get_action(state, score):
    """
    根據當前 full state（含 random tile）與 score，
    利用 MCTS_PUCT 選擇最佳玩家行動。
    """
    root = DecisionNode(state, score, parent=None, action=None, pure_state_edge=None)
    for _ in range(mcts_puct.iterations):
        mcts_puct.run_simulation(root)
    action = mcts_puct.best_action(root)
    return action

# -------------------------------
# 主程式執行與測試
# -------------------------------
if __name__ == "__main__":
    scores = []
    for ep in range(10):
        # env.reset() 回傳 full state（已含隨機 tile）
        state = env.reset()
        done = False
        while not done:
            best_act = get_action(state, env.score)
            state, score, done, _ = env.step(best_act)  # env.step() 內部已隨機插入 tile
        print("Episode {}: Final score: {}, Max tile: {}".format(ep+1, env.score, np.max(state)))
        scores.append(env.score)
    
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print("Average score:", avg_score)
    print("Standard deviation:", std_score)
