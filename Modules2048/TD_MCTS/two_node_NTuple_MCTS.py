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
import dotenv
dotenv.load_dotenv()

# 引入你的 N-Tuple approximator 與相關函式
from Modules2048.value_approximation.EXP_NTuple_TD0_8x6_2 import NTupleApproximator, simulate_move_without_tile
from Modules2048.value_approximation.EXP_NTuple_TD0_8x6_2 import patterns, OptimisticDefault

# -------------------------------
# N-Tuple Approximator 與相關工具（原始程式碼）
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

# (此處已從 Modules2048 引入 OptimisticDefault 與 NTupleApproximator)

# 複製並模擬玩家動作，不塞隨機 tile
def simulate_move_without_tile(env, action):
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

# 建立純狀態環境（僅存盤面與 score，不含隨機 tile）
def create_pure_env(pure_state, score):
    new_env = Game2048Env()
    new_env.board = pure_state.copy()
    new_env.score = score
    return new_env

# -------------------------------
# 以下開始為 MCTS 部分
# -------------------------------

# 定義決策節點（Decision Node）
class DecisionNode:
    def __init__(self, board, score, parent=None, move=None, pure_state_edge=None):
        """
        :param board: full state (含 random tile)
        :param score: 累計得分
        :param parent: 父節點（DecisionNode 或 ChanceNode）
        :param move: 從父節點採用的玩家動作（對 root 為 None）
        :param pure_state_edge: 代表玩家動作後尚未插入 random tile 的盤面，
               若未提供，則預設使用 full state 作為純狀態。
        """
        self.board = board.copy()
        self.score = score
        self.parent = parent
        self.move = move
        # pure_state_edge 為用於值評估的盤面，理想上應該是不含 random tile 的盤面；
        # 若傳入的 state 已經含 random tile，這裡可直接設定為該 state
        self.pure_state_edge = pure_state_edge if pure_state_edge is not None else board.copy()
        self.children = {}     # key: 玩家動作 (0~3)，value: 對應之 ChanceNode
        self.visits = 0
        self.total_value = 0.0
        self.fully_expanded = False

    def is_terminal(self):
        env = Game2048Env()
        env.board = self.board.copy()
        env.score = self.score
        legal_moves = [a for a in range(4) if env.is_move_legal(a)]
        return len(legal_moves) == 0

# 定義機率節點（Chance Node）
class ChanceNode:
    def __init__(self, board, score, parent, move, reward=0):
        self.board = board.copy()
        self.score = score
        self.parent = parent
        self.move = move
        self.reward = reward   # 用來儲存此邊的 incremental reward
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.fully_expanded = False

    def expand(self):
        """展開 chance node：依盤面空格與 tile (2,4) 的機率產生所有可能 outcome"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            # 若無空格，直接建立一個 decision node
            dn = DecisionNode(self.board, self.score, parent=self)
            self.children[('none', None)] = (dn, 1.0)
        else:
            for pos in empty_cells:
                for tile, tile_prob in [(2, 0.9), (4, 0.1)]:
                    outcome_key = (pos, tile)
                    if outcome_key not in self.children:
                        new_board = self.board.copy()
                        new_board[pos] = tile
                        # 此處產生的 decision node 為塞入 random tile 後玩家所看到的盤面
                        dn = DecisionNode(new_board, self.score, parent=self)
                        # outcome 機率為空格均分後乘上 tile 的個別機率
                        self.children[outcome_key] = (dn, (1.0/len(empty_cells)) * tile_prob)
        self.fully_expanded = True

# 輔助函數：模擬玩家動作並回傳 (新盤面, 新得分, 即時獎勵)
def simulate_move_from_board(board, score, action):
    env = Game2048Env()
    env.board = board.copy()
    env.score = score
    env = simulate_move_without_tile(env, action)
    new_board = env.board.copy()
    new_score = env.score
    reward = new_score - score
    return new_board, new_score, reward

# 輔助函數：取得盤面上合法玩家動作（根據盤面狀態）
def get_legal_moves(board):
    env = Game2048Env()
    env.board = board.copy()
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    return legal_moves

# -------------------------------
# 以下為 MCTS_PUCT 類別定義
# -------------------------------
class MCTS_PUCT:
    def __init__(self, env, approximator, iterations=200, c_puct=1.41, rollout_depth=5, gamma=0.99):
        self.env = env
        self.approximator = approximator  # 必須提供 value(board) 方法
        self.iterations = iterations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.max_weight = 0.0
        for pattern in approximator.weights:
            for weight in pattern.values():
                self.max_weight = max(self.max_weight, abs(weight))
        print(f"Max weight in approximator: {self.max_weight}")

    def select_node(self, node, depth=0):
        """
        從給定的 decision node 開始進行選擇，並記錄整條路徑（每項包含：(node, type, immediate_reward)）。
        
        在 decision node 層：
          - 若尚未展開，利用 simulate_move_from_board 展開所有合法動作，
            並在產生的 chance node 中保存玩家動作後所獲得的 incremental reward。
          - 再利用 PUCT 公式選擇最佳的 chance node。
        
        在 chance node 層：
          - 若尚未展開，先展開全部 outcome（展開時依據盤面空格數平均分配，再乘以 tile 的機率），
          - 從 chance node 的所有 outcome中根據機率抽樣選取下一個 decision node；
            此轉換通常沒有額外 reward，因此 immediate_reward 設為 0。

        回傳最終到達的 leaf node 及整條路徑。
        """
        path = []
        current = node
        # Root 的進入 reward 為 0
        path.append((current, 'decision', 0))
        while True:
            if depth >= self.rollout_depth or (isinstance(current, DecisionNode) and current.is_terminal()):
                break
            if isinstance(current, DecisionNode):
                if not current.children:
                    legal_moves = get_legal_moves(current.board)
                    for move in legal_moves:
                        new_board, new_score, immediate_reward = simulate_move_from_board(current.board, current.score, move)
                        # 在建立 chance node 時將 incremental reward 存入（代表玩家動作所帶來的分數增量）
                        current.children[move] = ChanceNode(new_board, new_score, parent=current, move=move, reward=immediate_reward)
                total_visits = current.visits
                best_val = -float('inf')
                best_move = None
                best_child = None
                for move, child in current.children.items():
                    q_val = child.total_value/ self.max_weight / child.visits if child.visits > 0 else 0.0
                    # prior = 1.0 / len(current.children)
                    # u_val = self.c_puct * prior * math.sqrt(total_visits + 1) / (1 + child.visits)
                    u_val = self.c_puct * math.sqrt(math.log(node.visits) / child.visits) if child.visits > 0 else float('inf')
                    puct_val = q_val + u_val
                    if puct_val > best_val:
                        best_val = puct_val
                        best_move = move
                        best_child = child
                path.append((best_child, 'chance', best_child.reward))
                current = best_child
                depth += 1
            elif isinstance(current, ChanceNode):
                if not current.fully_expanded:
                    current.expand()
                path.append((None, 'random_tile', 0))  # 這一層選擇 random tile，沒有 immediate reward
                outcomes = list(current.children.items())
                probs = [outcome[1][1] for outcome in outcomes]
                total_prob = sum(probs)
                norm_probs = [p / total_prob for p in probs]
                chosen_index = np.random.choice(len(outcomes), p=norm_probs)
                chosen_decision_node = outcomes[chosen_index][1][0]
                path.append((chosen_decision_node, 'decision', 0))
                current = chosen_decision_node
            
        # print('=' * 20)
        # print(f'Depth: {depth}, total value of current node: {current.total_value}, visits: {current.visits}')
        # print(f'last state for depth \n {path[-1][0].board}')
        # print(f'last state score for depth {path[-1][0].score}')
        # print(f'last state type for depth {path[-1][1]}')
            # print(f"Depth: {depth}, Current node visits: {current.visits}, Total value: {current.total_value}")
        return current, path

    def backpropagate(self, path, rollout_value):
        """
        沿著選擇路徑回傳更新，考慮每個 action 的 incremental reward。

        我們計算總回報 R 為：
            R = immediate_reward_i + gamma * (immediate_reward_{i+1} + gamma*(... + gamma^n*rollout_value))
        
        並從路徑尾端開始依次更新每個節點的訪問次數與總值。
        """
        R = rollout_value  # 從 leaf node 的 rollout 評估值開始
        # 從路徑的最末端往回更新，每個步驟將 immediate reward 加入
        for node, node_type, immediate_reward in reversed(path):
            R = immediate_reward + self.gamma * R
            # 若 node 為 None（例如在 random tile 層只記錄了一筆佔位資料），則跳過
            if node is not None:
                node.visits += 1
                node.total_value += R

    def run_simulation(self, root):
        """
        一次完整的模擬流程：
          1. 利用 select_node() 從 root 選擇至一個 leaf，並收集路徑上的所有 incremental reward；
          2. 當達到 rollout 條件時，使用 approximator.value() 得到估值；
          3. 利用 backpropagate() 沿整條路徑更新統計值。
        """
        leaf, path = self.select_node(root, depth=0)
        # Rollout 部分：遇到終局或達到深度上限時，直接以 approximator 的評估值作為 rollout_value
        rollout_value = self.approximator.value(leaf.board)
        self.backpropagate(path, rollout_value)
        return rollout_value

    def best_action(self, root):
        """
        根據 root decision node 下各 child (chance node) 的訪問次數選出最佳玩家動作。
        """
        best_move = None
        best_visits = -1
        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
        return best_move

# -------------------------------
# 定義 get_action 函數與模擬完整遊戲
# -------------------------------
# -------------------------------
# 主程式：讀取 checkpoint、初始化 approximator 與環境，並模擬 10 場完整遊戲
# -------------------------------

# Folder 與 checkpoint 路徑設定
folder_path = './Modules2048/checkpoints/value_approximation/Exp_N_tuple_TD5_6Tuples_8pattern_lr10-1_OI25_test_19566/'
MODE = os.getenv('MODE', 'test')
if MODE == 'LOCAL':
    CHECKPOINT_FILE = os.path.join(folder_path, 'value_net.pkl')
else:
    CHECKPOINT_FILE = 'value_net.pkl'

optimistic_init = 25
approximator = NTupleApproximator(4, patterns, optimistic_init)

# 注意：在後續程式中，approximator 會以 checkpoint 載入後的 approximator 來取代此處的 None
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "rb") as f:
        approximator.weights = pickle.load(f)
    
    print(f"Loaded weights from {CHECKPOINT_FILE}")
else:
    print(f"Checkpoint file {CHECKPOINT_FILE} not found. Using default weights.")

# 更新 MCTS_PUCT 裡的 approximator 與 env（環境實體）
# 初始化 MCTS_PUCT (參數可根據需要調整)
env = Game2048Env()
mcts_puct = MCTS_PUCT(env=env, approximator=approximator, iterations=100, c_puct=1.41, rollout_depth=5, gamma=0.99)
# get_action 函數更新：明確建立 decision node 並將 full state 設為 pure_state_edge
def get_action(state, score):
    """
    根據當前 full state（含 random tile）與 score，
    利用 MCTS_PUCT 選擇最佳玩家行動。
    為避免將 full state 誤認為 chance node，
    我們在建立根節點時，明確設定 pure_state_edge 為 state。
    """
    # 將 full state 作為 decision node，但明確設定 pure_state_edge (純狀態) 為 state
    root = DecisionNode(state, score, parent=None, move=None, pure_state_edge=state)
    for _ in range(mcts_puct.iterations):
        mcts_puct.run_simulation(root)
    action = mcts_puct.best_action(root)
    return action


if __name__ == "__main__":
  
    # 模擬 10 場完整遊戲並紀錄最終分數
    scores = []
    for ep in range(10):
        state = env.reset()  # 回傳 full state（含 random tile）
        done = False
        while not done:
            best_act = get_action(state, env.score)
            state, score, done, _ = env.step(best_act)  # env.step() 內部會隨機塞入 tile
        print("Episode {}: Final score: {}, Max tile: {}".format(ep+1, env.score, np.max(state)))
        scores.append(env.score)
    
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print("Average score:", avg_score)
    print("Standard deviation:", std_score)
