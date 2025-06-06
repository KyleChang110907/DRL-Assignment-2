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

# 建立 checkpoint 資料夾
folder_path = './Modules2048/checkpoints/value_approximation/Exp_N_tuple_TDlambda_6Tuples_8pattern_lr10-1_OI25/'
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
        初始化 N-Tuple approximator，採用樂觀初始化以鼓勵探索。
        """
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(OptimisticDefault(optimistic_init)) for _ in patterns]
        self.symmetry_patterns = [self.generate_symmetries(p) for p in patterns]

    def generate_symmetries(self, pattern):
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
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[r][c]) for (r, c) in coords)

    def value(self, board):
        total_val = 0.0
        for i, syms in enumerate(self.symmetry_patterns):
            for coords in syms:
                feat = self.get_feature(board, coords)
                total_val += self.weights[i][feat]
        return total_val

    def update(self, board, delta, alpha):
        total_features = sum(len(syms) for syms in self.symmetry_patterns)
        update_amount = (alpha * delta) / total_features
        for i, syms in enumerate(self.symmetry_patterns):
            for coords in syms:
                feat = self.get_feature(board, coords)
                self.weights[i][feat] += update_amount

# -------------------------------
# Helper Function: Simulate Move Without Random Tile
# -------------------------------
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

# -------------------------------
# 建立純狀態環境 (Pure Env)
# -------------------------------
def create_pure_env(pure_state, score):
    new_env = Game2048Env()
    new_env.board = pure_state.copy()
    new_env.score = score
    return new_env

# -------------------------------
# Compute Expected Q Value for Candidate Action (包含增量 reward)
# -------------------------------
def get_expected_q_for_action(pure_env, action, approximator, gamma):
    """
    以 pure_env (純環境，不含隨機 tile) 為輸入，
    針對候選 action：窮舉 pure_env.board 的所有空格，
      對每個空格依 2 (0.9) 與 4 (0.1) 插入，得到隨機盤面，
      然後以該隨機盤面建立臨時環境，
      接著執行 candidate action（使用 simulate_move_without_tile），
      並計算該動作的增量 reward (sim_env.score - pure_env.score) 與 approximator.value(sim_env.board) 的和，
      最後依照 tile 機率做加權平均回傳。
    """
    pure_state = pure_env.board.copy()
    empty_cells = list(zip(*np.where(pure_state == 0)))
    if not empty_cells:
        sim_env = simulate_move_without_tile(pure_env, action)
        inc_reward = sim_env.score - pure_env.score
        return inc_reward + approximator.value(sim_env.board)
    
    total = 0.0
    for pos in empty_cells:
        for tile, tile_prob in [(2, 0.9), (4, 0.1)]:
            board_with_tile = pure_state.copy()
            board_with_tile[pos] = tile
            temp_env = create_pure_env(board_with_tile, pure_env.score)
            sim_env = simulate_move_without_tile(temp_env, action)
            inc_reward = sim_env.score - pure_env.score
            new_pure = sim_env.board.copy()
            total += (1/len(empty_cells)) * tile_prob * (inc_reward + approximator.value(new_pure))
    return total

# -------------------------------
# Offline TD(λ) Learning Training Function (Batch Update After Each Episode)
# -------------------------------
def td_lambda_offline_learning(env, approximator, num_episodes=30000, alpha=0.1, gamma=0.99, lmbda=0.8):
    """
    離線版 TD(λ) 學習：
    將整局遊戲過程存下來，最後一次性更新 n-tuple approximator 的權重。
    
    參數:
      env: 遊戲環境
      approximator: NTupleApproximator 物件
      num_episodes: 訓練局數
      alpha: 學習率
      gamma: 折扣因子
      lmbda: TD(λ) 中的 λ 參數 (0 <= λ <= 1)
    """
    final_scores = []
    success_flags = []
    avg_scores_list = []
    time_start = time.time()

    total_features = sum(len(syms) for syms in approximator.symmetry_patterns)

    for episode in range(num_episodes):
        env.reset()
        state = env.board
        pure_state = state.copy()
        pure_env = create_pure_env(pure_state, env.score)
        episode_data = []  # 儲存每一步 (board, delta)
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            best_value = -float("inf")
            best_action = None
            for a in legal_moves:
                expected_q = get_expected_q_for_action(pure_env, a, approximator, gamma)
                if expected_q > best_value:
                    best_value = expected_q
                    best_action = a

            V_current = approximator.value(pure_env.board)
            delta = best_value - V_current
            episode_data.append((pure_env.board.copy(), delta))

            candidate_env = simulate_move_without_tile(env, best_action)
            best_candidate_pure = candidate_env.board.copy()

            _, new_score, done, _ = env.step(best_action)
            max_tile = max(max_tile, np.max(env.board))
            pure_env = create_pure_env(best_candidate_pure, new_score)
            state = env.board

        # 離線計算每一步的累積 TD 誤差 G_t 並更新權重
        T = len(episode_data)
        G = [0.0] * T
        for t in range(T):
            cum_update = 0.0
            discount = 1.0
            for j in range(t, T):
                _, delta_j = episode_data[j]
                cum_update += discount * delta_j
                discount *= gamma * lmbda
            G[t] = cum_update

        for t in range(T):
            board, _ = episode_data[t]
            update_amount = (alpha * G[t]) / total_features
            for i, syms in enumerate(approximator.symmetry_patterns):
                for coords in syms:
                    feat = approximator.get_feature(board, coords)
                    approximator.weights[i][feat] += update_amount

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode+1) % 100 == 0:
            interval = time.time() - time_start
            avg_score = np.mean(final_scores[-100:])
            avg_scores_list.append(avg_score)
            with open(CHECKPOINT_FILE, "wb") as f:
                pickle.dump(approximator.weights, f)
            np.save(final_score_sav_path, np.array(final_scores))
            print(f"Episode {episode+1}/{num_episodes} | Block Avg Score: {avg_score:.2f} | Success Rate: {np.mean(success_flags[-100:]):.2f} | Time: {interval:.2f}s")
            time_start = time.time()

    return final_scores

# -------------------------------
# Define 8 Patterns of 6 Tuples
# -------------------------------
pattern1 = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
pattern2 = [(1,0), (1,1), (1,2), (1,3), (2,0), (2,1)]
pattern3 = [(0,0), (1,0), (2,0), (3,0), (1,0), (1,1)]
pattern4 = [(0,0), (1,0), (1,1), (2,1), (2,2), (3,1)]
pattern5 = [(0,0), (1,0), (2,0), (1,1), (1,2), (2,2)]
pattern6 = [(0,0), (1,0), (1,1), (1,2), (1,3), (2,3)]
pattern7 = [(0,0), (1,0), (1,1), (1,2), (0,2), (1,3)]
pattern8 = [(0,0), (1,0), (0,1), (2,0), (2,1), (2,2)]
patterns = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8]

# -------------------------------
# Main Training Loop
# -------------------------------
if __name__ == "__main__":
    optimistic_init = 25
    approximator = NTupleApproximator(4, patterns, optimistic_init)
    env = Game2048Env()

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            approximator.weights = pickle.load(f)
        for d in approximator.weights:
            d.default_factory = OptimisticDefault(optimistic_init)
        print(f"Loaded weights from {CHECKPOINT_FILE}")
    else:
        print(f"Checkpoint file {CHECKPOINT_FILE} not found. Using default weights.")

    # 使用離線版 TD(λ) 進行訓練
    final_scores = td_lambda_offline_learning(env, approximator, num_episodes=100000, alpha=0.1, gamma=0.99, lmbda=0.8)

    plt.figure(figsize=(8, 4))
    plt.plot(final_scores)
    plt.xlabel("Episode")
    plt.ylabel("Final Score")
    plt.title("2048 TD(λ) Offline Learning with N-Tuple Approximator")
    plt.savefig(save_fig_path)
    plt.close()

    avg_100 = [np.mean(final_scores[i:i+100]) for i in range(0, len(final_scores), 100)]
    plt.figure(figsize=(8, 4))
    plt.plot(avg_100)
    plt.xlabel("Episode (x100)")
    plt.ylabel("Average Score")
    plt.title("Moving Average Score (per 100 episodes)")
    plt.savefig(save_fig_100_avg_path)
    plt.close()
