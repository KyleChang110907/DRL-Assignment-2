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
folder_path = './Modules2048/checkpoints/value_approximation/Exp_N_tuple_TD5_6Tuples_8pattern_lr10-1_OI25_test/'
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
        # 若無空格，則直接執行動作
        sim_env = simulate_move_without_tile(pure_env, action)
        inc_reward = sim_env.score - pure_env.score
        return inc_reward + approximator.value(sim_env.board)
    
    total = 0.0
    for pos in empty_cells:
        for tile, tile_prob in [(2, 0.9), (4, 0.1)]:
            board_with_tile = pure_state.copy()
            board_with_tile[pos] = tile
            # print(f'pure_env.score: {pure_env.score}')
            temp_env = create_pure_env(board_with_tile, pure_env.score)
            sim_env = simulate_move_without_tile(temp_env, action)
            inc_reward = sim_env.score - pure_env.score
            # print(f'inc_reward: {inc_reward}')
            new_pure = sim_env.board.copy()
            total += (1/len(empty_cells)) * tile_prob * (inc_reward + approximator.value(new_pure))
    return total

# -------------------------------
# Multi-Step TD-Learning Training Function (Using Pure Env and Expected Q)
# -------------------------------
def td_learning(env, approximator, num_episodes=30000, alpha=0.1, gamma=0.99):
    final_scores = []
    success_flags = []
    avg_scores_list = []
    time_start = time.time()

    # 初始時，從 env.reset() 得到的 state（隨機 tile），但建立一份 pure_env
    state = env.reset()
    pure_state = copy.deepcopy(state)
    pure_env = create_pure_env(pure_state, env.score)
    previous_score = env.score

    for episode in range(num_episodes):
        env.reset()
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            
            # print('='*40)
            # print(f'pure_env:\n{pure_env.board}')
            best_value = -float("inf")
            best_action = None
            # 計算每個候選 action 的期望 Q 值，基於純環境 pure_env
            for a in legal_moves:
                expected_q = get_expected_q_for_action(pure_env, a, approximator, gamma)
                if expected_q > best_value:
                    best_value = expected_q
                    best_action = a

            # 選到最佳 action 後，外層再對 pure_env 執行一次 simulate_move_without_tile 更新純環境
            candidate_env = simulate_move_without_tile(env, best_action)
            best_candidate_pure = candidate_env.board.copy()

            # 更新 approximator（以最佳候選動作後的純狀態作為 key）
            # print(f'pure_env 2 :\n{pure_env.board}')
            td_error = best_value - approximator.value(pure_env.board)
            approximator.update(pure_env.board, td_error, alpha)

            # 執行真正的動作：用 env.step() 推進遊戲（包含隨機 tile 插入）
            _, new_score, done, _ = env.step(best_action)
            previous_score = new_score
            max_tile = max(max_tile, np.max(env.board))

            # print(f'pure_env 1 :\n{pure_env.board}')
            # 更新 pure_env：以剛更新的 candidate pure state 與新 score 建立新的純環境
            pure_env = create_pure_env(best_candidate_pure, new_score)
            state = env.board

            # print(f'best_action: {best_action}')
            # print(f'best_candidate_pure:\n{best_candidate_pure}')
            # print(f'env.board:\n{env.board}')


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

    final_scores = td_learning(env, approximator, num_episodes=100000, alpha=0.025, gamma=0.99)

    plt.figure(figsize=(8, 4))
    plt.plot(final_scores)
    plt.xlabel("Episode")
    plt.ylabel("Final Score")
    plt.title("2048 Multi-Step TD Learning with N-Tuple Approximator (Expectimax Approx.)")
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
