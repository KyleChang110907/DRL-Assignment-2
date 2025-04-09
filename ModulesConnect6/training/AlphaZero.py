import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import combinations
import random
import math
import copy
import os
import matplotlib.pyplot as plt
import time
import logging
import sys

# 確保 log 資料夾存在
os.makedirs('./logs', exist_ok=True)

# 建立 logger 並設定等級
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 建立 file handler，將訊息寫入檔案
file_handler = logging.FileHandler('./logs/AlphaZero_debug.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 建立 stream handler，將訊息輸出到命令列
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(stream_formatter)

# 將 handler 加到 logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# Checkpoint file path
folder_path = './ModulesConnect6/checkpoints/alphazero/'
os.makedirs(folder_path, exist_ok=True)
CHECKPOINT_FILE = folder_path + 'connect6_net.pth'
loss_plot_path = folder_path + 'loss_plot.png'

# Global Parameters
BOARD_SIZE = 19         # Board dimensions.
WIN_COUNT = 6           # Number in a row to win.
HISTORY_LENGTH = 4      # Number of past states to stack.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SIMULATIONS = 100    # Number of MCTS simulations per move. (原本800)
CPUCT = 1.0             # Exploration constant for PUCT.
DIRICHLET_ALPHA = 0.3   # Dirichlet noise parameter.
DIRICHLET_EPSILON = 0.25
L2_REG = 1e-4           # L2 regularization coefficient.
MAX_REPLAY_BUFFER_SIZE = 10000  # Maximum replay buffer size.
EVAL_INTERVAL = 100      # Evaluate every 10 self-play games (原本500)

# ----------------------------
# Connect6 Game Implementation with Stacked History
# ----------------------------
class Connect6:
    def __init__(self, history_length=HISTORY_LENGTH):
        # Board: 0 = empty, 1 = player1, -1 = player2.
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1
        self.move_count = 0  # First move is single; later moves are pairs.
        self.history_length = history_length
        # Initialize history with empty boards.
        self.history = [np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int) for _ in range(history_length)]
    
    def copy(self):
        new_game = Connect6(history_length=self.history_length)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        new_game.history = [h.copy() for h in self.history]
        return new_game

    def get_legal_moves(self):
        if self.move_count == 0:
            return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if self.board[i, j] == 0]
        else:
            candidates = self.get_candidate_moves()
            legal = []
            for move_pair in combinations(candidates, 2):
                legal.append(move_pair)
            return legal

    def get_candidate_moves(self):
        candidates = set()
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i, j] != 0:
                    for d in directions:
                        ni, nj = i + d[0], j + d[1]
                        if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and self.board[ni, nj] == 0:
                            candidates.add((ni, nj))
        if not candidates:
            candidates.add((BOARD_SIZE // 2, BOARD_SIZE // 2))
        return list(candidates)

    def make_move(self, move):
        if self.move_count == 0:
            i, j = move
            if self.board[i, j] != 0:
                raise ValueError("Invalid move")
            self.board[i, j] = self.current_player
        else:
            for pos in move:
                i, j = pos
                if self.board[i, j] != 0:
                    raise ValueError("Invalid move")
                self.board[i, j] = self.current_player
        self.move_count += 1
        # Update history: discard oldest and add current board.
        self.history.pop(0)
        self.history.append(self.board.copy())
        self.current_player = -self.current_player

    def is_terminal(self):
        if self.check_win(1) or self.check_win(-1):
            return True
        if np.all(self.board != 0):
            return True
        return False

    def check_win(self, player):
        board = self.board
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] == player:
                    if j + WIN_COUNT <= BOARD_SIZE and all(board[i, j+k] == player for k in range(WIN_COUNT)):
                        return True
                    if i + WIN_COUNT <= BOARD_SIZE and all(board[i+k, j] == player for k in range(WIN_COUNT)):
                        return True
                    if i + WIN_COUNT <= BOARD_SIZE and j + WIN_COUNT <= BOARD_SIZE and all(board[i+k, j+k] == player for k in range(WIN_COUNT)):
                        return True
                    if i - WIN_COUNT >= -1 and j + WIN_COUNT <= BOARD_SIZE and all(board[i-k, j+k] == player for k in range(WIN_COUNT)):
                        return True
        return False

    def get_winner(self):
        if self.check_win(1):
            return 1
        elif self.check_win(-1):
            return -1
        else:
            return 0

    def get_state_tensor(self):
        """
        Stacks the last 'history_length' board states.
        For each board in history, two channels are computed:
         - Channel for cells occupied by current player's stones.
         - Channel for cells occupied by opponent's stones.
        Returns a tensor of shape (2 * history_length, BOARD_SIZE, BOARD_SIZE).
        """
        start_time = time.time()
        channels = []
        for board in self.history:
            channels.append((board == self.current_player).astype(np.float32))
            channels.append((board == -self.current_player).astype(np.float32))
        state = np.stack(channels, axis=0)
        state_tensor = torch.from_numpy(state).to(DEVICE)
        logger.debug(f"get_state_tensor took {time.time() - start_time:.4f} sec")
        return state_tensor

# -------------------------------
# Data Augmentation: Board Symmetries
# -------------------------------
def augment_state_policy(state, policy):
    start_time = time.time()
    augmented = []
    policy_board = policy.reshape((BOARD_SIZE, BOARD_SIZE))
    for k in range(4):
        rotated_state = torch.rot90(state, k, [1, 2])
        rotated_policy = np.rot90(policy_board, k)
        augmented.append((rotated_state, rotated_policy.flatten()))
        flipped_state = torch.flip(rotated_state, dims=[2])
        flipped_policy = np.fliplr(rotated_policy)
        augmented.append((flipped_state, flipped_policy.flatten()))
    logger.debug(f"augment_state_policy took {time.time() - start_time:.4f} sec")
    return augmented

# ----------------------------------
# Neural Network: Policy & Value Head with Stacked Input
# ----------------------------------
class Connect6Net(nn.Module):
    def __init__(self, history_length=HISTORY_LENGTH):
        super(Connect6Net, self).__init__()
        in_channels = 2 * history_length
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * BOARD_SIZE * BOARD_SIZE, 256)
        self.policy_head = nn.Linear(256, BOARD_SIZE * BOARD_SIZE)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value

# -------------------------------
# MCTS Node and Tree Search
# -------------------------------
class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, prior=0.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, net, num_simulations=NUM_SIMULATIONS):
        self.net = net
        self.num_simulations = num_simulations

    def run(self, game_state):
        start_run = time.time()
        root = MCTSNode(game_state.copy())
        legal_moves = game_state.get_legal_moves()
        policy_logits, _ = self.evaluate(game_state)
        move_priors = {}
        if game_state.move_count == 0:
            for move in legal_moves:
                idx = move[0] * BOARD_SIZE + move[1]
                move_priors[move] = F.softmax(policy_logits, dim=0)[idx].item()
        else:
            for move_pair in legal_moves:
                idx1 = move_pair[0][0] * BOARD_SIZE + move_pair[0][1]
                idx2 = move_pair[1][0] * BOARD_SIZE + move_pair[1][1]
                prob = (F.softmax(policy_logits, dim=0)[idx1].item() +
                        F.softmax(policy_logits, dim=0)[idx2].item())
                move_priors[move_pair] = prob

        total = sum(move_priors.values())
        for move in move_priors:
            move_priors[move] /= total
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(move_priors))
        for i, move in enumerate(move_priors):
            move_priors[move] = (1 - DIRICHLET_EPSILON) * move_priors[move] + DIRICHLET_EPSILON * noise[i]

        for move, prior in move_priors.items():
            new_state = game_state.copy()
            new_state.make_move(move)
            root.children[move] = MCTSNode(new_state, parent=root, move=move, prior=prior)

        # 執行 MCTS 模擬
        for sim in range(self.num_simulations):
            sim_start = time.time()
            node = root
            search_path = [node]
            while node.children:
                move, node = self.select_child(node)
                search_path.append(node)
            value = self.evaluate_leaf(node.game_state)
            self.backpropagate(search_path, value, node.game_state.current_player)
            logger.debug(f"MCTS simulation {sim+1}/{self.num_simulations} took {time.time() - sim_start:.4f} sec")
        total_run = time.time() - start_run
        logger.debug(f"MCTS.run took {total_run:.4f} sec for {self.num_simulations} simulations")
        return root

    def select_child(self, node):
        best_score = -float('inf')
        best_move = None
        best_child = None
        for move, child in node.children.items():
            score = child.value() + CPUCT * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def evaluate(self, game_state):
        start = time.time()
        state_tensor = game_state.get_state_tensor().unsqueeze(0)
        policy_logits, value = self.net(state_tensor)
        elapsed = time.time() - start
        logger.debug(f"MCTS.evaluate took {elapsed:.4f} sec")
        return policy_logits.squeeze(0), value.item()

    def evaluate_leaf(self, game_state):
        if game_state.is_terminal():
            winner = game_state.get_winner()
            if winner == 0:
                return 0
            return 1 if winner == game_state.current_player else -1
        _, value = self.evaluate(game_state)
        return value

    def backpropagate(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value if node.game_state.current_player == to_play else -value

    def select_action(self, root, temperature=1.0):
        moves = list(root.children.keys())
        visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)
        if temperature == 0:
            best_move = moves[np.argmax(visits)]
            policy = np.zeros_like(visits, dtype=np.float32)
            policy[np.argmax(visits)] = 1.0
            return best_move, policy
        visits = visits ** (1 / temperature)
        policy = visits / np.sum(visits)
        move = random.choices(moves, weights=policy)[0]
        return move, policy

# -------------------------------
# Rule-Based Agent: Random Near Opponent's Last Move
# -------------------------------
def rule_based_agent_move(game, last_opponent_move, distance_threshold=2):
    legal_moves = game.get_legal_moves()
    if last_opponent_move is None:
        return random.choice(legal_moves)
    near_moves = []
    for move in legal_moves:
        if game.move_count == 0:
            i, j = move
            li, lj = last_opponent_move
            if abs(i - li) + abs(j - lj) <= distance_threshold:
                near_moves.append(move)
        else:
            distances = []
            for stone in move:
                i, j = stone
                # 假設 last_opponent_move 為單一 tuple 或雙步走法 tuple
                li, lj = last_opponent_move[0] if isinstance(last_opponent_move, tuple) and isinstance(last_opponent_move[0], tuple) else last_opponent_move
                distances.append(abs(i - li) + abs(j - lj))
            if min(distances) <= distance_threshold:
                near_moves.append(move)
    if not near_moves:
        return random.choice(legal_moves)
    else:
        return random.choice(near_moves)

# -------------------------------
# Evaluation Routine vs. Rule-Based Agent
# -------------------------------
def evaluate_agent(net, num_games=20):
    wins, losses, draws = 0, 0, 0
    for _ in range(num_games):
        game = Connect6()
        mcts = MCTS(net, num_simulations=NUM_SIMULATIONS)
        last_move = None
        # 假設訓練後的代理總是以 player1 身份行棋
        while not game.is_terminal():
            if game.current_player == 1:
                root = mcts.run(game)
                move, _ = mcts.select_action(root, temperature=0)
                last_move = move
            else:
                move = rule_based_agent_move(game, last_move, distance_threshold=2)
                last_move = move
            game.make_move(move)
        winner = game.get_winner()
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
    logger.info(f"Evaluation vs. Rule-Based Agent - Wins: {wins}, Losses: {losses}, Draws: {draws}")

# -------------------------------
# Self-Play Routine with Data Augmentation
# -------------------------------
def self_play_game(net):
    start_selfplay = time.time()
    game = Connect6()
    states, mcts_policies, current_players = [], [], []
    mcts = MCTS(net)
    
    while not game.is_terminal():
        sp_start = time.time()
        root = mcts.run(game)
        move, policy = mcts.select_action(root, temperature=1.0)
        sp_eval = time.time() - sp_start
        logger.debug(f"self_play_game: MCTS + selection took {sp_eval:.4f} sec")
        state_tensor = game.get_state_tensor().cpu().numpy()
        policy_vector = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        if game.move_count == 0:
            idx = move[0] * BOARD_SIZE + move[1]
            policy_vector[idx] = 1.0
        else:
            idx1 = move[0][0] * BOARD_SIZE + move[0][1]
            idx2 = move[1][0] * BOARD_SIZE + move[1][1]
            policy_vector[idx1] = 0.5
            policy_vector[idx2] = 0.5
        aug_start = time.time()
        augmented = augment_state_policy(torch.tensor(state_tensor), policy_vector)
        logger.debug(f"self_play_game: augment_state_policy took {time.time() - aug_start:.4f} sec")
        for aug_state, aug_policy in augmented:
            states.append(aug_state.cpu().numpy())
            mcts_policies.append(aug_policy)
            current_players.append(game.current_player)
        game.make_move(move)
    total_sp = time.time() - start_selfplay
    logger.debug(f"self_play_game total time: {total_sp:.4f} sec")
    if game.check_win(1):
        logger.info("Player 1 wins!")
    elif game.check_win(-1):
        logger.info("Player 2 wins!")
    else:
        logger.info("Draw!")
    
    winner = game.get_winner()
    outcomes = [1 if winner == player else -1 if winner != 0 else 0 for player in current_players]
    return states, mcts_policies, outcomes

# -------------------------------
# Training Function with L2 Regularization
# -------------------------------
def train(net, optimizer, batch_states, batch_policies, batch_values):
    net.train()
    start_train = time.time()
    # 將列表轉換為 numpy array 後再轉換成 tensor，避免慢速操作
    state_tensor = torch.tensor(np.array(batch_states), dtype=torch.float32).to(DEVICE)
    target_policy = torch.tensor(np.array(batch_policies), dtype=torch.float32).to(DEVICE)
    target_value = torch.tensor(np.array(batch_values), dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    pred_policy, pred_value = net(state_tensor)
    loss_policy = -torch.mean(torch.sum(target_policy * F.log_softmax(pred_policy, dim=1), dim=1))
    loss_value = F.mse_loss(pred_value, target_value)
    l2_loss = sum(torch.sum(param**2) for param in net.parameters())
    loss = loss_value + loss_policy + L2_REG * l2_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    elapsed = time.time() - start_train
    logger.debug(f"train() took {elapsed:.4f} sec")
    return loss.item()

# -------------------------------
# Main Training Loop with Evaluation, Buffer Cleaning, Checkpoint Saving & Time Tracking
# -------------------------------
def main_training_loop(num_iterations=1000, games_per_iteration=10):
    net = Connect6Net(history_length=HISTORY_LENGTH).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    
    replay_buffer_states = []
    replay_buffer_policies = []
    replay_buffer_values = []
    total_games_played = 0
    iteration_losses = []  # To record average loss per iteration
    iteration_times = []   # To record time spent per iteration
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        logger.info(f"Iteration: {iteration}, LR: {scheduler.get_last_lr()[0]}")
        for i in range(games_per_iteration):
            sp_start = time.time()
            states, mcts_policies, outcomes = self_play_game(net)
            replay_buffer_states.extend(states)
            replay_buffer_policies.extend(mcts_policies)
            replay_buffer_values.extend(outcomes)
            total_games_played += 1
            sp_elapsed = time.time() - sp_start
            start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sp_start))
            logger.info(f"Self-play game {i+1}/{games_per_iteration} took {sp_elapsed:.2f} sec starting at {start_time_str}")
            if total_games_played % EVAL_INTERVAL == 0:
                logger.info(f"Evaluating at {total_games_played} self-play games...")
                evaluate_agent(net, num_games=10)
        
        # Clean replay buffer if too large.
        if len(replay_buffer_states) > MAX_REPLAY_BUFFER_SIZE:
            replay_buffer_states = replay_buffer_states[-MAX_REPLAY_BUFFER_SIZE:]
            replay_buffer_policies = replay_buffer_policies[-MAX_REPLAY_BUFFER_SIZE:]
            replay_buffer_values = replay_buffer_values[-MAX_REPLAY_BUFFER_SIZE:]
        
        # Training phase.
        logger.info("Training on replay buffer...")
        num_epochs = 5
        batch_size = 16
        train_start = time.time()
        indices = list(range(len(replay_buffer_states)))
        random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0
        for epoch in range(num_epochs):
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_states = [replay_buffer_states[j] for j in batch_idx]
                batch_policies = [replay_buffer_policies[j] for j in batch_idx]
                batch_values = [replay_buffer_values[j] for j in batch_idx]
                loss = train(net, optimizer, batch_states, batch_policies, batch_values)
                epoch_loss += loss
                num_batches += 1
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        iteration_losses.append(avg_loss)
        train_elapsed = time.time() - train_start
        iter_elapsed = time.time() - iter_start
        iteration_times.append(iter_elapsed)
        logger.info(f"Iteration {iteration}, Average Loss: {avg_loss:.4f}, Training Time: {train_elapsed:.2f} sec, Total Iteration Time: {iter_elapsed:.2f} sec")
        scheduler.step()
        
        torch.save(net.state_dict(), CHECKPOINT_FILE)
    
    # After training, plot the loss history.
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_losses, label="Average Loss per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Final save.
    torch.save(net.state_dict(), CHECKPOINT_FILE)
    logger.info("Training complete. Final network saved as connect6_net_alphaZero.pth.")

if __name__ == "__main__":
    main_training_loop(num_iterations=1000, games_per_iteration=10)
