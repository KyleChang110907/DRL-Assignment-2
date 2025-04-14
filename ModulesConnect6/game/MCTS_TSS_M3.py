#!/usr/bin/env python3
import sys
import numpy as np
import random
import copy
import io
import math
import datetime
import logging
import os

# Configure logging.
log_folder = "./logs/MCTS_TSS_in_T_collect/"
os.makedirs(log_folder, exist_ok=True)
log_filename = log_folder + "log_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
logging.basicConfig(filename=log_filename,
                    level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# =============================================================================
# 1. Connect6 Game Environment Definition
# =============================================================================
class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """
        Checks if a player wins:
          0 - No winner yet
          1 - Black wins
          2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts a column index to a letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))

    def label_to_index(self, col_char):
        """Converts a column letter to an index (considering skipped 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """
        Plays a move and checks game status.
          - color: 'B' or 'W'
          - move: move string, e.g. "A1" or "A1,B2"
        """
        if self.game_over:
            print("? Game over")
            return
        stones = move.split(',')
        positions = []
        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))
        for row, col in positions:
            self.board[row, col] = 1 if color.upper()=='B' else 2
        # 第一手只下一子，其餘每回合下兩子
        if np.count_nonzero(self.board) == 1:
            pass  # 第一手後不更換 turn
        else:
            self.turn = 3 - self.turn
        print("= ", end='', flush=True)

    def generate_move(self, color):
        """
        若盤面全空則直接下正中間（不進 MCTS），否則使用 MCTS 搜索，
        此處使用新的動作空間：以 Manhattan 距離<=3 的點加上 threat 候選，
        並在 best_child 中對 threat 行動給予極大 Q 獎勵。
        """
        if self.game_over:
            print("? Game over")
            return
        if np.count_nonzero(self.board) == 0:
            # 盤面全空，下正中間
            center = (self.size // 2, self.size // 2)
            move_str = move_to_str(center, self)
            self.play_move(color, move_str)
            logging.info(f"First move at center: {move_str}")
            print(f"{move_str}\n\n", end='', flush=True)
            print(move_str, file=sys.stderr)
            return
        root = MCTSNode(clone_game(self))
        best_move = mcts(root, itermax=5)
        move_str = move_to_str(best_move, self)
        self.play_move(color, move_str)
        logging.info(f"MCTS selected move: {move_str}")
        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)

    def show_board(self):
        """Prints the board state as text."""
        print("= ")
        for row in range(self.size-1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            return "env_board_size=19"
        if not command:
            return
        parts = command.split()
        cmd = parts[0].lower()
        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop: reads and processes GTP commands from stdin."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

# =============================================================================
# 2. MCTS and Action Space / Pre-MCTS Check Functions and Classes
# =============================================================================

# 新的動作空間函式：
# 若盤面全空，僅返回所有單點；
# 否則從 Manhattan 距離 3 以內所有空點生成兩子組合，再加入 threat 候選。
def get_action_space(game, player):
    if np.count_nonzero(game.board) == 0:
        return [(r, c) for r in range(game.size) for c in range(game.size) if game.board[r, c] == 0]
    stones = [(r, c) for r in range(game.size) for c in range(game.size) if game.board[r, c] != 0]
    candidates = set()
    for stone in stones:
        for nb in get_neighbors(game, stone, 3):
            candidates.add(nb)
    candidates = list(candidates)
    moves = []
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            moves.append((candidates[i], candidates[j]))
    threat = pre_mcts_collect(game, player, phase="defense")
    if threat is not None:
        moves.extend(threat)
        logging.info(f"Threat candidates: {threat} added :{moves}")
    return list(set(moves))

def move_to_str(move, game):
    """
    Converts a move into a string:
      - If move is a single point, returns "A1".
      - Otherwise, returns "A1,B2".
    """
    if isinstance(move[0], int):
        r, c = move
        return f"{game.index_to_label(c)}{r+1}"
    else:
        return ",".join(f"{game.index_to_label(c)}{r+1}" for r, c in move)

def clone_game(game):
    return copy.deepcopy(game)

def terminal(game):
    if game.check_win() != 0:
        return True
    if np.all(game.board != 0):
        return True
    return False

# 修改 best_child：對 threat 子節點給予額外 Q 獎勵
def best_child(node, c_param=1.41):
    Q_bonus = float('inf')
    choices_weights = []
    for child in node.children:
        base = child.wins / child.visits
        if hasattr(child, 'is_threat') and child.is_threat:
            base += Q_bonus
        w = base + c_param * math.sqrt((2 * math.log(node.visits)) / child.visits) if child.visits > 0 else 1000
        choices_weights.append(w)
    unvisited = [child for child in node.children if child.visits == 0]
    if unvisited:
        logging.debug("Best_child: Found unvisited child; choosing randomly.")
        return random.choice(unvisited)
    best = node.children[np.argmax(choices_weights)]
    logging.debug("Best_child: Chosen child with highest modified Q value.")
    return best

# 修改後的 tree_policy，不再在每個節點展開時呼叫 pre_MCTS_check
def tree_policy(node):
    iteration = 0
    while not terminal(node.game):
        iteration += 1
        if node.untried_moves:
            move = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
            logging.debug(f"Tree_policy iteration {iteration}: Expanding move {move}")
            new_game = clone_game(node.game)
            mstr = move_to_str(move, new_game)
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            color = 'B' if new_game.turn == 1 else 'W'
            new_game.play_move(color, mstr)
            sys.stdout = old_stdout
            child_node = MCTSNode(new_game, move, node)
            # 不再在 tree_policy 重新執行 pre_MCTS_check
            node.children.append(child_node)
            return child_node
        else:
            node = best_child(node)
    logging.debug("Tree_policy: Terminal node reached.")
    return node

def get_neighbors(game, stone, distance):
    r0, c0 = stone
    rows = np.arange(game.size)[:, None]
    cols = np.arange(game.size)
    dists = np.abs(rows - r0) + np.abs(cols - c0)
    mask = (game.board == 0) & (dists <= distance)
    indices = np.argwhere(mask)
    return [tuple(idx) for idx in indices]

def get_defensive_moves(game, attacker_move, defense_type):
    if defense_type == "conservative":
        candidate_set = set()
        if isinstance(attacker_move[0], int):
            candidate_set.update(get_neighbors(game, attacker_move, 2))
        else:
            for stone in attacker_move:
                candidate_set.update(get_neighbors(game, stone, 2))
        candidate_list = list(candidate_set)
        moves = []
        if len(candidate_list) < 2:
            return []
        for i in range(len(candidate_list)):
            for j in range(i+1, len(candidate_list)):
                moves.append((candidate_list[i], candidate_list[j]))
        return moves
    else:
        return get_action_space(game, game.turn)

# 以下 pre_MCTS_collect 保持原有邏輯，僅返回一組行動（兩子），檢查順序依原來設定
def check_window_margin_custom(window, board_coords, my_val, L):
    if len(window) != L:
        return None
    for cell in window:
        if cell not in (0, my_val):
            return None
    if np.count_nonzero(window == my_val) != (L - 2):
        return None
    if window[0] == 0 and window[-1] == 0:
        logging.debug(f"check_window_margin_custom: For L={L}, window {window} with coords {board_coords} triggers candidate.")
        return [board_coords[0], board_coords[-1]]
    else:
        return None

def check_window_intermediate_custom(window, board_coords, my_val, L, defense_mode=False):
    if len(window) != L:
        return None
    for cell in window:
        if cell not in (0, my_val):
            return None
    if np.count_nonzero(window == my_val) != (L - 2):
        return None
    if window[0] == 0 and window[-1] == 0:
        logging.debug(f"Intermediate condition already covered for window {window}.")
        return None
    else:
        empty_cells = [board_coords[i] for i in range(L) if window[i] == 0]
        if len(empty_cells) == 2:
            if defense_mode:
                valid_moves = [board_coords[i] for i in range(L) if i not in (0, L - 1)]
                return valid_moves
            else:
                return [empty_cells[0], empty_cells[1]]
        else:
            return None

def check_window_missing_one_custom(window, board_coords, my_val, L):
    if len(window) != L:
        return None
    for cell in window:
        if cell not in (0, my_val):
            return None
    if np.count_nonzero(window == my_val) != (L - 1):
        return None
    empty_cells = [board_coords[i] for i in range(L) if window[i] == 0]
    if len(empty_cells) != 1:
        return None
    return [empty_cells[0]]

def pre_mcts_collect(game, my_val, phase="win"):
    """
    收集候選走法：
      當 phase=="win" 時，僅採用 for_win=True 條件收集即時獲勝候選；
      當 phase=="defense" 時，採用 for_win=False 條件收集防守候選，
      並在防守模式下利用 defense_mode=True，避免返回窗口極端空位（例如只返回內側兩點）。
    掃描水平、垂直、主對角線與反對角線方向，使用窗口長度 6 與 7 (margin 與 intermediate)。
    為避免同一威脅重複出現，利用集合去除重複候選後回傳列表。
    """
    board = game.board
    size = game.size
    opp_val = 3 - my_val
    candidates = []

    # 根據 phase 判斷是否啟用 defense_mode
    defense_mode = True if phase=="defense" else False

    def scan_direction(dr, dc, L, for_win=True):
        moves_found = []
        val_to_check = my_val if for_win else opp_val
        if dr==0 and dc!=0:
            # horizontal
            for r in range(size):
                for c in range(size-L+1):
                    window = board[r, c:c+L]
                    coords = [(r, c+i) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
                    
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L, defense_mode )
                    if candidate is not None:
                        moves_found.extend(candidate)
                    candidate = check_window_missing_one_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        
        elif dc==0 and dr!=0:
            # vertical
            for c in range(size):
                for r in range(size-L+1):
                    window = board[r:r+L, c]
                    coords = [(r+i, c) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L , defense_mode )
                    if candidate is not None:
                        moves_found.extend(candidate)
                    candidate = check_window_missing_one_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        
        elif dr==dc:
            # main diagonal
            for r in range(size-L+1):
                for c in range(size-L+1):
                    window = np.array([board[r+i, c+i] for i in range(L)])
                    coords = [(r+i, c+i) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L , defense_mode )
                    if candidate is not None:
                        moves_found.extend(candidate)
                    candidate = check_window_missing_one_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        
        elif dr==1 and dc==-1:
            # anti-diagonal
            for r in range(size-L+1):
                for c in range(L-1, size):
                    window = np.array([board[r+i, c-i] for i in range(L)])
                    coords = [(r+i, c-i) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L , defense_mode)
                    if candidate is not None:
                        moves_found.extend(candidate)
                    candidate = check_window_missing_one_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)

        return moves_found


    def scan_direction_margin(dr, dc, L, for_win=True):
        moves_found = []
        val_to_check = my_val if for_win else opp_val
        if dr==0 and dc!=0:  # horizontal
            for r in range(size):
                for c in range(size-L+1):
                    window = board[r, c:c+L]
                    coords = [(r, c+i) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        elif dc==0 and dr!=0:  # vertical
            for c in range(size):
                for r in range(size-L+1):
                    window = board[r:r+L, c]
                    coords = [(r+i, c) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        elif dr==dc:  # main diagonal
            for r in range(size-L+1):
                for c in range(size-L+1):
                    window = np.array([board[r+i, c+i] for i in range(L)])
                    coords = [(r+i, c+i) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        elif dr==1 and dc==-1:  # anti-diagonal
            for r in range(size-L+1):
                for c in range(L-1, size):
                    window = np.array([board[r+i, c-i] for i in range(L)])
                    coords = [(r+i, c-i) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        return moves_found

    def scan_direction_intermediate(dr, dc, L, for_win=True):
        moves_found = []
        val_to_check = my_val if for_win else opp_val
        if dr==0 and dc!=0:  # horizontal
            for r in range(size):
                for c in range(size-L+1):
                    window = board[r, c:c+L]
                    coords = [(r, c+i) for i in range(L)]
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        elif dc==0 and dr!=0:  # vertical
            for c in range(size):
                for r in range(size-L+1):
                    window = board[r:r+L, c]
                    coords = [(r+i, c) for i in range(L)]
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        elif dr==dc:  # main diagonal
            for r in range(size-L+1):
                for c in range(size-L+1):
                    window = np.array([board[r+i, c+i] for i in range(L)])
                    coords = [(r+i, c+i) for i in range(L)]
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        elif dr==1 and dc==-1:  # anti-diagonal
            for r in range(size-L+1):
                for c in range(L-1, size):
                    window = np.array([board[r+i, c-i] for i in range(L)])
                    coords = [(r+i, c-i) for i in range(L)]
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        return moves_found
    
    def scan_direction_missing_one(dr, dc, L, for_win=True):
        moves_found = []
        val_to_check = my_val if for_win else opp_val
        if dr==0 and dc!=0:
            # horizontal
            for r in range(size):
                for c in range(size-L+1):
                    window = board[r, c:c+L]
                    coords = [(r, c+i) for i in range(L)]
                    candidate = check_window_missing_one_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        elif dc==0 and dr!=0:
            # vertical
            for c in range(size):
                for r in range(size-L+1):
                    window = board[r:r+L, c]
                    coords = [(r+i, c) for i in range(L)]
                    candidate = check_window_missing_one_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        elif dr==dc:
            # main diagonal
            for r in range(size-L+1):
                for c in range(size-L+1):
                    window = np.array([board[r+i, c+i] for i in range(L)])
                    coords = [(r+i, c+i) for i in range(L)]
                    candidate = check_window_missing_one_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        elif dr==1 and dc==-1:
            # anti-diagonal
            for r in range(size-L+1):
                for c in range(L-1, size):
                    window = np.array([board[r+i, c-i] for i in range(L)])
                    coords = [(r+i, c-i) for i in range(L)]
                    candidate = check_window_missing_one_custom(window, coords, val_to_check, L )
                    if candidate is not None:
                        moves_found.extend(candidate)
        return moves_found
    

    
    for_win = True if phase=="win" else False
    # candidates.extend(scan_direction_margin(0, 1, 6, for_win=for_win))
    # candidates.extend(scan_direction_margin(0, 1, 7, for_win=for_win))
    # candidates.extend(scan_direction_margin(1, 0, 6, for_win=for_win))
    # candidates.extend(scan_direction_margin(1, 0, 7, for_win=for_win))
    # candidates.extend(scan_direction_margin(1, 1, 6, for_win=for_win))
    # candidates.extend(scan_direction_margin(1, 1, 7, for_win=for_win))
    # candidates.extend(scan_direction_margin(1, -1, 6, for_win=for_win))
    # candidates.extend(scan_direction_margin(1, -1, 7, for_win=for_win))
    # candidates.extend(scan_direction_intermediate(0, 1, 6, for_win=for_win))
    # candidates.extend(scan_direction_intermediate(0, 1, 7, for_win=for_win))
    # candidates.extend(scan_direction_intermediate(1, 0, 6, for_win=for_win))
    # candidates.extend(scan_direction_intermediate(1, 0, 7, for_win=for_win))
    # candidates.extend(scan_direction_intermediate(1, 1, 6, for_win=for_win))
    # candidates.extend(scan_direction_intermediate(1, 1, 7, for_win=for_win))
    # candidates.extend(scan_direction_intermediate(1, -1, 6, for_win=for_win))
    # candidates.extend(scan_direction_intermediate(1, -1, 7, for_win=for_win))
    # candidates.extend(scan_direction_missing_one(0, 1, 6, for_win=for_win))
    # candidates.extend(scan_direction_missing_one(0, 1, 7, for_win=for_win))
    # candidates.extend(scan_direction_missing_one(1, 0, 6, for_win=for_win))
    # candidates.extend(scan_direction_missing_one(1, 0, 7, for_win=for_win))
    # candidates.extend(scan_direction_missing_one(1, 1, 6, for_win=for_win))
    # candidates.extend(scan_direction_missing_one(1, 1, 7, for_win=for_win))
    # candidates.extend(scan_direction_missing_one(1, -1, 6, for_win=for_win))
    # candidates.extend(scan_direction_missing_one(1, -1, 7, for_win=for_win))

    candidates.extend(scan_direction(0, 1, 6, for_win=for_win))
    candidates.extend(scan_direction(0, 1, 7, for_win=for_win))
    candidates.extend(scan_direction(1, 0, 6, for_win=for_win))
    candidates.extend(scan_direction(1, 0, 7, for_win=for_win))
    candidates.extend(scan_direction(1, 1, 6, for_win=for_win))
    candidates.extend(scan_direction(1, 1, 7, for_win=for_win))
    candidates.extend(scan_direction(1, -1, 6, for_win=for_win))
    candidates.extend(scan_direction(1, -1, 7, for_win=for_win))


    # 每兩個候選點組合成一個走法
    moves = []
    candidates = list(set(candidates))  # 去除重覆
    logging.info(f'Final candidates: {candidates}')
    if len(candidates)>=2:
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                if candidates[i] and candidates[j]:
                    move = (candidates[i], candidates[j])
                    moves.append(move)
    elif len(candidates)==1:
        logging.info(f"Phase {phase}: Found only one candidate: {candidates[0]}")
        # 如果只有一個候選點，則將其與慢哈頓距離小於三的所有空點組合
        r = 2 
        neighbors = get_neighbors(game, candidates[0], r)
        while len(neighbors)<2:
            r+= 1
            neighbors = get_neighbors(game, candidates[0], r)
        for nb in neighbors:
            if nb in candidates:
                continue
            move = (candidates[0], nb)
            moves.append(move)
    
    unique_moves = list(set(moves))  # 去除重覆
    logging.info(f"Phase {phase}: Collected {len(unique_moves)}unique candidate(s):{unique_moves}")
    return unique_moves


# --- get_legal_moves ---
def get_legal_moves(game):
    """
    產生合法走法：
      - 若第一手則回傳全盤空點。
      - 否則僅考慮與現有棋子曼哈頓距離 R (初始 R=3) 以內的空點，
        若候選不足（少於兩個），則逐步擴大 R，直到至少有兩個候選點。
      - 返回兩子組合（順序無關）。
    """
    num_stones = 1 if np.count_nonzero(game.board)==0 else 2
    empty_positions = [(r, c) for r in range(game.size) for c in range(game.size) if game.board[r, c]==0]
    if num_stones==1:
        return empty_positions
    R = 2
    candidates = set()
    stones = [(r, c) for r in range(game.size) for c in range(game.size) if game.board[r, c]!=0]
    while True:
        candidates = set()
        for stone in stones:
            for nb in get_neighbors(game, stone, R):
                candidates.add(nb)
        if candidates and len(candidates)>=2:
            break
        R += 1
        if R>game.size:
            candidates = set(empty_positions)
            break
    candidates = list(candidates)
    moves = []
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            moves.append((candidates[i], candidates[j]))
    return moves

def default_policy(game, rollout_player):
    current_game = clone_game(game)
    logging.info("Default policy: Starting random playout.")
    while not terminal(current_game):
        legal_moves = get_legal_moves(current_game)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        move_str = move_to_str(move, current_game)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        color = 'B' if current_game.turn == 1 else 'W'
        current_game.play_move(color, move_str)
        sys.stdout = old_stdout
    winner = current_game.check_win()
    logging.info(f"Default policy: Playout finished with winner: {winner}")
    if winner == rollout_player:
        return 1
    elif winner == 0:
        return 0.5
    else:
        return 0

def backup(node, reward, rollout_player):
    while node is not None:
        node.visits += 1
        if node.player == rollout_player:
            node.wins += reward
        else:
            node.wins += (1 - reward)
        node = node.parent

def mcts(root, itermax=500):
    logging.info(f"MCTS: Starting search for {itermax} iterations.")
    for i in range(itermax):
        node = tree_policy(root)
        if i % 50 == 0:
            logging.debug(f"MCTS iteration {i}: reached a node with {node.visits} visits.")
        rollout_reward = default_policy(node.game, root.player)
        backup(node, rollout_reward, root.player)
    best = max(root.children, key=lambda c: c.visits)
    logging.info(f"MCTS: Search finished. Best move selected with {best.visits} visits.")
    return best.move

class MCTSNode:
    def __init__(self, game, move=None, parent=None, restricted_actions=None):
        self.game = game              # Copy of current game state.
        self.move = move              # Move leading to this node.
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        if restricted_actions is not None:
            self.untried_moves = restricted_actions.copy()
        else:
            # 若盤面全空，使用單點；否則使用 get_action_space 生成兩子組合 + threat 候選
            if np.count_nonzero(game.board) == 0:
                self.untried_moves = [(r, c) for r in range(game.size) for c in range(game.size) if game.board[r,c] == 0]
            else:
                self.untried_moves = get_action_space(game, game.turn)
        self.player = game.turn       # The player to move at this node (1: Black, 2: White)

# 新的動作空間函式：若盤面全空，返回所有單點；否則從 Manhattan 距離 3 以內所有空點生成兩子組合，再加入 threat 候選。
def get_action_space(game, player):
    if np.count_nonzero(game.board) == 0:
        return [(r, c) for r in range(game.size) for c in range(game.size) if game.board[r,c] == 0]
    stones = [(r, c) for r in range(game.size) for c in range(game.size) if game.board[r,c] != 0]
    candidates = set()
    for stone in stones:
        for nb in get_neighbors(game, stone, 3):
            candidates.add(nb)
    candidates = list(candidates)
    moves = []
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            moves.append((candidates[i], candidates[j]))
    threat = pre_mcts_collect(game, player, phase="defense")
    if threat is not None:
        moves.extend(threat)
        logging.info(f"Threat candidates: {threat} added :{moves}")
    return list(set(moves))

def get_neighbors(game, stone, distance):
    r0, c0 = stone
    rows = np.arange(game.size)[:, None]
    cols = np.arange(game.size)
    dists = np.abs(rows - r0) + np.abs(cols - c0)
    mask = (game.board == 0) & (dists <= distance)
    indices = np.argwhere(mask)
    return [tuple(idx) for idx in indices]

def get_defensive_moves(game, attacker_move, defense_type):
    if defense_type == "conservative":
        candidate_set = set()
        if isinstance(attacker_move[0], int):
            candidate_set.update(get_neighbors(game, attacker_move, 2))
        else:
            for stone in attacker_move:
                candidate_set.update(get_neighbors(game, stone, 2))
        candidate_list = list(candidate_set)
        moves = []
        if len(candidate_list) < 2:
            return []
        for i in range(len(candidate_list)):
            for j in range(i+1, len(candidate_list)):
                moves.append((candidate_list[i], candidate_list[j]))
        return moves
    else:
        return get_action_space(game, game.turn)

# =============================================================================
# 3. Main Program Entry
# =============================================================================
def main():
    game = Connect6Game()
    game.run()
    logging.info("Game terminated.")
    logging.shutdown()

if __name__=="__main__":
    main()
