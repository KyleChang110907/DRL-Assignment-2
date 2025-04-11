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
log_folder = "./logs/MCTS_ITSS/"
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
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r-dr, c-dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c]==current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc]==current_color:
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
        # Update turn; in Connect6 (except initial move) each turn places 2 stones.
        self.turn = 3 - self.turn
        print("= ", end='', flush=True)

    def generate_move(self, color):
        """
        Before entering MCTS, perform pre-MCTS check for immediate win or defense.
        If found, execute that move pair directly; otherwise, enter MCTS search.
        """
        if self.game_over:
            print("? Game over")
            return
        my_val = 1 if color.upper()=='B' else 2
        logging.info("Starting pre-MCTS check.")
        check_result = pre_mcts_check(self, my_val)
        if check_result is not None:
            label, chosen_move_pair = check_result
            if label=="win":
                logging.info(f"Pre-MCTS: Immediate win move pair found: {chosen_move_pair}")
                move_str = move_to_str(chosen_move_pair, self)
                self.play_move(color, move_str)
                print(f"{move_str}\n\n", end='', flush=True)
                print(move_str, file=sys.stderr)
                return
            elif label=="defense":
                if len(chosen_move_pair) == 1:
                    logging.info(f"Pre-MCTS: Immediate defense move found: {chosen_move_pair}")
                    logging.info(f"Pre-MCTS ONLY ONE MOVE")
                    defense_move = chosen_move_pair[0]
                    logging.info(f"Pre-MCTS: Immediate defense move found: {defense_move}")
                    move_str_defense = move_to_str(defense_move, self)
                    # 模擬下完防守棋後，再用 MCTS 為剩餘的空位找一子
                    new_game = clone_game(self)
                    new_game.play_move(color, move_str_defense)
                    root = MCTSNode(new_game)
                    logging.info("MCTS (for second stone defense): Starting search for one stone.")
                    best_move = mcts(root, itermax=5)
                    # best_move 可能為兩子下法，僅取其中一子
                    if isinstance(best_move[0], int):
                        second_move = best_move
                    else:
                        second_move = best_move[0]
                    full_move = (defense_move, second_move)
                    move_str = move_to_str(full_move, self)
                    self.play_move(color, move_str)
                    logging.info(f"Pre-MCTS defense move completed: {move_str}")
                    print(f"{move_str}\n\n", end='', flush=True)
                    print(move_str, file=sys.stderr)
                    return
                else:
                    logging.info(f"Pre-MCTS: Immediate defense move pair found: {chosen_move_pair}")
                    move_str = move_to_str(chosen_move_pair, self)
                    self.play_move(color, move_str)
                    print(f"{move_str}\n\n", end='', flush=True)
                    print(move_str, file=sys.stderr)
                    return
        logging.info("Pre-MCTS check did not trigger; entering MCTS search.")
        root = MCTSNode(clone_game(self))
        logging.info("MCTS: Starting search.")
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
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col]==1 else "O" if self.board[row, col]==2 else "." for col in range(self.size))
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
        if command=="get_conf_str env_board_size:":
            return "env_board_size=19"
        if not command:
            return
        parts = command.split()
        cmd = parts[0].lower()
        if cmd=="boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd=="clear_board":
            self.reset_board()
        elif cmd=="play":
            if len(parts)<3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd=="genmove":
            if len(parts)<2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd=="showboard":
            self.show_board()
        elif cmd=="list_commands":
            self.list_commands()
        elif cmd=="quit":
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
# 2. MCTS and Quick Pre-MCTS Threat/Win Check Functions and Classes
# =============================================================================
def get_legal_moves(game):
    """
    根據遊戲狀態產生合法走法：
      - 第一手只下 1 子
      - 之後每回合必下 2 子
    為減少候選走法，非第一手時先僅考慮與現有棋子鄰近（曼哈頓距離 2 以內）的空格，
    再將候選點兩兩組合。
    """
    num_stones = 1 if np.count_nonzero(game.board) == 0 else 2
    empty_positions = [(r, c) for r in range(game.size) for c in range(game.size) if game.board[r, c] == 0]
    if num_stones == 1:
        return empty_positions  # 回傳單一點的列表
    else:
        candidates = set()
        for r in range(game.size):
            for c in range(game.size):
                if game.board[r, c] != 0:
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < game.size and 0 <= nc < game.size and game.board[nr, nc] == 0:
                                candidates.add((nr, nc))
        if not candidates:
            candidates = set(empty_positions)
        candidates = list(candidates)
        moves = []
        # 生成所有候選點的兩兩組合（順序無關）
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                moves.append((candidates[i], candidates[j]))
        return moves

def move_to_str(move, game):
    """
    Converts a move into a string:
      - Single-point (r, c) => "A1"
      - Two-stone move ((r1,c1),(r2,c2)) => "A1,B2"
    """
    if isinstance(move[0], int):
        r, c = move
        return f"{game.index_to_label(c)}{r+1}"
    else:
        return ",".join(f"{game.index_to_label(c)}{r+1}" for r, c in move)

def clone_game(game):
    """Deep-copies the game state."""
    return copy.deepcopy(game)

def terminal(game):
    """Checks whether the game state is terminal."""
    if game.check_win() != 0:
        return True
    if np.all(game.board!=0):
        return True
    return False

def best_child(node, c_param=1.41):
    """Selects the best child based on UCB1."""
    choices_weights = [
        (child.wins/child.visits) + c_param * math.sqrt((2 * math.log(node.visits))/child.visits)
        for child in node.children if child.visits>0
    ]
    unvisited = [child for child in node.children if child.visits==0]
    if unvisited:
        logging.debug("Best_child: Found unvisited child; choosing randomly.")
        return random.choice(unvisited)
    best = node.children[np.argmax(choices_weights)]
    logging.debug("Best_child: Chosen child with highest UCB1 weight.")
    return best

def tree_policy(node):
    """
    Extends the tree from the current node:
      - Expand untried moves if available;
      - Otherwise, select best child per UCB1.
    """
    iteration = 0
    while not terminal(node.game):
        iteration += 1
        if node.untried_moves:
            move = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
            logging.debug(f"Tree_policy iteration {iteration}: Expanding move {move}")
            new_game = clone_game(node.game)
            move_str = move_to_str(move, new_game)
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            color = 'B' if new_game.turn==1 else 'W'
            new_game.play_move(color, move_str)
            sys.stdout = old_stdout
            child_node = MCTSNode(new_game, move, node)
            node.children.append(child_node)
            return child_node
        else:
            node = best_child(node)
    logging.debug("Tree_policy: Terminal node reached.")
    return node

def get_neighbors(game, stone, distance):
    """
    Returns neighboring empty positions within the given Manhattan distance,
    using vectorized NumPy operations.
    """
    r0, c0 = stone
    rows = np.arange(game.size)[:, None]
    cols = np.arange(game.size)
    distances = np.abs(rows - r0) + np.abs(cols - c0)
    mask = (game.board==0) & (distances<=distance)
    indices = np.argwhere(mask)
    return [tuple(idx) for idx in indices]

def get_defensive_moves(game, attacker_move, defense_type):
    """
    Generates defensive candidate moves based on defense_type:
      - "conservative": only consider empty positions near the attacker's stones (Manhattan distance ≤2)
        and form two-stone moves.
      - Otherwise, returns all legal moves.
    """
    if defense_type=="conservative":
        candidate_set = set()
        if isinstance(attacker_move[0], int):
            candidate_set.update(get_neighbors(game, attacker_move, 2))
        else:
            for stone in attacker_move:
                candidate_set.update(get_neighbors(game, stone, 2))
        candidate_list = list(candidate_set)
        moves = []
        if len(candidate_list)<2:
            return []
        n = len(candidate_list)
        for i in range(n):
            for j in range(i+1, n):
                moves.append((candidate_list[i], candidate_list[j]))
        return moves
    else:
        return get_legal_moves(game)

# -------------------------------------------------
# 新增：快速威脅/獲勝檢查 (Pre-MCTS Check)
# 觸發條件：判斷兩種情況：
#  (1) 六子窗口：如果窗口長度為6，且該窗口沒有對手棋，
#      且窗口中己方棋的數量等於 4（其餘 2 格皆空），則返回 (board_coords[0], board_coords[5]).
#  (2) 七子窗口：如果窗口長度為7，且該窗口沒有對手棋，
#      且窗口中己方棋的數量等於 5（其餘 2 格皆空），則返回 (board_coords[0], board_coords[6]).
# ---------------------------------------------------------------------------------------------

def check_window_margin_custom(window, board_coords, my_val, L):
    """
    For a window of fixed length L, if the window contains only 0 or my_val,
    and exactly L-2 cells are my_val (i.e. the other 2 cells are empty),
    then return (board_coords[0], board_coords[-1]).
    Otherwise, return None.
    """
    if len(window) != L:
        return None
    # 檢查窗口是否純淨：只允許 0 或 my_val
    for cell in window:
        if cell not in (0, my_val):
            return None
    count_my = np.count_nonzero(window==my_val)
    if count_my != (L-2):
        return None
    # 檢查兩端是否皆為空
    if window[0] == 0 and window[-1] == 0:
    
        logging.debug(f"check_window_margin_custom: For L={L}, window {window} with coords {board_coords} triggers candidate.")
        return (board_coords[0], board_coords[-1])
    
    else:
            return None
    
def check_window_intermediate_custom(window, board_coords, my_val, L, for_win=True):
    if len(window) != L:
        return None
    # 檢查窗口是否純淨：只允許 0 或 my_val
    for cell in window:
        if cell not in (0, my_val):
            return None
    count_my = np.count_nonzero(window==my_val)
    if count_my != (L-2):
        return None
    # 檢查兩端是否皆為空
    if window[0] == 0 and window[-1] == 0:
        logging.debug(f"The consecutive condition should be found in the former check_window_margin_custom.")
        return None
    
    else:
        # find the two empty cells in the window
        logging.debug(f'board coords: {board_coords}')
        logging.debug(f'board values{[window[i] for i in range(L)]}')
        empty_cells = [board_coords[i] for i in range(L) if window[i] == 0]
        logging.debug(f"check_window_intermediate_custom: empty_cells={empty_cells}")
        # 如果empty cell的index為連續的兩個，且這兩個是最後或最先，則return None (忽略，因為等等還會看到)
        # 先得到empty cells 在window中的index
        empty_cells_index = [i for i in range(L) if window[i] == 0] 
        if len(empty_cells) == 2:
            if (empty_cells_index[0] + 1 == empty_cells_index[1]) and (empty_cells_index[0] == 0 or empty_cells_index[1] == L-1) and not for_win:
                if empty_cells_index[0] == 0:
                    logging.debug(f"check_window_intermediate_custom: For L={L}, window {window} with coords {board_coords} triggers candidate.")
                    return (empty_cells[1])
                else:
                    logging.debug(f"check_window_intermediate_custom: For L={L}, window {window} with coords {board_coords} triggers candidate.")
                    return (empty_cells[0])
            logging.debug(f"check_window_intermediate_custom: For L={L}, window {window} with coords {board_coords} triggers candidate.")
            return (empty_cells[0], empty_cells[1])
        else:
            return None


def pre_mcts_check(game, my_val):
    """
    Pre-MCTS check:
      Scan contiguous windows in horizontal, vertical, main diagonal, and anti-diagonal directions.
      先嘗試以窗口長度7檢查（對應連續五子條件），若滿足則返回 ("win", move_pair) 或 ("defense", move_pair)；
      若未找到，再嘗試以窗口長度6檢查（對應連續四子條件）。
    """
    board = game.board
    size = game.size
    opp_val = 3 - my_val
    logging.debug("Pre-MCTS check start.")
    
    def scan_direction_margin(dr, dc, L, for_win=True):
        moves_found = []
        val_to_check = my_val if for_win else opp_val
        # 先 margin check
        if dr==0 and dc!=0:  # horizontal
            for r in range(size):
                for c in range(size - L + 1):
                    window = board[r, c:c+L]
                    coords = [(r, c+i) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L)
                    if candidate is not None:
                        logging.debug(f"Scan ({dr},{dc}) at ({r},{c}) with L={L}: window {window} -> candidate pair {candidate}")
                        moves_found.append(candidate)
        elif dc==0 and dr!=0:  # vertical
            for c in range(size):
                for r in range(size - L + 1):
                    window = board[r:r+L, c]
                    coords = [(r+i, c) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L)
                    if candidate is not None:
                        logging.debug(f"Scan ({dr},{dc}) at ({r},{c}) with L={L}: window {window} -> candidate pair {candidate}")
                        moves_found.append(candidate)
        elif dr==dc:  # main diagonal
            for r in range(size - L + 1):
                for c in range(size - L + 1):
                    window = np.array([board[r+i, c+i] for i in range(L)])
                    coords = [(r+i, c+i) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L)
                    if candidate is not None:
                        logging.debug(f"Scan diag at ({r},{c}) with L={L}: window {window} -> candidate pair {candidate}")
                        moves_found.append(candidate)
        elif dr==1 and dc==-1:  # anti-diagonal
            for r in range(size - L + 1):
                for c in range(L-1, size):
                    window = np.array([board[r+i, c-i] for i in range(L)])
                    coords = [(r+i, c-i) for i in range(L)]
                    candidate = check_window_margin_custom(window, coords, val_to_check, L)
                    if candidate is not None:
                        logging.debug(f"Scan anti-diag at ({r},{c}) with L={L}: window {window} -> candidate pair {candidate}")
                        moves_found.append(candidate)
        return moves_found
    
    def scan_direction_intermediate(dr, dc, L, for_win=True):
        moves_found = []
        val_to_check = my_val if for_win else opp_val
        # 後intermediate check
        if dr==0 and dc!=0:  # horizontal
            for r in range(size):
                for c in range(size - L + 1):
                    window = board[r, c:c+L]
                    coords = [(r, c+i) for i in range(L)]
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L, for_win)
                    if candidate is not None:
                        logging.debug(f"Scan ({dr},{dc}) at ({r},{c}) with L={L}: window {window} -> candidate pair {candidate}")
                        moves_found.append(candidate)
        elif dc==0 and dr!=0:  # vertical
            for c in range(size):
                for r in range(size - L + 1):
                    window = board[r:r+L, c]
                    coords = [(r+i, c) for i in range(L)]
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L, for_win)
                    if candidate is not None:
                        logging.debug(f"Scan ({dr},{dc}) at ({r},{c}) with L={L}: window {window} -> candidate pair {candidate}")
                        moves_found.append(candidate)
        elif dr==dc:  # main diagonal
            for r in range(size - L + 1):
                for c in range(size - L + 1):
                    window = np.array([board[r+i, c+i] for i in range(L)])
                    coords = [(r+i, c+i) for i in range(L)]
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L, for_win)
                    if candidate is not None:
                        logging.debug(f"Scan diag at ({r},{c}) with L={L}: window {window} -> candidate pair {candidate}")
                        moves_found.append(candidate)
        elif dr==1 and dc==-1:  # anti-diagonal
            for r in range(size - L + 1):
                for c in range(L-1, size):
                    window = np.array([board[r+i, c-i] for i in range(L)])
                    coords = [(r+i, c-i) for i in range(L)]
                    candidate = check_window_intermediate_custom(window, coords, val_to_check, L, for_win)
                    if candidate is not None:
                        logging.debug(f"Scan anti-diag at ({r},{c}) with L={L}: window {window} -> candidate pair {candidate}")
                        moves_found.append(candidate)
        return moves_found

    
    # 先檢查己方勝利：先以 L=6 (連續四子條件)
    for direction in [(0,1), (1,0), (1,1), (1,-1)]:
        moves = scan_direction_margin(*direction, L=6, for_win=True)
        if moves:
            logging.info(f"Pre-MCTS win check (L=6) triggered: {moves[0]} in direction {direction}")
            return ("win", moves[0])
    # 若無，再以 L=7 (連續五子條件)
    for direction in [(0,1), (1,0), (1,1), (1,-1)]:
        moves = scan_direction_margin(*direction, L=7, for_win=True)
        if moves:
            logging.info(f"Pre-MCTS win check (L=7) triggered: {moves[0]} in direction {direction}")
            return ("win", moves[0])
    # 若無，再以 L=6 (六子中有四子) 檢查對手棋
    for direction in [(0,1), (1,0), (1,1), (1,-1)]:
        moves = scan_direction_intermediate(*direction, L=6, for_win=True)
        if moves:
            logging.info(f"Pre-MCTS win check (L=6) triggered: {moves[0]} in direction {direction}")
            return ("win", moves[0])
    # 若無，再以 L=7 (七子中有五子) 檢查對手棋
    for direction in [(0,1), (1,0), (1,1), (1,-1)]:
        moves = scan_direction_intermediate(*direction, L=7, for_win=True)
        if moves:
            logging.info(f"Pre-MCTS win check (L=7) triggered: {moves[0]} in direction {direction}")
            return ("win", moves[0])
        
    # 檢查對手威脅：同樣先以 L=6，再以 L=7，檢查對手棋
    for direction in [(0,1), (1,0), (1,1), (1,-1)]:
        moves = scan_direction_margin(*direction, L=6, for_win=False)
        if moves:
            logging.info(f"Pre-MCTS defense check (L=6) triggered: {moves[0]} in direction {direction}")
            return ("defense", moves[0])
    for direction in [(0,1), (1,0), (1,1), (1,-1)]:
        moves = scan_direction_margin(*direction, L=7, for_win=False)
        if moves:
            logging.info(f"Pre-MCTS defense check (L=7) triggered: {moves[0]} in direction {direction}")
            return ("defense", moves[0])
    for direction in [(0,1), (1,0), (1,1), (1,-1)]:
        moves = scan_direction_intermediate(*direction, L=6, for_win=False)
        if moves:
            logging.info(f"Pre-MCTS defense check (L=6) triggered: {moves[0]} in direction {direction}")
            return ("defense", moves[0])
    for direction in [(0,1), (1,0), (1,1), (1,-1)]:
        moves = scan_direction_intermediate(*direction, L=7, for_win=False)
        if moves:
            logging.info(f"Pre-MCTS defense check (L=7) triggered: {moves[0]} in direction {direction}")
            return ("defense", moves[0])
    
    logging.info("Pre-MCTS check found no immediate win or defense move.")
    return None

def default_policy(game, rollout_player):
    """
    Playout strategy:
      1. Before random playout, perform pre-MCTS check.
         If immediate win is detected, return 1; if immediate threat detected, return 0.
      2. Otherwise, perform random playout until terminal state and return:
         1 (win), 0.5 (draw), or 0 (loss).
    """
    current_game = clone_game(game)
    # logging.info("Default policy: Starting pre-MCTS check.")
    # check_result = pre_mcts_check(current_game, rollout_player)
    # if check_result is not None:
    #     label, _ = check_result
    #     if label=="win":
    #         logging.info("Default policy: Immediate win detected.")
    #         return 1
    #     elif label=="defense":
    #         logging.info("Default policy: Immediate threat detected.")
    #         return 0
    logging.info("Default policy: Pre-MCTS check passed; starting random playout.")
    while not terminal(current_game):
        legal_moves = get_legal_moves(current_game)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        move_str = move_to_str(move, current_game)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        color = 'B' if current_game.turn==1 else 'W'
        current_game.play_move(color, move_str)
        sys.stdout = old_stdout
    winner = current_game.check_win()
    logging.info(f"Default policy: Playout finished with winner: {winner}")
    if winner==rollout_player:
        return 1
    elif winner==0:
        return 0.5
    else:
        return 0

def backup(node, reward, rollout_player):
    """Backpropagates simulation results, updating node statistics."""
    while node is not None:
        node.visits += 1
        if node.player == rollout_player:
            node.wins += reward
        else:
            node.wins += (1 - reward)
        node = node.parent

def mcts(root, itermax=500):
    """Main MCTS search process."""
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
    def __init__(self, game, move=None, parent=None):
        self.game = game              # Copy of current game state.
        self.move = move              # Move leading to this node (single point or two-stone combination).
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = get_legal_moves(game)
        self.player = game.turn       # The player to move at this node (1: Black, 2: White)

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
