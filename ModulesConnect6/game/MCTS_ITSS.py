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

# Configure the logging module.
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
          color -- 'B' or 'W'
          move  -- move string (e.g. "A1" or "A1,B2")
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
            self.board[row, col] = 1 if color.upper() == 'B' else 2
        # Update turn: except for the first move, each turn plays two stones.
        self.turn = 3 - self.turn
        print("= ", end='', flush=True)

    def generate_move(self, color):
        """
        Uses MCTS combined with ITSS search to decide a move:
          1. Create an MCTS root node (copy of current state).
          2. Run MCTS search (simulation strategy incorporates ITSS).
          3. Play the move and output it.
        """
        if self.game_over:
            print("? Game over")
            return
        root = MCTSNode(clone_game(self))
        best_move = mcts(root, itermax=50)
        move_str = move_to_str(best_move, self)
        self.play_move(color, move_str)
        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)

    def show_board(self):
        """Prints the board state as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
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
        """Main loop: reads GTP commands from standard input."""
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
# 2. MCTS and ITSS Functions and Classes (with Vectorized Operations)
# =============================================================================

def get_legal_moves(game):
    """
    Generates legal moves:
      - First move: one stone; subsequent moves: two stones.
      - For non-first moves, only consider empty positions near existing stones (Manhattan distance ≤2),
        then form two-stone combinations.
    Uses vectorized operations to speed up filtering.
    """
    num_stones = 1 if np.count_nonzero(game.board) == 0 else 2
    if num_stones == 1:
        empties = np.argwhere(game.board == 0)
        return [tuple(pos) for pos in empties]
    else:
        # Get positions of all stones.
        stones = np.argwhere(game.board != 0)
        candidates = set()
        # For each stone, use vectorized window extraction.
        for s in stones:
            r0, c0 = s
            rmin = max(0, r0 - 2)
            rmax = min(game.size, r0 + 3)
            cmin = max(0, c0 - 2)
            cmax = min(game.size, c0 + 3)
            sub_board = game.board[rmin:rmax, cmin:cmax]
            # Find indices where sub_board==0 using np.argwhere.
            empties = np.argwhere(sub_board == 0)
            for pos in empties:
                candidates.add((rmin + pos[0], cmin + pos[1]))
        if not candidates:
            empties = np.argwhere(game.board == 0)
            candidates = {tuple(x) for x in empties}
        candidates = list(candidates)
        n = len(candidates)
        # To form combinations, we use a double loop (candidates count is often moderate).
        moves = []
        for i in range(n):
            for j in range(i+1, n):
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
    if np.all(game.board != 0):
        return True
    return False

def best_child(node, c_param=1.41):
    """Selects the best child based on UCB1."""
    choices_weights = [
        (child.wins / child.visits) + c_param * math.sqrt((2 * math.log(node.visits)) / child.visits)
        for child in node.children if child.visits > 0
    ]
    unvisited = [child for child in node.children if child.visits == 0]
    if unvisited:
        return random.choice(unvisited)
    return node.children[np.argmax(choices_weights)]

def tree_policy(node):
    """
    Extends the tree from the current node:
      - If there are untried moves, expand one at random.
      - Otherwise, select the best child according to UCB1.
    """
    while not terminal(node.game):
        if node.untried_moves:
            move = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
            new_game = clone_game(node.game)
            move_str = move_to_str(move, new_game)
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            color = 'B' if new_game.turn == 1 else 'W'
            new_game.play_move(color, move_str)
            sys.stdout = old_stdout
            child_node = MCTSNode(new_game, move, node)
            node.children.append(child_node)
            return child_node
        else:
            node = best_child(node)
    return node

def get_neighbors(game, stone, distance):
    """
    Returns a list of neighboring empty positions within the given Manhattan distance,
    using vectorized NumPy operations.
    """
    r0, c0 = stone
    # Create indices for the full board.
    rows = np.arange(game.size)
    cols = np.arange(game.size)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    # Compute Manhattan distances.
    distances = np.abs(rr - r0) + np.abs(cc - c0)
    # Mask for empty positions within given distance.
    mask = (game.board == 0) & (distances <= distance)
    indices = np.argwhere(mask)
    return [tuple(idx) for idx in indices]

def get_defensive_moves(game, attacker_move, defense_type):
    """
    Generates defensive candidate moves based on defense_type:
      - "conservative": only consider empty positions near the attacker's stones (Manhattan distance ≤2)
        and form two-stone moves.
      - Otherwise, return all legal moves.
    """
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
        n = len(candidate_list)
        for i in range(n):
            for j in range(i+1, n):
                moves.append((candidate_list[i], candidate_list[j]))
        return moves
    else:
        return get_legal_moves(game)

def ctss_check(game, move, attacker, defense_type):
    """
    CTSS check:
      1. Simulate the attacker's move.
      2. Generate defensive candidate moves (depending on defense_type).
      3. For each defensive move, simulate and check whether the attacker has an immediate winning response.
         If for every defensive move the attacker can win immediately, then the move is considered a CTSS solution.
    """
    new_game = clone_game(game)
    attacker_color = 'B' if attacker == 1 else 'W'
    new_game.play_move(attacker_color, move_to_str(move, new_game))
    defensive_moves = get_defensive_moves(new_game, move, defense_type)
    logging.debug(f"CTSS check for move {move}: {len(defensive_moves)} defensive moves")
    if not defensive_moves:
        return (new_game.check_win() == attacker)
    for i, def_move in enumerate(defensive_moves):
        sim_game = clone_game(new_game)
        defender_color = 'W' if attacker == 1 else 'B'
        sim_game.play_move(defender_color, move_to_str(def_move, sim_game))
        attacker_moves = get_legal_moves(sim_game)
        logging.debug(f"  Defense #{i}: {def_move} yields {len(attacker_moves)} attacker moves")
        winning_found = False
        for j, att_resp in enumerate(attacker_moves):
            logging.debug(f"    Attacker response #{j}: {att_resp}")
            sim_resp = clone_game(sim_game)
            sim_resp.play_move(attacker_color, move_to_str(att_resp, sim_resp))
            if sim_resp.check_win() == attacker:
                winning_found = True
                break
        if not winning_found:
            return False
    return True

def itss_search(game, attacker):
    """
    ITSS search:
      Iterates through candidate double-threat moves.
      Iteration 1 uses conservative defense.
      Iteration 2 uses normal defense.
      If any candidate move passes the CTSS check under the respective defense mode, return that move.
      Otherwise, return None.
    """
    max_iterations = 1  # Adjust as needed. better 2 
    for iteration in range(1, max_iterations + 1):
        defense_type = "conservative" if iteration == 1 else "normal"
        logging.debug(f"ITSS iteration {iteration} using defense type: {defense_type}")
        candidate_moves = get_legal_moves(game)  # For non-first moves: two-stone moves.
        for move in candidate_moves:
            if ctss_check(game, move, attacker, defense_type):
                logging.info(f"ITSS found a solution: {move} using defense {defense_type}")
                return move
    return None

def default_policy(game, rollout_player):
    """
    Simulation (playout) strategy:
      1. First, try ITSS search (integrated with CTSS) to detect an immediate winning move.
         If found, return a reward of 1.
      2. Otherwise, perform a random playout until terminal state and return reward:
           1 (win), 0.5 (draw), or 0 (loss).
    """
    current_game = clone_game(game)
    logging.info("ITSS search in default policy...")
    itss_solution = itss_search(current_game, rollout_player)
    if itss_solution is not None:
        logging.info("ITSS found an immediate winning move.")
        return 1
    logging.info("No immediate win found, performing random playout...")
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
    if winner == rollout_player:
        return 1
    elif winner == 0:
        return 0.5
    else:
        return 0

def backup(node, reward, rollout_player):
    """Backpropagates the simulation result, updating node statistics."""
    while node is not None:
        node.visits += 1
        if node.player == rollout_player:
            node.wins += reward
        else:
            node.wins += (1 - reward)
        node = node.parent

def mcts(root, itermax=500):
    """Main MCTS search process (simulation uses ITSS)."""
    for _ in range(itermax):
        node = tree_policy(root)
        rollout_reward = default_policy(node.game, root.player)
        backup(node, rollout_reward, root.player)
    best = max(root.children, key=lambda c: c.visits)
    return best.move

class MCTSNode:
    def __init__(self, game, move=None, parent=None):
        self.game = game              # Copy of the current game state.
        self.move = move              # The move that leads to this node (single point or two-stone combination).
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

if __name__ == "__main__":
    main()
