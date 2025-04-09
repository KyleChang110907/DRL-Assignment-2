import sys
import numpy as np
import random
import os
import torch

# You must have your RL modules (Connect6, Connect6Net, MCTS, etc.) available.
# For this example, we assume they are imported or defined in the same project.
# (See previous code for their definitions.)

# -------------------------------
# Helper Functions for AlphaZero Move Generation
# -------------------------------

# Global variables to hold the trained agent and its MCTS object.
ALPHAZERO_AGENT = None
ALPHAZERO_MCTS = None
# Path to the trained network checkpoint.
CHECKPOINT_PATH = './ModulesConnect6/checkpoints/connect6_net_alphaZero.pth'
# (Adjust CHECKPOINT_PATH as needed.)

def load_alphazero_agent():
    """Loads the trained AlphaZero agent if not already loaded."""
    global ALPHAZERO_AGENT, ALPHAZERO_MCTS
    if ALPHAZERO_AGENT is None:
        from ModulesConnect6.training.AlphaZero import Connect6Net, MCTS, HISTORY_LENGTH, DEVICE, NUM_SIMULATIONS
        ALPHAZERO_AGENT = Connect6Net(history_length=HISTORY_LENGTH).to(DEVICE)
        ALPHAZERO_AGENT.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        )
        ALPHAZERO_AGENT.eval()
        ALPHAZERO_MCTS = MCTS(ALPHAZERO_AGENT, num_simulations=NUM_SIMULATIONS)

def convert_game_to_rl(game):
    """
    Converts the Connect6Game board (with values 0,1,2) to a Connect6 board (with values 0,1,-1)
    expected by the RL agent.
    When game.turn == 1 (Black's turn), cells with 1 become 1 and cells with 2 become -1.
    When game.turn == 2 (White's turn), cells with 2 become 1 and cells with 1 become -1.
    """
    # Import the RL Connect6 class from your RL module.
    from ModulesConnect6.training.AlphaZero import Connect6, HISTORY_LENGTH
    rl_game = Connect6(history_length=HISTORY_LENGTH)
    for r in range(game.size):
        for c in range(game.size):
            if game.board[r, c] != 0:
                if game.turn == 1:
                    rl_game.board[r, c] = 1 if game.board[r, c] == 1 else -1
                else:
                    rl_game.board[r, c] = 1 if game.board[r, c] == 2 else -1
    # Set move_count: if no moves played, set 0; otherwise, 1 (a simple heuristic).
    rl_game.move_count = 0 if np.count_nonzero(game.board) == 0 else 1
    # Set current_player to 1 (the RL agent always sees itself as '1').
    rl_game.current_player = 1
    # For simplicity, fill the history with copies of the current board.
    for i in range(rl_game.history_length):
        rl_game.history[i] = rl_game.board.copy()
    return rl_game

def alphazero_generate_move(game):
    """
    Uses the trained AlphaZero agent to generate a move.
    It converts the current Connect6Game board into the RL environment's representation,
    runs MCTS to select a move (using greedy selection), and converts the move into
    the GTP move string.
    """
    # Ensure the agent is loaded.
    load_alphazero_agent()
    # Convert the game board.
    rl_game = convert_game_to_rl(game)
    # Run MCTS using the globally loaded agent.
    root = ALPHAZERO_MCTS.run(rl_game)
    move, _ = ALPHAZERO_MCTS.select_action(root, temperature=0)
    # Convert the move into a move string using the Connect6Game's index_to_label.
    if rl_game.move_count == 0:
        # For first move, move is a tuple (i, j).
        i, j = move
        move_str = f"{game.index_to_label(j)}{i+1}"
    else:
        # For subsequent moves, move is a pair of tuples.
        pos1, pos2 = move
        move_str = f"{game.index_to_label(pos1[1])}{pos1[0]+1},{game.index_to_label(pos2[1])}{pos2[0]+1}"
    return move_str

# -------------------------------
# Connect6Game Class (Interactive GTP)
# -------------------------------
class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
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
        """Checks if a player has won.
        Returns:
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
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
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

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a move using the trained AlphaZero agent."""
        move_str = alphazero_generate_move(self)
        self.play_move(color, move_str)
        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)

    def show_board(self):
        """Displays the board as text."""
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
        """Main loop that reads GTP commands from standard input."""
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

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
