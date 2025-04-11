#!/usr/bin/env python3
import sys
import numpy as np
import random
import copy
import io

# ----------------------------
# MCTS 相關輔助函式與類別
# ----------------------------

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
    將走法轉換為輸出格式：
      - 若 move 為單一點 (r, c)：轉換成 "A1" 格式
      - 若 move 為兩子 ((r1,c1), (r2,c2))：以逗號分隔輸出
    """
    if isinstance(move[0], int):
        r, c = move
        return f"{game.index_to_label(c)}{r+1}"
    else:
        return ",".join(f"{game.index_to_label(c)}{r+1}" for r, c in move)

def clone_game(game):
    """利用 deep copy 複製遊戲狀態"""
    return copy.deepcopy(game)

def terminal(game):
    """判斷目前遊戲是否已結束"""
    if game.check_win() != 0:
        return True
    if np.all(game.board != 0):
        return True
    return False

def best_child(node, c_param=1.41):
    """依 UCB1 選出最佳子節點"""
    choices_weights = [
        (child.wins / child.visits) + c_param * np.sqrt((2 * np.log(node.visits)) / child.visits)
        for child in node.children
    ]
    return node.children[np.argmax(choices_weights)]

def tree_policy(node):
    """
    搜索策略：當前節點有未展開的走法時隨機選一個展開，
    否則依 UCB1 原則選出最佳子節點進行擴展，
    直至遇到終局狀態。
    """
    while not terminal(node.game):
        if node.untried_moves:
            move = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
            new_game = clone_game(node.game)
            move_str = move_to_str(move, new_game)
            # 在模擬過程中暫時關閉輸出
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

def default_policy(game, rollout_player):
    """
    模擬策略：從當前狀態開始隨機模擬對局直至結束，
    回傳相對於 rollout_player 的獎勵：
      - 勝利：1
      - 失敗：0
      - 平手：0.5
    """
    current_game = clone_game(game)
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
    """反向傳播模擬結果，更新節點統計數據"""
    while node is not None:
        node.visits += 1
        if node.player == rollout_player:
            node.wins += reward
        else:
            node.wins += (1 - reward)
        node = node.parent

def mcts(root, itermax=500):
    """主 MCTS 搜索流程"""
    for _ in range(itermax):
        node = tree_policy(root)
        rollout_reward = default_policy(node.game, root.player)
        backup(node, rollout_reward, root.player)
    # 選擇拜訪數最高的子節點
    best = max(root.children, key=lambda c: c.visits)
    return best.move

class MCTSNode:
    def __init__(self, game, move=None, parent=None):
        # game 為當前狀態（Connect6Game 的複本）
        self.game = game
        self.move = move        # 導致此節點的走法，格式：單一點或兩子組合
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        # 當前節點尚未展開的合法走法
        self.untried_moves = get_legal_moves(game)
        # 紀錄該節點輪到的玩家（1 表示黑, 2 表示白）
        self.player = game.turn

# ----------------------------
# 修改後的 Connect6Game 類別（generate_move 改用 MCTS）
# ----------------------------

class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: 空, 1: 黑, 2: 白
        self.turn = 1  # 1: 黑, 2: 白
        self.game_over = False

    def reset_board(self):
        """清空棋盤並重設遊戲"""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """設定棋盤尺寸並重設遊戲"""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """
        檢查是否有玩家獲勝：
          0 - 尚未分勝負
          1 - 黑子獲勝
          2 - 白子獲勝
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
        """將欄位數字轉換成字母表示（跳過 'I' 字母）"""
        return chr(ord('A') + col + (1 if col >= 8 else 0))

    def label_to_index(self, col_char):
        """將字母轉為欄位索引（考慮跳過 'I'）"""
        col_char = col_char.upper()
        if col_char >= 'J':
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """
        落子並檢查遊戲狀態
          - color: 'B' 或 'W'
          - move: 落子位置字串（例如 "A1" 或 "A1,B2"）
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

        # 更新回合。注意：在 Connect6 中通常每回合落 2 子（第一手除外）
        self.turn = 3 - self.turn
        print("= ", end='', flush=True)

    def generate_move(self, color):
        """
        改用 MCTS 搜索決定最佳走法：
          1. 建立目前棋局的複本作為 MCTS 根節點
          2. 執行 MCTS 搜索（此處預設迭代 500 次，可依情況調整）
          3. 將得到的走法轉換成字串，執行落子並輸出結果
        """
        if self.game_over:
            print("? Game over")
            return
        
        root = MCTSNode(clone_game(self))
        best_move = mcts(root, itermax=1)
        move_str = move_to_str(best_move, self)
        self.play_move(color, move_str)
        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)

    def show_board(self):
        """以文字方式顯示棋盤狀態"""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """列出所有可用的指令"""
        print("= ", flush=True)

    def process_command(self, command):
        """解析並執行 GTP 指令"""
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
        """主迴圈：從標準輸入讀取並執行 GTP 指令"""
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

# ----------------------------
# 主程式入口
# ----------------------------

def main():
    game = Connect6Game()
    game.run()

if __name__ == "__main__":
    main()
