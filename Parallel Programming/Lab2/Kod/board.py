import os

class Board:
    def __init__(self, rows, cols, board=None):
        self.rows = rows
        self. cols = cols
        if board == None:
            self.board = [[0 for i in range(cols)] for j in range(rows)]
        else:
            self.board = [row[:] for row in board]

    def print_board(self):
        #os.system("cls")
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                print(self.board[i][j], end="  ")
            print()
        print()
    def move(self, col, player=1):
        if self.board[self.rows - 1][col] == 0:
            row = self.rows - 1
        else:
            row = 0
            while(self.board[row][col] == 0):
                row += 1
            row -= 1

        new_board = Board(self.rows, self.cols, self.board)
        new_board.board[row][col] = player
        return new_board
    
    def __repr__(self):
        board_str = ""
        for row in self.board:
            board_str += "  ".join(map(str, row)) + "\n"
        return board_str[:-1]
    
    def stalemate(self):
        for col in range(self.cols):
            if self.is_movable(col):
                return False
        return True

    def check_winner(self, col):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        row = 0
        while(self.board[row][col] == 0):
            row += 1
        
        r = row
        c = col
        for dx, dy in directions:
            for k in range(-3, 1):
                i = k
                sum = [0,0,0]
                #print(dx, dy, k)
                while(abs(i -k) < 4 and r + dx * i < self.rows and r + dx * i >= 0 and c + dy * i < self.cols and c + dy * i >= 0):
                    sum[self.board[r + dx * i][c + dy * i]] += 1
                 #   print(r + dx * i, c + dy * i, i, abs(i -k))
                    i += 1
                #print(sum)
                if sum[1] == 4:
                    return 1
                elif sum[2] == 4:
                    return 2
        return 0
    
    def is_movable(self, col):
        if self.board[0][col] != 0:
            return False
        return True


class State:
    def __init__(self, board, player, depth, max_depth, col_played):
        self.board = board
        self.value = 0
        self.player = player
        self.children = []
        self.depth = depth
        self.max_depth = max_depth
        self.col_played = col_played
        self.winner = False

        if depth > 0:
            if board.check_winner(col_played) != 0:
                self.winner = True
                if player == 1:
                    self.value = 1
                else:
                    self.value = -1
        
        
        if not self.winner and self.depth < max_depth:
            self.add_children()

        self.update_value()


    def update_value(self):
        if not self.winner:
            sum = 0
            for child in self.children:
                if child.winner:
                    if self.player == 1:
                        self.value = -1
                    else:
                        self.value = 1
                    break
                
                sum += child.value
            if self.value == 0 and len(self.children) > 0 :
                self.value = sum/len(self.children)

    def __repr__(self):
        string = "Board:\n" + str(self.board) + "\nValue:" + str(self.value) + "\nPlayer:" + str(self.player) + "\nDepth:" + str(self.depth) + "\nCol played:" + str(self.col_played)
        return string
    
    def add_children(self):
        for col in range(self.board.cols):
            if self.board.is_movable(col):
                self.children.append(State(self.board.move( col, self.player), 1 if self.player == 2 else 2, self.depth + 1, self.max_depth, col))


class Tree:
    def __init__(self, board, depth, player=2):
        self.root = State(board, player, 0, depth, None)
        self.depth = depth


        
    def get_best_move(self):
        best_move = 0
        val = -2
        #best_child = None
        for child in self.root.children:
            if child.value > val:
                best_move = child.col_played
                val = child.value
                #best_child = child
        return best_move

        #print(best_child)

    def print_rec(self, state):
        print(state)
        print()
        for child in state.children:
            self.print_rec(child)

    def get_children(self):
        children = []

        def dfs(state):
            if state.depth == self.depth:
                children.append(state)
            else:
                for child in state.children:
                    dfs(child)
        dfs(self.root)

        return children
    
    def update_children(self):
        children = []

        def dfs(state):
            if state.depth == self.depth -1:
                state.update_value()
            else:
                for child in state.children:
                    dfs(child)
                state.update_value()
        dfs(self.root)



if __name__=="__main__":
    b = Board(5, 5, [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [2, 2, 1, 2, 2]
])
    
    print(b.check_winner(3))


    rows = int(input("Rows: "))
    cols = int(input("Columns: "))

    b = Board(rows, cols)
    b.print_board()

    col = int(input("Columns: "))
    b = b.move( col, 1)
    b.print_board()

    i = 2
    while(not b.check_winner( col)):
        if i == 1:
            col = int(input("Columns: "))

        else:
            t = Tree(b, 6)
            col = t.get_best_move()
            
            #for c in t.get_children(2):
            #    print(c)
            #print(len(t.get_children(2)))
        b = b.move(col, i)
        b.print_board()

        i = 1 if i == 2 else 2
    print("winner", b.check_winner(col))
