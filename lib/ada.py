"""
Author: Wilhelm Ågren
Last edited: 26/05-2021
"""


class Ada:
    def __init__(self, player=False, max_val=float('-inf')):
        self.curr_player = player
        self.max_val = max_val
        self.best_move = None
        self.transposition_table = {}

    def set_player(self, player, max_val):
        self.curr_player = player
        self.max_val = max_val
        self.best_move = None

    def find_move(self, state, max_depth):
        self.__iterative_deepening__(state=state, max_depth=max_depth)
        return self.best_move

    def __iterative_deepening__(self, state, max_depth=3):
        for depth in range(max_depth):
            for move in state.branches():
                if self.best_move is None:
                    self.best_move = move
                state.board.push(move)
                evaluation = self.__minimax__(player=self.curr_player, state=state, depth=depth)
                state.board.pop()
                if self.curr_player:
                    if evaluation >= self.max_val:
                        self.best_move, self.max_val = move, evaluation
                else:
                    if evaluation <= self.max_val:
                        self.best_move, self.max_val = move, evaluation

    def __minimax__(self, player, state, alpha=float('-inf'), beta=float('inf'), depth=1):
        if depth == 0:
            return self.__gamma__(state)
        children = state.branches()
        if player:
            v = float('-inf')
            for child in children:
                state.board.push(child)
                v = max(v, self.__minimax__(not player, state, alpha, beta, depth - 1))
                state.board.pop()
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return v
        else:
            v = float('inf')
            for child in children:
                state.board.push(child)
                v = max(v, self.__minimax__(not player, state, alpha, beta, depth - 1))
                state.board.pop()
                beta = max(beta, v)
                if beta <= alpha:
                    break
            return v

    @staticmethod
    def __gamma__(state):
        return state.value()
