"""
Author: Wilhelm Ågren
Last edited: 28/05-2021
"""
from book import Book
import chess
import time
import random


class Ada:
    """
    TODO: Implement Monte-Carlo tree search
    TODO: Prune search tree with classifier!
    """
    def __init__(self, player=False, max_val=float('-inf')):
        self.curr_player = player
        self.max_val = max_val
        self.best_move = None
        self.transposition_table = {}
        self.book = Book()

    def set_player(self, player):
        self.curr_player = player
        self.max_val = float('-inf') if player else float('inf')
        self.best_move = None

    def __lookup_book__(self, prev_move, ply):
        print(f'{time.asctime()}  ::  previous move, {prev_move} at ply {ply}')
        possible_moves = []
        for transposition in self.book.openings:
            if transposition[ply] == prev_move:
                possible_moves.append(chess.Move.from_uci(transposition[ply + 1]))
        if possible_moves:
            self.best_move = random.choice(possible_moves)

    def find_move(self, prev_moves, ply, state):
        """
        ply is halfmove idx, i.e. how many totals moves have been performed.
        """
        if ply < 16:
            self.book.__update__(prev_moves)
            self.__lookup_book__(prev_moves[-1], ply - 1)
        if self.best_move is None:
            # self.__explore_deep__(state, max_depth=1)
            self.__explore_leaves__(state)
        return self.best_move

    def __explore_leaves__(self, state):
        for move in self.__topmoves__(state):
            state.board.push(move)
            my_val = self.__gamma__(state)
            state.board.pop()
            if my_val <= self.max_val:
                print('{}  ::  found better move, {}, {:.2f}<={:.2f}'.format(time.asctime(), move, my_val, self.max_val))
                self.best_move = move
                self.max_val = my_val

        if self.best_move is None:
            print(f'{time.asctime()}  ::  no move found(?), randomly choosing move..')
            self.best_move = random.choice(state.branches())

    def __explore_deep__(self, state, max_depth):
        for move in state.branches():
            state.board.push(move)
            evaluation = self.__minimax__(player=self.curr_player, state=state, depth=max_depth)
            state.board.pop()
            if self.curr_player:
                if evaluation >= self.max_val:
                    self.max_val = evaluation
                    self.best_move = move
            else:
                if evaluation <= self.max_val:
                    self.max_val = evaluation
                    self.best_move = move
        if self.best_move is None:
            self.best_move = random.choice(state.branches())

    def __minimax__(self, player, state, alpha=float('-inf'), beta=float('inf'), depth=0):
        if depth == 0:
            return self.__gamma__(state)
        children = self.__trim__(player, state)
        if player:
            v = float('-inf')
            for child in children:
                state.board.push(child)
                v = max(v, self.__minimax__(False, state, alpha, beta, depth - 1))
                state.board.pop()
                alpha = max(alpha, v)
                if beta <= alpha:
                    self.best_move = child
                    break
            return v
        else:
            v = float('inf')
            for child in children:
                state.board.push(child)
                v = min(v, self.__minimax__(True, state, alpha, beta, depth - 1))
                state.board.pop()
                beta = min(beta, v)
                if beta <= alpha:
                    self.best_move = child
                    break
            return v

    def __trim__(self, player, state):
        trimmed = []
        for move in state.branches():
            state.board.push(move)
            val = self.__gamma__(state)
            state.board.pop()
            if player and val < -0.2:
                continue
            if not player and val > 0.2:
                continue
            trimmed.append(move)
        return trimmed

    @staticmethod
    def __topmoves__(state):
        return state.best_moves()

    @staticmethod
    def __gamma__(state):
        return state.value()
