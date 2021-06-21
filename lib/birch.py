"""
Author: Wilhelm Ågren
Last edited: 17/06-2021
"""


import time
import torch
import chess
import random
from book import Book


class Birch(object):
    def __init__(self, player=False):
        self.player = player
        self.book, self.move, self.val = Book(), None, float('inf') if not player else float('-inf')

    def setplayer(self, player):
        self.player, self.move, self.val = player, None, float('inf') if not player else float('-inf')

    def search(self, prevmoves, ply, state):
        print(ply)
        if ply < 16:
            self.book.__update__(prevmoves)
            self.lookup_book(prevmoves[-1], ply)
        if self.move is None:
            self.explore_leaves(state)
        return self.move

    def lookup_book(self, prevmove, ply):
        possiblemoves = set()
        for transposition in self.book.openings:
            if transposition[ply - 1] == str(prevmove):
                possiblemoves.add(chess.Move.from_uci(transposition[ply]))
        if possiblemoves:
            self.move = random.choice(list(possiblemoves))

    def explore_leaves(self, state):
        children, statevals = state.branches(), []
        for child in children:
            state.board.push(child)
            statevals.append(state.value())
            state.board.pop()
        print(statevals)
        bestidx = torch.argmax(torch.tensor(statevals)) if self.player else torch.argmin(torch.tensor(statevals))
        print(bestidx)
        bestmove = children[bestidx]
        print('{}  ::  best move found {}, with val {}'.format(time.asctime(), bestmove, statevals[bestidx]))
        self.move = bestmove
