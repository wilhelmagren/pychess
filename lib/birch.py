"""
Author: Wilhelm Ågren
Last edited: 17/06-2021
"""


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
        if ply < 16:
            self.book.__update__(prevmoves)
            self.lookup_book(prevmoves[-1], ply)
        if self.move is None:
            self.explore_leaves(state)

    def lookup_book(self, prevmove, ply):
        possiblemoves = []
        for transposition in self.book.openings:
            if transposition[ply] == prevmove:
                possiblemoves.append(chess.Move.from_uci(transposition[ply + 1]))
        if possiblemoves:
            self.move = random.choice(possiblemoves)

    def explore_leaves(self, state):
        pass

