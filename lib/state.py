"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 28/05-2021
"""
import chess
import math
import numpy as np


class State(object):
    def __init__(self, board=None):
        self.board = chess.Board() if board is None else board
        self.piecemap = {}
        self.bitmap = self.serialize()

    def serialize(self) -> np.array:
        """
        8x8x12 bitmap representation of board. Extremely sparse. CNN favours sparse data...
        """
        assert self.board.is_valid()

        bitmap = np.zeros(shape=(8*8, 12))

        for idx in range(8*8):
            piece = self.board.piece_at(idx)
            if piece is not None:
                onehot_offset = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}[piece.symbol()]
                bitmap[idx, onehot_offset] = 1

        self.update_map()

        bitmap = np.reshape(bitmap, (12, 8, 8))
        return bitmap

    def update_map(self):
        self.piecemap.clear()
        for idx in range(8*8):
            piece = self.board.piece_at(idx)
            if piece is not None:
                if piece.symbol() not in self.piecemap:
                    self.piecemap.update({piece.symbol(): [(idx % 8, 7 - math.floor(idx / 8))]})
                else:
                    self.piecemap[piece.symbol()] += [(idx % 8, 7 - math.floor(idx / 8))]

    def branches(self) -> list:
        # Generator function board.legal_moves/0
        return list(self.board.legal_moves)

    def value(self) -> float:
        # TODO: Implement Neural Net here.
        return 0  # Currently all positions are drawn

    def __repr__(self):
        return self.board.__str__()


if __name__ == '__main__':
    s = State()
    print(s)
