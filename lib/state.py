"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 28/05-2021
"""
import chess
import math
import numpy as np
import torch
from net import TinyChessNet

"""
class Evaluator(object):
    def __init__(self):
        weights = torch.load('../nets/value_old.pth', map_location=lambda storage, loc: storage)
        self.model = resNet()
        self.model.load_state_dict(weights)

    def __call__(self, state):
        board = state.serialize()[None]
        output = self.model(torch.tensor(board).float())
        return float(output.data[0][0])
"""


class State(object):
    def __init__(self, board=None):
        # weights = torch.load('../nets/value.pth', map_location=lambda storage, loc: storage)
        # self.model = resNet()
        # self.model.load_state_dict(weights)
        self.board = chess.Board() if board is None else board
        self.piecemap = {}
        self.bitmap = self.serialize()

    def set_board(self, board):
        self.board = board

    def serialize(self) -> np.array:
        """
        7x8x8 bitmap representation of board. Extremely sparse. CNN favours sparse data...
        """

        bitmap = np.zeros(shape=(7, 8, 8))
        CLR_MOVE = {'b': -1, 'w': 1}
        for idx in range(8*8):
            piece = self.board.piece_at(idx)
            if piece is not None:
                onehot_offset = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                                 "p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}[piece.symbol()]
                bitmap[onehot_offset, idx % 8, math.floor(idx / 8)] = 1 if piece.symbol().isupper() else -1
                bitmap[6, idx % 8, math.floor(idx / 8)] = 1*CLR_MOVE[self.board.fen().split(' ')[1]]

        self.update_map()
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
        return 0

    def __repr__(self):
        return self.board.__str__()


if __name__ == '__main__':
    s = State()
    print(s)
