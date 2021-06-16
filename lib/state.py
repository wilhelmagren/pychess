"""
Author: Wilhelm Ågren
Last edited: 16/06-2021

State class wrapper for chess.Board instance.
"""


import math
import chess
import numpy as np


class State(object):
    def __init__(self, board=None):
        self.bitmap_shape = (12, 8, 8)
        self.board = chess.Board() if board is None else board

    def __float__(self) -> float:
        return float(self.value())

    def __str__(self) -> str:
        return str(self.board)

    def __abs__(self) -> float:
        return abs(self.value())

    def __round__(self, n=None) -> float:
        return round(self.value(), 2) if n is None else round(self.value(), n)

    def __floor__(self) -> float:
        return math.floor(self.value())

    def __ceil__(self) -> float:
        return math.ceil(self.value())

    def __ge__(self, other) -> bool:
        return self.value() >= other.value()

    def __le__(self, other) -> bool:
        return self.value() <= other.value()

    def __gt__(self, other) -> bool:
        return self.value() > other.value()

    def __lt__(self, other) -> bool:
        return self.value() < other.value()

    def __eq__(self, other) -> bool:
        return round(self.value()) == round(other.value())

    def __ne__(self, other) -> bool:
        return round(self.value()) != round(other.value())

    #| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    def serialize(self) -> np.array:
        """
        bit(s) 0-5: 'one-hot' encoding of the piece at the square, positive for white pieces and negative for black
        bit 6: player turn to move, 1 for white -1 for black
        bit 7: long or short castle available for current player (bit 7), 0 or 1
        bit 8: legal en passant available, 0 or 1
        bit 9: promotion available, 0 or 1
        bit 10: king is in check, 0 or 1

        Last 5 bits greatly impacts the outcome for the board, and those features should be learned more easily due to
        the increased loss during training(?). Is this too complex for my generated data? 11x8x8 = 704bits, maybe not...
        """
        bitmap, c2m, fen = np.zeros(shape=self.bitmap_shape), {'b': -1, 'w': 1}, self.board.fen()
        to_move, castles_available = c2m[fen.split(' ')[1]], self.board.has_castling_rights(self.board.turn)
        enpassant_available, legal_moves = self.board.has_legal_en_passant(), list(self.board.legal_moves)
        promotion_available = not all(list(move.promotion is None for move in legal_moves))
        is_check, piece_offset = self.board.is_check(), {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                                                         "p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}
        for idx in range(64):
            x_idx, y_idx = idx % 8, math.floor(idx / 8)
            piece = self.board.piece_at(idx)
            if piece:
                # not an empty square, set first 1-6 bits according to piece color & type
                bitmap[piece_offset[piece.symbol()], x_idx, y_idx] = 1 if piece.symbol().isupper() else -1
            bitmap[6, x_idx, y_idx] = to_move
            bitmap[7, x_idx, y_idx] = 1*castles_available
            bitmap[8, x_idx, y_idx] = 1*enpassant_available
            bitmap[9, x_idx, y_idx] = 1*promotion_available
            bitmap[10, x_idx, y_idx] = 1*is_check

        return bitmap

    @staticmethod
    def value() -> float:
        return 0.0


def main():
    s = State()
    s.serialize()


if __name__ == '__main__':
    main()
