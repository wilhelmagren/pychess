"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 24/05-2021
"""
import chess


class State(object):
    def __init__(self):
        self.board = chess.Board()

    def serialize(self):
        pass

    def branches(self) -> list:
        # Generator function board.legal_moves/0
        return list(self.board.legal_moves)

    def value(self) -> float:
        # TODO: Implement Neural Net here.
        return 0  # Currently all positions are drawn


if __name__ == '__main__':
    s = State()
    print(s.value())
