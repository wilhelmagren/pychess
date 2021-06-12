"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 12/06-2021
"""


class MCTS(object):
    """
    Monte Carlo Tree Search object, called with current chess state/board and arbitry keyword args.
    """
    def __init__(self, board, **kwargs):
        # Current board,
        self.board = board


def test():
    print(' | Monte Carlo Tree Search unit testing ...')


if __name__ == '__main__':
    test()
