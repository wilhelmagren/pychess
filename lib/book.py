"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 28/05-2021
"""
import pickle


class Book:
    """
    Read the opening book. Used for the engine, i.e. ada.py
    Currently there are 34700 different openings in the book.
    Searching through it to find a match would take a long time...
    Just put every sequence in its own list. Then when looking up opener, pass
    the ply idx, i.e. what move it is, then extract all those
    """
    def __init__(self):
        self.raw_openings = self.__read__()
        self.openings = self.__process__()

    def __update__(self, prev_moves):
        """
        remove all transpositions which does not contain prev_moves
        """
        if prev_moves is []:
            return
        updated = []
        for transposition in self.openings:
            transposition_list = []
            for plydx, move in enumerate(transposition[:len(prev_moves)]):
                if move != prev_moves[plydx]:
                    break
                transposition_list.append(move)
            if len(transposition_list) == len(prev_moves):
                updated.append(transposition)
        self.openings = updated

    def __process__(self):
        openings = []
        for transposition in self.raw_openings:
            openings.append(list(map(self.__translate__, transposition)))
        return openings

    @staticmethod
    def __translate__(move):
        return move.uci()

    @staticmethod
    def __read__():
        return pickle.load(open('../parsed/opening_book.p', 'rb'))


if __name__ == '__main__':
    b = Book()
    moves = ['e2e4', 'c7c6', 'd2d4', 'd7d5', 'e4e5']
    b.__update__(prev_moves=moves)
    print(b.openings)
