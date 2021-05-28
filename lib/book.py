"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 26/05-2021
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

    def __update__(self, prev_move, ply):
        """
        remove all transpositions which does not contain prev_move at ply
        """
        updated = []
        for transposition in self.openings:
            if transposition[ply] == prev_move:
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
    print(b.openings)
