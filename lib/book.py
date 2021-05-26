"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 26/05-2021
"""
import pickle


class Book:
    """
    Read the opener book. Used for the engine, i.e. ada.py
    """
    def __init__(self):
        self.openings = self.__read__()

    @staticmethod
    def __read__():
        return pickle.load(open('../parsed/opening_book.p', 'rb'))


if __name__ == '__main__':
    b = Book()
    print(b.openings)
