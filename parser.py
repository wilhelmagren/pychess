import os
import chess.pgn
import numpy as np
from state import State


def generate_data():
    values = {'1/2-1/2': 0, '1-0': 1, '0-1': -1}
    for fn in os.listdir('data'):
        pgn = open(os.path.join('data', fn))
        for game in chess.pgn.read_game(pgn):
            print(game)
