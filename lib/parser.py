"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 24/05-2021
"""
import os
import pickle
import argparse
import chess.pgn
import chess.engine
import numpy as np
from state import State


OPENING_BOOK_FILEPATH = 'data/opening_book.pgn'


def generate_data(regression=False, num_samples=0.0):
    state = State()
    result_values, X, Y_classification, Y_regression, num_games, reading_games = \
        {'1/2-1/2': 0, '1-0': 1, '0-1': -1}, [], [], [], 0, True
    for fn in os.listdir('../data'):
        if not reading_games:
            break
        pgn = open(os.path.join('../data', fn))
        while reading_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            res = game.headers['Result']
            if res not in result_values:
                continue
            value = result_values[res]
            for idx, move in enumerate(game.mainline_moves()):
                board.push(move)
                state.set_board(board)
                bitmap = state.serialize()
                X.append(bitmap)
                Y_classification.append(value)
            print(f'parsing game: {num_games},\ttotal positions: {len(X)}')
            if len(X) > num_samples:
                reading_games = False
            num_games += 1
    X = np.array(X)
    Y_classification, Y_regression = np.array(Y_classification), np.array(Y_regression)
    return X, Y_classification, Y_regression


def generate_book():
    openings = []
    num_games = 0
    tot_pos = 0
    with open(os.path.join('../', OPENING_BOOK_FILEPATH)) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            moves = []
            for idx, move in enumerate(game.mainline_moves()):
                moves.append(move)
                tot_pos += 1
            openings.append(moves)
            num_games += 1
            print(f'parsing game: {num_games},\ttotal positions: {tot_pos}')
    pickle.dump(obj=openings, file=open('../parsed/opening_book.p', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PGN parser')
    parser.add_argument('-r', '--regression', action='store_true', default=False, help='parse regression targets')
    args = parser.parse_args()
    X, Y, yR = generate_data(regression=False, num_samples=100e3)
    np.savez_compressed('../parsed/dataset_100K.npz', X, Y)

    # generate_book()
