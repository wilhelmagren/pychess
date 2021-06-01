"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 24/05-2021
"""
import os
import time
import pickle
import argparse
import chess.pgn
import chess.engine
import numpy as np
from state import State
import matplotlib.pyplot as plt

DATA_FILEPATH = '../data/ficsgamesdb_2018_chess_nomovetimes_201349.pgn'
OPENING_BOOK_FILEPATH = 'data/opening_book.pgn'
STOCKFISH_FILEPATH = '../stockfish/stockfish_13_win_x64_avx2.exe'
SKIP_GAMES = 0


MATED_VALUES = [-30.0, 30.0]
CLR_MOVE = {
    'b': -1,
    'w': 1
}
c2i = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}


def generate_data(num_games):
    state = State()
    X, Y, tot_pos = [], [], 0
    with open(DATA_FILEPATH) as pgn:
        print(f'{time.asctime()}  ::  skipping {SKIP_GAMES} games')
        for skip in range(SKIP_GAMES):
            game = chess.pgn.read_game(pgn)
        print(f'{time.asctime()}  ::  parsing real games now...')
        for game_idx in range(num_games):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                state.set_board(board)
                bitmap = state.serialize()
                board.push(move)
                pos = list(move.uci())[:2]
                sqr_idx = c2i[pos[0]] + 8*(int(pos[1]) - 1)
                X.append(bitmap)
                Y.append(int(sqr_idx))
                tot_pos += 1
            print(f'{time.asctime()}  ::  parsing game {game_idx + 1},\ttotal positions {tot_pos}')
            num_games += 1
    X = np.array(X)
    Y = np.array(Y)
    plot_data(Y)
    print(f'{time.asctime()}  ::  done parsing')
    print(X.shape, Y.shape)
    return X, Y


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


def plot_data(y):
    plt.hist(y, bins=64, color='maroon')
    plt.xlabel('target evaluation')
    plt.ylabel('num labels')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PGN parser')
    parser.add_argument('-r', '--regression', action='store_true', default=False, help='parse regression targets')
    args = parser.parse_args()
    X, Y = generate_data(num_games=5000)
    np.savez_compressed('../parsed/dataset_batch1_5K_C.npz', X, Y)

    # generate_book()
