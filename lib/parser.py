"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 30/05-2021
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

TACTICS_FILEPATH = '../data/tactics.pgn'
DATA_FILEPATH = '../data/ficsgamesdb_2018_chess_nomovetimes_201349.pgn'
OPENING_BOOK_FILEPATH = 'data/opening_book.pgn'
STOCKFISH_FILEPATH = '../stockfish/stockfish_13_win_x64_avx2.exe'
SKIP_GAMES = 50000


MATED_VALUES = [-15.0, 15.0]
CLR_MOVE = {
    'b': -1,
    'w': 1
}


def read_tactics(num_games):
    state = State()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_FILEPATH)
    X, Y, tot_pos, = [], [], 1
    with open(TACTICS_FILEPATH) as pgn:
        for skip in range(SKIP_GAMES):
            game = chess.pgn.read_game(pgn)
        for gidx in range(num_games):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            fen = game.headers['FEN']
            board = chess.Board(fen)
            for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()
                state.set_board(board)
                bitmap = state.serialize()
                to_move = CLR_MOVE[fen.split(' ')[1]]
                score = engine.analyse(board, chess.engine.Limit(depth=6))['score'].white()
                eval = ''
                try:
                    eval = str(str(score.score() / 100))
                except:
                    if score.mate() > 0:
                        eval = MATED_VALUES[1]
                    elif score.mate() < 0:
                        eval = MATED_VALUES[0]
                    else:
                        if to_move > 0:
                            eval = MATED_VALUES[0]
                        else:
                            eval = MATED_VALUES[1]
                eval = float(eval)
                X.append(bitmap)
                Y.append(eval)
                tot_pos += 1
            print(f'{time.asctime()}  ::  parsing game {gidx + 1},\ttotal positions {tot_pos}')
            gidx += 1
    engine.quit()
    X = np.array(X)
    Y = np.array(Y)
    plot_data(Y, 60)
    print(f'{time.asctime()}  ::  done parsing')
    print(X.shape, Y.shape)
    return X, Y


def generate_data(num_games):
    state = State()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_FILEPATH)
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
            num_even = 1
            num_uneven = 1
            for idx, move in enumerate(game.mainline_moves()):
                board.push(move)
                fen = board.fen()
                state.set_board(board)
                bitmap = state.serialize()
                to_move = CLR_MOVE[fen.split(' ')[1]]
                score = engine.analyse(board, chess.engine.Limit(depth=6))['score'].white()
                eval = ''
                try:
                    eval = str(str(score.score() / 100))
                except:
                    if score.mate() > 0:
                        eval = MATED_VALUES[1]
                    elif score.mate() < 0:
                        eval = MATED_VALUES[0]
                    else:
                        if to_move > 0:
                            eval = MATED_VALUES[0]
                        else:
                            eval = MATED_VALUES[1]
                eval = float(eval)
                if -2 <= eval <= 2:
                    num_even += 1
                else:
                    num_uneven += 1
                if -2 <= eval <= 2 and num_even/(num_even + num_uneven) > 0.7:
                    continue
                X.append(bitmap)
                Y.append(eval)
                tot_pos += 1
            print(f'{time.asctime()}  ::  parsing game {game_idx + 1},\ttotal positions {tot_pos}')
            num_games += 1
    engine.quit()
    X = np.array(X)
    Y = np.array(Y)
    plot_data(Y, 60)
    print(f'{time.asctime()}  ::  done parsing')
    print(X.shape, Y.shape)
    return X, Y


def generate_classification(num_samples):
    state = State()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_FILEPATH)
    game_idx = 1
    X, Y, tot_pos = [], [], 0
    results = {'1/2-1/2': 0, '1-0': 1, '0-1': -1}
    with open(DATA_FILEPATH) as pgn:
        print(f'{time.asctime()}  ::  parsing real games now...')
        while 1:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            result = game.headers['Result']
            for idx, move in enumerate(game.mainline_moves()):
                board.push(move)
                state.set_board(board)
                bitmap = state.serialize()
                X.append(bitmap)
                Y.append(results[result])
                tot_pos += 1
            if tot_pos >= num_samples:
                break
            print(f'{time.asctime()}  ::  parsing game {game_idx + 1},\ttotal positions {tot_pos}')
            game_idx += 1
    X = np.array(X)
    Y = np.array(Y)
    plot_data(Y, 3)
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


def plot_data(y, bins):
    plt.hist(y, bins=bins, color='maroon')
    plt.xlabel('target evaluation')
    plt.ylabel('num labels')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PGN parser')
    parser.add_argument('-r', '--regression', action='store_true', default=False, help='parse regression targets')
    args = parser.parse_args()
    X, Y = read_tactics(num_games=10000)
    np.savez_compressed('../parsed/dataset_tactics_batch6_10K.npz', X, Y)

    # generate_book()
