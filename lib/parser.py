"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 17/06-2021
"""
import os
import time
import pickle
import chess.pgn
import chess.engine
import numpy as np
from state import State
import matplotlib.pyplot as plt


TACTICS_FILEPATH = '../data/tactics.pgn'
DATA_FILEPATH = '../data/ficsgamesdb_2020_blitz_nomovetimes_210322.pgn'
OPENING_BOOK_FILEPATH = 'data/opening_book.pgn'
STOCKFISH_FILEPATH = '../stockfish/stockfish_13_win_x64_avx2.exe'
SKIP_GAMES = 30000


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
                state.setboard(board)
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
                if eval > 30:
                    eval = 30
                if eval < -30:
                    eval = -30
                """
                if -5 <= eval <= 5:
                    num_even += 1
                else:
                    num_uneven += 1
                if -5 <= eval <= 5 and num_even/(num_even + num_uneven) > 0.6:
                    continue
                """
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


def generate_data_class(num_games):
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
                state.setboard(board)
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


def plot_data(y, n):
    plt.hist(y, bins=n, color='maroon')
    plt.xlabel('target evaluation')
    plt.ylabel('num labels')
    plt.show()


if __name__ == '__main__':
    X, Y = generate_data(num_games=5000)
    np.savez_compressed('../parsed/dataset01_batch18_5K_R.npz', X, Y)
    exit()
