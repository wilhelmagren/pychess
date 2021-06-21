"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 20/06-2021

Multi-threaded parser for evaluating chess positions using the Stockfish 13 engine.
Reads a local Portable Game Notation (PGN) file from disk and iterates over all the
mainline moves for each game in the file. Stores the Forsyth–Edwards Notation (FEN)
strings with corresponding Stockfish 13 centipawn (cp) evaluation in thread-specific
dictionaries which later can be merged using func import_table.
"""
import os
import time
import threading

import chess.engine
import chess.pgn
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator(object):
    """
    Wrapper class for generating chess position data.
    Spawns x amount of threads and parses y amount of games on each thread.
    Every 50 game each thread dumps the dictionary to file with np.savez_compressed.

    There is slight overhead when starting the script. This is due to the fact that the
    higher indexed threads have to iterate over all the games in the PGN file which they
    should not parse. This spins them in a loop until all unecessary games have been 'read'.
    Number of games to skip and length of for loop is: thread_num x num_games_to_parse, where
    thread_num is 0 indexed for first thread.
    """
    def __init__(self, pgnfile, categorical=False, **kwargs):
        self.categorical = categorical
        self.pgnfile = pgnfile
        self.store = {}
        self.numgames = int(kwargs.get('numgames'))
        self.threads = int(kwargs.get('threads'))

    def import_table(self):
        dics, completedic, tpe = [], {}, 'C' if self.categorical else 'R'
        for file in os.listdir('../parsed/'):
            if file.__contains__('BIG'):
                continue
            if file.__contains__('_C_TEST'.format(tpe)):
                print(' | parsing data from filepath {}'.format(file))
                data = np.load(os.path.join('../parsed/', file), allow_pickle=True)
                dics.append(data['arr_0'])
        for dic in dics:
            completedic = {**completedic, **dic[()]}
        print(' | loaded {} dictionarie(s), total of {} unique positions'.format(len(dics), len(completedic)))
        self.store = completedic

    def export_table(self):
        print(' | exporting total progress, processed {} games'.format(len(self.store)))
        np.savez_compressed('../parsed/dataset_standard_{}_TEST'.format('C' if self.categorical else 'R'), self.store)

    def rebuild_table(self):
        inital_dict, rebuilt_dict = {}, {}
        for file in os.listdir('../parsed/'):
            if file.__contains__('dataset_standard_R_TEST'):
                print(' | parsing data from filepath {}'.format(file))
                data = np.load(os.path.join('../parsed/', file), allow_pickle=True)['arr_0']
                inital_dict = {**inital_dict, **data[()]}
        for idx, fen in enumerate(inital_dict):
            print(' | parsing {}'.format(idx))
            if inital_dict[fen] > 150:
                rebuilt_dict[fen] = 2
            elif inital_dict[fen] < -150:
                rebuilt_dict[fen] = 0
            else:
                rebuilt_dict[fen] = 1
        self.categorical = True
        self.store = rebuilt_dict
        self.export_table()

    def parse(self, tnum, skip, pgnfile):

        def export_thread_table():
            print(' | exporting progress for thread {}, processed {} games'.format(tnum, games_by_thread))
            np.savez_compressed('../parsed/dataset_TEST_thread{}_R'.format(tnum), tscore_dict)

        engine = chess.engine.SimpleEngine.popen_uci('../stockfish/stockfish_13_win_x64_avx2.exe')

        gamenum, games_by_thread, num_pos = 0, 0, 0
        tscore_dict = {}

        with open(pgnfile) as pgn:
            print(' | thread {} skipping {} games'.format(tnum, skip))
            for _ in range(skip + 50000):
                _ = chess.pgn.read_game(pgn)
            while gamenum < self.numgames:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    FEN = board.fen()
                    if FEN not in self.store and FEN not in tscore_dict:
                        to_move = board.turn
                        score = engine.analyse(board, chess.engine.Limit(depth=10))['score'].white()
                        if score.mate() is None:
                            score = score.score()
                        else:
                            if score.mate() > 0:
                                score = 10000
                            elif score.mate() < 0:
                                score = -10000
                            else:
                                if to_move > 0:
                                    score = -10000
                                else:
                                    score = 10000
                        if score is None:
                            print(' SCORE IS NONE!')
                            exit()
                        tscore_dict[FEN] = score
                        num_pos += 1
                gamenum += 1
                games_by_thread += 1
                print(' | thread {}, processed {} games and {} positions'.format(tnum, gamenum, num_pos))

                if games_by_thread % 50 == 0:
                    export_thread_table()

        engine.close()

    def execute_parallel(self):
        procs = []
        for t in range(self.threads):
            threadpgn = self.pgnfile[t] if type(self.pgnfile) is list else self.pgnfile
            print(' | thread {} started, parsing file: {}'.format(t, threadpgn))
            thread = threading.Thread(target=self.parse, args=(t, t*self.numgames, threadpgn))
            procs.append(thread)
            thread.start()
        for proc in procs:
            proc.join()


def plot(y):
    plt.style.use('ggplot')
    plt.hist(y, bins=3, color='gray', edgecolor='black', linewidth='1.2')
    plt.title('Categorical train distribution')
    plt.xlabel('black-      draw      white+')
    plt.ylabel('num samples')
    plt.xlim((-1, 3))
    # plt.ylim((0, 1.3*y.shape[0]/3))
    plt.show()


def clean(y):
    new_y = {}
    for idx, fen in enumerate(y):
        newfen = ''
        print(' | game {}'.format(idx))
        for idx, subs in enumerate(fen.split(' ')[:-3]):
            if idx == 0:
                newfen += subs
            else:
                newfen += ' ' + subs
        new_y[newfen] = y[fen]
    print(' | parsed {} unique positions'.format(len(new_y)))
    return new_y


def remake(data, categorical):
    X, Y, starttime = [], [], time.time()
    for bidx, fen in enumerate(data):
        if len(fen.split('/')) < 6:
            continue
        bitmap = np.zeros((19, 8, 8), dtype=int)
        board, piece_offset = chess.Board(fen), {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                                                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}

        for idx in reversed(range(64)):
            x_idx, y_idx = idx % 8, int(np.floor(idx / 8))
            piece = board.piece_at(idx)
            if piece:
                bitmap[piece_offset[piece.symbol()], x_idx, y_idx] = 1
        bitmap[12, :, :] = int(board.turn)
        bitmap[13, :, :] = int(board.has_kingside_castling_rights(chess.WHITE))
        bitmap[14, :, :] = int(board.has_queenside_castling_rights(chess.WHITE))
        bitmap[15, :, :] = int(board.has_kingside_castling_rights(chess.BLACK))
        bitmap[16, :, :] = int(board.has_queenside_castling_rights(chess.BLACK))
        bitmap[17, :, :] = 1 if board.is_check() and board.turn else 0
        bitmap[18, :, :] = 1 if board.is_check() and not board.turn else 0
        X.append(bitmap[None])
        Y.append(data[fen])
        print(' | position {}'.format(bidx + 1))
    X = np.concatenate(X, axis=0)
    Y = np.array(Y)
    print(' | done parsing in {:.1f}s, {}  {}'.format(time.time() - starttime, X.shape, Y.shape))
    np.savez_compressed('../parsed/dataset_standard_{}_BIG_TEST.npz'.format('C' if categorical else 'R'), X, Y)
    plot(Y)


if __name__ == '__main__':
    parser = DataGenerator('../data/ficsgamesdb_2020_standard_nomovetimes_210720.pgn',
                           True, numgames=2500, threads=2)
    # parser.execute_parallel()
    parser.import_table()
    # parser.store = clean(parser.store)
    # parser.export_table()
    # parser.rebuild_table()
    # plot(parser.store.values())
    remake(parser.store, True)
    # plot(parser.store)
    # parser.execute_parallel()
