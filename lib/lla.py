import os
import threading

import chess.engine
import chess.pgn
import numpy as np


class DataGenerator(object):
    def __init__(self, pgnfile, categorical=False, **kwargs):
        self.categorical = categorical
        self.pgnfile = pgnfile
        self.store = {}
        self.numgames = int(kwargs.get('numgames'))
        self.threads = int(kwargs.get('threads'))

    def import_table(self):
        dics, completedic = [], {}
        for file in os.listdir('../parsed/'):
            if file.__contains__('dataset00_thread'):
                print(' | parsing data from filepath {}'.format(file))
                data = np.load(os.path.join('../parsed/', file), allow_pickle=True)
                dics.append(data['arr_0'])
        for dic in dics:
            completedic = {**completedic, **dic[()]}
        print(' | loaded {} dictionaries, total of {} unique positions'.format(len(dics), len(completedic)))
        self.store = completedic

    def export_table(self):
        print(' | exporting total progress, processed {} games'.format(len(self.store)))
        np.savez_compressed('../parsed/dataset00_standard_{}K_{}'.format(self.numgames, 'C' if self.categorical else 'R'), self.store)

    def parse(self, tnum, skip, pgnfile):

        def export_thread_table():
            print(' | exporting progress for thread {}, processed {} games'.format(tnum, games_by_thread))
            np.savez_compressed('../parsed/dataset00_thread{}'.format(tnum), tscore_dict)

        engine = chess.engine.SimpleEngine.popen_uci('../stockfish/stockfish_13_win_x64_avx2.exe')

        gamenum, games_by_thread, num_pos = 0, 0, 0
        tscore_dict = {}

        with open(pgnfile) as pgn:
            for _ in range(skip):
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
                        try:
                            score = score.score()
                        except:
                            if score.mate() > 0:
                                score = 10000
                            elif score.mate() < 0:
                                score = -10000
                            else:
                                if to_move > 0:
                                    score = -10000
                                else:
                                    score = 10000
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


if __name__ == '__main__':
    parser = DataGenerator('../data/ficsgamesdb_2020_standard_nomovetimes_210720.pgn',
                           False, numgames=1000, threads=5)
    parser.import_table()
    # parser.execute_parallel()
