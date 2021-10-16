"""
datagenerator script. run this standalone.
read PGN files of recorded blitz games from variously rated
players. spawn N number of threads for reading M amount of data.
total games read from one session is equal to N*M, which has to 
be smaller than the number of games in one PGN file.

the fics2019-blitz dataset contains 4787152 number of games,
which mean you can safely parse up to 4.5 million games.

Author: Wilhelm Ågren, wagren@kth.se
Last edited: 15-10-2021
"""
import os
import sys
import time
import threading
import chess.pgn
import chess.engine
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import WPRINT

FILEPATHS = os.listdir('../../data/pgn-data/')
EXTREME_VALUES = [-6000, 6000]


def num_games_in_pgn(pgnfile):
    with open('../../data/pgn-data/'+pgnfile+'/'+pgnfile) as pgn:
        
        count = 1
        game = chess.pgn.read_headers(pgn)
        while game is not None:
            print("current count: {}".format(count)) if count % 1000 == 0 else None
            count += 1
            game = chess.pgn.read_headers(pgn)
        print("{} number of games in {}".format(count, pgnfile))



class DataGenerator:
    def __init__(self, pgnfile, **kwargs):
        self._store         = dict()
        self._filepath      = pgnfile
        self._ngames        = kwargs.get('numgames', 500000)
        self._nthreads      = kwargs.get('threads', 2)
        self._categorical   = kwargs.get('categorical', False)
        self._regression    = kwargs.get('regression', False)

        assert self._categorical != self._regression, EPRINT("can't do both regresssion and categorical labels...", str(self))


    def __str__(self):
        return "DataGenerator"


    def plot(self):
        vals = self._store.values()
        plt.hist(vals, bins=13000, color='gray', edgecolor='black', linewidth='1')
        plt.title('Regression label distribution')
        plt.xlabel('label')
        plt.ylabel('num samples')
        plt.xlim((-6500, 6500))
        plt.savefig('{}-dist.png'.format('2019'))

    
    def import_data(self):
        WPRINT("importing data dictionary with FEN+labels...", str(self), True)
        completedict = dict()
        files = list(f for f in os.listdir('.') if '.npz' in f)
        dicts = list(map(lambda f: np.load(f, allow_pickle=True)['arr_0'], files))
        for dic in dicts:
            completedict = {**completedict, **dic[()]}
        WPRINT("done importing", str(self), True)
        self._store = completedict

    
    def export_data(self):
        WPRINT("exporting data ditctionary with FEN+labels", str(self), True)
        completedict = self._merge_thread_dicts()
        np.savez_compressed('{}_blitz_FEN-{}'.format(self._filepath.split('_')[1], 'C' if self._categorical else 'R'), self._store)


    def scale_data(self):
        WPRINT("scaling down extreme points in data", str(self), True)
        scaleddict = {k: v if v < EXTREME_VALUES[1] else EXTREME_VALUE[1] for k, v in self._store.items()}
        scaleddict = {k: v if v > EXTREME_VALUES[0] else EXTREME_VALUES[0] for k, v in scaleddict.items()}

        assert len(scaleddict) == len(self._store), EPRINT("missed values when scaling data, dictionaries not matching size", str(self))

        self._store = scaleddict
        WPRINT("done scaling down data", str(self), True) 


    def _merge_thread_dicts(self):
        WPRINT("merging the thread-created dictionaries", str(self), True)
        dics, completedict = list(self._store[key] for key in self._store.keys()), dict()
        for dic in dics:
            completedict = {**completedict, **dic}
        return completedict


    def _parse(self, threadid, offset):
        engine = chess.engine.SimpleEngine.popen_uci('../../stockfish/stockfish_14_x64_avx2.exe')
        thread_store, n_games, n_pos = dict(), 0, 0
        with open('../../data/pgn-data/'+self._filepath+'/'+self._filepath) as pgn:
            WPRINT("thread={} skipping {} games".format(threadid, offset), str(self), True)
            for _ in range(offset):
                _ = chess.pgn.read_headers(pgn)
            while n_games < self._ngames:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    FEN = board.fen()
                    t_fen = ''
                    for idx, subs in enumerate(FEN.split(' ')[:-3]):
                        t_fen += subs if idx == 0 else ' ' + subs
                    if t_fen not in self._store and t_fen not in thread_store:
                        to_move = board.turn
                        score = engine.analyse(board, chess.engine.Limit(depth=5))['score'].white()
                        if score.mate() is None:
                            score = score.score()
                        else:
                            if score.mate() > 0:
                                score = EXTREME_VALUES[1]
                            elif score.mate() < 0:
                                score = EXTREME_VALUES[0]
                            else:
                                score = EXTREME_VALUES[1] if to_move < 0 else EXTREME_VALUES[0]
                        thread_store[t_fen] = score
                        n_pos += 1
                n_games +=1
                WPRINT("thread={} processed {} games and {} positions".format(threadid, n_games, n_pos), str(self), True)
        engine.close()
        WPRINT("thread={} done!".format(threadid), str(self), True)
        self._store[threadid] = thread_store


    def t_generate(self):
        WPRINT("spawning {} threads".format(self._nthreads), str(self), True)
        procs = []
        for t in range(self._nthreads):
            thread = threading.Thread(target=self._parse, args=(t, t*self._ngames))
            procs.append(thread)
            thread.start()
        for proc in procs:
            proc.join()
        WPRINT("all threads done!", str(self), True)
        self._store = self._merge_thread_dicts()



if __name__ == "__main__":
    datagen = DataGenerator(FILEPATHS[0], threads=10, numgames=10000, regression=True)
    datagen.import_data()
    datagen.scale_data()
    #datagen.t_generate()
    datagen.plot()
    #datagen.export_data()

