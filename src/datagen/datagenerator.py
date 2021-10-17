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
import random
import argparse
import threading
import chess.pgn
import chess.engine
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import WPRINT, EPRINT, num_games_in_PGN

FILEPATHS = os.listdir('../../data/pgn-data/')
EXTREME_VALUES = [-6000.0, 6000.0]
ALLOW = 100000
EXALLOW = 20000
NTHRESH = -50
PTHRESH =  50

class DataGenerator:
    def __init__(self, pgnfile, **kwargs):
        self._store         = dict()
        self._filepath      = pgnfile
        self._ngames        = kwargs.get('ngames', 10000)
        self._nthreads      = kwargs.get('nthreads', 10)
        self._categorical   = kwargs.get('categorical', False)
        self._regression    = kwargs.get('regression', False)

        if self._categorical == self._regression:
            raise ValueError("can't do both regression and categorical labels...")


    def __str__(self):
        return "DataGenerator"


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


    def plot(self, data, fname):
        plt.hist(data, bins=1000, color='gray', edgecolor='black', linewidth='1.2')
        plt.title('Regression label distribution')
        plt.xlabel('label')
        plt.ylabel('num samples')
        plt.xlim((-2, 2))
        plt.savefig(fname+'.png')

    
    def import_data(self, f=None):
        WPRINT("importing data dictionary with FEN+labels", str(self), True)
        completedict = dict()
        files = list(f for f in os.listdir('.') if '.npz' in f) if f is None else f
        dicts = list(map(lambda f: np.load(f, allow_pickle=True)['arr_0'], files))
        for dic in dicts:
            completedict = {**completedict, **dic[()]}
        WPRINT("done importing", str(self), True)
        self._store = completedict

    
    def export_data(self, fname):
        WPRINT("exporting data dictionary with FEN+labels", str(self), True)
        np.savez_compressed(fname, self._store)


    def rerange_data(self):
        WPRINT("setting range for data", str(self), True)
        nplabels = np.array(list(self._store.values()), dtype=np.float16)
        nplabels[nplabels > EXTREME_VALUES[1]] = EXTREME_VALUES[1]
        nplabels[nplabels < EXTREME_VALUES[0]] = EXTREME_VALUES[0]
        self._store = {k: nplabels[i] for i, k in enumerate(self._store.keys())}
        WPRINT("done reranging data", str(self), True) 

    
    def scale_data_min_max(self, a=-1, b=1):
        WPRINT("scaling down data labels using min-max (normalization)", str(self), True)
        nplabels = np.array(list(self._store.values()), dtype=np.float16)
        FEATURE_MAX, FEATURE_MIN = nplabels.max(), nplabels.min()
        nplabels = a + ((nplabels - FEATURE_MIN)*(b - a)) / (FEATURE_MAX - FEATURE_MIN)
        self._store = {k: nplabels[i] for i, k in enumerate(self._store.keys())}
        WPRINT("done scaling down data", str(self), True)


    def scale_data_studentized_residual(self):
        WPRINT("scaling down data labels using studentized residual (normalization)", str(self), True)
        nplabels = np.array(list(self._store.values()))
        nplabels = (nplabels - np.mean(nplabels)) / np.std(nplabels)
        self._store = {k: nplabels[i] for i, k in enumerate(self._store.keys())}
        WPRINT("done scaling down data", str(self), True)


    def shuffle_data(self):
        WPRINT("shuffling the data", str(self), True)
        keys = list(self._store.keys())
        random.shuffle(keys)
        shuffleddict = {k: self._store[k] for k in keys}
        self._store = shuffleddict
        WPRINT("done shuffling data", str(self), True)

    
    def downsample_data(self):
        """
        remove some samples from the extreme middle point in our regression set
        """
        WPRINT("downsampling the regression labels", str(self), True)
        ncount, pcount, nexcount, pexcount, sampleddict = 0, 0, 0, 0, dict()
        for k, v in self._store.items():
            if NTHRESH < v < 0:
                if ncount < ALLOW:
                    ncount += 1
                    sampleddict[k] = v
            elif 0 <= v < PTHRESH:
                if pcount < ALLOW:
                    pcount += 1
                    sampleddict[k] = v
            elif v <= EXTREME_VALUES[0]:
                if nexcount < EXALLOW:
                    nexcount += 1
                    sampleddict[k] = v
            elif v >= EXTREME_VALUES[1]:
                if pexcount < EXALLOW:
                    pexcount += 1
                    sampleddict[k] = v
            else:
                sampleddict[k] = v
        self._store = sampleddict
        WPRINT("done downsampling", str(self), True)


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
    parser = argparse.ArgumentParser(prog='datagenerator', usage='%(prog)s [options]')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='print debugs in verbose mode')
    parser.add_argument('-nt', '--nthreads', action='store', type=int,
                        dest='nthreads', help='set number of threads to use for generation', default=10)
    parser.add_argument('-ng', '--ngames', action='store', type=int,
                        dest='ngames', help='set the number of games to read for each thread', default=10000)
    parser.add_argument('-r', '--regression', action='store_true',
                        dest='regression', help='set labeling to numerical values')
    parser.add_argument('-c', '--classification', action='store_true',
                        dest='classification', help='set labeling to classes')
    parser.add_argument('-f', '--files', nargs='+', action='store', type=str, dest='files', 
                        help='set the files to generate FEN+label samples from', default=FILEPATHS)
    args = parser.parse_args()
    
    if args.regression == args.classification:
        raise ValueError("you can't use both regression and classification labels, use -h for help")

    datagen = DataGenerator(args.files[0], nthreads=args.nthreads, ngames=args.ngames, regression=args.regression, classifcation=args.classification)
    datagen.import_data(f=["2019-downsampled_FEN-R.npz"])
    # datagen.plot(datagen._store.values(), "loaded-downsampled")
    #datagen.shuffle_data()
    # datagen.rerange_data()
    # datagen.plot(datagen._store.values(), "after-reranging")
    # datagen.downsample_data()
    datagen.scale_data_min_max()
    # datagen.scale_data_studentized_residual()
    datagen.plot(datagen._store.values(), "2019-dist-downsampled-scaled")
    datagen.export_data("2019-downsampled-scaled_FEN-R.npz")

