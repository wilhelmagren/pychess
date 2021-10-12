"""
Main Pychess module with implemented argparse CLI for
starting specific game instances.

Author: Wilhelm Ågren, wagren@kth.se
Last edited: 11-10-2021
"""
import os
import sys
import argparse

from src.gui 	import PychessGUI
from src.tui 	import PychessTUI
from src.utils	import *



def parse_args():
    parser = argparse.ArgumentParser(prog='pychess', usage='%(prog)s mode players [options]', 
                        description="pychess arguments for setting running mode and number of players", allow_abbrev=False)
    parser.add_argument('mode', action='store', type=str,
                        help='set running mode to either TUI or gui')
    parser.add_argument('players', action='store', type=int,
                        help="set number of players to either 1 or 2")
    parser.add_argument('-v', '--verbose', action='store_true', 
                        dest='verbose', help='print in verbose mode')
    parser.add_argument('-n', '--names', nargs=2, action='store', type=str,
                        dest='names', help='set the player names')
    args = parser.parse_args()
    return args


def create(args):
    players 	= args.players
    names 		= args.names
    names		= ['white', 'black'] if names is None else names
    verbose		= args.verbose
    mode		= args.mode
    WPRINT("creating new {} instance".format(mode), "Pychess\t", True)
    if mode == 'tui':
        return PychessTUI(players, names, verbose=verbose)
    elif mode == 'gui':
        return PychessGUI(players, names, verbose=verbose)
    EPRINT("invalid mode, use -h for help", "Pychess\t")
    sys.exit(0)


def shutdown(pychess_instance):
    WPRINT("shutting down {} instance".format(pychess_instance._mode), "Pychess\t", True)
    sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    mode = create(args)
    mode.start()
    shutdown(mode)

