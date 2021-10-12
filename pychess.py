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
    parser.add_argument('-t','--time', action='store', type=int, default=300,
                        dest='time', help='set the time format, in seconds')
    parser.add_argument('-i', '--increment', action='store', type=int, default=5,
                        dest='increment', help='set the time increment, in seconds')
    args = parser.parse_args()
    return args


def create(args):
    players = args.players
    names   = args.names
    names   = ['white', 'black'] if names is None else names
    verbose = args.verbose
    mode    = args.mode
    t_time  = args.time
    t_incr  = args.increment
    if t_time < 30:
        EPRINT("invalid time format, must be greater than 30 seconds", "Pychess\t")
        sys.exit(0)
    if players != 1 and players != 2:
        EPRINT("invalid number of players, must be 1 or 2", "Pychess\t")
        sys.exit(0)
    WPRINT("creating new {} instance".format(mode), "Pychess\t", True)
    if mode == 'tui':
        return PychessTUI(players, names, verbose=verbose, time=t_time, increment=t_incr)
    elif mode == 'gui':
        return PychessGUI(players, names, verbose=verbose, time=t_time, increment=t_incr)
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

