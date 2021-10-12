"""
Main Pychess module with implemented argparse CLI for
starting specific game instances.

to run pychess from the command line run the following:
$ python3 pychess.py <mode> <number of player(s)> [options]
where arg mode is: ['tui', 'gui'] 
and arg num players is: ['1', '2']
valid options are:
    -v,             --verbose
    -n name1 name2, --names name1 name2
    -t time,        --time time
    -i increment,   --increment increment

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
    """
    @spec  parse_args()  =>  namespace
    func sets up the argparses module for Command Line Interface (CLI)
    and parses the given arguments. the parsed arguments are returned
    as a namespace containing the provided args.
    """
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
    parser.add_argument('-t', '--time', nargs=1, action='store', type=int,
                        dest='time', help='set the time format, in seconds')
    parser.add_argument('-i', '--increment', nargs=1, action='store', type=int,
                        dest='increment', help='set the time increment, in seconds')
    args = parser.parse_args()
    return args


def create(args):
    """
    @spec  create(namespace)  =>  PychessTUI/PychessGUI
    func creates the specified instance, either TUI or GUI, based
    on provided CLI args. function returns those created objects 
    if the specified mode and number of players were valid.
    """
    players 	= args.players
    names 		= args.names
    names		= ['white', 'black'] if names is None else names
    verbose		= args.verbose
    mode		= args.mode
    time        = args.time
    increment   = args.increment
    WPRINT("creating new {} instance".format(mode), "Pychess\t", True)
    if players != 1 or players != 2:
        EPRINT("invalid number of players, use -h for help", "Pychess\t")
        sys.exit(0)
    if mode == 'tui':
        return PychessTUI(players, names, verbose=verbose, time=time, increment=increment)
    elif mode == 'gui':
        return PychessGUI(players, names, verbose=verbose, time=time, increment=increment)
    else:
        EPRINT("invalid mode, use -h for help", "Pychess\t")
        sys.exit(0)


def shutdown(pychess_instance):
    """
    @spec  shutdown(PychessTUI/PychessGUI)  =>  none
    func simply shuts down the current instance of Pychess
    and notifies user by printing to terminal. 
    """
    WPRINT("shutting down {} instance".format(pychess_instance._mode), "Pychess\t", True)
    sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    mode = create(args)
    mode.start()
    shutdown(mode)

