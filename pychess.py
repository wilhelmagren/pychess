"""
Main Pychess module with implemented argparse CLI for
starting specific game instances. Either run the 
application in Terminal User Interface (TUI) or
Graphical User Interface (GUI) mode. This has to
be specified when starting the program from the 
command line, together with how many players are
going to be playing. An example of running the 
program, without running in /bin/bash, is:

kali@kali:~$ python3 pychess tui 2 --verbose --time 300 --increment 15 --names ulysses tux

to have two players playing in TUI mode with 5min time 
format, 15s increment, and player names are 
white: ulysses, black: tux. the double dash (--) 
CLI options are optionals, hence the name, but the mode
and number of players is required.

Author: Wilhelm Ågren, wagren@kth.se
Last edited: 12-10-2021
"""
import os
import sys
import argparse

from src.gui        import PychessGUI
from src.tui        import PychessTUI
from src.utils      import *



def parse_args():
    """
    @spec  parse_args()  =>  namespace
    func creates a new argparses object for CLI interaction,
    sets up the valid arguments and parses them into 
    a namespace object. this object is used for getting the
    positional and optional arguments. running the main 
    file (pychess.py) in helper mode (-h, --help) yields:

    usage: pychess [mode] [players] [options]

    pychess arguments for setting running mode and other game settings

    positional arguments:
        mode                                set running mode to either TUI or GUI
        players                             set number of players to either 1 or 2

    optional arguments:
        -h, --help                          show this help message and exit
        -v, --verbose                       print debugs in verbose mode
        -n NAME NAME, --names NAME NAME     set the player names
        -t TIME, --time TIME                set the time format (in seconds)
        -i INCREMENT, --increment INCREMENT set the time increment (in seconds)
    """
    parser = argparse.ArgumentParser(prog='pychess', usage='%(prog)s mode players [options]', 
                        description="pychess arguments for setting running mode and number of players", allow_abbrev=False)
    parser.add_argument('mode', action='store', type=str,
                        help='set running mode to either TUI or GUI')
    parser.add_argument('players', action='store', type=int,
                        help="set number of players to either 1 or 2")
    parser.add_argument('-v', '--verbose', action='store_true', 
                        dest='verbose', help='print debugs in verbose mode')
    parser.add_argument('-n', '--names', nargs=2, action='store', type=str,
                        dest='names', help='set the player names', default=[DEFAULT_WHITE, DEFAULT_BLACK])
    parser.add_argument('-t','--time', action='store', type=int, default=DEFAULT_TIME,
                        dest='time', help='set the time format (in seconds)')
    parser.add_argument('-i', '--increment', action='store', type=int, default=DEFAULT_INCREMENT,
                        dest='increment', help='set the time increment (in seconds)')
    args = parser.parse_args()
    return args


def create(args):
    """
    @spec  create(namespace)  =>  PychessMode
    func takes the parsed CLI arguments as a namespace
    and processes them. exiting program if provided user args
    are invalid. returns the requested PychessMode as main
    game loop for running the application. 
    """
    players = args.players
    names   = args.names
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

    if mode == 'TUI':
        WPRINT("creating new {} instance".format(mode), "Pychess\t", True)
        return PychessTUI(players, names, verbose=verbose, time=t_time, increment=t_incr, white=names[0], black=names[1])
    elif mode == 'GUI':
        WPRINT("creating new {} instance".format(mode), "Pychess\t", True)
        return PychessGUI(players, names, verbose=verbose, time=t_time, increment=t_incr, white=names[0], black=names[1])
    EPRINT("invalid mode, use -h for help", "Pychess\t")
    sys.exit(0)


def shutdown(pychess_mode):
    """
    @spec  shutdown(PychessMode)  =>  none
    func is simply notifying to the user that the program is
    now terminating the created PychessMode instance and 
    exiting totally. since program came this system exit
    code is set to 1 because no errors.

    TODO: implement cleaning up .tmp files created by program.
    """
    WPRINT("shutting down {} instance".format(pychess_mode._mode), "Pychess\t", True)
    sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    mode = create(args)
    mode.start()
    shutdown(mode)

