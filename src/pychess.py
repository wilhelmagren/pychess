import os
import sys
import argparse

from pychess_gui import PychessGUI
from pychess_cli import PychessCLI


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(prog='pychess', usage='%(prog)s mode players [options]', 
									description="pychess arguments for setting running mode and number of players", allow_abbrev=False)
	parser.add_argument('mode', action='store', type=str,
						help='set running mode to either cli or gui')
	parser.add_argument('players', action='store', type=int,
						help="set number of players to either 1 or 2")
	parser.add_argument('-v', '--verbose', action='store_true', 
						dest='verbose', help='print in verbose mode')
	parser.add_argument('-n', '--names', nargs=2, action='store', type=str,
						dest='names', help='set the player names')
	args = parser.parse_args()
	return args


PychessInstance = object
def shutdown(pychess_instance: PychessInstance) -> None:
	print("[*]  shutting down pychess instance {}".format(pychess_instance.mode))
	sys.exit(1)


if __name__ == "__main__":
	args = parse_args()
	pychess_instance = PychessGUI() if args.mode == 'gui' else PychessCLI()
	pychess_instance.start()
	shutdown(pychess_instance)
	