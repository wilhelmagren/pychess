from .pychess_game 	import PychessGame
from .pychess_utils	import *


class PychessCLI:
	def __init__(self, players, names, verbose=False, **kwargs):
		self._game 		= None
		self._mode 		= 'cli'
		self._players 	= players
		self._names		= names
		self._verbose 	= verbose


	def _get_and_push_move(self):
		move = input(WHITE_TO_PLAY if self._game.get_turn() else BLACK_TO_PLAY)
		self._game.make_move(move) 


	def _run(self):
		while not self._game.is_terminal():
			self._get_and_push_move()
		WPRINT("game is done, cleaning up and terminating ...", "PychessCLI\t", True)


	def start(self):
		WPRINT("creating new game instance", "PychessCLI\t", True)
		try:
			self._game = PychessGame(players=self._players, verbose=self._verbose, white=self._names[0], black=self._names[1])
		except:
			EPRINT("could not create new game instance, terminating ...", "PychessCLI\t")
		self._run()
