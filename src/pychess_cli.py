from .pychess_game 	import PychessGame
from .pychess_utils	import *


class PychessCLI:
	def __init__(self, players, names, verbose=False, **kwargs):
		self._game 		= None
		self._mode 		= 'cli'
		self._players 	= players
		self._names		= names
		self._verbose 	= verbose


	def start(self):
		WPRINT("creating new game instance", "PychessCLI\t", True)
		try:
			self._game = PychessGame(players=self._players, verbose=self._verbose, white=self._names[0], black=self._names[1])
		except:
			EPRINT("could not create new game instance, terminating ...", "PychessCLI\t")
		self._game.run()
		WPRINT("game is done, cleaning up and terminating ...", "PychessCLI\t", True)
		
