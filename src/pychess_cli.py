from .pychess_game import PychessGame


class PychessCLI:
	def __init__(self, num_players, player_names, verbose=False, **kwargs):
		self._game = None
		self._mode = 'cli'


	def start(self):
		print("[*]  starting  [pychess]  instance in mode {}".format(self.mode))
		self._game = PychessGame(verbose=verbose, white=player_names[0], black=player_names[1])
