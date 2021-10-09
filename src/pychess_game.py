import time
import chess


DEFAULT_WHITE		= 'white'
DEFAULT_BLACK		= 'black'
DEFAULT_TIME		= 300
DEFAULT_INCREMENT	= 5


class PychessGame:
	"""!! definition for class  PychessGame
	used as a main game object in both 'pychess_gui' and 'pychess_cli'
	the object is used as a wrapper of the chess board state and also
	implements necessary helper functions for running the game.
	game information is stored in self._info dictionary and is 
	initialized with **kwargs or DEFAULT values. the time for
	both players are also managed there. this information dictionary 
	may see more key-value pair being added to it as deemed necessary.

	public  funcs:
		$  PychessGame.get_info				=>  dict
		$  PychessGame.get_state			=>	chess.Board

	private funcs:
		$  PychessGame._VPRINT				=> 	none
		$  PychessGame._init_info			=>	dict
		$  PychessGame._update_info			=>	none
		$  PychessGame._set_info			=>  bool
		$  PychessGame._legal_moves			=>  list
		$  PychessGame._push_move			=>  bool

	implemented dunder funcs:
		$  PychessGame.__init__				=>	PychessGame
		$  PychessGame.__str__				=>	str

	"""
	def __init__(self, board=None, verbose=False, **kwargs):
		self._state 	= chess.Board() if board is None else board
		self._verbose 	= verbose
		self._info		= self._init_info(kwargs)


	def __str__(self):
		return str(self._state)


	def _VPRINT(self, msg):
		""" private func
		@spec  _VPRINT(PychessGame, str)  =>  none
		"""
		print(msg) if self._verbose else None


	def _init_info(self, kwargs):
		""" private func
		@spec  _init_info(PychessGame, dict)  =>  dict
		"""
		self._VPRINT("[*]  PychessGame  initializing information dictionary ...")
		infodict = dict()
		infodict['FEN']				= self._state.fen()
		infodict['turn']			= self._state.turn
		infodict['white']			= kwargs.get('white', DEFAULT_WHITE)
		infodict['black']			= kwargs.get('black', DEFAULT_BLACK)
		infodict['time-prev-move']	= time.time()
		infodict['time-start']		= kwargs.get('time', DEFAULT_TIME)
		infodict['time-increment']	= kwargs.get('increment', DEFAULT_INCREMENT)
		infodict['time-white']		= infodict['time-start']
		infodict['time-black']		= infodict['time-start']
		infodict['state-history']	= [self._state]
		infodict['move-history']	= list()
		self._VPRINT("[*]  PychessGame  intialization done")
		return infodict

	def _update_info(self, move):
		""" private func
		@spec  _update_info(PychessGame, chess.Move)  =>  none
		"""
		self._VPRINT("[*]  PychessGame  updating information dictionary ...")
		self._info['FEN']			= self._state.fen()
		self._info['turn']			= self._state.turn
		self._info['time-white']	= self._info['time-start'] - (time.time() - self._info['time-prev-move']) + self._info['time-increment'] if not self._info['turn'] else self._info['time-white']
		self._info['time-black']	= self._info['time-start'] - (time.time() - self._info['time-prev-move']) + self._info['time-increment'] if self._info['turn'] else self._info['time-black']
		self._info['state-history'].append(self._state)
		self._info['move-history'].append(move)
		self._VPRINT("[*]  PychessGame  updating done")


	def _set_info(self, key, val):
		""" private func
		@spec  _set_info(PychessGame, str, *)  =>  bool
		"""
		self._VPRINT("[*]  PychessGame  setting information ...")
		if key not in self._info.keys():
			print("[!]  PychessGame  could not set information, key is not present in dictionary ...")
			return False
		
		self._info[key] = val
		self._VPRINT("[*]  PychessGame  setting information done")
		return True


	def _legal_moves(self):
		""" private func
		@spec  _legal_moves(PychessGame)  =>  list
		"""
		return list(self._state.legal_moves)


	def _push_move(self, move):
		""" private func
		@spec  _push_move(PychessGame, chess.Move)  =>  bool
		"""
		if move in self._legal_moves():
			self._VPRINT("[*]  PychessGame  pushing move ...")
			self._state.push(move)
			self._update_info(move)
			self._VPRINT("[*]  PychessGame  pushing and updating done")
			return True


		return False


	def start_clock(self):
		""" public func
		@spec  start_clock(PychessGame)  =>  none
		"""
		self._VPRINT("[*]  PychessGame  starting clock")
		self._info['time-prev-move'] = time.time()


	def get_info(self):
		""" public func
		@spec  get_info(PychessGame)  =>  dict
		"""
		self._VPRINT("[*]  PychessGame  getting information")
		return self._info


	def get_state(self):
		""" public func
		@spec  get_state(PychessGame)  =>  chess.Board
		"""
		self._VPRINT("[*]  PychessGame  getting state")
		return self._state


	def make_move(self, move):
		""" public func
		@spec  make_move(PychessGame, chess.Move)  =>  bool
		"""
		self._VPRINT("[*]  PychessGame  making move {}".format(move.uci()))
		return self._push_move(move)


if __name__ == "__main__":
	pg = PychessGame(white='ulysses', black='beq', verbose=True)
	a = pg.get_info()
	b = pg.get_state()
	