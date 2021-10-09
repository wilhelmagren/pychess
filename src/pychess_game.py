import time
import chess
import numpy as np


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
		$  PychessGame._init_info			=>	dict
		$  PychessGame._update_info			=>	none
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


	def _init_info(self, kwargs):
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
		return infodict

	def _update_info(self, move):
		self._info['FEN']			= self._state.fen()
		self._info['turn']			= self._state.turn
		self._info['time-white']	= self._info['time-start'] - (time.time() - self._info['time-prev-move']) + self._info['time-increment'] if not self._info['turn'] else self._info['time-white']
		self._info['time-black']	= self._info['time-start'] - (time.time() - self._info['time-prev-move']) + self._info['time-increment'] if self._info['turn'] else self._info['time-black']
		self._info['state-history'].append(self._state)
		self._info['move-history'].append(move)


	def _legal_moves(self):
		return list(self._state.legal_moves)


	def _push_move(self, move):
		if move in self._legal_moves():
			self._state.push(move)
			self._update_info(move)
			return True

		return False


	def start_clock(self):
		self._info['time-prev-move'] = time.time()


	def get_info(self):
		return self._info


	def get_state(self):
		return self._state


	def make_move(self, move):
		return self._push_move(move)


if __name__ == "__main__":
	pg = PychessGame(white='ulysses', black='beq')
