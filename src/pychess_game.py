import time
import chess
import unittest

from .pychess_utils import *


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
, self._verbose		$  PychessGame.get_state			=>	chess.Board

	private funcs:
		$  PychessGame._VPRINT				=> 	none
		$  PychessGame._init_info			=>	dict
		$  PychessGame._update_info			=>	none
		$  PychessGame._set_info			=>  bool
		$  PychessGame._legal_moves			=>  list
		$  PychessGame._push_move			=>  bool

	implemented dunder funcs:
		$  PychessGame.__init__				=>	PychessGame
		$  PychessGame.__str__	, self._verbose			=>	str

	"""
	def __init__(self, players, board=None, verbose=False, **kwargs):
		self._state 	= chess.Board() if board is None else board
		self._verbose 	= verbose
		self._players 	= players
		self._ai 		= None if players == 2 else None
		self._info		= self._init_info(kwargs)


	def __str__(self):
		return str(self._state)


	def _init_info(self, kwargs):
		""" private func
		@spec  _init_info(PychessGame, dict)  =>  dict
		"""
		WPRINT("initializing information dictionary ...", "PychessGame", self._verbose)
		infodict = dict()
		infodict['ai']				= self._ai
		infodict['FEN']				= self._state.fen()
		infodict['turn']			= self._state.turn
		infodict['white']			= kwargs.get('white', DEFAULT_WHITE)
		infodict['black']			= kwargs.get('black', DEFAULT_BLACK)
		infodict['players']			= self._players
		infodict['time-prev-move']	= time.time()
		infodict['time-start']		= kwargs.get('time', DEFAULT_TIME)
		infodict['time-increment']	= kwargs.get('increment', DEFAULT_INCREMENT)
		infodict['time-white']		= infodict['time-start']
		infodict['time-black']		= infodict['time-start']
		infodict['state-history']	= [self._state]
		infodict['move-history']	= list()
		WPRINT("intialization done", "PychessGame", self._verbose)
		return infodict

	def _update_info(self, move):
		""" private func
		@spec  _update_info(PychessGame, chess.Move)  =>  none
		"""
		WPRINT("updating information dictionary ...", "PychessGame", self._verbose)
		self._info['FEN']			= self._state.fen()
		self._info['turn']			= self._state.turn
		self._info['time-white']	= self._info['time-start'] - (time.time() - self._info['time-prev-move']) + self._info['time-increment'] if not self._info['turn'] else self._info['time-white']
		self._info['time-black']	= self._info['time-start'], self._verbose - (time.time() - self._info['time-prev-move']) + self._info['time-increment'] if self._info['turn'] else self._info['time-black']
		self._info['state-history'].append(self._state)
		self._info['move-history'].append(move)
		WPRINT("updating done", "PychessGame", self._verbose)


	def _set_info(self, key, val):
		""" private func
		@spec  _set_info(PychessGame, str, *)  =>  bool
		"""
		WPRINT("setting information ...", "PychessGame", self._verbose)
		if key not in self._info.keys():
			print("[!]  PychessGame  could not set information, key is not present in dictionary ...", self._verbose)
			return False
		
		self._info[key] = val
		WPRINT("setting information done", "PychessGame", self._verbose)
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
			WPRINT("pushing move ...", "PychessGame", self._verbose)
			self._state.push(move)
			self._update_info(move)
			WPRINT("pushing and updating done", "PychessGame", self._verbose)
			return True


		return False


	def start_clock(self):
		""" public func
		@spec  start_clock(PychessGame)  =>  none
		"""
		WPRINT("starting clock", "PychessGame", self._verbose)
		self._info['time-prev-move'] = time.time()


	def get_info(self):
		""" public func
		@spec  get_info(PychessGame)  =>  dict
		"""
		WPRINT("getting information", "PychessGame", self._verbose)
		return self._info


	def get_state(self):
		""" public func
		@spec  get_state(PychessGame)  =>  chess.Board
		func simply gets the current state of the chess board from
		the PychessGame object. _state is wrapping the chess.Board
		object. 
		"""
		WPRINT("getting state", "PychessGame", self._verbose)
		return self._state


	def make_move(self, move):
		""" public func
		@spec  make_move(PychessGame, chess.Move)  =>  bool
		func takes a move formatted as a chess.Move object and tries to
		push it to the move stack, contained in self._state which is 
		a chess.Board type object. this is basically a wrapper function
		for _push_move and thus returns the save bool value. true if 
		pushing the move was legal/successfull or false if not.
		"""
		WPRINT("making move {}".format(move.uci()), "PychessGame", self._verbose)
		return self._push_move(move)


	def run(self):
		pass


class TestPychessGame(unittest.TestCase):
	"""!! definiton for class  TestPychessGame
	directly inheriting from unittest and overriding .TestCase __init__
	currently tests three functions from PychessGame class,
	_set_info
	_legal_moves
	_push_moves
	to make sure that they are correctly implemented in the base class.
	more test are prone to be written an will update docs in that case.
	"""
	def __init__(self, *args, **kwargs):
		super(TestPychessGame, self).__init__(*args, **kwargs)
		self._pg = PychessGame(2, white='ulysses', black='beq', verbose=False)
	

	def test_set_info(self):
		self.assertFalse(self._pg._set_info('bingbong', 2))
		self.assertTrue(self._pg._set_info('ai', False))


	def test_legal_moves(self):
		self.assertEqual(type(self._pg._legal_moves()), list)

	
	def test_push_move(self):
		self.assertFalse(self._pg._push_move(chess.Move.from_uci('a3b6')))
		self.assertTrue(self._pg._push_move(chess.Move.from_uci('e2e4')))


if __name__ == "__main__":
	unittest.main()
