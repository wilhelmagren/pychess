import curses

from .game 	import PychessGame
from .utils	import *


class PychessTUI:
	def __init__(self, players, names, verbose=False, **kwargs):
		self._game 		= None
		self._mode 		= 'tui'
		self._players 	= players
		self._names		= names
		self._verbose 	= verbose
		self._screen	= None
		self._clock		= False
		self._terminal  = False


	def _initscreen(self):
		self._screen = curses.initscr()
		curses.echo()


	def _blit(self):
		#!!! reset the screen
		self._screen.clear()

		#!!! draw the board state 
		# starting  at  ( 2,  2)
		# stretching to (10, 16)
		for y, row in enumerate(str(self._game.get_state()).split("\n")):
			self._screen.addstr(2+y, 3, row)
		#!!! ===================================

		#!!! draw the borders of the board state 
		for y in range(2, 11):
			self._screen.addstr(y,  2, "|")
			self._screen.addstr(y, 18, "|")
		for x in range(3, 18):
			self._screen.addstr(1,  x, "_")
			self._screen.addstr(10, x, "_")
		self._screen.addstr( 1,  2, "_")
		self._screen.addstr( 1, 18, "_")
		self._screen.addstr(10,  2, "|")
		self._screen.addstr(10, 18, "|")
		#!!! ===================================

		#!!! draw the previous move
		prev_move 	= self._game.get_prev_move()
		self._screen.addstr(12, 1, "previous move: {}".format(prev_move if type(prev_move) == str else prev_move.uci()))
		#!!! ===================================

		#!!! draw the time format
		self._screen.addstr(13, 1, "time format: " + self._game.get_info('time-format'))
		#!!! ===================================

		#!!! draw the player times
		w_m, w_s = divmod(self._game.get_info('time-white'), 60)
		b_m, b_s = divmod(self._game.get_info('time-black'), 60)
		self._screen.addstr(4, 24, "white time:  {}:{}  ".format(w_m, w_s))
		self._screen.addstr(5, 24, "black time:  {}:{}  ".format(b_m, b_s))
		self._screen.addstr(6, 24, "{}".format(WHITE_TO_PLAY if self._game.get_turn() else BLACK_TO_PLAY))
		self._screen.addstr(7, 24, "> ")
		#!!! ===================================

		#!!! draw outcome of game if game is terminal state
		if self._terminal:
			self._screen.addstr(10, 24, "GAME OVER: {}".format(self._game.get_outcome_str()))
			self._screen.addstr(11, 24, "Start a new game? [Y/n]")
			self._screen.addstr(12, 24, "> ")


		self._screen.refresh()
		curses.napms(100)

	def _blit_quit(self):
		#!!! reset the screen
		self._screen.clear()

		#!!! draw the goodbye text
		self._screen.addstr(9, 15, "Thanks for playing Pychess in TUI mode! Bye bye ...")
		self._screen.refresh()
		curses.napms(2000)


	def _get_and_push_move(self):
		move = self._screen.getstr(7, 26).decode()
		if self._clock is False:
			self._game.start_clock()
			self._clock = True
		self._game.make_move(move) 


	def _query_new_game(self):
		self._terminal = True
		self._blit()
		resp = self._screen.getstr(12, 27).decode()
		if resp == 'Y':
			# Start a new game
			self._restart()
		else:
			self._blit_quit()
			return


	def _restart(self):
		self._game 		= PychessGame(players=self._players, verbose=self._verbose, white=self._names[0], black=self._names[1])
		self._clock 	= False
		self._terminal 	= False
		self._run(False)


	def _run(self, f_game):
		self._initscreen() if f_game else None
		while not self._game.is_terminal():
			self._blit()
			self._get_and_push_move()
		self._query_new_game()
		WPRINT("game is done, cleaning up and terminating ...", "PychessTUI\t", True)
		curses.endwin()


	def start(self):
		WPRINT("creating new game instance", "PychessTUI\t", True)
		try:
			self._game = PychessGame(players=self._players, verbose=self._verbose, white=self._names[0], black=self._names[1])
		except:
			EPRINT("could not create new game instance, terminating ...", "PychessTUI\t")
			return
		self._run(True)
