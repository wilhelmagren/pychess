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


	def _blit(self, white_off_t, black_off_t):
		self._screen.clear()
		# Draw the borders
		for x in range(1, 50):
			self._screen.addstr(0,  x, "_")
			self._screen.addstr(12, x, "_")
		for y in range(1, 12):
			self._screen.addstr(y,  0, "|")
			self._screen.addstr(y, 50, "|")

		self._screen.addstr(0,   0, "+")
		self._screen.addstr(0,  50, "+")
		self._screen.addstr(12,  0, "+")
		self._screen.addstr(12, 50, "+")

		# Draw the board state
		for y, row in enumerate(str(self._game.get_state()).split("\n")):
			self._screen.addstr(1+y, 2, row)

		# Draw previous move
		prev_move 	= self._game.get_prev_move()
		self._screen.addstr(10, 2, "previous move: {}".format(prev_move if type(prev_move) == str else prev_move.uci()))

		# Draw time format
		t_format 	= self._game.get_info('time-start')
		t_increment = self._game.get_info('time-increment')
		self._screen.addstr(11, 2, "time format: {}s + {}".format(t_format, t_increment))

		# Draw the border of board state
		for y in range(1, 10):
			self._screen.addstr(y, 18, "|")
		for x in range(1, 17):
			self._screen.addstr(9, x, "_")

		# Draw the player times
		w_m, w_s = divmod(self._game.get_info('time-white') - white_off_t, 60)
		b_m, b_s = divmod(self._game.get_info('time-black') - black_off_t, 60)
		self._screen.addstr(3, 26, "white time:  {}:{}  ".format(w_m, w_s))
		self._screen.addstr(4, 26, "black time:  {}:{}  ".format(b_m, b_s))
		self._screen.addstr(5, 26, "{}".format(WHITE_TO_PLAY if self._game.get_turn() else BLACK_TO_PLAY))
		self._screen.addstr(6, 26, "> ")
		

		self._screen.refresh()
		curses.napms(100)


	def _get_and_push_move(self):
		move = self._screen.getstr(6, 28).decode()
		if self._clock is False:
			self._game.start_clock()
			self._clock = True
		self._game.make_move(move) 


	def _run(self):
		self._screen = curses.initscr()
		curses.echo()
		while not self._game.is_terminal():
			self._blit(0, 0)
			self._get_and_push_move()
		WPRINT("game is done, cleaning up and terminating ...", "PychessTUI\t", True)
		curses.endwin()


	def start(self):
		WPRINT("creating new game instance", "PychessTUI\t", True)
		try:
			self._game = PychessGame(players=self._players, verbose=self._verbose, white=self._names[0], black=self._names[1])
		except:
			EPRINT("could not create new game instance, terminating ...", "PychessTUI\t")
		self._run()
