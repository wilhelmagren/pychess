import curses

from .pychess_game 	import PychessGame
from .pychess_utils	import *


class PychessCLI:
	def __init__(self, players, names, verbose=False, **kwargs):
		self._game 		= None
		self._mode 		= 'cli'
		self._players 	= players
		self._names		= names
		self._verbose 	= verbose
		self._screen	= None
		self._clock		= False


	def _blit(self, white_off_t, black_off_t):
		self._screen.clear()
		for x in range(60):
			self._screen.addstr(0, x, "$")
			self._screen.addstr(25, x, "$")
			self._screen.addstr(16, x, "=")
		for y in range(26):
			self._screen.addstr(y, 0, "$")
			self._screen.addstr(y, 60, "$")

		for y, row in enumerate(str(self._game.get_state()).split("\n")):
			self._screen.addstr(4+y, 4, row)

		w_m, w_s = divmod(self._game.get_info('time-white') - white_off_t, 60)
		b_m, b_s = divmod(self._game.get_info('time-black') - black_off_t, 60)
		for x in range(21):
			self._screen.addstr(6, 30 + x, "_")
			self._screen.addstr(12, 30 + x, "_")

		prev_move 	= self._game.get_prev_move()
		self._screen.addstr(14, 3, "| previous move: {}".format(prev_move if type(prev_move) == str else prev_move.uci()))
		t_format 	= self._game.get_info('time-start')
		t_increment = self._game.get_info('time-increment')
		self._screen.addstr(15, 3, "| time format: {}s + {}".format(t_format, t_increment))

		self._screen.addstr(7, 30, "| white time:  {}:{}  ".format(w_m, w_s))
		self._screen.addstr(8, 30, "| black time:  {}:{}  ".format(b_m, b_s))
		self._screen.addstr(9, 30, "|")
		self._screen.addstr(7, 50, "|")
		self._screen.addstr(8, 50, "|")
		self._screen.addstr(9, 50, "|")
		self._screen.addstr(10, 50, "|")
		self._screen.addstr(11, 50, "|")

		self._screen.addstr(10, 30, "| {}".format(WHITE_TO_PLAY if self._game.get_turn() else BLACK_TO_PLAY))
		self._screen.addstr(11, 30, "| > ")
		

		self._screen.refresh()
		curses.napms(100)


	def _get_and_push_move(self):
		move = self._screen.getstr(11, 34).decode()
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
		WPRINT("game is done, cleaning up and terminating ...", "PychessCLI\t", True)
		curses.endwin()


	def start(self):
		WPRINT("creating new game instance", "PychessCLI\t", True)
		try:
			self._game = PychessGame(players=self._players, verbose=self._verbose, white=self._names[0], black=self._names[1])
		except:
			EPRINT("could not create new game instance, terminating ...", "PychessCLI\t")
		self._run()
