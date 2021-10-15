"""
PychessTUI class implementation for main module Pychess.
Initialized from pychess main file based on user input
parsed with argpase CLI. user has to set mode=tui when
running to start this Terminal User Interface mode.

Author: Wilhelm Aagren, wagren@kth.se
Last edited: 12-10-2021
"""
import curses

from .mode      import PychessMode
from .game      import PychessGame
from .utils     import *



class PychessTUI(PychessMode):
    """!!! definition for class  PychessTUI
    class directly inherits from mother PychessMode which wraps all
    generic game information attributes, and mode specific attributes
    are implemented in this class. see mode.py for more details.

    used as Terminal User Interface mode for Pychess. TUI is implemented
    using a standard library in Python3.9 called 'curses' which allows
    direct printing (x,y) location on the terminal. see original
    documentation for more information. the PychessTUI is initialized
    from Pychess when user directs mode tui on the CLI. this object
    creates a screen buffer to write to and manages it. main game
    loop is performed in private function _run and only leaves
    whenever a game is over, and user doesn't want to play more,
    or the user issues a SIGINT with ctrl+c. due to the TUI
    taking up the terminal interface no debug/verbosity printing
    is available at run time, and is only shown whenever the 
    initialized curses windows is closed.

    public  funcs:
        $  PychessTUI.start                 =>  none

    private funcs:
        $  PychessTUI._initscreen           =>  none
        $  PychessTUI._blit                 =>  none
        $  PychessTUI._blit_quit            =>  none
        $  PychessTUI._get_and_push_move    =>  none
        $  PychessTUI._query_new_game       =>  none
        $  PychessTUI._restart              =>  none
        $  PychessTUI._run                  =>  none

    dunder  funcs:
        $  PychessTUI.__init__              =>  PychessTUI

    """
    def __init__(self, players, names, verbose=False, **kwargs):
        super().__init__(players, names, verbose=verbose, mode='PychessTUI')
        self._screen    = None
        self._screendic = dict()
        self._stdout    = StdOutWrapper()
        self._kwargs    = kwargs


    def _initscreen(self):
        """ private func
        @spec  _initscreen(PychessTUI)  =>  none

        """
        self._stdout.WPUT("initializing screen with curses", str(self), self._verbose)
        self._screen = curses.initscr()
        curses.echo()
        height, width = self._screen.getmaxyx()
        if height < 30 or width < 60:
            self._stdout.EPUT("running in too small terminal, please resize to atleast (30x60)", str(self))
            curses.endwin()
            raise ValueError
        
        if width  % 2 == 0:
            width -= 1
        if height % 2 == 0:
            height -= 1

        self._screendic['width']        = width
        self._screendic['height']       = height
        self._screendic['mid-width']    = 1 + width  // 2
        self._screendic['mid-height']   = 1 + height // 2
        self._screendic['board-width']  = 15
        self._screendic['board-height'] = 8
        self._screendic['board-start']  = (self._screendic['mid-height'] - self._screendic['board-height'], self._screendic['mid-width'] - self._screendic['board-width'])
        self._screendic['prev-move']    = (self._screendic['mid-height'] + 1, self._screendic['mid-width'] - self._screendic['board-width'])
        self._screendic['time-format']  = (self._screendic['mid-height'] + 2, self._screendic['mid-width'] - self._screendic['board-width'])
        self._screendic['white-time']   = (self._screendic['mid-height'] - 6, self._screendic['mid-width'] + 2)
        self._screendic['black-time']   = (self._screendic['mid-height'] - 5, self._screendic['mid-width'] + 2)
        self._screendic['to_move']      = (self._screendic['mid-height'] - 4, self._screendic['mid-width'] + 2)
        self._screendic['query-move']   = (self._screendic['mid-height'] - 3, self._screendic['mid-width'] + 2)
        self._screendic['player-names'] = (self._screendic['mid-height'] - self._screendic['board-height'] - 2, self._screendic['mid-width'] - self._screendic['board-width'])
        self._screendic['quitting']     = (self._screendic['mid-height'], self._screendic['mid-width'] - 30)
        self._stdout.WPUT("done initializing screen", str(self), self._verbose)


    def _blit(self):
        """ private func
        @spec  _blit(PychessTUI)  =>  none
        func puts the game state, player times etc.
        on to the screen using self._screen and 
        curses library. two possible states for
        blitting to the screen: either chess.Board
        is terminal, or it is not. if it is terminal
        then the player(s) will be queried if they
        want to play another game [Y/n].
        """
        self._stdout.WPUT("writing to screen", str(self), self._verbose)
        #!!! reset the screen
        self._screen.clear()

        #!!! draw the board state 
        b_y, b_x = self._screendic['board-start']
        for y, row in enumerate(str(self._game.get_state()).split("\n")):
            self._screen.addstr(b_y + y, b_x, row)

        #!!! draw the previous move
        prev_move = self._game.get_prev_move()
        p_y, p_x = self._screendic['prev-move']
        self._screen.addstr(p_y, p_x, "previous move: {}".format(prev_move if type(prev_move) == str else prev_move.uci()))

        #!!! draw the time format
        t_y, t_x = self._screendic['time-format']
        self._screen.addstr(t_y, t_x, "time format: " + self._game.get_info('time-format'))

        #!!! draw the player times
        w_m, w_s = divmod(self._game.get_info('time-white'), 60)
        b_m, b_s = divmod(self._game.get_info('time-black'), 60)
        wt_y, wt_x = self._screendic['white-time']
        bt_y, bt_x = self._screendic['black-time']
        self._screen.addstr(wt_y, wt_x, "white time:  {}:{}  ".format(w_m, w_s))
        self._screen.addstr(bt_y, bt_x, "black time:  {}:{}  ".format(b_m, b_s))
        m_y, m_x = self._screendic['query-move']
        self._screen.addstr(m_y, m_x, "> ")
      
        #!!! draw the player names
        n_y, n_x = self._screendic['player-names']
        name1, name2 = self._game.get_info('white'), self._game.get_info('black')
        self._screen.addstr(n_y, n_x, "(W) {}  vs  (B) {}".format(name1, name2))

        #!!! draw outcome of game if game is terminal state
        if self._terminal:
            self._screen.addstr(n_y - 1, n_x, "GAME OVER: {}".format(self._game.get_info('winner')))
            self._screen.addstr(bt_y + 1, bt_x, "Start a new game? [Y/n]")
        else:
            self._screen.addstr(bt_y + 1, bt_x, "{}".format(WHITE_TO_PLAY if self._game.get_info('turn') else BLACK_TO_PLAY))

        self._screen.refresh()
        curses.napms(100)

    def _blit_quit(self):
        """ private func
        @spec  _blit_quit(PychessTUI)  =>  none
        func draws the quitting information to the screen
        using the curses library and self._screen
        only happens if user doesn't want to play another 
        game after terminal outcome, or user issues
        SIGINT, i.e. CTRL+C when running the application.
        """
        self._stdout.WPUT("writing termination message to screen", str(self), self._verbose)
        #!!! reset the screen
        self._screen.clear()

        #!!! draw the goodbye text
        t_y, t_x = self._screendic['quitting']
        self._screen.addstr(t_y - 5, t_x, "Thanks for playing Pychess using the Terminal User Interface!")
        self._screen.addstr(t_y - 4, t_x, "                    $ sudo rm -rf")
        self._screen.refresh()
        curses.napms(3000)


    def _get_and_push_move(self):
        """ private func
        @spec  _get_and_push_move(PychessTUI)  =>  none
        func reads a string from the user and tries
        to push it to the chess.Board if it is legal.
        if there is no move made yet then the clock is off,
        false, and thus we have to start it.
        """
        self._stdout.WPUT("getting user move", str(self), self._verbose)
        q_y, q_x = self._screendic['query-move']
        move = self._screen.getstr(q_y, q_x + 2).decode()
        if self._clock is False:
            self._stdout.WPUT("starting the game clock", str(self), self._verbose)
            self._game.start_clock()
            self._clock = True
        self._stdout.WPUT("trying to push user move ...", str(self), self._verbose)
        self._game.make_move(move) 


    def _query_new_game(self):
        """ private func
        @spec  _query_new_game(PychessTUI)  =>  none
        func queries the user for playing a new game,
        either [Y/n], in which case the entire
        PychessGame object is reinitialized for 
        cleaning up all put values.
        """
        self._stdout.WPUT("quering user for new game [Y/n]?", str(self), self._verbose)
        self._terminal = True
        self._blit()
        q_y, q_x = self._screendic['query-move']
        resp = self._screen.getstr(q_y, q_x + 2).decode()
        if resp == 'Y':
            # Start a new game
            self._restart()
        else:
            self._quit()
            return


    def _restart(self):
        """ private func
        @spec  _restart(PychessTUI)  =>  none
        func simply restarts the chess.Board game
        if user queries it. only called from
        _query_new_game
        """
        self._stdout.WPUT("creating new game instance and restarting", str(self), self._verbose)
        self._game = PychessGame(players=self._players, verbose=self._verbose, white=self._kwargs.get('white'), black=self._kwargs.get('black'), stdout=self._stdout, time=self._kwargs.get('time'), increment=self._kwargs.get('increment'))
        self._clock     = False
        self._terminal  = False
        self._run(False)


    def _quit(self):
        """ private func
        @spec  _quit(PychessTUI)  =>  none
        func blits the quitting information to 
        the screen and terminates the initialized
        curses window instance. flushes the 
        stdoutwrapper buffer to the terminal
        once the window is terminated.
        """
        self._stdout.WPUT("quitting the program", str(self), self._verbose)
        self._blit_quit()
        curses.endwin()
        self._stdout.WRITE()


    def _run(self, f_game):
        """ private func
        @spec  _run(PychessTUI, bool)  =>  none
        func is the main running loop for the game.
        handles looping until terminal game, and
        then queries the user for a new game or 
        termination. any exception in this function
        is catched and will cause the program to quit.
        """
        try:
            self._initscreen() if f_game else None
        except ValueError:
            self._stdout.EPUT("ValueError exception caught in self._initscreen", str(self))
            self._stdout.WRITE()
            return
        try:
            self._initscreen() if f_game else None
            while not self._game.is_terminal():
                self._blit()
                self._get_and_push_move()
            self._query_new_game()
            self._stdout.WPUT("removing instance of self, exiting", str(self), self._verbose)
            curses.endwin()
        except KeyboardInterrupt:
            self._stdout.EPUT("KeyboardInterrupt exception SIGINT caught", str(self))
        self._quit()


    def start(self):
        """ public func
        @spec  start(PychessTUI)  =>  none
        func is called from main running file
        (pychess.py) if the user has requested a
        TUI instance. if PychessGame throws any exception
        when initializing the function will catch it and
        terminate the program. no retrying is made here,
        user will have to restart program and try again.
        """
        self._stdout.WPUT("creating new game instance", str(self), self._verbose)
        try:
            self._game = PychessGame(players=self._players, verbose=self._verbose, white=self._kwargs.get('white'), black=self._kwargs.get('black'), stdout=self._stdout, time=self._kwargs.get('time'), increment=self._kwargs.get('increment'))
        except:
            self._stdout.EPUT("could not create new game instance, exiting ...", str(self))
            self._stdout.WRITE()
            return
        self._run(True)
