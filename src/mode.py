"""
PychessMode wrapper close for TUI and GUI mode.
Since both modes are prone to be instantiated 
when running the CLI from pychess.py it is 
necessary to create this wrapper for correct
typing. The PychessTUI and PychessGUI objects
directly inherit from this main class,
which contains the generic information for the 
game. TUI contains attributes regarding
'curses' screen and StdOutWrapper, whereas
the GUI contains attributes regarding 
Qt5/pygame2 instances.

TODO:   implement generic funcs used by both modes
        in this parent class.

Author: Wilhelm Ågren, wagren@kth.se
Last edited: 12-10-2021
"""


class PychessMode:
    def __init__(self, players, names, verbose=False, mode='none', **kwargs):
        self._mode      = mode
        self._players   = players
        self._names     = names
        self._verbose   = verbose
        self._game      = None
        self._clock     = False
        self._terminal  = False

