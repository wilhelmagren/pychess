"""
PychessUtils file implementing diverse utility functions
used in various situations and instances of other Pychess*
instantiations. default values variables are set here
and the StdOutWrapper for PychessMode=TUI is also implemented
here as it is necessary for both PychessTUI and PychessGame.

Author: Wilhelm Ågren, wagren@kth.se
Last edited: 12-10-2021
"""
import os
import sys



"""!!! global variables
used in pychess.py when creating PychessMode
and user has not supplied all optional args.
"""
DEFAULT_WHITE		= 'white'
DEFAULT_BLACK		= 'black'
DEFAULT_TIME		= 300
DEFAULT_INCREMENT	= 5
WHITE_TO_PLAY		= "Whites move: "
BLACK_TO_PLAY		= "Blacks move: "


class StdOutWrapper:
    """!!! definition for class StdOutWrapper
    class is used for printing to terminal whenever
    PychessMode=TUI. the text-buffer is built up
    during program runtime until it terminates at which
    point the buffer is printed to standard output.
    """
    def __init__(self):
        self._text = ""


    def WPUT(self, msg, tpe, verbose):
        """ public func
        @spec  WPUT(StdOutWrapper, str, str, bool)  =>  none
        func takes a user created message, the type of
        class instance invoking this function and the
        verbosity setting of the invoking class.
        if verbose is true then the message is 
        appended to the buffer as a *working* string.
        """
        string = "[*]  {}  {}\n".format(tpe, msg)
        if verbose:
            self._text += string


    def EPUT(self, msg, tpe):
        """ public func
        @spec  EPUT(StdOutWrapper, str, str)  =>  none
        func takes a user created error message, the type
        of class instance invoking this function and appends
        the message to the buffer as a *error* string.
        """
        string = "[!]  {}  {}\n".format(tpe, msg)
        self._text += string


    def WRITE(self):
        """ public func
        @spec  WRITE(StdOutWrapper)  =>  none
        func writes every line of text in the buffer to the
        standard output, i.e. print, if nothing else is 
        specified. strips final newlines from buffer.
        """
        rows = self._text.split("\n")
        for row_idx, row in enumerate(rows):
            print(row) if row_idx != len(rows) - 1 else None



"""!!! global funcs
WSTRING(str, str, bool) =>  str
ESTRING(str, str)       =>  str
WPRINT(str, str, bool)  =>  none
EPRINT(str, str)        =>  none
"""
def WSTRING(msg, tpe, verbose):
    return "[*]  {}  {}".format(tpe, msg) if verbose else None


def ESTRING(msg, tpe):
    return "[!]  {}  {}".format(tpe, msg)


def WPRINT(msg, tpe, verbose):
    print("[*]  {}  {}".format(tpe, msg)) if verbose else None


def EPRINT(msg, tpe):
    print("[!]  {}  {}".format(tpe, msg))

