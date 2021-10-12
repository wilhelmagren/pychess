import os
import sys


DEFAULT_WHITE		= 'white'
DEFAULT_BLACK		= 'black'
DEFAULT_TIME		= 300
DEFAULT_INCREMENT	= 5
WHITE_TO_PLAY		= "Whites move: "
BLACK_TO_PLAY		= "Blacks move: "



class StdOutWrapper:
    """
    ye this is the stdoutwrapper
    """
    def __init__(self):
        self._text = ""

    def put(self, msg):
        """
        """
        if msg is None:
            return
        self._text += msg + "\n"

    def write(self):
        """
        """
        rows = self._text.split("\n")
        for row_idx, row in enumerate(rows):
            print(row) if row_idx != len(rows) - 1 else None


"""
"""
def WSTRING(msg, tpe, verbose):
    return "[*]  {}  {}".format(tpe, msg) if verbose else None


def WPRINT(msg, tpe, verbose):
    print("[*]  {}  {}".format(tpe, msg)) if verbose else None


def ESTRING(msg, tpe):
    return "[!]  {}  {}".format(tpe, msg)

def EPRINT(msg, tpe):
    print("[!]  {}  {}".format(tpe, msg))
