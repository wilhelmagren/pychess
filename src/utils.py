import os
import sys


DEFAULT_WHITE		= 'white'
DEFAULT_BLACK		= 'black'
DEFAULT_TIME		= 300
DEFAULT_INCREMENT	= 5
WHITE_TO_PLAY		= "Whites move: "
BLACK_TO_PLAY		= "Blacks move: "


def WPRINT(msg, tpe, verbose):
	print("[*]  {}  ".format(tpe) + msg) if verbose else None


def EPRINT(msg, tpe):
	print("[!]  {}  ".format(tpe) + msg)