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
	"""
	def __init__(self):
		self._text = ""

	def put(self, msg):
		"""
		"""
		self._text += "\n" + msg

	def write(self):
		"""
		"""
		for row in self._text.split("\n"):
			print(row)


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