"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 25/05-2021
"""
import pygame as pg
import sys
import os
import argparse
import time
from state import State
import chess


class Game:
    def __init__(self, singleplayer=False, multiplayer=False):
        self.sp, self.mp = singleplayer, multiplayer


if __name__ == '__main__':
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    parser = argparse.ArgumentParser(description='CLI chess options')
    parser.add_argument('-1', '--singleplayer', action='store_true', default=False, help='specify to play against AI')
    parser.add_argument('-2', '--multiplayer', action='store_true', default=False, help='specify to play against a friend')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='specify verbose printing')
    args = parser.parse_args()

    if not args.singleplayer and not args.multiplayer:
        print(f'{time.asctime()}  ::  ERROR! no mode specified, either \'-1\' or \'-2\' required...')
        exit()

    gobject = Game(singleplayer=args.singleplayer, multiplayer=args.multiplayer)

