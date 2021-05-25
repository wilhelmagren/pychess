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


class Game(object):
    def __init__(self, WIDTH=800, HEIGHT=800, singleplayer=False, multiplayer=False, verbose=False):
        self.WIDTH, self.HEIGHT, self.sp, self.mp, self.verbose = WIDTH, HEIGHT, singleplayer, multiplayer, verbose
        self.state = State()
        self.clock, self.screen, self.sprite_group = self.init_gameobjects()

    def init_gameobjects(self):
        clock = pg.time.Clock()
        pg_screen = pg.display.set_mode(size=(self.WIDTH, self.HEIGHT))
        pg.display.set_caption('pyChess')
        pg.display.set_icon(pg.image.load(os.path.join('../', 'images/icon.png')))
        sprite_group = pg.sprite.Group()
        return clock, pg_screen, sprite_group

    def spunk_screen(self):
        self.screen.blit(pg.image.load(os.path.join('../', 'images/board.png')))

    def spunk_all(self):
        self.spunk_screen()
        self.update_spunk()

    def update_spunk(self):
        pass

    def make_move(self, player):
        # just make a move
        move = self.state.branches()[0]
        self.state.board.push(move)
        if self.verbose:
            print(f'{time.asctime()} :: player {player} made move {move}, ')
            print(self.state.board)

    def play(self):
        while not self.state.board.is_game_over():
            time.sleep(1)
            # player 1 move
            self.make_move(player='W')
            self.update_spunk()
            pg.display.update()
            # player 2 move
            self.make_move(player='B')
            self.update_spunk()
            pg.display.update()
        print(f'{time.asctime()} ::  GAME OVER, result is {self.state.board.result()}')


if __name__ == '__main__':
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    parser = argparse.ArgumentParser(description='CLI chess options')
    parser.add_argument('-s', '--singleplayer', action='store_true', default=False, help='specify to play against AI')
    parser.add_argument('-m', '--multiplayer', action='store_true', default=False, help='specify to play against a friend')
    parser.add_argument('-W', '--width', action='store_true', default=800, help='width of gfx window')
    parser.add_argument('-H', '--height', action='store_true', default=800, help='height of gfx window')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='specify verbose printing')
    args = parser.parse_args()

    if not args.singleplayer and not args.multiplayer:
        print(f'{time.asctime()}  ::  ERROR! no game-mode specified, either \'-1\' or \'-2\' required...')
        exit()

    gobject = Game(WIDTH=args.width, HEIGHT=args.height, singleplayer=args.singleplayer, multiplayer=args.multiplayer, verbose=args.verbose)
    gobject.play()

