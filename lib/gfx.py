"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 25/05-2021
"""
import pygame as pg
import os
import argparse
import time
from state import State


class Square(pg.sprite.Sprite):
    def __init__(self, piece_symbol, coordinates):
        super().__init__()
        self.image = pg.image.load(self.get_image(piece_symbol))
        self.rect = self.image.get_rect()
        self.rect.center = (coordinates[0]*100 + 50, coordinates[1]*100 + 50)

    @staticmethod
    def get_image(symbol):
        if symbol == 'R':
            return os.path.join('../', 'images/wr.png')
        if symbol == 'r':
            return os.path.join('../', 'images/br.png')
        if symbol == 'N':
            return os.path.join('../', 'images/wn.png')
        if symbol == 'n':
            return os.path.join('../', 'images/bn.png')
        if symbol == 'B':
            return os.path.join('../', 'images/wb.png')
        if symbol == 'b':
            return os.path.join('../', 'images/bb.png')
        if symbol == 'Q':
            return os.path.join('../', 'images/wq.png')
        if symbol == 'q':
            return os.path.join('../', 'images/bq.png')
        if symbol == 'K':
            return os.path.join('../', 'images/wk.png')
        if symbol == 'k':
            return os.path.join('../', 'images/bk.png')
        if symbol == 'P':
            return os.path.join('../', 'images/wp.png')
        if symbol == 'p':
            return os.path.join('../', 'images/bp.png')


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
        self.screen.blit(pg.image.load(os.path.join('../', 'images/chessboard.png')), (0, 0))

    def update_spunk(self):
        self.sprite_group.empty()
        for piece in self.state.piecemap:
            for coordinate in self.state.piecemap[piece]:
                self.sprite_group.add(Square(piece_symbol=piece, coordinates=coordinate))

    def spunk_all(self):
        self.spunk_screen()
        self.update_spunk()
        self.sprite_group.draw(self.screen)
        pg.display.update()

    def make_move(self, player):
        # just make a move
        move = self.state.branches()[0]
        self.state.board.push(move)
        if self.verbose:
            print(f'{time.asctime()} :: player {player} made move {move}, ')
            print(self.state.board)

    def play(self):
        self.clock.tick(60)
        while not self.state.board.is_game_over():
            self.spunk_all()
            time.sleep(100)
            for event in pg.event.get():
                if event.type == pg.MOUSEBUTTONUP:
                    print(f'{time.asctime()}  ::  got event mousebuttonup')
            # player 1 move
            self.make_move(player='W')
            self.spunk_all()
            # player 2 move
            self.make_move(player='B')
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

