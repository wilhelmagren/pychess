"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 25/05-2021
"""
import pygame as pg
import os
import argparse
import time
import random
import math
import chess
from ada import Ada
from state import State


class Square(pg.sprite.Sprite):
    def __init__(self, piece_symbol, coordinates):
        super().__init__()
        self.image = pg.image.load(self.get_image(piece_symbol))
        self.rect = self.image.get_rect()
        self.rect.center = (coordinates[0] * 100 + 50, coordinates[1] * 100 + 50)

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


class Information(pg.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pg.image.load(os.path.join('../', 'images/info.png'))
        self.rect = self.image.get_rect()
        self.rect.center = (1000, 400)
        self.m2s = {True: 'White to move', False: 'Black to move'}
        self.to_move = True
        self.to_move_text = self.m2s[True]
        self.to_move_pos = (900, 50)
        self.white_time = 10
        self.black_time = 10

    def __update__(self, to_move):
        self.to_move = to_move
        self.to_move_text = self.m2s[to_move]


class Game(object):
    """
    800x800 chess board + 400x800 information
    """
    def __init__(self, WIDTH=800, HEIGHT=800, singleplayer=False, multiplayer=False, verbose=False):
        self.WIDTH, self.HEIGHT, self.sp, self.mp, self.verbose = WIDTH, HEIGHT, singleplayer, multiplayer, verbose
        self.state, self.moves,  = State(), [],
        self.idx_to_char = {
            0: 'a',
            1: 'b',
            2: 'c',
            3: 'd',
            4: 'e',
            5: 'f',
            6: 'g',
            7: 'h'
        }
        self.char_to_idx = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
            'h': 7
        }
        self.clock, self.screen, self.sprite_group, self.info, self.font = self.init_gameobjects()
        self.active_square, self.to_move = None, True
        self.engine = Ada()

    def init_gameobjects(self):
        clock = pg.time.Clock()
        pg_screen = pg.display.set_mode(size=(self.WIDTH, self.HEIGHT))
        pg.display.set_caption('Pepega chess')
        pg.display.set_icon(pg.image.load(os.path.join('../', 'images/icon.png')))
        sprite_group = pg.sprite.Group()
        info = Information()
        pg.font.init()
        font = pg.font.SysFont('Calibri', 35)
        pg.mixer.init()
        return clock, pg_screen, sprite_group, info, font

    def spunk_screen(self):
        self.screen.blit(pg.image.load(os.path.join('../', 'images/chessboard.png')), (0, 0))

    def update_spunk(self):
        self.sprite_group.empty()
        for piece in self.state.piecemap:
            for coordinate in self.state.piecemap[piece]:
                self.sprite_group.add(Square(piece_symbol=piece, coordinates=coordinate))
        self.sprite_group.add(self.info)

    def draw_info(self):
        self.screen.blit(self.font.render(self.info.to_move_text, True, (0, 0, 0)), self.info.to_move_pos)

    def spunk_all(self):
        self.state.update_map()
        self.spunk_screen()
        self.update_spunk()
        self.sprite_group.draw(self.screen)
        self.draw_info()
        pg.display.update()

    def update_turn(self):
        self.to_move = not self.to_move
        self.info.__update__(self.to_move)

    def parse_move(self, xf, yf, xto, yto):
        return chess.Move.from_uci(self.idx_to_char[xf] + str(8 - yf) + self.idx_to_char[xto] + str(8 - yto))

    @staticmethod
    def mouse_on_square(mx, my, sx, sy):
        return sx - 50 <= mx <= sx + 50 and sy - 50 <= my <= sy + 50

    @staticmethod
    def get_from_idx(coordinates):
        x, y = coordinates
        if 0 <= x <= 800 and 0 <= y <= 800:
            return int(math.floor(x / 100)), int(math.floor(y / 100))

    def play_sounds(self):
        if self.state.board.is_check():
            pg.mixer.music.load(os.path.join('../', 'sounds/hitmarker.mp3'))
        else:
            pg.mixer.music.load(os.path.join('../', 'sounds/move.wav'))
        pg.mixer.music.play(0)

    def highlight_moves(self, c):
        fromx, fromy = c
        possible_moves = self.state.branches()
        to_squares = []
        for move in possible_moves:
            s = list(str(move))
            if self.idx_to_char[fromx] == s[0] and (8 - fromy) == int(s[1]):
                to_squares.append((self.char_to_idx[s[2]], int(s[3])))
        for square in to_squares:
            self.screen.blit(pg.image.load(os.path.join('../', 'images/highlight_move.png')), [100*square[0], 100*(8 - square[1])])

    def get_to_idx(self, c):
        mx, my = c
        for square in self.sprite_group:
            [sx, sy] = square.rect.center
            if self.mouse_on_square(mx, my, sx, sy):
                self.screen.blit(pg.image.load(os.path.join('../', 'images/highlight.png')), [sx - 50, sy - 50])
                # Highlight possible moves
                self.highlight_moves(self.get_from_idx(c))
                pg.display.update()
                pg.event.set_blocked(None)
                pg.event.set_allowed(pg.MOUSEBUTTONUP)
                pg.event.set_allowed(pg.QUIT)
                event = pg.event.wait()
                pg.event.set_allowed(None)
                return self.get_from_idx(pg.mouse.get_pos())
        return False, False

    def make_move(self, coordinates):
        # just make a move
        xto, yto = self.get_to_idx(coordinates)

        if xto is False and yto is False:
            return False

        mx, my = self.get_from_idx(coordinates)
        if (mx, my) == (xto, yto):
            print(f'{time.asctime()}  ::  WARNING! Incorrect move, can not move to starting position...')
            return False

        move = self.parse_move(mx, my, xto, yto)
        if move in self.state.branches():
            self.state.board.push(move)
            self.state.update_map()
            self.moves.append(self.state.board.peek())
            return True
        return False

    def make_computer_move(self, player):
        # just make a move
        try:
            move = random.choice(self.state.branches())
            self.state.board.push(move)
            if self.verbose:
                print(f'\n{time.asctime()}  ::  player {player} made move {move}, ')
                print(self.state.board)
        except:
            return

    def play_hvh(self):
        self.clock.tick(60)
        while not self.state.board.is_game_over():
            self.spunk_all()
            for event in pg.event.get():
                if event.type == pg.MOUSEBUTTONUP:
                    valid_move = self.make_move(coordinates=pg.mouse.get_pos())
                    if valid_move:
                        self.play_sounds()
                        print(f'{time.asctime()}  ::  valid move, {self.moves[-1]}')
                        self.update_turn()
                if event.type == pg.QUIT:
                    print(f'{time.asctime()}  ::  PLAYER INTERRUPT, terminating process...')
                    exit(1)
        self.game_over(self.state.board)

    def play_hvc(self):
        self.clock.tick(60)
        player_move = False
        while not self.state.board.is_game_over():
            self.spunk_all()
            for event in pg.event.get():
                if event.type == pg.MOUSEBUTTONUP:
                    valid_move = self.make_move(coordinates=pg.mouse.get_pos())
                    if valid_move:
                        self.play_sounds()
                        print(f'{time.asctime()}  ::  valid move, {self.moves[-1]}')
                        self.update_turn()
                        player_move = True
                if event.type == pg.QUIT:
                    print(f'{time.asctime()}  ::  PLAYER INTERRUPT, terminating process...')
                    exit(1)
            if player_move:
                self.spunk_all()
                self.make_computer_move(player='B')
                self.update_turn()
                player_move = False
        self.game_over(self.state.board)

    def play_cvc(self):
        self.clock.tick(60)
        while not self.state.board.is_game_over():
            self.spunk_all()
            time.sleep(1)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    print(f'{time.asctime()}  ::  PLAYER INTERRUPT, terminating process...')
                    exit(1)
            self.make_computer_move(player='W')
            self.spunk_all()
            time.sleep(1)
            self.make_computer_move(player='B')
        self.game_over(self.state.board)

    @staticmethod
    def game_over(board):
        print(f'{time.asctime()}  ::  GAME OVER, result is {board.result()}')
        exit(1)


if __name__ == '__main__':
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    parser = argparse.ArgumentParser(description='CLI chess options')
    parser.add_argument('-c', '--computer', action='store_true', default=False,
                        help='let computer play against herself')
    parser.add_argument('-s', '--singleplayer', action='store_true', default=False, help='specify to play against AI')
    parser.add_argument('-m', '--multiplayer', action='store_true', default=False,
                        help='specify to play against a friend')
    parser.add_argument('-W', '--width', action='store_true', default=1200, help='width of gfx window')
    parser.add_argument('-H', '--height', action='store_true', default=800, help='height of gfx window')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='specify verbose printing')
    args = parser.parse_args()

    if not args.singleplayer and not args.multiplayer and not args.computer:
        print(f'{time.asctime()}  ::  ERROR! no game-mode specified, either \'-s\', \'c\', or \'-m\' required...')
        exit()

    gobject = Game(WIDTH=args.width, HEIGHT=args.height, singleplayer=args.singleplayer, multiplayer=args.multiplayer,
                   verbose=args.verbose)
    if args.computer:
        gobject.play_cvc()
    else:
        if args.singleplayer:
            gobject.play_hvc()
        elif args.multiplayer:
            gobject.play_hvh()
