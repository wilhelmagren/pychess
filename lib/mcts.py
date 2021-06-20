"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 19/06-2021

    "Considers environments by a finite Markov Decision Process (MPD), as a tuple <S, A, T, R, gamma>
    over states S, actions A, transition function T : S x A x S -> [0, 1], reward function R : S x A x S -> |R,
    and a discount factor gamma in (0, 1). The aim is to learn a policy pi : S x A -> [0, 1] which specifies
    the probability of executing an action a in A given state s in S so as to maximize r in R.
    Suppose that after time-step t we observe the sequence of rewards r_t+1, r_t+2, r_t+3,...
    The expected return |E[Rt] is then the discounted sum of rewards, where Rt = SUM{gamma^i * r_t+i+1}.

    For a given policy pi, we can calculate the value of any state s as the expected reward,
        V^pi (s) = |E_pi [Rt | s_t = s].
    A policy pi* is said to be optimal if,
        all s in S, V^pi (s) = V (s) = max_pi V^pi(s)."
        - Steven James et al. 'An Analysis of Monte Carlo Tree Search' February 2017
"""


import copy
import math
import time
import torch
import chess
import random
import numpy as np
from net import ChessNet


class MCTS(object):
    """
    Monte Carlo Tree Search object, called with current chess state/board and arbitry keyword args.
    MCTS iteratively builds a search tree by executing four phases.
    1. Selection, 2. Expansion, 3. Simulation, 4. Backpropagation
    """
    def __init__(self, root, **kwargs):
        self.root, self.node_history = root, [root]
        self.result_record, self.state_record = dict(), dict()
        self.gamma = kwargs.get('discount', 1.4)
        self.calculation_time = kwargs.get('time', 30)
        self.max_moves = kwargs.get('moves', 100)
        self.result_record[(root.player, root.fen)], self.state_record[(root.player, root.fen)] = 0, 0
        self.net = ChessNet()
        self.net.load_state_dict(torch.load('../nets/ChessNet.pth', map_location=lambda storage, loc: storage))

    def __str__(self):
        s = ''
        s += ' | Num states explored: {}\n'.format(len(self.state_record))
        s += ' | Result and state record:\n'
        for rplayer, rstate in self.result_record:
            s += ' | ({}, {})\t\t=> ratio: {:.2f}\n'.format(rplayer, rstate, self.result_record[(rplayer, rstate)]/self.state_record[(rplayer, rstate)])
        return s

    def __del__(self):
        print('-+----- deleting MCTS class instance -------')

    def treesearch(self, maxsec):
        starttime = time.time()
        while time.time() - starttime <= maxsec:
            selected, node, path = False, self.node_history[-1], [(n.player, n.fen) for n in self.node_history]
            path.append((node.player, node.fen))
            prev_node = node
            # 1. Selection
            while 1:
                node = self.selection(node)
                if node is None:
                    # Leaf node, could not find any suitable children through UCB1 -> so give it previous node and explore
                    node = prev_node
                    break
                path.append((node.player, node.fen))
                # If any of the children of the selected node are in the state_history, then redo it!
                if not any(self.state_record.get((child.player, child.fen)) for child in node.children()):
                    break
                prev_node = node

            assert node is not None

            # 2. Expansion
            path.append(self.expansion(node))

            # 3. Simulation
            result = self.simulation(node)

            # 4. Backpropagation
            self.backprop(path, result)

    def selection(self, node):
        """

        """
        # starttime = time.time()
        next_node, highest_UCB1 = None, float('-inf')
        for child in node.children():
            if (child.player, child.fen) not in self.state_record:
                continue
            payout, num_plays = self.result_record[(child.player, child.fen)], self.state_record[(child.player, child.fen)]
            logexp = (2*math.log(sum(self.state_record[(player, state)] for player, state in self.state_record)))
            UCB1 = payout/num_plays + self.gamma*math.sqrt(logexp/self.state_record[(child.player, child.fen)])
            if UCB1 >= highest_UCB1:
                highest_UCB1 = UCB1
                next_node = child
        # print(' | selection took {}s'.format(time.time() - starttime))
        return next_node

    def expansion(self, node):
        """

        """
        # starttime = time.time()
        next_node = random.choice(node.children())
        self.state_record[(next_node.player, next_node.fen)] = 0
        self.result_record[(next_node.player, next_node.fen)] = 0
        # print(' | expansion took {}s'.format(time.time() - starttime))
        return next_node.player, next_node.fen

    def simulation(self, node):
        """

        """
        # starttime = time.time()
        results = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
        statecopy = copy.copy(node.state)
        outcome = statecopy.outcome()
        while outcome is None:
            # Perform pseudo-random moves til final outcome
            moves, vals = list(statecopy.legal_moves), []
            for move in moves:
                statecopy.push(move)
                vals.append(self.net(torch.tensor(serialize(statecopy)[None]).float()))
                statecopy.pop()
            bestmove = moves[np.argmax(vals)] if statecopy.turn else moves[np.argmin(vals)]
            statecopy.push(bestmove)
            outcome = statecopy.outcome()
        # print(' | simulation took {:.2f}s, resulted in {}'.format(time.time() - starttime, outcome.result()))
        return results[outcome.result()]

    def backprop(self, path, result):
        """

        """
        # starttime = time.time()
        for player, state in path:
            self.state_record[(player, state)] += 1
            if player and result == 1:
                self.result_record[(player, state)] += 1
            if (not player) and result == -1:
                self.result_record[(player, state)] += 1
        # print(' | backprop took {}s'.format(time.time() - starttime))

    def bestnode(self):
        bestnode = None
        bestdiv = float('-inf')
        for child in self.node_history[-1].children():
            if (child.player, child.fen) not in self.state_record:
                continue
            results = self.result_record[(child.player, child.fen)]
            states = self.state_record[(child.player, child.fen)]
            div = results/states
            if div >= bestdiv:
                bestdiv = div
                bestnode = child
        return bestnode


class Node(object):
    """

    """
    def __init__(self, state, action=None):
        self.state = state
        self.fen = state.fen()
        self.player = state.turn
        self.req_action = action

    def __str__(self):
        return self.fen

    def children(self):
        children = []
        for move in self.state.legal_moves:
            self.state.push(move)
            children.append(Node(self.state.copy(), move))
            self.state.pop()
        return children


def serialize(state):
    bitmap, c2m, fen = np.zeros(shape=(11, 8, 8)), {'b': -1, 'w': 1}, state.fen()
    to_move, castles_available = c2m[fen.split(' ')[1]], state.has_castling_rights(state.turn)
    enpassant_available, legal_moves = state.has_legal_en_passant(), list(state.legal_moves)
    promotion_available = not all(list(move.promotion is None for move in legal_moves))
    is_check, piece_offset = state.is_check(), {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                                                     "p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}
    for idx in range(64):
        x_idx, y_idx = idx % 8, math.floor(idx / 8)
        piece = state.piece_at(idx)
        if piece:
            # not an empty square, set first 1-6 bits according to piece color & type
            bitmap[piece_offset[piece.symbol()], x_idx, y_idx] = 1 if piece.symbol().isupper() else -1
        bitmap[6, x_idx, y_idx] = to_move
        bitmap[7, x_idx, y_idx] = 1 * castles_available
        bitmap[8, x_idx, y_idx] = 1 * enpassant_available
        bitmap[9, x_idx, y_idx] = 1 * promotion_available
        bitmap[10, x_idx, y_idx] = 1 * is_check

    return bitmap


def test():
    print(' | Monte Carlo Tree Search unit testing ...')
    root = Node(chess.Board())
    m = MCTS(root)
    game = chess.Board()
    cont = 0
    while not game.is_game_over():
        print(game)
        m.treesearch(10)
        topnode = m.bestnode()
        m.node_history.append(topnode)
        game.push(topnode.req_action)
        if cont % 10 == 0:
            print(m)
        cont += 1
    print('result {}'.format(game.result()))


if __name__ == '__main__':
    test()
