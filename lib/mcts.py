"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 13/06-2021

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
import chess
import random


class MCTS(object):
    """
    Monte Carlo Tree Search object, called with current chess state/board and arbitry keyword args.
    MCTS iteratively builds a search tree by executing four phases.
    1. Selection, 2. Expansion, 3. Simulation, 4. Backpropagation
    """
    def __init__(self, root, **kwargs):
        self.root = root
        self.result_record, self.state_record = dict(), dict()
        self.gamma = kwargs.get('discount', 1.4)
        self.calculation_time = kwargs.get('time', 30)
        self.max_moves = kwargs.get('moves', 100)
        self.result_record[(root.player, root.fen)], self.state_record[(root.player, root.fen)] = 0, 0

    def __str__(self):
        s = ''
        s += ' | Num states explored: {}\n'.format(len(self.state_record))
        s += ' | Result and state record:\n'
        for rplayer, rstate in self.result_record:
            s += ' | ({}, {})\t\t\t=> ratio: {:.2f}\n'.format(rplayer, rstate, self.result_record[(rplayer, rstate)]/self.state_record[(rplayer, rstate)])
        return s

    def __del__(self):
        print(' | deleting MCTS class instance ...')

    def treesearch(self, num_sim):
        for _ in range(num_sim):
            selected, node, path = False, self.root, []
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

        return next_node

    def expansion(self, node):
        """

        """
        next_node = random.choice(node.children())
        self.state_record[(next_node.player, next_node.fen)] = 0
        self.result_record[(next_node.player, next_node.fen)] = 0
        return (next_node.player, next_node.fen)

    def simulation(self, node):
        """

        """
        results = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
        new_node = copy.copy(node)
        outcome = new_node.state.outcome()
        while outcome is None:
            # Perform pseudo-random moves til final outcome
            new_node = random.choice(new_node.children())
            outcome = new_node.state.outcome()
        print(' | simulation result: {}'.format(outcome.result()))
        return results[outcome.result()]

    def backprop(self, path, result):
        """

        """
        for player, state in path:
            self.state_record[(player, state)] += 1
            if player and result == 1:
                self.result_record[(player, state)] += 1
            if (not player) and result == -1:
                self.result_record[(player, state)] += 1

    def bestnode(self):
        bestnode = None
        bestdiv = float('-inf')
        for child in self.root.children():
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


def test():
    print(' | Monte Carlo Tree Search unit testing ...')
    root = Node(chess.Board())
    m = MCTS(root)
    starttime = time.time()
    m.treesearch(10)
    print(' | performed iterative MCTS in {:.1f}s'.format(time.time() - starttime))
    print(m)


if __name__ == '__main__':
    test()
