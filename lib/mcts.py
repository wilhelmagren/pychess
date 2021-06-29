from __future__ import annotations

import sys
import time
import copy
import chess
import random

import numpy as np

class MCTS():
  def __init__(self, root, time_limit, **kwargs):
    self.node_visits, self.node_results, self.root = {}, {}, root
    self.depths, self.time_limit, self.gamma = {}, time_limit, kwargs.get('gamma', 1.4)
    self.mapping =  {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
  
  def __str__(self):
    s = ''
    for depth in range(len(self.depths)):
      s += '\nDEPTH {}\n'.format(depth)
      for pos in self.depths[depth]:
        s += '{} \t {}%\n'.format(pos.repr, round(100*self.node_results[(pos.repr, pos.ply)]/self.node_visits[(pos.repr, pos.ply)], 1))
    return s

  def update(self, new_root):
    self.node_visits[(new_root.repr, new_root.ply)] = 0
    self.node_results[(new_root.repr, new_root.ply)] = 0
    if new_root.ply not in self.depths:
      self.depths[new_root.ply] = [new_root]
    else:
      self.depths[new_root.ply].append(new_root)
    self.root = new_root

  def search(self):
    tottime = time.time()
    while time.time() - tottime < self.time_limit:
      starttime = time.time()
      self.execute(self.root)
      print(' | mcts execution took \t{}ms'.format(round(1000*(time.time() - starttime), 1)))

  def execute(self, node) -> None:
    """
      perform one iteration of the Monte Carlo Tree Search algorithm,
      where the 4 corresponding phases are:
        1. selection
        2. expansion
        3. simulation
        4. backpropagation
    """
    selected_node = self.select(node)
    expanded_node = self.expand(selected_node)
    if expanded_node is not None:
      result = self.simulate(expanded_node)
    self.backprop(selected_node if expanded_node is None else expanded_node, self.mapping[selected_node.board.outcome().result()] if expanded_node is None else result)
  
  def select(self, node) -> MCTSNode:
    while not node.is_terminal:
      if node.is_expanded:
        node = self.get_child(node)
      else:
        return node
    # Returning terminal node here? Catch it somewhere?
    return node

  def expand(self, node) -> None:
    actions = node.possible_actions()
    for action in actions:
      if action not in node.children:
        new_node = MCTSNode(board=node.take_action(action), parent=node)
        self.node_visits[(new_node.repr, new_node.ply)] = 0
        self.node_results[(new_node.repr, new_node.ply)] = 0
        node.children[action] = new_node
        if new_node.ply not in self.depths:
          self.depths[new_node.ply] = [new_node]
        else:
           self.depths[new_node.ply].append(new_node)
        if len(node.children) == len(actions):
          # We have added all possible children to the node, now it is fully expanded!
          node.is_expanded = True
        return new_node

  def simulate(self, node) -> int:
    return node.simulate()

  def backprop(self, node, result) -> None:
    while 1:
      self.node_visits[(node.repr, node.ply)] += 1
      if result == 1 and node.player:
        self.node_results[(node.repr, node.ply)] += 1
      if result == -1 and (not node.player):
        self.node_results[(node.repr, node.ply)] += 1
      if not node.parent:
        return
      node = node.parent
    return
  
  def get_child(self, node):
    UCB1, best_vals = float('-inf'), []
    for child in node.children.values():
      logexp = 2*np.log(self.node_visits[(node.repr, node.ply)])
      val = self.node_results[(child.repr, child.ply)]/self.node_visits[(child.repr, child.ply)] + self.gamma*np.sqrt(logexp/self.node_visits[(child.repr, child.ply)])
      if val > UCB1:
        UCB1 = val
        best_vals = [child]
      elif val == UCB1:
        best_vals.append(child)
    return random.choice(best_vals)
    

class MCTSNode():
  def __init__(self, board, parent):
    self.board, self.parent = board, parent
    self.ply, self.player = board.ply(), board.turn
    self.is_terminal, self.is_expanded = self.is_terminal(), self.is_terminal()
    self.repr, self.children = board.fen(), {}
  
  def __str__(self) -> str:
    return str(self.board)

  def possible_actions(self) -> list:
    return list(self.board.legal_moves)
  
  def simulate(self) -> int:
    # TODO: implement actual simulation
    results = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
    tmp_board = copy.deepcopy(self.board)
    outcome = tmp_board.outcome()
    while outcome is None:
      action = random.choice(list(tmp_board.legal_moves))
      tmp_board.push(action)
      outcome = tmp_board.outcome()
    return results[outcome.result()]
    
  def take_action(self, action) -> chess.Board:
    tmp_board = copy.deepcopy(self.board)
    tmp_board.push(action)
    return tmp_board

  def is_terminal(self) -> bool:
    return True if self.board.outcome() else False


class ChessGame():
  def __init__(self, root):
    self.mcts = MCTS(None, int(sys.argv[1]))
    self.mcts.update(root)
  
  def push(self, move):
    tmpboard = copy.deepcopy(self.mcts.root.board)
    tmpboard.push(move)
    new_root = MCTSNode(tmpboard, self.mcts.root)
    self.mcts.update(new_root)

  def play(self):
    outcome = self.mcts.root.board.outcome()
    print(self.mcts.root.board)
    while outcome is None:
      # Get player move
      self.push(chess.Move.from_uci(str(input(' | please make move: '))))
      print(self.mcts.root.board)
      outcome = self.mcts.root.board.outcome()
      if outcome is not None:
        break
      self.mcts.search()
      self.mcts.update(self.mcts.get_child(self.mcts.root))
      print(self.mcts.root.board)


if __name__ == '__main__':
  # 'rnbqk1nr/pppp1ppp/8/2b1p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 3')
  game = ChessGame(MCTSNode(chess.Board(), None))
  game.play()

