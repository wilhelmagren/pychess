"""
PychessAgent implementation, limited lookahead 
agent using static evaluation of chess positions
using trained CNN regressor.

Author: Wilhelm Ågren, wagren@kth.se
Last edited: 21-20-2021
"""
import chess
import torch
import numpy as np

from .utils import WPRINT, EPRINT, CHESS_WHITE, CHESS_BLACK, CHESS_HEIGHT, CHESS_WIDTH, PIECE_OFFSET
from .net import ChessRegressorCNN



class PychessAgent:
    def __init__(self, model=None, color=CHESS_BLACK, verbose=False, **kwargs):
        self.color = color
        self.model = ChessRegressorCNN() if model is None else model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._verbose = verbose

        WPRINT("loading model weights for {} and moving to device {}".format(str(self.model), self.device), str(self), self._verbose)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('./models/ChessRegressorCNN_R.pth', map_location=torch.device(self.device)))
        WPRINT("done initializing net and agent", str(self), self._verbose)

    def __str__(self):
        return "PychessAgent"
   
    def __call__(self, state):
        WPRINT("looking up best move", str(self), self._verbose)
        best_val = float('-inf') if self.color else float('inf')
        best_move = None
        for move in state.legal_moves:
            state.push(move)
            serialized = self._serialize_state(state)
            value = self.eval(serialized)
            print(value, move)
            if self.color:
                if value >= best_val:
                    best_move = move
                    best_val = value
            else:
                if value <= best_val:
                    best_move = move
                    best_val = value
            state.pop()
        if best_move is None:
            raise ValueError('no better move found, verify forward pass in neural net!')
        
        WPRINT("found best move {}".format(best_move.uci()), str(self), self._verbose)
        return best_move

    def _serialize_state(self, state):
        bitmap = np.zeros((18, 64), dtype=np.uint8)
        for idx in range(CHESS_HEIGHT*CHESS_WIDTH):
            p = state.piece_at(idx)
            if p:
                bitmap[PIECE_OFFSET[p.symbol()], idx] = 1
        bitmap[12, :] = int(state.turn)
        bitmap[13, :] = int(state.is_check())
        bitmap[14, :] = int(state.has_kingside_castling_rights(chess.WHITE))
        bitmap[15, :] = int(state.has_kingside_castling_rights(chess.BLACK))
        bitmap[16, :] = int(state.has_queenside_castling_rights(chess.WHITE))
        bitmap[16, :] = int(state.has_queenside_castling_rights(chess.BLACK))
        bitmap = bitmap.reshape(18, 8, 8)
        return bitmap[None]
    
    def eval(self, bitmap):
        self.model.eval()
        tensor = torch.Tensor(bitmap)
        value = self.model(tensor)
        return value.data


if __name__ == "__main__":
    """
    [+] eval: -2.90, r1bqk1nr/pppp1ppp/2n5/2b1N3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 4
        Agent finds best move! c6e5 capturing the hanging knight.

    [+] eval: -11.0, r1b1k1nr/pppp1ppp/5q2/2b1n2Q/2B1P3/2N5/PPPP1PPP/R1B1K2R b KQkq - 3 6
        Agent finds best move! f6f2 bringing the game close to mate.

    [+] eval: +17.0, r1b1k1nr/pppp1ppp/4q3/2b1Q3/2B1P3/2N5/PPPP1PPP/R1B1K2R w KQkq - 1 8
        Agent find best move! c4e6 obviously capturing the hanging queen.

    [!] eval: +2.10, r1b3nr/pp4pp/2pk4/3p4/8/2N5/PPPP1PPP/R1B1R1K1 w - - 2 14
        Agent does NOT find best move. c3b5 sacks the knight... 

    [+] eval: +8.60, rnbqkbnr/pppp1p1p/6p1/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 3
        Agent finds best move. h5e5, but evaluation is very off. network says almost equal,
        probably because it has learned to penalize moving queen early? queen safety 
        is bad in this position.

    [!] eval: +1.60, r1b2rk1/ppppqp1p/1bn2np1/6B1/N1BpP3/5Q2/PPP1NPPP/R4RK1 b - - 3 10
        Agent does NOT find best move. d7d5 sacks material, but is an ambitious strike to the center of the board...

    [+] eval: +0.80, r1b2rk1/pppp1p1p/1b3qp1/4n1B1/N1BpP3/8/PPP1NPPP/R4RK1 w - - 0 12
        Agent find best move. g5e6 taking the queen after the trade.

    [+] eval: +0.40, rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
        Agent prefers italian opener. e7e5 is response as black in starting position.

    [!] eval: +M1  , r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4
        Agent does NOT make the mate in one... instead does c4f7 and captures with bishop.
    """
    agent = PychessAgent(color=CHESS_WHITE, verbose=True)
    move = agent(chess.Board('r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4'))
    print(move.uci())

