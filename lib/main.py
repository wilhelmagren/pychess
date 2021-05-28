"""
pychess rewrite, with CNN regression model for position evaluation
Author: Wilhelm Ågren, wagren@kth.se
"""
import chess.svg
from flask import Flask, Response, request
from state import State
app = Flask(__name__)
state = State()


def computer_move():
    move = state.branches()[0]
    print(move)
    state.board.push(move)


@app.route('/')
def webpage():
    res = '<html><head>'
    res += '<head><script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script></head>'
    res += '<style>button {font-size: 60px;}</style>'
    res += '</head><body>'
    res += '<img width=600 height=600 src="/board.svg"></img><br/>'
    res += '<button onclick=\'$.post("/move"); location.reload();\'>Make Engine Move</button>'
    return res


@app.route('/board.svg')
def board():
    return Response(chess.svg.board(board=state.board), mimetype='image/svg+xml')


@app.route('/move', methods=['POST'])
def move():
    computer_move()
    return ""


if __name__ == '__main__':
    app.run()
