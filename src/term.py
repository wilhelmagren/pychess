import curses
import chess


board = chess.Board()

print("prepping to initialize screen ...")
scr = curses.initscr()

rows, cols = scr.getmaxyx()


"""
30 width?
12 for board, 18 for time and move input

"""
for x in range(50):
	scr.addstr(0, x, "$")
	scr.addstr(25, x, "$")
	scr.addstr(16, x, "=")
for y in range(26):
	scr.addstr(y, 0, "$")
	scr.addstr(y, 50, "$")

for y, row in enumerate(str(board).split("\n")):
	scr.addstr(4+y, 4, row)

scr.addstr(14, 3, "<!> black to move")
scr.addstr(7, 25, "<!> white time:  4:13")
scr.addstr(8, 25, "<!> black time:  5:21")


scr.refresh()
curses.napms(10000)
curses.endwin()

print("window ended")
