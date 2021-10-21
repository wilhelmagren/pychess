![Pychess logo](images/pychess.png)
# ♟️ Pychess, the minimal python chess GUI/TUI 
Have you every wondered where all the great GUI applications written in Python are? Or are you an oldschooler that loves immersing yourself in a black and green 16bit terminal? Then Pychess offers just the thing for you. An easy to setup chess GUI or TUI implemented with minimalism in mind (currently only supports Terminal User Interface).

Follow the below steps in order to install all dependencies and set up Pychess:
```
$ git clone https://github.com:willeagren/pychess.git
$ cd pychess
$ python3 -m pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
$ python3 -m pip install chess==1.7.0
$ python3 -m pip install numpy==1.21.2
$ python3 -m pip install matplotlib==3.4.3
```

If you are running this in a Windows terminal, then you need to install additional **curses** package:
```
$ python3 -m pip install windows-curses==2.2.0
```

To run unittests for all modules run **test.sh**, and to start with default settings in TUI mode for 2 players run **run.sh**.
Starting with default options requires you to run the python file like this
```
$ python3 pychess [mode] [players] [options]

positional arguments:
        mode                                set running mode to either TUI or GUI
        players                             set number of players to either 1 or 2
    optional arguments:
        -h, --help                          show this help message and exit
        -v, --verbose                       print debugs in verbose mode
        -n NAME NAME, --names NAME NAME     set the player names
        -t TIME, --time TIME                set the time format (in seconds)
        -i INCREMENT, --increment INCREMENT set the time increment (in seconds)
```

# Deep CNN for limited look ahead playing
Static evalution of chess positions to play with limited look ahead, i.e. only searching the moves available in the current state.


### Contact and license
Author: Wilhelm Ågren, wagren@kht.se
<br>License: MIT
