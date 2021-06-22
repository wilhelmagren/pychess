# Deep Neural Networks and Chess, a Comparative Exploration
This repository contains the code, parsed data, and trained model(s) for exploring classificaiton and regression on chess positions.
The motivation for this work comes from a newfound interest in Deep Learning and a curiously burning passion for the beautiful zero-sum game chess!

## Datasets
Two general training datasets have been generated and parsed. Both of them contain ~50k publically played chess games and was downloaded from the FICS Games Database. One dataset was created for classification and one for regression. All in all, the datasets contain 3 million unique positions. A testing dataset was created as well for both corresponding training datasets. This set of data contains ~300k unique positions.
To create the labels I designed a multithreaded parser which, together with **Stockfish 13**, parsed each game and clamped the corresponding engine evaluation. **Stockfish 13** was allowed to run at depth 10, so a fair amount of depth is already encoded in the data labels. Because of this, the proposed model only has to explore a shallow search tree for the same amount of future information!

The datasets can be found in the directory **parsed/**.

## General

The main libraries used are, **pytorch**, **numpy**, **python-chess**, and **pygame2**.

To create the evaluation labels in the training/testing datasets I used the renowned and publicly available open source engine 'Stockfish 13'.

The model was trained on a GeForce GTX 1060 6GB GPU and a Ryzen 5 1600 6-core 3.50GHz CPU.

Inspiration and general guidelines for model architectures were found in research. *"Playing Chess with Limited Look Ahead"* by Arman Maesumi 2020, and *"Learning to Evaluate Chess Positions with Deep Neural Networks and Limited Lookahead"* by M. Sabatelli et al. 2018.

---
## TODO:
- [ ] Start working on LaTeX document
- [X] Deside on architecture for MLP
- [X] Train classifiers, batchnorm/no-bn, dropout/no-dropout, etc.
- [ ] Compare, analyse, and conclude results
- [ ] Implement model with tree-search
- [x] Implement Monte Carlo Tree Search
- [ ] Write findings in LaTeX document
- [x] Have fun!


Author: Wilhelm Ågren, wagren@kht.se
