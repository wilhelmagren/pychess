# Static Evaluation of Chess Positions Using Deep Neural Networks
This repository contains the code, parsed data, and trained model for statically evaluating chess positions. 

The main libraries used are, **pytorch**, **numpy**, **python-chess**, and **pygame2**.

To create the evaluation labels in the training/testing datasets I used the renowned and publicly available open source engine 'Stockfish 13'.

The model was trained on a GeForce GTX 1060 6GB GPU and a Ryzen 5 1600 6-core 3.50GHz CPU.

Inspiration and general guidelines for model architectures were found in research. *"Playing Chess with Limited Look Ahead"* by Arman Maesumi 2020, and *"Learning to Evaluate Chess Positions with Deep Neural Networks and Limited Lookahead"* by M. Sabatelli et al. 2018.

---
## TODO:
- [ ] Start working on LaTeX document
- [x] Train MLP for classification
- [ ] Train CNN for classification
- [ ] Experiment with MLP
- [ ] Experiment with CNN
- [ ] Compare MLP & CNN results
- [ ] Implement model with tree-search
- [x] Implement Monte Carlo Tree Search
- [ ] Write findings in LaTeX document
- [x] Have fun!


Author: Wilhelm Ågren, wagren@kht.se
