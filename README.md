# Deep-Chess
## Abstract

Since IBM's DeepBlue beat [Kasparov](https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)) in 1997 chess engines have been a fascinating tool for the exploration of the intersection between computer science, mathematics, and game theory. In recent years neural network based approaches have become more prevalent, such as [Alphazero](https://en.wikipedia.org/wiki/AlphaZero)  or [Leela](https://en.wikipedia.org/wiki/Leela_Chess_Zero). However many of these approaches use bit board encodings and deep recurrent CNN model designs. The purpose of this project will be to explore a more novel approach to both board encodings and model design. 

## Main Goals

 - Design and Implement Transformer Chess Engine in Pytorch 
 - Train the model via both supervised and RL approaches
 - Benchmark engine capabilities with puzzles and games against known engines
 - Present the findings and work in clear and reproducible way with both this document and well made Github repo
## Data Acquisition
The 'Oracle'  for the purpose of this project is the [Stockfish Engine](https://stockfishchess.org/) it is the best available source of target moves and board evaluations. However due to this project requiring a large amount of data, it isn't feasible to create the dataset solely on my compute. The following is the data sources used for this project:
 - [Lichess Stockfish Dataset](https://database.lichess.org/#evals)
 -  [Kaggle Board Evaluation](https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations) 
 - Self Created 

  ## Visualizations
