# Deep-Chess
# Abstract

Since IBM's DeepBlue beat Kasparov in 1997, chess engines have been a fascinating tool for exploring the intersection between computer science, mathematics, and game theory. In recent years, neural network-based approaches have become more prevalent, such as AlphaZero or Leela. However, many of these approaches use bitboard encodings in combination with deep recurrent CNN model designs. The purpose of this project is to explore a more novel approach to both board encodings and model design.

**GitHub for this article can be found [here](https://github.com/JoeyNiestroy/TransformerChessEngine).**

---

# Introduction

Traditionally, chess boards have been encoded with bitboard encodings. Although shown to be very effective with CNN-based models such as AlphaZero [1], the encodings themselves end up being extremely sparse, and the spatial element of the board forces techniques like CNNs to be used for meaningful feature extraction. Transformers have been shown to have state-of-the-art (SOTA) performance in not only NLP but also CV tasks [2]. We chose this architecture to explore its application in chess engines.  

In this project, we introduce a transformer-based chess model that can predict moves and values. Through a structured experiment, we compare its performance to traditional deep CNN-based chess models in both action and value prediction.

---

# Methods

## Data

### Action Data

The dataset for actions was the Lichess Evaluated dataset [3], which provides boards and the legal moves ranked by the Stockfish Chess engine. Boards are converted into three tensors:  

1. **Position Indexes**  
2. **Piece Indexes**  
3. **Mask**  

The two index-based tensors retrieve board embeddings (described below), while the mask distinguishes legal moves during model training and inference. The highest-ranked move is encoded by its class index. The final sample count was approximately 20 million.  
*No repeat samples are saved.*

### Value Data

The dataset for values was retrieved from Kaggle [4], providing boards and Stockfish evaluations. The boards are processed similarly to the action data, and values are scaled to be between [-1,1]. The final sample count was approximately 16 million.  
*No repeat samples are saved.*

### Note on Move Encodings

A common way to represent chess moves is the UCI format (e.g., `G1F3`). In this form, there are only 1968 possible moves. Thus, the action model is framed as a multiclass classification problem with \(n = 1968\). The mask is a binary vector of length 1968, with `1`s at indexes representing legal moves and `-100` elsewhere.

---

## Model Architecture

The Transformer Chess Model is implemented using the PyTorch library. It consists of embedding matrices, transformer encoder layers (either traditional or Switch), and output layers.

### Embeddings

We use a two-embedding system to represent a single square in a board state:

1. **Relative Piece Embedding**: A trainable vector of weights in \(\mathbb{R}^n\), where every unique piece has a 'friendly' and 'enemy' embedding.  
2. **Fixed Positional Embeddings**: Another trainable vector of weights in \(\mathbb{R}^n\), where each square (e.g., `A1`) has an assigned embedding.  

This allows boards to be viewed from either player's perspective. For example, the `A1` square on the starting board from White's perspective combines the friendly rook embedding with the `A1` positional embedding. Final board states are represented as a matrix \(M^{64 \times n}\).  
*\(n\) represents the dimensions of the embeddings.*

### Transformer Encoder

A multi-layer transformer encoder processes the embeddings:

- **Normalization**: Pre- and post-normalization using RMSProp.  
- **Activation**: GeLU.  
- **Feedforward Network**: Sized 4 times the embedding dimension.  
- **Layers and Heads**: 8 layers with 4 attention heads.  
- **Embedding Dimensions**: 128 (small models) or 256 (large models).  

### M.O.E. Transformer Encoder

The second explored transformer encoder is based on the Switch Transformer architecture [5]. This approach focuses on the parameter efficiency and scaling potential of our model.

### Output Layers

The transformed output from the encoder '\(64 \times n\)' is flattened to a single layer of size \(64 \cdot n\) and projected into either the action or value space:

- **Action Models**: 1968 neurons, with outputs masked using the legal move mask to exclude invalid moves.  
- **Value Models**: A single output neuron with Tanh activation for regression.

A [ CLS ] style token was also explored in subexperiments 

---

# Training Procedure

The following settings were used for all models:

- **Optimizer**: Adam (\(\beta_1 = 0.9\), \(\beta_2 = 0.999\), weight decay \(= 1 \times 10^{-2}\))  
- **Learning Rate**: \(1 \times 10^{-4}\)  
- **Gradient Clipping Value**: 2.0  
- **Batch Size**: 1024  
- **Total Training Steps**: 3906  
- **Action Model Loss**: Cross-Entropy  
- **Value Model Loss**: Mean Squared Error  
- **Test Dataset Size**: 200,000  

---

# Results

| **Model Type**          | **Total Parameters** | **Mean Squared Error (Value)** | **Cross Entropy Loss (Policy)** |
|--------------------------|----------------------|--------------------------------|----------------------------------|
| Base Encoder Small       | 18M                 | 0.1593                         | 2.12                            |
| Base Encoder Medium      | 42M                 | 0.1586                         | **2.089**                       |
| Moe Encoder Small        | 180M                | 0.2088                         | 2.415                           |
| Moe Encoder Large        | 558M                | 0.2094                         | 2.408                           |
| Alpha-0 Small            | 21M                 | **0.1391**                     | 4.199                           |
| Alpha-0 Medium           | 40M                 | 0.1416                         | 4.194                           |

---

# Conclusion

Although the two models were competitive for value estimation, the transformer-based chess models were significantly better for action prediction. This demonstrates the potential of transformer-based models to become the next SOTA chess engine. Further work will test ELO performance and RL approaches.

---

# References

1. McGrath, [2022].  
2. Dosovitskiy, A., et al., [2021].  
3. Lichess Evaluated Dataset, [Lichess.org](https://database.lichess.org/).  
4. Kaggle Chess Dataset, [Kaggle.com](https://www.kaggle.com/).  
5. Fedus, W., et al., [2022].
