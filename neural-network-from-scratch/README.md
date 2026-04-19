# Feedforward Neural Network from Scratch

A fully connected feedforward neural network implemented from scratch with backpropagation, supporting 0–2 hidden layers, both classification and regression, and optional autoencoder pre-training.

## Algorithm Details

### Architecture
- **Input layer:** one node per feature (after one-hot encoding)
- **Hidden layers:** 0, 1, or 2 fully connected layers with configurable node counts
- **Output layer:**
  - Classification: one output node per class (softmax applied at loss calculation)
  - Regression: single output node (linear activation)
  - Autoencoder: one output node per input feature (reconstructs the input)
- Each node stores its own weights and bias, and tracks the last input/output values for backpropagation

### Forward Pass
- Hidden nodes apply **sigmoid activation**: `σ(w·x + b)`
- Output nodes compute a **weighted sum** (no activation — softmax is applied externally for classification)
- Forward pass propagates inputs through each layer sequentially

### Backpropagation
- Output error is computed per node:
  - **Classification:** `target − softmax(output)` using one-hot targets
  - **Regression:** `actual − predicted`
  - **Autoencoder:** `input − reconstruction`
- Each node's `update_weights` recursively propagates the error signal back through parent nodes, updating weights via the delta rule: `Δw = η × error × sigmoid_derivative × input`

### Loss Functions
- **Cross-entropy loss** (classification): sum of log-probabilities of correct classes under softmax
- **MSE** (regression): mean squared error over the test set
- **Reconstruction loss** (autoencoder): root sum of squared reconstruction errors

### Autoencoder Support
- Train a network as an autoencoder (output = input) to learn a compressed representation
- `append_autoencoder_to_network(size_hidden_layer_2)` strips the reconstruction output layer, adds a new hidden layer, and attaches a classification/regression output layer — enabling unsupervised pre-training followed by supervised fine-tuning

### Model Persistence
- `save_nodes(file_location)` serializes all weights to CSV
- `overwrite_nodes_from_dt(file_location)` restores a saved network, including layer structure and weights

## Project Structure

```
neural-network-from-scratch/
├── src/
│   ├── Network.py        # Core Network class (architecture, forward pass, backprop, loss)
│   ├── Node.py           # Individual neuron (sigmoid, weighted sum, weight updates)
│   ├── NetworkTuning.py  # Training loop, hyperparameter tuning, epoch management
│   └── DataLoader.py     # Data loading, one-hot encoding, normalization, stratified splits
├── data/
│   ├── raw/              # Original UCI dataset files
│   ├── processed/        # Train/tuning splits
│   └── models/           # Saved network weights (CSV checkpoints)
└── README.md
```

## Key Classes

**`Network`** (`src/Network.py`)
- `estimate_sample(sample)` — full forward pass through all layers
- `feedforward_and_backpropagate(sample, learning_rate)` — single training step
- `backpropagate(error_vec, learning_rate)` — weight update pass
- `calc_hit_rate(testing_set)` — classification accuracy
- `calc_mse(testing_set)` — regression MSE
- `calc_cross_entropy_loss(testing_set)` — classification loss
- `append_autoencoder_to_network(size_hidden_layer_2)` — autoencoder → supervised network conversion
- `save_nodes` / `overwrite_nodes_from_dt` — model checkpointing

**`Node`** (`src/Node.py`)
- `calc_sigmoid(inputs)` — sigmoid activation with overflow protection
- `calc_weighted_sum_of_inputs(inputs)` — linear weighted sum
- `update_weights(node_error, learning_rate)` — delta rule weight update with recursive backprop

## Results

Averaged across 10 stratified train/test splits. "Simple" = no hidden layers (linear model); "Multilayer" = best result across architectures with 1–2 hidden layers. Classification metric is hit rate; regression metric is MSE.

| Dataset | Task | Simple Network | Multilayer Network | Null Model |
|---------|------|----------------|--------------------|------------|
| Breast Cancer Wisconsin | Classification | 96.8% | **97.0%** | 65.5% |
| Car Evaluation | Classification | 87.2% | **88.4%** | 69.9% |
| House Votes 1984 | Classification | 95.0% | **96.7%** | 61.3% |
| Abalone | Regression (MSE ↓) | **0.50** | 1.00 | 1.00 |
| Machine CPU | Regression (MSE ↓) | **8,792.14** | 14,567.94 | 28,770.50 |
| Forest Fires | Regression (MSE ↓) | — | 1.85 | 1.84 |

[Read the paper](./Project3-paper.pdf)

## Datasets

| Dataset | Task | Loss Function |
|---------|------|---------------|
| Breast Cancer Wisconsin | Classification | Cross-entropy |
| Car Evaluation | Classification | Cross-entropy |
| House Votes 1984 | Classification | Cross-entropy |
| Abalone | Regression | MSE |
| Forest Fires | Regression | MSE |
| Machine CPU | Regression | MSE |

## Dependencies

```
pandas
numpy
```
