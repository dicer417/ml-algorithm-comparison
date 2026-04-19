# ML Algorithms from Scratch

Three machine learning algorithms implemented from scratch in Python — no scikit-learn, no deep learning frameworks. Each project supports both classification and regression and is evaluated on six UCI benchmark datasets.

## Projects

| Project | Algorithm | Key Techniques |
|---------|-----------|----------------|
| [knn-from-scratch](./knn-from-scratch/) | k-Nearest Neighbors | Minkowski distance, Value Distance Metric (VDM), Gaussian kernel regression, training set editing |
| [decision-tree-from-scratch](./decision-tree-from-scratch/) | Decision Tree (CART) | Information gain ratio, squared error splitting, cost-complexity pruning |
| [neural-network-from-scratch](./neural-network-from-scratch/) | Feedforward Neural Network | Sigmoid activation, full backpropagation, softmax + cross-entropy, autoencoder support |

## Datasets

All three algorithms are benchmarked on the same six UCI datasets covering both classification and regression tasks:

| Dataset | Task | Samples | Features |
|---------|------|---------|----------|
| Breast Cancer Wisconsin | Classification | 699 | 9 numeric |
| Car Evaluation | Classification | 1,728 | 6 nominal |
| House Votes 1984 | Classification | 435 | 16 nominal |
| Abalone | Regression | 4,177 | 8 mixed |
| Forest Fires | Regression | 517 | 12 mixed |
| Machine CPU | Regression | 209 | 6 numeric |

## Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, numpy
- **No external ML libraries** (no scikit-learn, TensorFlow, PyTorch, etc.)
