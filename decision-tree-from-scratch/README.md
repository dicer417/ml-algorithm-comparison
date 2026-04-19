# Decision Tree from Scratch

A CART-style decision tree implementation supporting both classification and regression, with cost-complexity pruning and mixed attribute type handling.

## Algorithm Details

### Splitting Criteria

**Classification trees** select splits by maximizing **gain ratio** (information gain normalized by split information value):
- Categorical features: entropy-based information gain across all unique values
- Numeric features: binary split at the feature mean, gain ratio computed for `< cutoff` vs `>= cutoff`
- Gain ratio normalization penalizes features with many values, reducing bias toward high-cardinality splits

**Regression trees** select splits by minimizing **mean squared error (MSE)**:
- Each branch predicts the mean of samples that fall into it
- Split is chosen to minimize total weighted MSE across branches

### Tree Construction
- Recursive partitioning via `create_tree(data)`
- Each non-leaf node stores its split feature, boundary, and the full query condition to reach it
- Leaf nodes store the majority class (classification) or mean value (regression) of their samples
- Stops when all features are exhausted or a node has only one sample

### Pruning
Cost-complexity pruning implemented in `pruning_tuning.py`:
- `make_parent_leaf(node_dt, parent_node_num)` collapses a subtree by removing children and converting the parent to a leaf
- Evaluates tree performance on a held-out tuning set before and after each pruning step
- Retains the pruned structure if it maintains or improves performance

## Project Structure

```
decision-tree-from-scratch/
├── src/
│   ├── regression_tree.py  # Core RegressionTree class
│   ├── pruning_tuning.py   # Cost-complexity pruning and hyperparameter tuning
│   └── data_loader.py      # Data loading and preprocessing
├── data/
│   ├── raw/                # Original UCI dataset files
│   ├── processed/          # Train/tuning splits
│   ├── trees/              # Saved unpruned tree nodes (CSV)
│   ├── trees_pruned/       # Saved pruned tree nodes (CSV)
│   ├── split_training_sets/
│   └── accuracy/           # Performance metrics
└── README.md
```

## Key Classes

**`RegressionTree`** (`src/regression_tree.py`)
- `calc_categorical_feature_gain(data, feature)` — entropy-based information gain
- `calc_categorical_feature_gain_ratio(data, feature)` — normalized gain ratio
- `calc_numerical_feature_squared_error(data, feature, cutoff)` — MSE for regression splits
- `select_split_feature(data, used_features)` — selects best split via gain ratio or MSE
- `generate_next_node(data, parent_node_num, used_features, full_condition)` — creates a single node
- `create_tree(data)` — builds the full tree iteratively
- `predict_sample(sample)` — traverses the tree to predict a single sample
- `get_tree_hit_rate_or_mse(test_data)` — evaluates on a full test set
- `make_parent_leaf(node_dt, parent_node_num)` — collapses a subtree for pruning
- `load_nodes(num, pruned)` / `save_nodes(file_location)` — model persistence

## Results

Averaged across 10 stratified train/test splits. Pruning consistently improves or matches raw tree accuracy, especially on classification tasks. Classification metric is hit rate; regression metric is MSE.

| Dataset | Task | Raw Tree | Pruned Tree | Null Model |
|---------|------|----------|-------------|------------|
| Breast Cancer Wisconsin | Classification | 94.0% | **94.2%** | 65.5% |
| Car Evaluation | Classification | 80.2% | **87.9%** | 69.9% |
| House Votes 1984 | Classification | 90.1% | **95.6%** | 61.3% |
| Abalone | Regression (MSE ↓) | 0.54 | **0.46** | 99.66 |
| Forest Fires | Regression (MSE ↓) | 4.15 | **3.47** | 2.04 |
| Machine CPU | Regression (MSE ↓) | **9,724.60** | 10,406.19 | 28,107.85 |

[Read the paper](./Project2-paper.pdf)

## Datasets

| Dataset | Task | Split Criterion |
|---------|------|-----------------|
| Breast Cancer Wisconsin | Classification | Gain ratio |
| Car Evaluation | Classification | Gain ratio |
| House Votes 1984 | Classification | Gain ratio |
| Abalone | Regression | MSE |
| Forest Fires | Regression | MSE |
| Machine CPU | Regression | MSE |

## Dependencies

```
pandas
numpy
```
