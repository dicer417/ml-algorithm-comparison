# k-Nearest Neighbors from Scratch

A full k-Nearest Neighbors (kNN) implementation supporting both classification and regression, with mixed-type attribute handling and training set editing.

## Algorithm Details

### Distance Metric
- **Numeric attributes:** Minkowski distance with configurable exponent `p`
- **Nominal attributes:** Value Distance Metric (VDM) — measures categorical distance using per-class conditional frequencies rather than treating categories as unordered
- **Ordinal attributes (month, day):** Circular-aware scaling to handle wrap-around (e.g., December → January)
- **One-hot-encoded attributes:** Binary Hamming distance
- All numeric features are z-score normalized before distance calculations

### Prediction
- **Classification:** Plurality vote among the `k` nearest neighbors
- **Regression:** Gaussian kernel-weighted average of the `k` nearest neighbors' target values, where closer neighbors receive higher weight via `exp(distance / (σ × std_dev))`

### Training Set Editing
Implements a condensing/pruning algorithm to reduce training set size:
- Iterates through training samples; removes a sample if kNN (on the remaining set) predicts it correctly
- Terminates when the target reduction ratio is reached or no further pruning is possible
- Useful for reducing inference time without significant accuracy loss

## Project Structure

```
knn-from-scratch/
├── src/
│   ├── models/
│   │   ├── knn.py          # Core kNN class (NearestNeighbor2)
│   │   └── knn_tuning.py   # Hyperparameter tuning and cross-validation
│   └── data/
│       └── data_loader.py  # Data loading, preprocessing, stratified splitting
├── data/
│   ├── raw/                # Original UCI dataset files
│   ├── processed/          # Train/tuning splits
│   └── estimates/          # Model predictions output
└── README.md
```

## Key Classes

**`NearestNeighbor2`** (`src/models/knn.py`)
- `calc_distance(sample1, sample2, exponent)` — computes mixed-type distance
- `calc_nominal_frequencies(data)` — precomputes VDM frequency tables
- `calculate_neighbors(sample, data, exponent)` — returns sorted neighbor table
- `determine_category(nearest_neighbors, k)` — plurality vote classification
- `estimate_function_value(nearest_neighbors, k, standard_dev_mult)` — Gaussian kernel regression
- `edit_training_data(portion_to_leave, error_threshold, k, exponent)` — training set condensing

## Datasets

| Dataset | Task | Attribute Types |
|---------|------|-----------------|
| Breast Cancer Wisconsin | Classification | Numeric |
| Car Evaluation | Classification | Nominal |
| House Votes 1984 | Classification | Nominal |
| Abalone | Regression | Mixed (nominal + numeric) |
| Forest Fires | Regression | Mixed (ordinal + numeric) |
| Machine CPU | Regression | Numeric |

## Dependencies

```
pandas
numpy
```
