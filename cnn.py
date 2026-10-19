"""MNIST CNN digit classification from scratch (NumPy).

This script is a cleaned, .py-friendly version of a notebook-style workflow:
- Loads train/test CSVs (Kaggle-style: id, label, pixel columns like "1x1" .. "28x28")
- Visualizes a few samples (optional)
- Trains a 2-hidden-layer convolutional neural network with sigmoid + softmax and (optional) inverted dropout
- Writes a submission CSV

Usage:
  python cnn.py

Adjust constants in CONFIG below if needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# CONFIG
# ----------------------------
TRAIN_CSV_PATH = "data/train_mnist.csv"
TEST_CSV_PATH = "data/test_mnist.csv"
SUBMISSION_CSV_PATH = "data/submission.csv"

RANDOM_SEED = 1

# Model / training hyperparameters
LEARNING_RATE = 0.1
HIDDEN_UNITS = 64
EPOCHS = 100
BATCH_SIZE = 256
EARLY_STOPPING = 15

# Dropout
USE_DROPOUT = True
DROPOUT_PROB = 0.2  # probability to drop a hidden unit

# Visualization
SHOW_PLOTS = True


# ----------------------------
# Utilities
# ----------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation for the hidden layer.

    Returns values in (0, 1). We clip x to avoid overflow in exp().
    """
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(sigmoid_output: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid computed from its output a = sigmoid(z).

    d/dz sigmoid(z) = a * (1 - a)
    """
    return sigmoid_output * (1.0 - sigmoid_output)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax.

    Converts logits (batch_size, n_classes) into probabilities that sum to 1.
    Numeric stability trick: subtract row-wise max before exp.
    """
    stabilized = logits - logits.max(axis=1, keepsdims=True)
    exp_scores = np.exp(stabilized)
    return exp_scores /exp_scores.sum(axis=1, keepdims=True)


def ReLu(x: np.ndarray)-> np.ndarray:
    return np.maximum(0, x)


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """One-hot encode integer labels 0..(num_classes-1).

    labels: shape (N,)
    returns: shape (N, num_classes)
    """
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def make_inverted_dropout_mask(shape: tuple[int, int], drop_prob: float)->np.ndarray:
    """Inverted dropout mask.
    
        - Each unit is kept with probability (1 - drop_prob)
        - We divide by (1 - drop_prob) to keep the activation expectation unchanged.
    
        shape: (batch_size, hidden_units)
        returns mask with same shape.
    """
    keep_mask = (np.random.rand(*shape) > drop_prob).astype(np.float32) # 1 if kept, 0 if dropped
    return keep_mask / (1.0 - drop_prob)


def extract_and_sort_pixel_columns(df: pd.DataFrame)->list[str]:
    """Extracts pixel columns like '1x1'..'28x28' and sorts them by (row, col)."""
    pixel_columns = [
        col for col in df.columns
        if "x" in col and col.split("x")[0].isdigit() and col.split[1].isdigit()
    ]
    return sorted(pixel_columns, key=lambda s: (int(s.split("x")[0]), int(s.split("x")[1])))


# ----------------------------
# Data loading + optional visualization
# ----------------------------

def load_mnist_csvs(train_path: str, test_path: str, verbose = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if verbose == True:
        print("Train shape:", train_df.shape)
        print("Train columns (first 20):", train_df.columns.tolist()[:20], "...")
        print(train_df.head(2))

        print("\nTest shape:", test_df.shape)
        print("Test columns (first 20):", test_df.columns.tolist()[:20], "...")
        print(test_df.head(2))

    return train_df, test_df


def show_single_example(images_flat: np.ndarray, labels: np.ndarray, train_df: pd.DataFrame, index: int) -> None:
    image_2d = images_flat[index].reshape(28, 28)
    plt.imshow(image_2d, cmap="gray")
    plt.title(f"Label: {labels[index]} | id: {train_df.loc[index, 'id']}")
    plt.axis("off")
    plt.show()


def show_grid_examples(images_flat: np.ndarray, labels: np.ndarray, grid_size: int = 5) -> None:
    num_images = grid_size * grid_size
    sampled_indices = np.random.choice(len(images_flat), num_images, replace=False)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    for ax, idx in zip(axes.ravel(), sampled_indices):
        ax.imshow(images_flat[idx].reshape(28, 28), cmap="gray")
        ax.set_title(str(labels[idx]), fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ----------------------------
# Model: 3 hidden layer CNN

# X -> Conv -> ReLu -> Pool -> Flatten -> Dense -> Loss
# Loss -> Dense -> Flatten -> Pool -> Relu -> Conv
# ----------------------------

def calculate_one_kernel(
    matrix: np.ndarray,
    weights: np.ndarray,
    row: int = 0,
    col: int = 0,
):
    kernel_height, kernel_width = weights.shape
    window = matrix[row:row + kernel_height, col:col + kernel_width]

    return np.sum(window * weights)


def calculate_kernel(matrix: np.ndarray, weights: np.ndarray):
    rows = matrix.shape[0] - weights.shape[0] + 1
    cols = matrix.shape[1] - weights.shape[1] + 1

    feature_map = np.zeros((rows, cols), dtype=np.float64)

    for col in range(cols):
        for row in range(rows):
            feature_map[row, col] = calculate_one_kernel(matrix, weights, row, col)

    return feature_map


def calculate_filters(matrix: np.ndarray, weights: np.ndarray):
    feature_map = {} # n * rows * cols = 8 * 26 * 26
    biases = np.zeros(8)
    for i in range(weights.shape[0]):
        feature_map[i] = calculate_kernel(matrix, weights[i]) + biases[i]
    return np.stack(feature_map)


def pooling(matrix, ksize, method = 'max', pad=False): # pad when dimensions not // by blocs size
    m, n = matrix.shape[:2]
    ky, kx = ksize

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = (ny * ky, nx * kx) + matrix.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = matrix
    else:
        ny = m // ky
        nx = n // kx
        mat_pad = matrix[:ny*ky, :nx*kx, ...]

    new_shape = (ny, ky, nx, kx) + matrix.shape[2:]
    # 0:ny      which line group
    # 1:ky      which line inside the bloc
    # 2:nx      which column group
    # 3:kx      which column inside the bloc

    if method == "max":
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))
    return result


def forward_pass():
    ...