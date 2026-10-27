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
    stabilized = logits - logits.max(axis=1, keepdims=True)
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
        if "x" in col and col.split("x")[0].isdigit() and col.split("x")[1].isdigit()
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


def calculate_filters(matrix: np.ndarray, weights: np.ndarray, biases: np.ndarray):
    feature_map = {} # n * rows * cols = 8 * 26 * 26
    for i in range(weights.shape[0]):
        feature_map[i] = calculate_kernel(matrix, weights[i]) + biases[i]
    return np.stack(list(feature_map.values()))


def pooling(matrix, ksize, pad=False): # pad when dimensions not // by blocs size
    m, n = matrix.shape[:2]
    ky, kx = ksize

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, ky)   # blocks needed to cover all height
        nx = _ceil(n, kx)   # blocks needed to cover all width
        size = (ny * ky, nx * kx) + matrix.shape[2:]    # dimensions needed for complete blocks
        mat_pad = np.full(size, np.nan)     # full map NaN
        mat_pad[:m, :n, ...] = matrix       # fill only top left corner
    else:
        ny = m // ky    # n complete blocks for each dimension
        nx = n // kx
        mat_pad = matrix[:ny*ky, :nx*kx, ...]   # remove incompletes rows/cols

    # 0:ny      which line group
    # 1:ky      which line inside the bloc
    # 2:nx      which column group
    # 3:kx      which column inside the bloc

    blocks = mat_pad.reshape(ny, ky, nx, kx)    # seperate each dimension en block number
    blocks = blocks.transpose(0, 2, 1, 3)       # regroup different bloc axis
    blocks = blocks.reshape(ny, nx, ky * kx)    # flatten : ky * kx vectors

    result = np.nanmax(blocks, axis=-1)      # block maximum
    indices = np.nanargmax(blocks, axis=-1)  # block maximum indices

    # convert indices into coordinates in the entry map
    block_rows, block_cols = np.indices((ny, nx))

    rows = block_rows * ky + indices // kx  # find global row : block beginning + local row
    cols = block_cols * kx + indices % kx   # find global col : block beginning + local col

    # coordinates in pairs -> shape: (ny, nx, 2)
    max_positions = np.stack((rows, cols), axis=-1)

    return result, max_positions


def flatten(pooling_maps):
    # /!\ Suppose that we add batch axis before
    initial_shape = pooling_maps.shape          # keep shape for the gradient
    flattened_maps = pooling_maps.reshape(pooling_maps.shape[0], -1)     # (8, 13, 13) -> (1352,)
    return initial_shape, flattened_maps


def dense(x, w, b):
    # X: (N, 1352)
    # W: (1352, 10)
    # b: (10,)
    # Z: (N, 10)

    # Z = XW + b
    Z = x @ w + b
    return Z



def forward_pass(
        X: np.ndarray,
        weights_conv: np.ndarray,
        weights_dense: np.ndarray,
        biases_conv: np.ndarray,
        biases_dense: np.ndarray,
        ):
    conv = calculate_filters(X, weights_conv, biases_conv)
    relu = ReLu(conv)

    pooled_maps = []
    max_positions = []

    for f in range(relu.shape[0]):
        pooled_map, positions = pooling(relu[f], (2, 2))
        pooled_maps.append(pooled_map)
        max_positions.append(positions)

    pooled_maps = np.stack(pooled_maps)     # pooled_maps : (8, 13, 13)
    max_positions = np.stack(max_positions) # max_positions : (8, 13, 13, 2)

    initial_shape, flattened = flatten(pooled_maps[None, ...])   # flattened : (1, 1352)
    
    z = dense(flattened, weights_dense, biases_dense)
    out = softmax(z)

    cache = {
        "X": X,
        "conv": conv,
        "flattened": flattened,
        "max_positions": max_positions,
        "initial_shape": initial_shape,
    }
    return out, cache


def cross_entropy(x: np.ndarray, y: np.ndarray):
    x = np.clip(x, 1e-12, 1.0)
    produit = y * np.log(x)
    L = np.sum(produit, axis=1)
    return np.mean(-L)


def softmax_cross_entropy_backward(x: np.ndarray, y: np.ndarray):
    """
    How much loss change based on score changing.
    x: probabilities
    y: one-hot labels
    dZ: logits gradient (N, 10)
    """
    N = x.shape[0]
    dZ = (x - y) / N
    return dZ


def dense_backward(
        dZ: np.ndarray,
        flattened: np.ndarray,
        weights_dense: np.ndarray
    ):
    dW = flattened.T @ dZ       # (1352, N) @ (N, 10) = (1352, 10)
    db = np.sum(dZ, axis=0)     # bias on every image, we add every contribution
    dF = dZ @ weights_dense.T   # (N, 1O) @ (10, 1352) = (N, 1352)
    return dF, dW, db


def pooling_backward(
        d_pool: np.ndarray,
        max_positions: np.ndarray,
        input_shape: tuple[int, int],
    ):
    """
    Creates a table of zeros of input_shape shape.
    For every d_pool value, we get the coresponding coordinates in max_positions.
    We add this value at these coordinates in the outing table.
    """
    table = np.zeros(input_shape, dtype=d_pool.dtype)

    for i in range(d_pool.shape[0]):
        for j in range(d_pool.shape[1]):
            row, col = max_positions[i, j]
            table[row, col] += d_pool[i, j]

    return table


def relu_backward(dA: np.ndarray, Z: np.ndarray):
    """
    dA : Gradient from pooling_backward
    Z: Values before ReLU, kept in cache["conv"]

    If ReLu is positive : gradient goes through
    Else : zero
    """
    return dA * (Z > 0)
    