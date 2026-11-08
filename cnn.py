"""MNIST CNN digit classification from scratch (NumPy).

Loads MNIST-style CSVs, optionally displays samples, and trains:
Conv1 -> ReLU -> Pool -> Conv2 -> ReLU -> Flatten -> Dense -> Softmax.
Both convolutions use 3x3 kernels, stride 1, and no padding.
Training uses per-image SGD and cross-entropy, without dropout or early stopping.
The final parameters produce a submission CSV.

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
SUBMISSION_CSV_PATH = "data/submission_cnn.csv"

RANDOM_SEED = 1

# Model / training hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 100
NUM_FILTERS = 16   # F1: first convolution output channels
NUM_FILTERS2 = 8   # F2: second convolution output channels

EARLY_STOPPING = 15

# Visualization
SHOW_PLOTS = False


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
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def make_inverted_dropout_mask(shape: tuple[int, int], drop_prob: float)->np.ndarray:
    """Inverted dropout mask.

        - Each unit is kept with probability (1 - drop_prob)
        - We divide by (1 - drop_prob) to keep the activation expectation unchanged.

        shape: (batch_size, hidden_units)
        returns mask with same shape.
    """
    keep_mask = (np.random.rand(*shape) > drop_prob).astype(np.float64) # 1 if kept, 0 if dropped
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
# Model: two convolutional layers and one dense output layer
# F1 = NUM_FILTERS, F2 = NUM_FILTERS2, D = F2 * 11 * 11.
# X -> Conv1 -> ReLU1 -> Pool -> Conv2 -> ReLU2 -> Flatten -> Dense -> Softmax
# Backward: loss -> logits -> Dense -> Flatten -> ReLU2 -> Conv2 -> Pool -> ReLU1 -> Conv1
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
    feature_map = {}  # (F1, out_h, out_w); loop-based reference implementation
    for i in range(weights.shape[0]):
        feature_map[i] = calculate_kernel(matrix, weights[i]) + biases[i]
    return np.stack(list(feature_map.values()))


def calculate_filters_vectorized(matrix: np.ndarray, weights: np.ndarray, biases: np.ndarray):
    kh, kw = weights.shape[1:]
    windows = np.lib.stride_tricks.sliding_window_view(matrix, (kh, kw))
    # einsum describe multiplication and axes to sum
    # ijab : position (i, j), local pixel (a, b)
    # fab  : filter f, local weight (a, b)
    # fij  : filter f result, position (i, j)
    feature_maps = np.einsum("ijab, fab->fij", windows, weights)
    # feature_maps[f, i, j] = sum on a,b of windows[i,j,a,b] × weights[f,a,b]
    return feature_maps + biases[:, None, None] # (F1,) -> (F1, 1, 1), broadcast over spatial positions


def conv_multichannel(X: np.ndarray, weights: np.ndarray, biases: np.ndarray):
    # X: (C, H, W)
    # weights: (F, C, kh, kw)
    _, _, kh, kw = weights.shape

    windows = np.lib.stride_tricks.sliding_window_view(X, (kh, kw), axis=(1, 2))
    # windows: (C, out_h, out_w, kh, kw)

    out = np.einsum("cijab, fcab->fij", windows, weights)
    # c : entry canal.
    # i,j : position dans carte de sortie.
    # a,b : position locale dans filtre.
    # f : filtre de sortie.
    # c,a,b disappear → sum on these 3 axis
    return out + biases[:, None, None]


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
    # Input includes a batch axis: (N, F2, 11, 11) -> (N, D).
    initial_shape = pooling_maps.shape  # Restore this shape during backward.
    flattened_maps = pooling_maps.reshape(pooling_maps.shape[0], -1)
    return initial_shape, flattened_maps


def dense(x, w, b):
    # X: (N, D)
    # W: (D, 10)
    # b: (10,)
    # Z: (N, 10)

    # Z = XW + b
    Z = x @ w + b
    return Z



def forward_pass(
        X: np.ndarray,
        weights_conv: np.ndarray,
        weights_dense: np.ndarray,
        weights_conv2: np.ndarray,
        biases_conv: np.ndarray,
        biases_dense: np.ndarray,
        biases_conv2: np.ndarray,
        ):
    conv = calculate_filters_vectorized(X, weights_conv, biases_conv)
    relu = ReLu(conv)

    pooled_maps = []
    max_positions = []

    for f in range(relu.shape[0]):
        pooled_map, positions = pooling(relu[f], (2, 2))
        pooled_maps.append(pooled_map)
        max_positions.append(positions)

    pooled_maps = np.stack(pooled_maps)     # (F1, 13, 13)
    max_positions = np.stack(max_positions) # (F1, 13, 13, 2)

    conv2 = conv_multichannel(pooled_maps, weights_conv2, biases_conv2)  # (F2, 11, 11)
    relu2 = ReLu(conv2)
    initial_shape, flattened = flatten(relu2[None, ...])   # (1, F2, 11, 11) -> (1, D)

    z = dense(flattened, weights_dense, biases_dense)
    out = softmax(z)

    cache = {
        "X": X,
        "conv": conv,
        "flattened": flattened,
        "max_positions": max_positions,
        "initial_shape": initial_shape,
        "pooled_maps": pooled_maps,
        "conv2": conv2,
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
    dW = flattened.T @ dZ       # (D, N) @ (N, 10) -> (D, 10)
    db = np.sum(dZ, axis=0)     # bias on every image, we add every contribution
    dF = dZ @ weights_dense.T   # (N, 10) @ (10, D) -> (N, D)
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


def pooling_backward_vectorized(
        d_pool: np.ndarray,
        max_positions: np.ndarray,
        input_shape: tuple[int, int],
    ):
    table = np.zeros(input_shape, dtype=d_pool.dtype)

    # maxima positions
    rows = max_positions[..., 0]
    cols = max_positions[..., 1]

    np.add.at(table, (rows, cols), d_pool) # add each gradient to it's position
    return table


def relu_backward(dA: np.ndarray, Z: np.ndarray):
    """
    dA: upstream activation gradient, with the same shape as Z.
    Z: values before the corresponding ReLU (cache["conv"] or cache["conv2"]).

    If ReLu is positive : gradient goes through
    Else : zero
    """
    return dA * (Z > 0)


def conv_backward(
        X: np.ndarray,              # forward image (28, 28)
        weights_conv: np.ndarray,   # first-layer filter weights (F1, 3, 3)
        d_conv: np.ndarray,         # gradients after ReLU1 backward (F1, 26, 26)
    ):
    """
    Calculate 3 different things:
    - dW: filter gradients, weights_conv shape
    - db: bias gradients, (F1,) shape
    - dX: entry image gradient, X shape
    """
    n_filters, kh, kw = weights_conv.shape
    _, out_h, out_w = d_conv.shape

    dW = np.zeros_like(weights_conv)
    db = np.zeros(n_filters)
    dX = np.zeros_like(X)

    for f in range(n_filters):
        # bias gradient
        db[f] = np.sum(d_conv[f])

        for i in range(out_h):
            for j in range(out_w):
                gradient = d_conv[f, i, j]

                # image window used in forward
                window = X[i:i+kh, j:j+kw]

                # gradient of filter weights
                dW[f] += gradient * window

                # gradient of input image
                dX[i:i+kh, j:j+kw] += (gradient*weights_conv[f])

    return dX, dW, db


def conv_backward_vectorized(
        X: np.ndarray,
        weights_conv: np.ndarray,
        d_conv: np.ndarray,
    ):
    _, kh, kw = weights_conv.shape
    _, out_h, out_w = d_conv.shape

    windows = np.lib.stride_tricks.sliding_window_view(X, (kh, kw))

    # sum on positions i, j.
    dW = np.einsum("fij,ijab->fab", d_conv, windows)

    # bias used everywhere -> sum of every gradient
    db = d_conv.sum(axis=(1, 2))
    # each pixel receive contributions from multiple windows and filters
    # accumulation with +=
    dX = np.zeros_like(X)

    # one passage for each filter position : 3x3
    for a in range(kh):
        for b in range(kw):
            contribution = np.einsum("fij, f->ij", d_conv, weights_conv[:, a, b])
            dX[a:a + out_h, b:b + out_w] += contribution

    return dX, dW, db


def backward_pass(
        out: np.ndarray,            # predicted proba
        y: np.ndarray,              # real class (1, 10)
        cache: dict,                # forward values for derivatives
        weights_dense: np.ndarray,  # gradients to dense entry
        weights_conv: np.ndarray,   # for conv_backward
        weights_conv2: np.ndarray,
    ):
    """
    Loss + Softmax → Dense → Flatten → ReLU2 → Conv2 → Pool → ReLU1 → Conv1
    """
    dZ = softmax_cross_entropy_backward(out, y) # gradient with respect to logits (1, 10)

    dF, dW_dense, db_dense = dense_backward(dZ, cache["flattened"], weights_dense)
    # for Z = F @ W + b:
        # dW_dense = F.T @ dZ : each entry contribute to each score
        # db_dense = dZ.sum(axis=0) : bias added to the scores
        # dF = dZ @ W.T : gradient of flattened Conv2 activations (1, D)

    d_relu2 = dF.reshape(cache["initial_shape"])[0]
    d_conv2 = relu_backward(d_relu2, cache["conv2"])
    # (1, D) -> (1, F2, 11, 11) -> (F2, 11, 11)
    # flatten only change values positions → original positions.
    # [0] take out batch axis add on forward step

    d_pool, dW_conv2, db_conv2 = conv_multichannel_backward(
        cache["pooled_maps"], weights_conv2, d_conv2
    )  # d_pool: (F1, 13, 13)


    d_relu = np.zeros_like(cache["conv"])

    for f in range(d_pool.shape[0]):
        d_relu[f] = pooling_backward_vectorized(
            d_pool[f],
            cache["max_positions"][f],
            cache["conv"][f].shape
            )
        # d_relu: (F1, 26, 26)
        # in each block : only max selected influence output, the rest get 0

    d_conv = relu_backward(d_relu, cache["conv"]) # cache["conv"] = values before ReLu

    dX, dW_conv, db_conv = conv_backward_vectorized(cache["X"], weights_conv, d_conv)
    # dW_conv (F1, 3, 3): first-layer weight gradients
    # db_conv (F1,): first-layer bias gradients
    # dX (28, 28): image gradient
    grads = {
        "weights_conv": dW_conv,
        "biases_conv": db_conv,
        "weights_dense": dW_dense,
        "biases_dense": db_dense,
        "weights_conv2": dW_conv2,
        "biases_conv2": db_conv2,
    }
    return dX, grads


def conv_multichannel_backward(X: np.ndarray, weights: np.ndarray, d_out: np.ndarray):
    _, _, kh, kw = weights.shape
    _, out_h, out_w = d_out.shape

    windows = np.lib.stride_tricks.sliding_window_view(X, (kh, kw), axis=(1, 2))

    dW = np.einsum("fij,cijab->fcab", d_out, windows)
    db = d_out.sum(axis=(1, 2))

    dX = np.zeros_like(X)

    for a in range(kh):
        for b in range(kw):
            contribution = np.einsum("fij,fc->cij", d_out, weights[:, :, a, b])
            dX[:, a:a + out_h, b:b + out_w] += contribution
    # dX redistribute gradients to each canal by summy filters contributions
    return dX, dW, db


def initialize_parameters(seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    num_features = NUM_FILTERS2 * 11 * 11
    # He scaling for ReLU convolutions; dense fan-in is F2 * 11 * 11.
    return {
        "weights_conv": rng.normal(0, np.sqrt(2 / 9), (NUM_FILTERS, 3, 3)),
        "biases_conv": np.zeros(NUM_FILTERS),
        "weights_conv2": rng.normal(0, np.sqrt(2 / (NUM_FILTERS * 3 * 3)), (NUM_FILTERS2, NUM_FILTERS, 3, 3)),
        "biases_conv2": np.zeros(NUM_FILTERS2),
        "weights_dense": rng.normal(0, np.sqrt(1 / num_features), (num_features, 10)),
        "biases_dense": np.zeros(10),
    }


def train_cnn(
    train_inputs: np.ndarray,         # (N, 28, 28), normalized
    train_labels_onehot: np.ndarray,  # (N, 10)
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    seed: int = RANDOM_SEED,
    ):
    """
    train_inputs: (N, 28, 28), normalized to [0, 1]
    train_labels_onehot: (N, 10)
    """
    params = initialize_parameters(seed)
    rng = np.random.default_rng(seed)
    num_samples = len(train_inputs)

    for epoch in range(epochs):
        indices = rng.permutation(num_samples)
        total_loss = 0.0

        for i in indices:
            X = train_inputs[i]             # (28, 28)
            y = train_labels_onehot[i:i+ 1] # (1, 10)

            out, cache = forward_pass(X, **params)
            total_loss += float(cross_entropy(out , y))

            _, grads = backward_pass(out, y, cache, params["weights_dense"], params["weights_conv"], params["weights_conv2"])

            for name in params:
                params[name] -= learning_rate * grads[name]

        mean_loss = total_loss / num_samples
        active_fraction = np.mean(cache["conv"] > 0)
        print(f"Epoch {epoch + 1}/{epochs} | loss = {mean_loss:.4f} | ReLU1 positives dernière image={active_fraction:.1%}")

    return params


def predict_probabilities(
        inputs: np.ndarray,
        params: dict,
    )-> np.ndarray:
    """inputs: (N, 28, 28). Returns: (N, 10)."""
    probabilities = np.empty((len(inputs), 10))

    for i, image in enumerate(inputs):
        out, _ = forward_pass(image, **params)
        probabilities[i] = out[0]

    return probabilities


def predict_labels(
    inputs: np.ndarray,
    params: dict,
) -> np.ndarray:
    probabilities = predict_probabilities(inputs, params)
    return np.argmax(probabilities, axis=1) # select highest-probability class for each image.


def main() -> None:
    print("Loading data...")
    train_df, test_df = load_mnist_csvs(TRAIN_CSV_PATH, TEST_CSV_PATH)
    print("Data Loaded...")

    print("Extracting pixels...")
    train_pixel_cols = extract_and_sort_pixel_columns(train_df)
    test_pixel_cols = extract_and_sort_pixel_columns(test_df)
    print("Pixels extracted...")

    # Train data
    print("Preparing inputs...")
    train_inputs = (train_df[train_pixel_cols].to_numpy().astype(np.float64)) / 255.0  # (N, 784)
    train_labels_int = train_df["label"].to_numpy().astype(int)  # (N,)
    train_labels_onehot = one_hot_encode(train_labels_int, num_classes=10)  # (N, 10)
    print("Inputs prepared...")

    # Optional visualization
    if SHOW_PLOTS:
        show_single_example(train_inputs, train_labels_int, train_df, index=13)
        show_grid_examples(train_inputs, train_labels_int, grid_size=5)

    # CNN needs (N, 28, 28) : currently (N, 784)
    print("Last inputs reshape...")
    train_inputs = train_inputs.reshape(-1, 28, 28)

    # Train the model
    print("Training model...")
    params = train_cnn(
        train_inputs=train_inputs,
        train_labels_onehot=train_labels_onehot,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        seed=RANDOM_SEED,
    )

    # Test data
    test_inputs = (test_df[test_pixel_cols].to_numpy().astype(np.float64) / 255.0).reshape(-1, 28, 28)
    print("Predicting tests labels...")
    test_predictions = predict_labels(test_inputs, params)

    print("Writing submission...")
    submission_df = test_df[["id"]].copy()
    submission_df["label"] = test_predictions
    submission_df.to_csv(SUBMISSION_CSV_PATH, index=False)

    print(f"\nSaved submission to: {SUBMISSION_CSV_PATH}")


if __name__ == "__main__":
    main()
