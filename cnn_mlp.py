"""MNIST digit classification with a CNN or a feature-based MLP (NumPy).

Loads MNIST-style CSVs, optionally displays samples, and trains:
Conv1 -> ReLU -> Pool -> Conv2 -> ReLU -> Flatten -> Dense -> Softmax.
Both convolutions use 3x3 kernels, stride 1, and no padding.
Training uses per-image SGD, random translations, and cross-entropy, without dropout.
Early stopping monitors validation loss; the best validation checkpoint produces a submission CSV.

Usage:
  python cnn_mlp.py

Set MLP to True to train and fuse both models, or False to train only the CNN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Literal


# ----------------------------
# CONFIG
# ----------------------------
TRAIN_CSV_PATH = "data/train_mnist.csv"
TEST_CSV_PATH = "data/test_mnist.csv"
SUBMISSION_CSV_PATH = "data/submission_cnn_mlp.csv"

RANDOM_SEED = 1

# Model / training hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 100
EPOCHS_MLP = 5000
NUM_FILTERS = 16   # F1: first convolution output channels
NUM_FILTERS2 = 8   # F2: second convolution output channels

EARLY_STOPPING = 15  # Patience: consecutive epochs without lower validation loss.

MLP = True
MLP_LEARNING_RATE = 0.1

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


def split_train_validation(
        labels: np.ndarray,
        validation_fraction: float = 0.1,
        seed: int = RANDOM_SEED,
    ):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be ]0,1[")
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)

        if len(indices) < 2:
            raise ValueError("Each class needs à least 2 examples")
        rng.shuffle(indices)

        num_validation = round(len(indices) * validation_fraction)
        num_validation = max(1, min(num_validation, len(indices) - 1))

        validation_indices.extend(indices[:num_validation])
        train_indices.extend(indices[num_validation:])

    train_indices = np.array(train_indices, dtype=int)
    validation_indices = np.array(validation_indices, dtype=int)
    # np.array instead of np.ndarray :
        # np.array converts the list values into an array of indices.
        # np.ndarray interprets the list as the shape of a new array.

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return train_indices, validation_indices


def evaluate(
        inputs: np.ndarray,
        targets_onehot: np.ndarray,
        params: dict,
    ):
    return classification_metrics(predict_probabilities(inputs, params), targets_onehot)


def classification_metrics(probabilities: np.ndarray, targets_onehot: np.ndarray):
    loss = float(cross_entropy(probabilities, targets_onehot))

    predictions = np.argmax(probabilities, axis = 1)
    targets = np.argmax(targets_onehot, axis=1)
    accuracy = float(np.mean(predictions == targets)) # proportion of correctly classified

    return loss, accuracy


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


def random_translation(
        image: np.ndarray,
        rng: np.random.Generator,
        max_shift: int = 2,
)->np.ndarray:
    height, width = image.shape

    if not 0 <= max_shift < min(height, width):
        raise ValueError("max_shift must be nonnegative and smaller than image dimensions")

    if max_shift == 0:
        return image.copy()

    dy, dx = rng.integers(-max_shift, max_shift + 1, size=2)

    translated = np.zeros_like(image)

    src_y = slice(max(0, -dy), min(height, height - dy))
    src_x = slice(max(0, -dx), min(width, width - dx))

    dst_y = slice(max(0, dy), min(height, height + dy))
    dst_x = slice(max(0, dx), min(width, width + dx))

    translated[dst_y, dst_x] = image[src_y, src_x]

    return translated


def train_cnn(
    train_inputs: np.ndarray,         # (N, 28, 28), normalized
    train_labels_onehot: np.ndarray,  # (N, 10)
    validation_inputs: np.ndarray,
    validation_labels_onehot: np.ndarray,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    seed: int = RANDOM_SEED,
    max_shift: int = 2,
    patience: int = EARLY_STOPPING,
    ):
    """
    train_inputs: (N, 28, 28), normalized to [0, 1]
    train_labels_onehot: (N, 10)
    Returns an in-memory copy of the parameters with the lowest validation loss.
    Stops after patience consecutive epochs without improvement.
    """
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")

    params = initialize_parameters(seed)
    best_params = None
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    rng = np.random.default_rng(seed)
    augmentation_rng = np.random.default_rng(seed + 1)
    # 2 generators → translations doesn't change shuffle
    num_samples = len(train_inputs)

    for epoch in range(epochs):
        indices = rng.permutation(num_samples)

        for i in indices:
            X = random_translation(
                train_inputs[i],
                augmentation_rng,
                max_shift=max_shift
            )
            y = train_labels_onehot[i:i+ 1] # (1, 10)

            out, cache = forward_pass(X, **params)

            _, grads = backward_pass(out, y, cache, params["weights_dense"], params["weights_conv"], params["weights_conv2"])

            for name in params:
                params[name] -= learning_rate * grads[name]

        train_loss, train_accuracy = evaluate(train_inputs, train_labels_onehot, params)

        validation_loss, validation_accuracy = evaluate(validation_inputs, validation_labels_onehot, params)

        if not np.isfinite(validation_loss):
            raise FloatingPointError("Validation loss is not finite")

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            # Copy arrays: later in-place SGD updates must not change the checkpoint.
            best_params = {name: value.copy() for name, value in params.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.2%} | "
            f"val loss={validation_loss:.4f}, acc={validation_accuracy:.2%}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping: {patience} epochs without validation loss improvement.")
            break

    print(f"Restoring epoch {best_epoch} | best val loss={best_loss:.4f}")
    return best_params


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

# ----- MLP ------

from collections import deque

def has_hole(image: np.ndarray, threshold: float = 0.5)->bool:
    """image: (28, 28), normalized to [0, 1].
    Then we propagate to each pixel to see if it's 0, if it's 1 we don't cross.
    At the end, is pixels aren't visited it means there is a hole."""
    # True = digit; False = background.
    # Padding adds an empty border connected to the exterior.
    binary = np.pad(image>threshold, pad_width=1, mode="constant", constant_values=False)

    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)

    queue = deque([(0,0)])
    visited[0,0] = True

    while queue:
        row, col = queue.popleft()

        for row_offset, col_offset in (
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1),   # right
        ):
            neighbor_row = row + row_offset
            neighbor_col = col + col_offset

            if (
                0 <= neighbor_row < height
                and 0 <= neighbor_col < width
                and not binary[neighbor_row, neighbor_col]
                and not visited[neighbor_row, neighbor_col]
            ):
                visited[neighbor_row, neighbor_col] = True
                queue.append((neighbor_row, neighbor_col))

    # Background unreachable from exterior belongs to a hole
    return bool(np.any(~binary & ~visited))


def vertical_symmetry_score(image: np.ndarray)->float:
    _, width = image.shape
    half_width = width // 2

    if half_width == 0:
        return 0.0

    left_side = image[:, :half_width]
    right_side = image[:, -half_width:][:, ::-1] # flip the image

    left_pixels = left_side.ravel()
    right_pixels = right_side.ravel()

    # correlation is undefined when either vector is constant
    if np.std(left_pixels) == 0 or np.std(right_pixels) == 0:
        return float(np.allclose(left_pixels, right_pixels))

    correlation = np.corrcoef(left_pixels, right_pixels)[0, 1]

    return float(correlation)


def horizontal_symmetry_score(image: np.ndarray)->float:
    height, _ = image.shape
    half_height = height // 2

    if half_height == 0:
        return 0.0

    up_side = image[:half_height, :]
    down_side = image[-half_height:, :][::-1, :] # flip the image

    up_pixels = up_side.ravel()
    down_pixels = down_side.ravel()

    # correlation is undefined when either vector is constant
    if np.std(up_pixels) == 0 or np.std(down_pixels) == 0:
        return float(np.allclose(up_pixels, down_pixels))

    correlation = np.corrcoef(up_pixels, down_pixels)[0, 1]

    return float(correlation)


def image_density(image: np.ndarray, threshold: float = 0.5)-> float:
    return float(np.mean(image > threshold))


def projection_profile(
    image: np.ndarray,
    threshold: float = 0.5,
    type: Literal["mean", "sum"] = "mean",
) -> np.ndarray:
    """Return active-pixel proportions or counts per row, then per column."""
    binary = image > threshold

    if type == "mean":
        row_counts = binary.mean(axis=1) # proportion instead of mean mnist(28,28) -> 56 features
        col_counts = binary.mean(axis=0)
    elif type == "sum":
        row_counts = binary.sum(axis=1)
        col_counts = binary.sum(axis=0)
    else:
        raise ValueError("type must be mean or sum")

    return np.concatenate([row_counts, col_counts])



def create_features(train_inputs: np.ndarray):
    feature_rows = []

    for image in train_inputs:
        features = [
            float(has_hole(image)),
            vertical_symmetry_score(image),
            horizontal_symmetry_score(image),
            image_density(image),
            *projection_profile(image),
        ]
        feature_rows.append(features)

    X_features = np.array(feature_rows, dtype=np.float64)

    return X_features


def predict_mlp_probabilities(inputs: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    hidden = sigmoid(dense(inputs, params["weights_input_hidden"], params["bias_hidden"]))
    return softmax(dense(hidden, params["weights_hidden_output"], params["bias_output"]))


def train_mlp(
    train_inputs: np.ndarray,
    train_labels_onehot: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels_onehot: np.ndarray,
    hidden_units: int = 64,
    batch_size: int = 64,
    use_dropout: bool = False,
    dropout_prob: float = 0.2,
    learning_rate: float = MLP_LEARNING_RATE,
    epochs: int = EPOCHS_MLP,
    seed: int = RANDOM_SEED,
    patience: int = EARLY_STOPPING,
) -> dict[str, np.ndarray]:
    """Train a one-hidden-layer MLP on features and return its best validation checkpoint."""
    if min(hidden_units, batch_size, epochs, patience) < 1:
        raise ValueError("hidden_units, batch_size, epochs and patience must be positive")
    if use_dropout and not 0 <= dropout_prob < 1:
        raise ValueError("dropout_prob must be in [0, 1)")

    rng = np.random.default_rng(seed)
    num_samples, num_features = train_inputs.shape
    num_classes = train_labels_onehot.shape[1]

    weights_input_hidden = rng.normal(0, 1 / np.sqrt(num_features), (num_features, hidden_units))
    bias_hidden = np.zeros(hidden_units)
    weights_hidden_output = rng.normal(0, 1 / np.sqrt(hidden_units), (hidden_units, num_classes))
    bias_output = np.zeros(num_classes)
    params = {
        "weights_input_hidden": weights_input_hidden,
        "bias_hidden": bias_hidden,
        "weights_hidden_output": weights_hidden_output,
        "bias_output": bias_output,
    }

    best_params = {}
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            batch = indices[start:start + batch_size]
            inputs = train_inputs[batch]
            targets = train_labels_onehot[batch]

            hidden = sigmoid(dense(inputs, weights_input_hidden, bias_hidden))
            dropout_mask = None
            if use_dropout:
                keep_prob = 1 - dropout_prob
                dropout_mask = (rng.random(hidden.shape) < keep_prob) / keep_prob
            hidden_output = hidden if dropout_mask is None else hidden * dropout_mask
            probabilities = softmax(dense(hidden_output, weights_hidden_output, bias_output))

            grad_logits = softmax_cross_entropy_backward(probabilities, targets)
            grad_hidden, grad_weights_output, grad_bias_output = dense_backward(
                grad_logits, hidden_output, weights_hidden_output
            )
            if dropout_mask is not None:
                grad_hidden *= dropout_mask
            grad_hidden_logits = grad_hidden * sigmoid_derivative(hidden)
            _, grad_weights_hidden, grad_bias_hidden = dense_backward(
                grad_hidden_logits, inputs, weights_input_hidden
            )

            weights_input_hidden -= learning_rate * grad_weights_hidden
            bias_hidden -= learning_rate * grad_bias_hidden
            weights_hidden_output -= learning_rate * grad_weights_output
            bias_output -= learning_rate * grad_bias_output

        train_loss, train_accuracy = classification_metrics(
            predict_mlp_probabilities(train_inputs, params), train_labels_onehot
        )
        validation_loss, validation_accuracy = classification_metrics(
            predict_mlp_probabilities(validation_inputs, params), validation_labels_onehot
        )
        if not np.isfinite(validation_loss):
            raise FloatingPointError("MLP validation loss is not finite")

        print(
            f"MLP epoch {epoch}/{epochs} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.2%} | "
            f"val loss={validation_loss:.4f}, acc={validation_accuracy:.2%}"
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_params = {name: value.copy() for name, value in params.items()}

        if epoch - best_epoch >= patience:
            print(f"MLP early stopping: {patience} epochs without validation loss improvement.")
            break

    print(f"MLP restoring epoch {best_epoch} | best val loss={best_loss:.4f}")
    return best_params


def fuse_output_heads(
    cnn_params: dict[str, np.ndarray],
    mlp_params: dict[str, np.ndarray],
    alpha: float = 0.9,
):
    """Build a dense head that combines CNN and MLP logits."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    weights = np.vstack([  # (CNN features + MLP hidden units, classes)
        alpha * cnn_params["weights_dense"],
        (1 - alpha) * mlp_params["weights_hidden_output"],
    ])
    biases = (
        alpha * cnn_params["biases_dense"]
        + (1 - alpha) * mlp_params["bias_output"]
    )
    return weights, biases


def predict_fused_probabilities(
        images,
        features,
        cnn_params,
        mlp_params,
        alpha=0.9,
):
    if len(images) != len(features):
        raise ValueError("Each image must have one feature row")

    weights, biases = fuse_output_heads(cnn_params, mlp_params, alpha)
    mlp_hidden = sigmoid(dense(features, mlp_params["weights_input_hidden"], mlp_params["bias_hidden"]))

    probabilities = np.empty((len(images), weights.shape[1]))

    for i, image in enumerate(images):
        _, cache = forward_pass(image, **cnn_params)

        combined = np.concatenate([cache["flattened"], mlp_hidden[i:i + 1]], axis=1)

        scores = dense(combined, weights, biases)

        probabilities[i] = softmax(scores)[0]
    return probabilities


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

    # train / val split
    train_indices, validation_indices = split_train_validation(train_labels_int, seed=RANDOM_SEED)
    fit_inputs = train_inputs[train_indices]
    fit_targets = train_labels_onehot[train_indices]

    validation_inputs = train_inputs[validation_indices]
    validation_targets = train_labels_onehot[validation_indices]

    test_inputs = (test_df[test_pixel_cols].to_numpy(dtype=np.float64) / 255.0).reshape(-1, 28, 28)

    if MLP:
        print("Extracting features and training MLP...")
        validation_features = create_features(validation_inputs)
        mlp_params = train_mlp(
            create_features(fit_inputs),
            fit_targets,
            validation_features,
            validation_targets,
        )

        print("Training CNN...")
        cnn_params = train_cnn(fit_inputs, fit_targets, validation_inputs, validation_targets)

        best_loss = float("inf")
        best_alpha = None
        for alpha in (0.8, 0.9, 0.95, 1.0):
            probabilities = predict_fused_probabilities(
                validation_inputs,
                validation_features,
                cnn_params,
                mlp_params,
                alpha=alpha,
            )
            loss, accuracy = classification_metrics(probabilities, validation_targets)
            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha
            print(f"alpha={alpha:.2f} | val loss={loss:.4f} | acc={accuracy:.2%}")

        print(f"Predicting with alpha={best_alpha:.2f}...")
        probabilities = predict_fused_probabilities(
            test_inputs,
            create_features(test_inputs),
            cnn_params,
            mlp_params,
            alpha=best_alpha,
        )
        test_predictions = probabilities.argmax(axis=1)
    else:
        print("Training CNN...")
        params = train_cnn(fit_inputs, fit_targets, validation_inputs, validation_targets)
        test_predictions = predict_labels(test_inputs, params)

    print("Writing submission...")
    submission_df = test_df[["id"]].copy()
    submission_df["label"] = test_predictions
    submission_df.to_csv(SUBMISSION_CSV_PATH, index=False)
    print(f"\nSaved submission to: {SUBMISSION_CSV_PATH}")


if __name__ == "__main__":
    main()
