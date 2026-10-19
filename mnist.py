"""MNIST digit classification from scratch (NumPy).

This script is a cleaned, .py-friendly version of a notebook-style workflow:
- Loads train/test CSVs (Kaggle-style: id, label, pixel columns like "1x1" .. "28x28")
- Visualizes a few samples (optional)
- Trains a 1-hidden-layer neural network with sigmoid + softmax and (optional) inverted dropout
- Writes a submission CSV

Usage:
  python mnist.py

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
EPOCHS = 30
BATCH_SIZE = 256

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
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """One-hot encode integer labels 0..(num_classes-1).

    labels: shape (N,)
    returns: shape (N, num_classes)
    """
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def make_inverted_dropout_mask(shape: tuple[int, int], drop_prob: float) -> np.ndarray:
    """Inverted dropout mask.

    - Each unit is kept with probability (1 - drop_prob)
    - We divide by (1 - drop_prob) to keep the activation expectation unchanged.

    shape: (batch_size, hidden_units)
    returns mask with same shape.
    """
    keep_mask = (np.random.rand(*shape) > drop_prob).astype(np.float32)  # 1 if kept, 0 if dropped
    return keep_mask / (1.0 - drop_prob)


def extract_and_sort_pixel_columns(df: pd.DataFrame) -> list[str]:
    """Extracts pixel columns like '1x1'..'28x28' and sorts them by (row, col)."""
    pixel_columns = [
        col
        for col in df.columns
        if "x" in col and col.split("x")[0].isdigit() and col.split("x")[1].isdigit()
    ]
    return sorted(pixel_columns, key=lambda s: (int(s.split("x")[0]), int(s.split("x")[1])))


# ----------------------------
# Data loading + optional visualization
# ----------------------------
def load_mnist_csvs(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

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
# Model: 1 hidden layer NN
# ----------------------------
def forward_pass(
    batch_inputs: np.ndarray,
    weights_input_hidden: np.ndarray,
    bias_hidden: np.ndarray,
    weights_hidden_output: np.ndarray,
    bias_output: np.ndarray,
    use_dropout: bool,
    dropout_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Forward pass.

    Returns:
      hidden_activation, output_probabilities, dropout_mask (or None)
    """
    hidden_logits = batch_inputs @ weights_input_hidden + bias_hidden  # (batch, hidden_units)
    hidden_activation = sigmoid(hidden_logits)  # (batch, hidden_units)

    dropout_mask = None
    if use_dropout:
        dropout_mask = make_inverted_dropout_mask(hidden_activation.shape, dropout_prob)
        hidden_activation *= dropout_mask

    output_logits = hidden_activation @ weights_hidden_output + bias_output  # (batch, 10)
    output_probabilities = softmax(output_logits)
    return hidden_activation, output_probabilities, dropout_mask


def compute_train_accuracy(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    weights_input_hidden: np.ndarray,
    bias_hidden: np.ndarray,
    weights_hidden_output: np.ndarray,
    bias_output: np.ndarray,
) -> float:
    """Accuracy on the full training set (no dropout)."""
    hidden_activation = sigmoid(train_inputs @ weights_input_hidden + bias_hidden)
    output_probabilities = softmax(hidden_activation @ weights_hidden_output + bias_output)
    predicted_labels = np.argmax(output_probabilities, axis=1)
    return float((predicted_labels == train_labels).mean())


def train_one_hidden_layer_nn(
    train_inputs: np.ndarray,
    train_labels_onehot: np.ndarray,
    train_labels_int: np.ndarray,
    learning_rate: float,
    hidden_units: int,
    epochs: int,
    batch_size: int,
    use_dropout: bool,
    dropout_prob: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Train a 1-hidden-layer NN with sigmoid + softmax.

    Returns a dict containing the best parameters found on training accuracy.
    """
    np.random.seed(seed)

    num_samples, num_features = train_inputs.shape
    num_classes = train_labels_onehot.shape[1]

    # Initialize parameters
    weights_input_hidden = (0.01 * np.random.randn(num_features, hidden_units)).astype(np.float32)
    bias_hidden = np.zeros((1, hidden_units), dtype=np.float32)

    weights_hidden_output = (0.01 * np.random.randn(hidden_units, num_classes)).astype(np.float32)
    bias_output = np.zeros((1, num_classes), dtype=np.float32)

    # Sanity check shapes
    print("\nShapes check")
    print("train_inputs:", train_inputs.shape)  # (N, 784)
    print("weights_input_hidden:", weights_input_hidden.shape)  # (784, hidden_units)
    print("bias_hidden:", bias_hidden.shape)  # (1, hidden_units)
    print("weights_hidden_output:", weights_hidden_output.shape)  # (hidden_units, 10)
    print("bias_output:", bias_output.shape)  # (1, 10)

    best_train_accuracy = -1.0
    best_params: dict[str, np.ndarray] = {}

    for epoch_idx in range(epochs):
        shuffled_indices = np.random.permutation(num_samples)
        shuffled_inputs = train_inputs[shuffled_indices]
        shuffled_targets_onehot = train_labels_onehot[shuffled_indices]

        for start in range(0, num_samples, batch_size):
            batch_inputs = shuffled_inputs[start : start + batch_size]
            batch_targets_onehot = shuffled_targets_onehot[start : start + batch_size]
            current_batch_size = len(batch_inputs)

            # ----- Forward
            hidden_activation, output_probabilities, dropout_mask = forward_pass(
                batch_inputs,
                weights_input_hidden,
                bias_hidden,
                weights_hidden_output,
                bias_output,
                use_dropout,
                dropout_prob,
            )

            # ----- Backward (softmax + cross-entropy)
            # dL/d(output_logits) = (p - y) / batch_size
            grad_output_logits = (output_probabilities - batch_targets_onehot) / current_batch_size

            # Gradients for output layer params
            grad_weights_hidden_output = hidden_activation.T @ grad_output_logits  # (hidden_units, 10)
            grad_bias_output = grad_output_logits.sum(axis=0, keepdims=True)  # (1, 10)

            # Backprop into hidden activation
            grad_hidden_activation = grad_output_logits @ weights_hidden_output.T  # (batch, hidden_units)

            # If dropout used, apply same mask to the gradient
            if use_dropout and dropout_mask is not None:
                grad_hidden_activation *= dropout_mask

            # Backprop through sigmoid
            grad_hidden_logits = grad_hidden_activation * sigmoid_derivative(hidden_activation)

            # Gradients for hidden layer params
            grad_weights_input_hidden = batch_inputs.T @ grad_hidden_logits  # (784, hidden_units)
            grad_bias_hidden = grad_hidden_logits.sum(axis=0, keepdims=True)  # (1, hidden_units)

            # ----- Parameter update (SGD)
            weights_hidden_output -= learning_rate * grad_weights_hidden_output
            bias_output -= learning_rate * grad_bias_output
            weights_input_hidden -= learning_rate * grad_weights_input_hidden
            bias_hidden -= learning_rate * grad_bias_hidden

        # Monitoring: train accuracy (no dropout)
        train_accuracy = compute_train_accuracy(
            train_inputs,
            train_labels_int,
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output,
        )

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            best_params = {
                "weights_input_hidden": weights_input_hidden.copy(),
                "bias_hidden": bias_hidden.copy(),
                "weights_hidden_output": weights_hidden_output.copy(),
                "bias_output": bias_output.copy(),
            }

        print(f"epoch {epoch_idx + 1}/{epochs} | acc(train)= {train_accuracy:.3f}")

    return best_params


def predict_probabilities(
    inputs: np.ndarray,
    weights_input_hidden: np.ndarray,
    bias_hidden: np.ndarray,
    weights_hidden_output: np.ndarray,
    bias_output: np.ndarray,
) -> np.ndarray:
    hidden_activation = sigmoid(inputs @ weights_input_hidden + bias_hidden)
    return softmax(hidden_activation @ weights_hidden_output + bias_output)


def predict_labels(
    inputs: np.ndarray,
    weights_input_hidden: np.ndarray,
    bias_hidden: np.ndarray,
    weights_hidden_output: np.ndarray,
    bias_output: np.ndarray,
) -> np.ndarray:
    return np.argmax(
        predict_probabilities(inputs, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output),
        axis=1,
    )


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    train_df, test_df = load_mnist_csvs(TRAIN_CSV_PATH, TEST_CSV_PATH)

    train_pixel_cols = extract_and_sort_pixel_columns(train_df)
    test_pixel_cols = extract_and_sort_pixel_columns(test_df)

    # Train data
    train_inputs = (train_df[train_pixel_cols].to_numpy().astype(np.float32)) / 255.0  # (N, 784)
    train_labels_int = train_df["label"].to_numpy().astype(int)  # (N,)
    train_labels_onehot = one_hot_encode(train_labels_int, num_classes=10)  # (N, 10)

    # Optional visualization
    if SHOW_PLOTS:
        show_single_example(train_inputs, train_labels_int, train_df, index=13)
        show_grid_examples(train_inputs, train_labels_int, grid_size=5)

    # Train the model
    best_params = train_one_hidden_layer_nn(
        train_inputs=train_inputs,
        train_labels_onehot=train_labels_onehot,
        train_labels_int=train_labels_int,
        learning_rate=LEARNING_RATE,
        hidden_units=HIDDEN_UNITS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        use_dropout=USE_DROPOUT,
        dropout_prob=DROPOUT_PROB,
        seed=RANDOM_SEED,
    )

    # Test data
    test_inputs = (test_df[test_pixel_cols].to_numpy().astype(np.float32)) / 255.0

    test_predictions = predict_labels(
        test_inputs,
        best_params["weights_input_hidden"],
        best_params["bias_hidden"],
        best_params["weights_hidden_output"],
        best_params["bias_output"],
    )

    submission_df = test_df[["id"]].copy()
    submission_df["label"] = test_predictions
    submission_df.to_csv(SUBMISSION_CSV_PATH, index=False)

    print(f"\nSaved submission to: {SUBMISSION_CSV_PATH}")


if __name__ == "__main__":
    main()