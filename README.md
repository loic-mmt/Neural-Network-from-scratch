# MNIST Number Classification from Scratch

An educational implementation of a handwritten-digit classifier in both **Python/NumPy** and **C++/Armadillo**. The model is trained from scratch: the forward pass, softmax output, backpropagation, gradient descent, and dropout are implemented directly, without TensorFlow, PyTorch, or scikit-learn.

## What this project covers

- loading MNIST-style data from CSV files;
- reconstructing and displaying `28 × 28` grayscale digits;
- building a fully connected neural network from basic matrix operations;
- training with sigmoid activations, softmax, cross-entropy gradients, and dropout;
- tracking training accuracy and retaining the best parameters;
- predicting test labels and exporting a submission CSV;
- comparing equivalent Python and C++ implementations.

## Model architecture

```text
784 input pixels → hidden layer → 10 output classes
                    sigmoid          softmax
```

| Implementation | Hidden units | Optimisation | Dropout | Output |
| --- | ---: | --- | ---: | --- |
| Python/NumPy | 64 | Mini-batch gradient descent, 30 epochs | 20% | `data/submission.csv` |
| C++/Armadillo | 128 | Full-batch gradient descent, 6,000 iterations | 20% | `data/submission_cpp.csv` |

The implementations intentionally use slightly different hyperparameters, so their predictions are not expected to be identical.

## Project structure

```text
.
├── mnist.py          # Python/NumPy implementation
├── mnist.cpp                 # C++/Armadillo implementation
├── data/
│   ├── train_mnist.csv       # Labels and 784 pixel columns
│   ├── test_mnist.csv        # IDs and 784 pixel columns
│   ├── sample_submission.csv
│   ├── submission.csv
│   └── submission_cpp.csv
└── README.md
```

## Dataset format

Each image is represented by 784 columns named from `1x1` to `28x28`. Pixel values are expected to be in the `[0, 255]` range.

- Training data: an `id` column, a `label` column, and 784 pixel columns.
- Test data: an `id` column and 784 pixel columns.
- Submission: two columns, `id` and `label`.

> [!NOTE]
> Some pandas exports name the identifier column `Unnamed: 0`. Rename that column to `id` before running the programs, because both implementations expect `id`.

## Run the Python implementation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install numpy pandas matplotlib
```

Run the training and prediction pipeline from the repository root:

```bash
python mnist.py
```

The script displays a digit and a random image grid, trains the network, prints the training accuracy after every epoch, and writes `data/submission.csv`.

## Build and run the C++ implementation

Install a C++17 compiler and the Armadillo linear-algebra library, then compile:

```bash
c++ -std=c++17 -O3 mnist.cpp -o mnist -larmadillo
```

Run the executable from the repository root:

```bash
./mnist
```

The program prints an ASCII preview of a digit, reports loss and training accuracy during optimisation, and writes `data/submission_cpp.csv`.

## How training works

1. Pixel columns are ordered and reshaped into 784-element input vectors.
2. Pixel intensities are scaled to the `[0, 1]` range.
3. A hidden layer applies the sigmoid activation.
4. Inverted dropout randomly disables hidden activations during training.
5. The output layer produces class probabilities with softmax.
6. Backpropagation computes the weight and bias gradients.
7. Gradient descent updates the parameters, while the best training checkpoint is retained.

## Current limitations

- Model selection is based on training accuracy; there is no validation split yet.
- The network has a single hidden layer and fixed hyperparameters.
- The code is designed for learning and experimentation, not production inference.
- The Python and C++ training loops are not exact performance equivalents.

## Possible next steps

- add a validation set and report reproducible evaluation metrics;
- add command-line options for hyperparameters and file paths;
- save and reload trained model parameters;
- add automated tests for activations, gradients, and CSV parsing;
- benchmark the Python and C++ implementations under matching settings.
