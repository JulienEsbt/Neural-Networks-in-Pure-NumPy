# Neural Network Foundations in NumPy

> **Status: Incomplete academic project (2022)** — preserved as coursework,
> not as a finished neural-network implementation.

This repository contains the first stages of an academic exercise on building a
feed-forward neural network for MNIST without a machine-learning framework. It
was developed by Julien Esterbet and Hugo Legeard.

## Implemented work

- parsing IDX image and label files;
- flattening and normalising MNIST inputs;
- one-hot encoding of target labels;
- normally distributed weight initialisation;
- helpers for sigmoid, sigmoid derivative and softmax activations.

## Planned but not completed

The current source stops before the central parts of the assignment:

- forward propagation through the complete network;
- backpropagation;
- weight updates;
- training loops and mini-batches;
- evaluation on the test set.

The repository should therefore not be described as a working multi-layer
neural-network implementation.

## Files

- `numpy_nn_Esterbet.py` — partial implementation;
- `tp2_sujet.pdf` — original assignment brief;
- MNIST test images and labels, plus training labels.

The MNIST training images required by the script are not included. The Python
file also contains absolute local paths from the original development machine.

## Reproducing the exercise

Requirements:

- Python 3
- NumPy
- Matplotlib
- the complete MNIST IDX dataset

Before running the script, replace the absolute dataset paths with paths valid
on your machine. No automated tests or reproducible training command are
provided in the current version.

## License

No open-source license has been selected. The assignment brief and MNIST data
remain subject to their respective authors' and distributors' terms.
