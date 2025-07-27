"""
File: evaluate.py
Author: STEVEN ZHAO
Date: 2025-07-27
Description: Evaluation logic for trained CNN.

This module provides functions to evaluate the performance of the trained CNN model on the MNIST dataset.

Example:
    $ python evaluate.py
"""


import numpy as np
from utils.logger_util import setup_logger
from utils.dataset import MNISTLoader
from models.cnn import SimpleCNN

# ------------------ Logging Setup ------------------
logger = setup_logger(__name__)

# ------------------ Functions ------------------


def evaluate(model, dataset, batch_size=32):
    """Evaluates the model on test data.

    Args:
        model: Trained SimpleCNN model
        dataset: MNISTLoader instance
        batch_size: Number of samples per batch

    Returns:
        Accuracy (float)
    """
    _, (X_test, y_test) = dataset.load_data()

    correct = 0
    total = 0

    for start in range(0, len(X_test), batch_size):
        end = start + batch_size
        batch_X = X_test[start:end]
        batch_y = y_test[start:end]

        for x, y in zip(batch_X, batch_y):
            x = x.reshape(1, 28, 28)
            probs = model.forward(x)  # no label → only predict
            pred = np.argmax(probs)
            correct += int(pred == y)
            total += 1

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc


if __name__ == '__main__':
    # Initialize model and dataset
    logger.info("Starting evaluation...")

    model = SimpleCNN()
    dataset = MNISTLoader()
    evaluate(model, dataset)

    logger.info("Evaluation completed.")