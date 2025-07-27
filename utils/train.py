"""
File: train.py
Author: STEVEN ZHAO
Date: 2025-07-27
Description: Training loop for CNN on MNIST dataset

This module provides the training loop for the CNN model on the MNIST dataset.

Example:
    $ python train.py
"""

import numpy as np
from models.cnn import SimpleCNN
from utils.dataset import MNISTLoader
from utils.logger_util import setup_logger

# ------------------ Logging Setup ------------------
logger = setup_logger(__name__)


def train(model, dataset, epochs=5, batch_size=32, lr=0.01):
    """Training loop using SGD.

    Args:
        model: Instance of SimpleCNN
        dataset: MNISTLoader object with loaded data
        epochs: Number of training iterations
        batch_size: Number of samples per batch
        lr: Learning rate
    """
    (X_train, y_train), _ = dataset.load_data()

    for epoch in range(epochs):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        epoch_loss = 0
        correct = 0
        total = 0

        for start in range(0, len(X_train), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            batch_X = X_train[batch_idx]
            batch_y = y_train[batch_idx]

            for x, y in zip(batch_X, batch_y):
                x = x.reshape(1, 28, 28)
                loss, probs = model.forward(x, label=y)
                epoch_loss += loss
                pred = np.argmax(probs)
                correct += int(pred == y)
                total += 1

                # === handling SFG: update each layer's parameters ===

        acc = correct / total
        avg_loss = epoch_loss / total
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {acc:.4f}")


if __name__ == '__main__':
    # Initialize dataset and model
    logger.info("Starting training...")

    dataset = MNISTLoader()
    model = SimpleCNN()
    train(model, dataset, epochs=5, batch_size=32, lr=0.01)

    logger.info("Training completed.")