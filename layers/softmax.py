"""
File: softmax.py
Author: STEVEN ZHAO
Date: 2025-07-27
Description: Softmax layer for neural networks (forward only)

This module provides a simple implementation of the softmax activation function, commonly used in the output layer of neural networks for multi-class classification.

"""

import numpy as np


def softmax(logits):
    """Applies softmax to logits (numerically stable).

    Args:
        logits: Input vector (1D numpy array)

    Returns:
        Softmax probabilities (same shape as input)
    """
    exp_shifted = np.exp(logits - np.max(logits))  # stability trick
    return exp_shifted / np.sum(exp_shifted)


def cross_entropy_loss(probs, label):
    """Computes cross-entropy loss.

    Args:
        probs: Softmax probabilities (1D numpy array)
        label: Ground-truth class index (int)

    Returns:
        Scalar loss value (float)
    """
    return -np.log(probs[label] + 1e-12)  # add epsilon to prevent log(0)


class SoftmaxCrossEntropy:
    """Combined Softmax Activation and Cross Entropy Loss.

    Applies softmax then computes cross-entropy loss.
    """

    def forward(self, logits, label):
        """Forward pass.

        Args:
            logits: Output vector from last layer (1D numpy array)
            label: Ground-truth class index (int)

        Returns:
            Tuple of (loss value, softmax probabilities)
        """
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, label)
        return loss, probs
    

    def predict(self, logits):
        """Predict class probabilities (for inference)."""
        return softmax(logits)