"""
File: fc.py
Author: STEVEN ZHAO
Date: 2025-07-27
Description: Fully connected layer for neural networks (forward only)

This module provides a simple implementation of a fully connected (dense) layer for neural networks, including the forward pass.

"""

import numpy as np


class FullyConnected:
    """Fully Connected (Dense) Layer.

    Maps flattened input vector to output vector.
    """

    def __init__(self, input_dim, output_dim):
        """Initializes weights and biases.

        Args:
            input_dim: Size of input vector.
            output_dim: Size of output vector.
        """
        limit = 1 / np.sqrt(input_dim)
        self.weight = np.random.uniform(-limit, limit, (output_dim, input_dim))
        self.bias = np.random.uniform(-limit, limit, output_dim)

    def forward(self, x):
        """Performs forward pass.

        Args:
            x: Input array of shape (input_dim,)

        Returns:
            Output array of shape (output_dim,)
        """
        return np.dot(self.weight, x) + self.bias