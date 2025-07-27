"""
File: activation.py
Author: STEVEN ZHAO
Date: 2025-07-27
Description: RELU activation function for neural networks   (forward Only)

This module provides the implementation of the ReLU (Rectified Linear Unit) activation function, commonly used in neural networks.

"""

import numpy as np

class ReLU:
    """ReLU (Rectified Linear Unit) Activation Function.

    Applies element-wise activation: f(x) = max(0, x)
    """

    def __init__(self):
        pass

    def forward(self, x):
        """Forward pass of the ReLU activation.

        Args:
            x: Input tensor (Numpy array of any shape)

        Returns:
            Activated tensor (same shape as input)
        """
        return np.maximum(0, x)