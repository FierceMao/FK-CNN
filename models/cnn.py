"""
File: cnn.py
Author: STEVEN ZHAO
Date: 2025-07-27
Description: Assembely of CNN components for MNIST classification (forward only)

This module provides a simple implementation of a convolutional neural network (CNN) for classifying MNIST digits.

"""

import numpy as np
from layers.conv import Conv2D
from layers.activation import ReLU
from layers.pool import MaxPool2D
from layers.fc import FullyConnected
from layers.softmax import SoftmaxCrossEntropy


class SimpleCNN:
    """Simple CNN Model Architecture (1 input, 6 conv, pool, fc, softmax).

    Structure:
    Input (1x28x28)
    -> Conv2D(6x5x5) + ReLU
    -> MaxPool2D(2x2)
    -> FullyConnected -> Softmax
    """

    def __init__(self):
        self.conv = Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu = ReLU()
        self.pool = MaxPool2D(kernel_size=2, stride=2)
        self.fc = None  # delayed initialization
        self.softmax = SoftmaxCrossEntropy()

    def forward(self, x, label=None):
        """Performs full forward pass through CNN.

        Args:
            x: Input image (1, 28, 28)
            label: Ground-truth label (int, optional)

        Returns:
            If label is provided: (loss, probs)
            Else: probs
        """
        out = self.conv.forward(x)
        out = self.relu.forward(out)
        out = self.pool.forward(out)
        out = out.reshape(-1)  # flatten

        # according to the real architecture, the fully connected layer is initialized only once
        if self.fc is None:
            self.fc = FullyConnected(input_dim=out.size, output_dim=10)

        logits = self.fc.forward(out)

        if label is not None:
            loss, probs = self.softmax.forward(logits, label)
            return loss, probs
        else:
            return self.softmax.predict(logits)