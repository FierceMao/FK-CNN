"""
File: pool.py
Author: STEVEN ZHAO
Date: 2025-07-27
Description: Max pooling layer for neural networks (forward only)

This module provides the implementation of a 2D max pooling layer, commonly used in convolutional neural networks.

"""

import numpy as np

class MaxPool2D:
    """2D Max Pooling Layer (Single-batch version, forward only).

    Applies max pooling over each channel separately.
    """

    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """Forward pass for max pooling.

        Args:
            x: Input array of shape (C, H, W)

        Returns:
            Output array after max pooling, shape: (C, H_out, W_out)
        """
        C, H, W = x.shape
        K = self.kernel_size
        S = self.stride

        # handle border cases: if the input size is not divisible by the kernel size and stride, zero-pad the input
        pad_h = (S - (H - K) % S) % S
        pad_w = (S - (W - K) % S) % S

        if pad_h > 0 or pad_w > 0:
            x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
            H += pad_h
            W += pad_w

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        # Vectorized: unfold input into sliding windows
        shape = (C, H_out, W_out, K, K)
        strides = (
            x.strides[0],
            S * x.strides[1],
            S * x.strides[2],
            x.strides[1],
            x.strides[2],
        )
        patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)

        # calculate the max-value of each patch
        out = patches.reshape(C, H_out, W_out, -1).max(axis=-1)

        return out