"""
File: conv.py
Author: STEVEN ZHAO
Date: 2025-07-27
Description: 2D convolutional layer for neural networks in the forward pass

This module provides a simple implementation of a 2D convolutional layer for neural networks, including the forward pass and basic setup.

"""

import numpy as np

def pad2d(input_tensor, pad):
    """Pads the input tensor with zeros.

    Args:
        input_tensor: Numpy array of shape (C, H, W).
        pad: Either an integer or a tuple of four integers (pad_top, pad_bottom, pad_left, pad_right).

    Returns:
        Padded input tensor.
    """
    if isinstance(pad, int):
        pad_top = pad_bottom = pad_left = pad_right = pad
    elif isinstance(pad, tuple) and len(pad) == 4:
        pad_top, pad_bottom, pad_left, pad_right = pad
    else:
        raise ValueError("pad must be an int or a 4-tuple")

    return np.pad(
        input_tensor,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant'
    )


class Conv2D:
    """2D Convolutional Layer (Single-batch version, forward only).

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of convolutional kernel (assumed square).
        stride: Stride of the convolution.
        padding: Zero-padding size (int or tuple).
        weight: Convolution kernels (out_channels, in_channels, k, k).
        bias: Bias for each output channel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        limit = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = np.random.uniform(-limit, limit,
                                        (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.random.uniform(-limit, limit, out_channels)

    def forward(self, x):
        """Performs the forward pass of the Conv2D layer (vectorized).

        Args:
            x: Input array of shape (in_channels, H, W)

        Returns:
            Output array of shape (out_channels, H_out, W_out)
        """
        C, H, W = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        if isinstance(P, int):
            pad_top = pad_bottom = pad_left = pad_right = P
        else:
            pad_top, pad_bottom, pad_left, pad_right = P

        x_padded = pad2d(x, P)
        H_out = (H + pad_top + pad_bottom - K) // S + 1
        W_out = (W + pad_left + pad_right - K) // S + 1

        output = np.zeros((self.out_channels, H_out, W_out))

        # Vectorized: unfold input into patches
        patches = np.lib.stride_tricks.as_strided(
            x_padded,
            shape=(C, H_out, W_out, K, K),
            strides=(
                x_padded.strides[0],
                S * x_padded.strides[1],
                S * x_padded.strides[2],
                x_padded.strides[1],
                x_padded.strides[2],
            ),
            writeable=False
        )
        patches = patches.transpose(1, 2, 0, 3, 4).reshape(H_out * W_out, -1)  # (H_out*W_out, C*K*K)
        weights = self.weight.reshape(self.out_channels, -1)  # (out_channels, C*K*K)

        out = patches @ weights.T + self.bias  # (H_out*W_out, out_channels)
        output = out.T.reshape(self.out_channels, H_out, W_out)

        return output