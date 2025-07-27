"""
File: datset.py
Author: STEVEN ZHAO
Date: 2025-07-27
Description: Load MNIST dataset

Example:
    $ python datset.py
"""

import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from utils.logger_util import setup_logger

os.makedirs("data_set", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ------------------ Logging Setup ------------------
logger = setup_logger(__name__)

class MNISTLoader:
    """MNIST dataset loader."""

    def __init__(self, data_dir: str = "data_set/"):
        self.data_dir = data_dir
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None


    def load_data(self):
        """load MNIST dataset."""

        logger.info("Loading MNIST dataset...")
        self.train_images, self.train_labels = self._load_mnist('train')
        self.test_images, self.test_labels = self._load_mnist('t10k')
        logger.info("MNIST dataset loaded successfully.")
        return (self.train_images, self.train_labels), (self.test_images, self.test_labels)
    

    def _load_mnist(self, kind: str):
        """load certain type of MNIST data"""

        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory {self.data_dir} does not exist.")
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")

        images_path = os.path.join(self.data_dir, f'{kind}-images-idx3-ubyte.gz')
        labels_path = os.path.join(self.data_dir, f'{kind}-labels-idx1-ubyte.gz')

        with gzip.open(images_path, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

        with gzip.open(labels_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return images, labels
    
    
    def get_batch(self, batch_size: int = 32, train: bool = True):
        """get a batch of data"""
        if train:
            indices = np.random.choice(len(self.train_images), batch_size)
            return self.train_images[indices], self.train_labels[indices]
        else:
            indices = np.random.choice(len(self.test_images), batch_size)
            return self.test_images[indices], self.test_labels[indices]


    def show_sample(self, index: int = 0, train: bool = True):
        """display a sample image from the dataset"""

        if train:
            image = self.train_images[index]
            label = self.train_labels[index]
        else:
            image = self.test_images[index]
            label = self.test_labels[index]

        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')
        plt.savefig("output/sample.png")


# ------------------ Main Entry ------------------
if __name__ == "__main__":
    logger.info("Starting MNIST dataset loading...")

    loader = MNISTLoader()
    (x_train, y_train), (x_test, y_test) = loader.load_data()
    logger.info(f"Train data shape: {x_train.shape}, Train labels shape: {y_train.shape}")
    logger.info(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")

    # Show a sample image from the training set
    # loader.show_sample(index=0, train=True)

    logger.info("MNIST dataset is ready for use.")
    # Now you can use x_train, y_train, x_test, y_test for training and testing your model.