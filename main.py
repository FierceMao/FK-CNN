"""
File: main.py
Author: STEVEN ZHAO
Date: 2025-07-27
Description: Entry point for training and evaluation.

This module provides the main entry point for training and evaluating the CNN model on the MNIST dataset.

Example:
    $ python main.py
"""


from utils.logger_util import setup_logger
from models.cnn import SimpleCNN
from utils.dataset import MNISTLoader
from utils.train import train
from utils.evaluate import evaluate


# ------------------ Logging Setup ------------------
logger = setup_logger(__name__)

def main():
    logger.info("Initializing dataset and model...")
    dataset = MNISTLoader()
    model = SimpleCNN()

    print("[INFO] Starting training...")
    train(model, dataset, epochs=5, batch_size=32, lr=0.01)

    print("[INFO] Evaluating model...")
    evaluate(model, dataset)


if __name__ == "__main__":
    main()