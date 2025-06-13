"""
High-performance music genre classifier training script.

This script runs the full training pipeline for the genre classifier, incorporating
all the best practices for achieving 80%+ accuracy:
1. Spectrogram-based input (instead of MFCC features)
2. Transfer learning with ResNet50
3. Mixup augmentation
4. Attention mechanism
5. Advanced regularization
"""

import argparse
import os
import logging
from pathlib import Path
from src.train import train_model, load_data, evaluate_model, save_results

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train music genre classifier")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/fma_medium",
        help="Path to FMA audio files",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="data/fma_metadata/tracks.csv",
        help="Path to FMA metadata CSV file",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument(
        "--augment", action="store_true", help="Apply data augmentation"
    )
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use class weights for imbalanced data",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    return parser.parse_args()


def main():
    """Main function to run the training pipeline."""
    args = parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Override settings for high-performance training
    args.mode = "spectrogram"
    args.model_type = "enhanced"
    args.augment = True
    args.class_weights = True

    logger.info("Starting high-performance training pipeline")
    logger.info(f"Using spectrogram mode with {args.data_dir}")

    # Load data
    X, y = load_data(
        data_file=None,  # Not used in spectrogram mode
        mode="spectrogram",
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
    )

    # Train model with high-performance settings
    results = train_model(X, y, args)

    # Save results
    save_results(results, args.save_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
