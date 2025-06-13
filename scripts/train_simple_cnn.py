#!/usr/bin/env python3
"""
Simple CNN training script for music genre classification demo.

This script trains a simplified CNN model (SimpleCNN) for quick demos,
targeting ~70% accuracy while being fast to train.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import argparse
import logging
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

from src.models import SimpleCNN

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a simple CNN model for music genre classification"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/preprocessed/features.pkl",
        help="Path to preprocessed features pickle file",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    return parser.parse_args()


def load_data(data_file):
    """Load preprocessed data from pickle file."""
    logger.info(f"Loading data from {data_file}")
    try:
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        X = np.array(data["features"])
        y = np.array(data["labels"])

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        logger.info(f"Loaded {len(X)} samples with {len(set(y))} classes")

        # Save scaler and label encoder
        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("checkpoints/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

        return X, y, label_encoder
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def train_model(X, y, args):
    """Train the SimpleCNN model."""
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_size = X.shape[1]
    num_classes = len(np.unique(y))

    model = SimpleCNN(input_size=input_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Track metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # Train model
    logger.info(f"Training SimpleCNN model for {args.epochs} epochs")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # Save final model
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = Path(args.save_dir) / "simple_cnn_best.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "val_acc": val_accs[-1],
        "input_size": input_size,
        "num_classes": num_classes,
        "mode": "features",
    }

    torch.save(checkpoint, model_path)
    logger.info(f"Model saved to {model_path}")

    # Plot training curves
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    ax2.plot(train_accs, label="Train")
    ax2.plot(val_accs, label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(Path(args.save_dir) / f"training_curves_{timestamp}.png")

    results = {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
        "final_val_acc": val_accs[-1],
    }

    return results


def main():
    """Main function."""
    args = parse_args()

    # Load data
    X, y, label_encoder = load_data(args.data_file)

    # Train model
    results = train_model(X, y, args)

    final_accuracy = results["final_val_acc"]

    logger.info(
        f"Training completed with final validation accuracy: {final_accuracy:.4f}"
    )

    # Ensure the accuracy is correctly saved in the model
    # The accuracy is already saved in the checkpoint during train_model

    # We don't need the accuracy.txt file anymore since we're loading from checkpoint
    acc_file = Path(args.save_dir) / "accuracy.txt"
    if acc_file.exists():
        logger.info(
            "Removing accuracy.txt file as we're now using model checkpoint for accuracy"
        )
        os.remove(acc_file)


if __name__ == "__main__":
    main()
