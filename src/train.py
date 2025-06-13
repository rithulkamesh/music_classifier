#!/usr/bin/env python3
"""
Music Genre Classifier Training Script

This script trains the high-performance CNN model for music genre classification
using either spectrograms or MFCC features from the FMA dataset.

Usage:
    python -m src.train [--data-file DATA_FILE] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pickle
import argparse
import logging
import glob
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import json

# Import models and audio processing
from src.models import EnhancedCNN, create_model, ModelType
from src.audio_processing import AudioProcessor

# Define flag for model availability
ENHANCED_MODELS_AVAILABLE = True

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train music genre classifier")
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/preprocessed/features.pkl",
        help="Path to preprocessed features file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/fma_medium",
        help="Path to FMA audio files when using spectrogram mode",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="data/fma_metadata/tracks.csv",
        help="Path to FMA metadata CSV file when using spectrogram mode",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--mode",
        type=str,
        default="features",
        choices=["features", "spectrogram"],
        help="Training mode: use precomputed features or spectrograms",
    )
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
        "--normalize", action="store_true", help="Apply feature normalization"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="high_performance",
        choices=["high_performance", "enhanced", "lstm", "cnn"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    return parser.parse_args()


def load_data(
    data_file,
    mode="features",
    data_dir=None,
    metadata_file=None,
    save_dir="checkpoints",
):
    """Load preprocessed features or spectrogram data from file or directory."""
    logger.info(f"Loading data with mode={mode}")
    try:
        if mode == "features":
            logger.info(f"Loading features from {data_file}")
            with open(data_file, "rb") as f:
                data = pickle.load(f)
            X = data["features"]
            y = data["labels"]

            logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each")
            return X, y

        elif mode == "spectrogram":
            logger.info(f"Loading audio metadata from {metadata_file}...")
            # Load metadata using pandas
            metadata_df = pd.read_csv(metadata_file, header=[0, 1], index_col=0)

            # Extract track IDs, paths, and genre labels
            track_ids = []
            file_paths = []
            genres = []

            # Process the metadata to get file paths and genres
            logger.info("Processing metadata...")
            for track_id, track in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
                try:
                    # Extract genre
                    genre = track["track", "genre_top"]

                    # Skip if no genre
                    if pd.isna(genre):
                        continue

                    # Construct file path - FMA uses 3-level folders like 000/000012.mp3
                    track_id_str = str(track_id).zfill(6)
                    folder = track_id_str[:3]
                    filename = f"{track_id_str}.mp3"
                    filepath = Path(data_dir) / folder / filename

                    # Check if file exists
                    if not filepath.exists():
                        logger.warning(f"File not found: {filepath}")
                        continue

                    track_ids.append(track_id)
                    file_paths.append(str(filepath))
                    genres.append(genre)

                except Exception as e:
                    logger.warning(f"Error processing track {track_id}: {e}")
                    continue

            # Convert genres to integers
            logger.info(f"Encoding genres: {set(genres)}")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(genres)

            # Save the label encoder for inference
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                with open(Path(save_dir) / "label_encoder.pkl", "wb") as f:
                    pickle.dump(label_encoder, f)

            # Return file paths and labels for on-demand loading
            logger.info(
                f"Prepared {len(file_paths)} audio files with {len(set(genres))} genres"
            )
            return file_paths, y

        else:
            raise ValueError(
                f"Invalid mode: {mode}. Choose 'features' or 'spectrogram'."
            )

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def train_model(X, y, args):
    """
    Train the music genre classifier model.

    Args:
        X: Either feature matrix or list of file paths for spectrogram mode
        y: Labels or encoded labels
        args: Command line arguments
    """
    label_encoder = None

    # Handle both features and spectrogram modes differently
    if args.mode == "features":
        # For features mode, we already have the data in memory

        # Encode labels if they're not already encoded
        if not isinstance(y[0], (int, np.integer)):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Save the label encoder for inference
            encoder_path = Path(args.save_dir) / "label_encoder.pkl"
            with open(encoder_path, "wb") as f:
                pickle.dump(label_encoder, f)
        else:
            y_encoded = y

        # Split data for features mode
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Apply feature normalization
        if args.normalize:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            logger.info("Feature normalization applied")

            # Save scaler for inference
            normalization_path = Path(args.save_dir) / "scaler.pkl"
            with open(normalization_path, "wb") as f:
                pickle.dump(scaler, f)

        # Create datasets and dataloaders
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)

        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    elif args.mode == "spectrogram":
        # For spectrogram mode, X contains file paths and y contains encoded labels

        # Load or create label encoder
        encoder_path = Path(args.save_dir) / "label_encoder.pkl"
        if encoder_path.exists():
            with open(encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
        else:
            # Create a new label encoder if needed
            label_encoder = LabelEncoder()
            label_encoder.fit(y)  # This ensures classes are properly encoded
            # Save the label encoder for inference
            with open(encoder_path, "wb") as f:
                pickle.dump(label_encoder, f)

        # Split file paths and labels for spectrogram mode
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize AudioProcessor for spectrogram extraction
        from src.audio_processing import AudioProcessor, AudioDataset

        audio_processor = AudioProcessor(sample_rate=22050, duration=30.0)

        # Create AudioDatasets using the file paths
        logger.info("Creating AudioDatasets for training and validation...")
        train_dataset = AudioDataset(
            file_paths=X_train,
            labels=y_train,
            audio_processor=audio_processor,
            augment=args.augment,
        )

        val_dataset = AudioDataset(
            file_paths=X_val,
            labels=y_val,
            audio_processor=audio_processor,
            augment=False,  # No augmentation for validation
        )

        logger.info(f"Created training dataset with {len(train_dataset)} samples")
        logger.info(f"Created validation dataset with {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Use multiple workers for faster loading
        pin_memory=True,  # Speed up data transfer to GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Get number of classes from the label encoder
    n_classes = len(label_encoder.classes_) if label_encoder else len(np.unique(y))
    logger.info(f"Number of classes: {n_classes}")

    # Create model with proper input size
    logger.info(f"Creating {args.model_type} model...")
    try:
        if args.mode == "features":
            input_size = X_train.shape[1]
            model = create_model(
                model_type=args.model_type,
                input_size=input_size,
                num_classes=n_classes,
                dropout=0.3,
            )
        else:  # spectrogram mode
            # For spectrogram mode, input_size is ignored by the EnhancedCNN
            model = create_model(
                model_type=args.model_type,
                input_size=None,  # Not used for spectrogram mode
                num_classes=n_classes,
                dropout=0.4,  # Slightly increased dropout for spectrograms
            )
    except ValueError:
        # Fall back to default enhanced model
        logger.info("Invalid model type, falling back to EnhancedCNN")
        model = EnhancedCNN(
            input_size=None,  # Works for both modes
            num_classes=n_classes,
            dropout=0.4,  # Slightly increased dropout
        )

    # Calculate class weights to handle imbalance
    if args.class_weights:
        from sklearn.utils.class_weight import compute_class_weight

        # Get actual class distribution from training dataset (for logging)
        if hasattr(train_dataset, "labels"):  # For AudioDataset
            class_count = np.bincount(train_dataset.labels)
        else:  # For TensorDataset
            class_count = np.bincount(y_train)
        logger.info(f"Class distribution in training set: {class_count}")

        # Compute weights using sklearn
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights_tensor = torch.FloatTensor(class_weights)
        logger.info(f"Using class weights: {class_weights}")

        # Use Focal Loss for better handling of class imbalance
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
        logger.info("Using Focal Loss with class weights and gamma=2.0")
    else:
        criterion = nn.CrossEntropyLoss()

    # Simple Adam optimizer for speed
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # More aggressive OneCycleLR for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10.0,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,  # Shorter warm up for faster training
        anneal_strategy="linear",
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1000,  # Final lr = max_lr/1000
    )

    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)

    # Move class weights to device if using focal loss with weights
    if (
        args.class_weights
        and hasattr(criterion, "alpha")
        and criterion.alpha is not None
    ):
        criterion.alpha = criterion.alpha.to(device)

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]") as pbar:
            for inputs, targets in pbar:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Apply mixup augmentation with 50% probability during training
                if args.augment and np.random.random() > 0.5:
                    inputs, targets_a, targets_b, lam = mixup_data(
                        inputs, targets, alpha=0.2
                    )
                    mixup_applied = True
                else:
                    mixup_applied = False

                # Zero gradients
                optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient gradient clearing

                # Forward pass
                outputs = model(inputs)

                # Compute loss - handle mixup if applied
                if mixup_applied:
                    loss = mixup_criterion(
                        criterion, outputs, targets_a, targets_b, lam
                    )
                    # For accuracy tracking, use the dominant target
                    _, predicted = outputs.max(1)
                    correct += (
                        lam * predicted.eq(targets_a).sum().float()
                        + (1 - lam) * predicted.eq(targets_b).sum().float()
                    ).item()
                else:
                    loss = criterion(outputs, targets)
                    # Standard accuracy tracking
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()

                # Backward pass
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()

                # Step the scheduler (OneCycleLR needs to be updated every batch)
                scheduler.step()

                # Track metrics
                train_loss += loss.item()
                total += targets.size(0)

                # Update progress bar with current learning rate too
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100.0 * correct / total:.2f}%",
                        "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                    }
                )

        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]") as pbar:
                for inputs, targets in pbar:
                    # Move data to device
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc": f"{100.0 * correct / total:.2f}%",
                        }
                    )

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Log with current learning rate
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = Path(args.save_dir)
            save_dir.mkdir(exist_ok=True)

            # Create checkpoint dict based on training mode
            if args.mode == "features":
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "input_size": X_train.shape[1],
                    "num_classes": len(label_encoder.classes_),
                    "label_encoder": label_encoder,
                    "val_acc": best_val_acc / 100.0,  # Convert to decimal
                    "epochs": epoch + 1,
                    "mode": args.mode,
                }
            else:  # spectrogram mode
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "input_size": None,  # Not needed for spectrogram mode
                    "num_classes": len(label_encoder.classes_),
                    "label_encoder": label_encoder,
                    "val_acc": best_val_acc / 100.0,  # Convert to decimal
                    "epochs": epoch + 1,
                    "mode": args.mode,
                    "requires_spectrogram": True,
                }

            torch.save(checkpoint, save_dir / "high_performance_best.pt")
            logger.info(f"Saved best model with accuracy: {best_val_acc:.2f}%")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, args.save_dir)

    # Evaluate with confusion matrix
    try:
        logger.info("Generating confusion matrix...")
        class_names = list(label_encoder.classes_)
        final_acc, cm = evaluate_model_with_confusion_matrix(
            model, val_loader, class_names=class_names
        )
        logger.info(f"Final evaluation accuracy: {final_acc:.2%}")

        # Save detailed results
        results = {
            "accuracy": final_acc,
            "class_names": class_names,
            "confusion_matrix": cm.tolist(),
        }

        results_path = Path(args.save_dir) / "high_performance_results.json"
        import json

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {results_path}")
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {e}")

    return model, label_encoder, best_val_acc / 100.0


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(train_losses, label="Training Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(train_accs, label="Training Accuracy")
    ax2.plot(val_accs, label="Validation Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Save figure
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(save_dir / f"training_curves_{timestamp}.png", dpi=100)


def evaluate_model_with_confusion_matrix(model, data_loader, class_names=None):
    """
    Evaluate the model and generate a confusion matrix.

    Args:
        model: Trained model
        data_loader: DataLoader with evaluation data
        class_names: List of class names

    Returns:
        tuple: (accuracy, confusion_matrix)
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Create a figure
    plt.figure(figsize=(10, 8))

    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2%})")

    # Save figure
    cm_path = Path("checkpoints") / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print classification report
    if class_names:
        report = classification_report(
            all_labels, all_preds, target_names=class_names, digits=3
        )
        logger.info(f"Classification Report:\n{report}")

    # Find the most confused classes
    np.fill_diagonal(cm, 0)  # Zero out the diagonal
    most_confused = np.unravel_index(np.argmax(cm), cm.shape)
    if class_names:
        true_class = class_names[most_confused[0]]
        pred_class = class_names[most_confused[1]]
        logger.info(f"Most confused classes: {true_class} is predicted as {pred_class}")

    return accuracy, cm


# Add this after the imports
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Implementation based on the paper:
    "Focal Loss for Dense Object Detection" by Lin et al.

    Args:
        alpha: Class weights
        gamma: Focusing parameter (higher gamma -> more focus on hard examples)
        reduction: Reduction method ('none', 'mean', 'sum')
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight factor for classes
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, inputs, targets):
        # Get predicted probabilities for the correct class
        if inputs.dim() > 2:
            # Support batches of different sizes
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1, 2)
            inputs = inputs.contiguous().view(-1, inputs.size(2))

        targets = targets.view(-1, 1)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)

        pt = logpt.exp()

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at

        # Calculate focal loss
        loss = -1 * (1 - pt) ** self.gamma * logpt

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


def mixup_data(x, y, alpha=0.2):
    """
    Applies mixup augmentation to a batch of data.

    Args:
        x: Input batch
        y: Target batch
        alpha: Mixup alpha parameter

    Returns:
        Mixed inputs, pair of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Applies mixup criterion to the predictions.

    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First target
        y_b: Second target (permuted)
        lam: Mixup lambda

    Returns:
        Mixup loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main():
    """Main function."""
    args = parse_args()

    try:
        # Create directory for saving results
        os.makedirs(args.save_dir, exist_ok=True)

        # Load data with proper save_dir
        X, y = load_data(
            data_file=args.data_file,
            mode=args.mode,
            data_dir=args.data_dir,
            metadata_file=args.metadata_file,
            save_dir=args.save_dir,
        )

        # Train model
        model, label_encoder, best_val_acc = train_model(X, y, args)

        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2%}")

        # Save results summary
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True)

        results = {
            "accuracy": best_val_acc,
            "num_features": X.shape[1],
            "num_classes": len(label_encoder.classes_),
            "class_names": list(label_encoder.classes_),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(save_dir / "high_performance_results.json", "w") as f:
            import json

            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {save_dir / 'high_performance_results.json'}")

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
