#!/usr/bin/env python3
"""
Generate UMAP embeddings for music visualization.

This script loads a trained model, extracts features from audio files,
runs inference, and generates UMAP embeddings for visualization.

Usage:
    python -m src.generate_embeddings [--data-file DATA_FILE] [--output-file OUTPUT_FILE]
"""

import torch
import numpy as np
import pickle
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import umap
from sklearn.preprocessing import StandardScaler

from src.models import EnhancedCNN

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate UMAP embeddings")
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/preprocessed/features.pkl",
        help="Path to preprocessed features file",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="checkpoints/high_performance_best.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/embeddings/umap_embeddings.pkl",
        help="Output file for UMAP embeddings",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter",
    )
    return parser.parse_args()


def load_model(model_file):
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {model_file}")
    try:
        checkpoint = torch.load(model_file, map_location="cpu")

        input_size = checkpoint.get("input_size", 52)
        num_classes = checkpoint.get("num_classes", 13)

        model = EnhancedCNN(input_size=input_size, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model, checkpoint["label_encoder"]

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def extract_embeddings(model, features):
    """Extract embeddings from the penultimate layer of the model."""
    logger.info(f"Extracting embeddings for {len(features)} samples")

    embeddings = []

    with torch.no_grad():
        for feature in tqdm(features):
            feature_tensor = torch.FloatTensor(feature).unsqueeze(0)

            # Get output from the penultimate layer (128 dims)
            x = model.input_norm(feature_tensor)
            x1 = torch.relu(model.bn1(model.fc1(x)))
            x1 = model.dropout1(x1)
            x2 = torch.relu(model.bn2(model.fc2(x1)))
            x2 = model.dropout2(x2)
            x2 = x2 + x1  # Residual connection
            x3 = torch.relu(model.bn3(model.fc3(x2)))
            x3 = model.dropout3(x3)
            x4 = torch.relu(model.bn4(model.fc4(x3)))

            # Extract the 128-dimensional embedding
            embedding = x4.squeeze().cpu().numpy()
            embeddings.append(embedding)

    return np.array(embeddings)


def generate_umap(embeddings, n_neighbors=15, min_dist=0.1):
    """Generate UMAP embeddings for visualization."""
    logger.info(
        f"Generating UMAP embeddings with n_neighbors={n_neighbors}, min_dist={min_dist}"
    )

    # Normalize the embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    # Generate UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42,
    )

    umap_embeddings = reducer.fit_transform(normalized_embeddings)

    return umap_embeddings


def main():
    """Main function."""
    args = parse_args()

    try:
        # Create output directory
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load data
        with open(args.data_file, "rb") as f:
            data = pickle.load(f)

        features = data["features"]
        labels = data["labels"]

        # Load model
        model, label_encoder = load_model(args.model_file)

        # Extract embeddings
        embeddings = extract_embeddings(model, features)

        # Generate UMAP embeddings
        umap_embeddings = generate_umap(
            embeddings, n_neighbors=args.n_neighbors, min_dist=args.min_dist
        )

        # Save results
        result = {
            "x": umap_embeddings[:, 0],
            "y": umap_embeddings[:, 1],
            "genres": labels,
            "original_embeddings": embeddings,
        }

        with open(output_path, "wb") as f:
            pickle.dump(result, f)

        logger.info(f"UMAP embeddings saved to {output_path}")

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


if __name__ == "__main__":
    main()
