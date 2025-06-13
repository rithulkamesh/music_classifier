#!/usr/bin/env python3
"""
Preprocessing script for the music genre classifier.

This script processes audio files from the FMA dataset, extracts features,
and saves them to a pickle file for model training.

Usage:
    python -m src.preprocess [--data-dir DATA_DIR] [--output-file OUTPUT_FILE] [--n-samples N]
"""

import pickle
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

from src.audio_processing import AudioProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess music files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory containing FMA files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/preprocessed/features.pkl",
        help="Output file path for preprocessed features",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None, help="Number of samples to process"
    )
    parser.add_argument(
        "--n-workers", type=int, default=4, help="Number of parallel workers"
    )
    return parser.parse_args()


def load_metadata(data_dir):
    """Load and process FMA metadata."""
    logger.info("Loading metadata...")
    data_dir = Path(data_dir)
    metadata_dir = data_dir / "fma_metadata"

    # Load tracks metadata
    tracks_file = metadata_dir / "tracks.csv"
    if not tracks_file.exists():
        logger.error(f"Tracks file not found: {tracks_file}")
        raise FileNotFoundError(f"Tracks file not found: {tracks_file}")

    tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1])

    # Filter for medium subset
    medium_subset = tracks.loc[tracks[("set", "subset")] == "medium"].copy()

    # Get genre information
    genres = medium_subset[("track", "genre_top")].dropna()

    logger.info(f"Found {len(genres)} tracks with genre information")
    logger.info(f"Genre distribution: {genres.value_counts().to_dict()}")

    return genres


def process_audio_file(args):
    """Process a single audio file and extract features."""
    audio_path, sample_rate, duration = args

    try:
        # Create processor
        processor = AudioProcessor(sample_rate=sample_rate, duration=duration)

        # Load and process audio
        audio, sr = processor.load_audio(audio_path, normalize=True)

        # Extract MFCC features
        mfccs = processor.extract_mfcc(audio)

        # Extract other features
        spectral = processor.extract_spectral_features(audio)
        chroma = processor.extract_chroma_features(audio)

        # Combine all features
        all_features = np.concatenate(
            [
                mfccs.flatten(),
                spectral.flatten(),
                chroma.flatten(),
            ]
        )

        return all_features

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return None


def process_dataset(genres, data_dir, n_samples=None, n_workers=4):
    """Process the dataset and extract features from audio files."""
    data_dir = Path(data_dir)
    fma_medium_dir = data_dir / "fma_medium"

    if not fma_medium_dir.exists():
        logger.error(f"FMA medium directory not found: {fma_medium_dir}")
        raise FileNotFoundError(f"FMA medium directory not found: {fma_medium_dir}")

    # Prepare file paths
    file_paths = []
    labels = []

    for track_id, genre in genres.items():
        # FMA file structure: track ID -> first 3 digits form the folder name
        folder = str(track_id).zfill(6)[:3]
        file_path = fma_medium_dir / folder / f"{str(track_id).zfill(6)}.mp3"

        if file_path.exists():
            file_paths.append(file_path)
            labels.append(genre)

    logger.info(f"Found {len(file_paths)} valid audio files")

    # Sample if needed
    if n_samples and n_samples < len(file_paths):
        # Stratified sampling to maintain genre distribution
        unique_genres = set(labels)
        sampled_paths = []
        sampled_labels = []

        for genre in unique_genres:
            genre_indices = [i for i, label in enumerate(labels) if label == genre]
            n_genre_samples = int(n_samples * (len(genre_indices) / len(labels)))
            n_genre_samples = max(
                1, n_genre_samples
            )  # Ensure at least one sample per genre

            random.seed(42)  # For reproducibility
            selected_indices = random.sample(
                genre_indices, min(n_genre_samples, len(genre_indices))
            )

            for idx in selected_indices:
                sampled_paths.append(file_paths[idx])
                sampled_labels.append(labels[idx])

        file_paths = sampled_paths
        labels = sampled_labels

        logger.info(f"Sampled {len(file_paths)} files from dataset")

    # Process audio files in parallel
    logger.info(f"Processing audio files with {n_workers} workers...")
    features = []

    # Prepare arguments for parallel processing
    process_args = [(path, 22050, 30.0) for path in file_paths]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_audio_file, arg) for arg in process_args]

        # Process results as they complete
        valid_indices = []
        with tqdm(total=len(futures), desc="Extracting features") as pbar:
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result is not None:
                    features.append(result)
                    valid_indices.append(i)
                pbar.update(1)

    # Filter labels for successful extractions
    filtered_labels = [labels[i] for i in valid_indices]

    logger.info(f"Successfully processed {len(features)} files")

    return np.array(features), filtered_labels


def main():
    """Main function."""
    args = parse_args()

    try:
        # Create output directory if it doesn't exist
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load metadata
        genres = load_metadata(args.data_dir)

        # Process dataset
        features, labels = process_dataset(
            genres, args.data_dir, args.n_samples, args.n_workers
        )

        # Save processed data
        data = {
            "features": features,
            "labels": labels,
        }

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Preprocessed data saved to {output_path}")
        logger.info(f"Feature shape: {features.shape}")
        logger.info(f"Number of classes: {len(set(labels))}")

    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
