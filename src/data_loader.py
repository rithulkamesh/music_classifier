"""
Efficient FMA dataset loading and preprocessing with caching.

This module handles loading the Free Music Archive (FMA) dataset,
extracting MFCC features, and providing efficient data access.
"""

import pandas as pd
import numpy as np
import librosa
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st

logger = logging.getLogger(__name__)


class FMADataLoader:
    """
    Efficient loader for the FMA dataset with caching and preprocessing.
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the FMA data loader.

        Args:
            data_dir: Path to the data directory containing FMA files
        """
        self.data_dir = Path(data_dir)
        self.fma_medium_dir = self.data_dir / "fma_medium"
        self.metadata_dir = self.data_dir / "fma_metadata"
        self.cache_dir = self.data_dir / "cache"

        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)

        # Load metadata
        self.tracks_df = None
        self.genres_df = None
        self.features_df = None
        self.genre_mapping = {}

        self._load_metadata()
        self._load_genre_mapping()

    def _load_metadata(self):
        """Load FMA metadata files."""
        try:
            # Load tracks metadata
            tracks_path = self.metadata_dir / "tracks.csv"
            if tracks_path.exists():
                self.tracks_df = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
                logger.info(f"Loaded tracks metadata: {len(self.tracks_df)} tracks")

            # Load genres metadata
            genres_path = self.metadata_dir / "genres.csv"
            if genres_path.exists():
                self.genres_df = pd.read_csv(genres_path, index_col=0)
                logger.info(f"Loaded genres metadata: {len(self.genres_df)} genres")

            # Load features if available
            features_path = self.metadata_dir / "features.csv"
            if features_path.exists():
                self.features_df = pd.read_csv(
                    features_path, index_col=0, header=[0, 1, 2]
                )
                logger.info(f"Loaded features metadata: {len(self.features_df)} tracks")

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise

    def _load_genre_mapping(self):
        """Load or create genre ID to name mapping."""
        genre_file = self.data_dir / "genre_names.json"

        if genre_file.exists():
            with open(genre_file, "r") as f:
                loaded_data = json.load(f)

            # Check if it's a list or dictionary
            if isinstance(loaded_data, list):
                # Convert list to dictionary with integer keys
                self.genre_mapping = {i: name for i, name in enumerate(loaded_data)}
            elif isinstance(loaded_data, dict):
                # Convert string keys to integers if needed
                self.genre_mapping = {
                    int(k) if k.isdigit() else k: v for k, v in loaded_data.items()
                }
            else:
                self.genre_mapping = {}
        else:
            # Create mapping from genres DataFrame
            if self.genres_df is not None:
                self.genre_mapping = dict(
                    zip(self.genres_df.index, self.genres_df["title"])
                )

                # Save mapping
                with open(genre_file, "w") as f:
                    json.dump(self.genre_mapping, f, indent=2)

        logger.info(f"Loaded genre mapping: {len(self.genre_mapping)} genres")

    def get_subset_metadata(self, subset: str = "medium") -> pd.DataFrame:
        """
        Get metadata for a specific FMA subset.

        Args:
            subset: FMA subset ('small', 'medium', 'large', 'full')

        Returns:
            DataFrame with subset metadata
        """
        if self.tracks_df is None:
            raise ValueError("Tracks metadata not loaded")

        # Filter by subset
        subset_mask = self.tracks_df[("set", "subset")] <= subset
        subset_df = self.tracks_df[subset_mask].copy()

        # Add genre information
        if ("track", "genre_top") in subset_df.columns:
            subset_df["genre_name"] = subset_df[("track", "genre_top")].map(
                self.genre_mapping
            )

        return subset_df

    def get_audio_path(self, track_id: int) -> Path:
        """
        Get the file path for a track ID.

        Args:
            track_id: FMA track ID

        Returns:
            Path to the audio file
        """
        # FMA directory structure: fma_medium/XXX/XXXXXX.mp3
        track_str = f"{track_id:06d}"
        folder = track_str[:3]
        filename = f"{track_str}.mp3"

        return self.fma_medium_dir / folder / filename

    def load_audio(
        self, track_id: int, sr: int = 22050, duration: Optional[float] = 30.0
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file for a given track ID.

        Args:
            track_id: FMA track ID
            sr: Sample rate
            duration: Duration to load (None for full track)

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_path = self.get_audio_path(track_id)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            y, sr_actual = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
            return y, sr_actual
        except Exception as e:
            logger.error(f"Error loading audio {track_id}: {e}")
            raise

    def extract_mfcc_features(
        self,
        audio: np.ndarray,
        sr: int = 22050,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """
        Extract MFCC features from audio.

        Args:
            audio: Audio time series
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT

        Returns:
            MFCC features array
        """
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
            )

            # Take mean across time dimension for fixed-size features
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)

            # Combine mean and std for richer features
            features = np.concatenate([mfcc_mean, mfcc_std])

            return features

        except Exception as e:
            logger.error(f"Error extracting MFCC features: {e}")
            raise

    def get_cached_features(self, track_id: int) -> Optional[np.ndarray]:
        """
        Get cached MFCC features for a track.

        Args:
            track_id: FMA track ID

        Returns:
            Cached features or None if not found
        """
        cache_file = self.cache_dir / f"mfcc_{track_id}.npy"

        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.warning(f"Error loading cached features for {track_id}: {e}")

        return None

    def cache_features(self, track_id: int, features: np.ndarray):
        """
        Cache MFCC features for a track.

        Args:
            track_id: FMA track ID
            features: MFCC features to cache
        """
        cache_file = self.cache_dir / f"mfcc_{track_id}.npy"

        try:
            np.save(cache_file, features)
        except Exception as e:
            logger.warning(f"Error caching features for {track_id}: {e}")

    def get_features_for_track(
        self, track_id: int, use_cache: bool = True
    ) -> np.ndarray:
        """
        Get MFCC features for a single track, with caching.

        Args:
            track_id: FMA track ID
            use_cache: Whether to use cached features

        Returns:
            MFCC features array
        """
        # Check cache first
        if use_cache:
            cached_features = self.get_cached_features(track_id)
            if cached_features is not None:
                return cached_features

        # Load audio and extract features
        try:
            audio, sr = self.load_audio(track_id)
            features = self.extract_mfcc_features(audio, sr)

            # Cache the features
            if use_cache:
                self.cache_features(track_id, features)

            return features

        except Exception as e:
            logger.error(f"Error getting features for track {track_id}: {e}")
            raise

    def prepare_dataset(
        self,
        max_samples_per_genre: int = 500,
        use_cache: bool = True,
        n_workers: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
        """
        Prepare the complete dataset with features and labels.

        Args:
            max_samples_per_genre: Maximum samples per genre to use
            use_cache: Whether to use cached features
            n_workers: Number of parallel workers for feature extraction

        Returns:
            Tuple of (features, labels, track_ids, genre_names)
        """
        # Get subset metadata
        subset_df = self.get_subset_metadata("medium")

        # Filter out tracks without genre information
        valid_mask = subset_df[("track", "genre_top")].notna()
        subset_df = subset_df[valid_mask]

        # Filter out tracks with missing audio files
        existing_tracks = []
        for track_id in subset_df.index:
            audio_path = self.get_audio_path(track_id)
            if audio_path.exists():
                existing_tracks.append(track_id)

        subset_df = subset_df.loc[existing_tracks]
        logger.info(f"Found {len(subset_df)} tracks with existing audio files")

        # Limit samples per genre
        if max_samples_per_genre > 0:
            subset_df = subset_df.groupby(("track", "genre_top")).head(
                max_samples_per_genre
            )

        track_ids = subset_df.index.tolist()
        genre_ids = subset_df[("track", "genre_top")].tolist()

        logger.info(f"Preparing dataset with {len(track_ids)} tracks")

        # Extract features
        features_list = []
        valid_track_ids = []
        valid_genre_ids = []

        # Use progress bar in Streamlit if available
        if "streamlit" in globals() and hasattr(st, "progress"):
            progress_bar = st.progress(0)
            status_text = st.empty()

        def extract_features_worker(track_id):
            """Worker function for parallel feature extraction."""
            try:
                features = self.get_features_for_track(track_id, use_cache)
                return track_id, features
            except Exception as e:
                logger.warning(f"Failed to extract features for track {track_id}: {e}")
                return track_id, None

        # Extract features in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_track = {
                executor.submit(extract_features_worker, track_id): track_id
                for track_id in track_ids
            }

            completed = 0
            for future in as_completed(future_to_track):
                track_id, features = future.result()

                if features is not None:
                    features_list.append(features)
                    valid_track_ids.append(track_id)

                    # Find corresponding genre
                    track_idx = track_ids.index(track_id)
                    valid_genre_ids.append(genre_ids[track_idx])

                completed += 1

                # Update progress
                if "progress_bar" in locals():
                    progress = completed / len(track_ids)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {completed}/{len(track_ids)} tracks")

        if not features_list:
            raise ValueError("No valid features extracted")

        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(valid_genre_ids)

        # Get genre names
        genre_names = [
            self.genre_mapping.get(genre_id, f"Genre_{genre_id}")
            for genre_id in valid_genre_ids
        ]

        logger.info(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")

        return X, y, valid_track_ids, genre_names


@st.cache_data
def get_audio_features(audio_file, sr: int = 22050) -> Dict[str, np.ndarray]:
    """
    Extract various audio features for visualization and analysis.

    Args:
        audio_file: Audio file or audio array
        sr: Sample rate

    Returns:
        Dictionary containing different audio features
    """
    try:
        # Load audio if it's a file
        if isinstance(audio_file, (str, Path)):
            y, sr = librosa.load(audio_file, sr=sr)
        else:
            y = audio_file

        features = {}

        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["mfcc"] = mfccs

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid"] = spectral_centroids

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spectral_rolloff"] = spectral_rolloff

        zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
        features["zero_crossings"] = zero_crossings

        # Tempo and beat
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = tempo
        features["beats"] = beats

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features["chroma"] = chroma

        # Raw audio for waveform
        features["audio"] = y
        features["sr"] = sr

        return features

    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        raise
