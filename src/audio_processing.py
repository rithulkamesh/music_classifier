"""
Audio processing utilities for music genre classification.

This module provides functions for feature extraction (MFCC and spectrograms),
audio augmentation, and preprocessing for the music classification pipeline.
"""

import numpy as np
import librosa
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Union, List, Any
import logging
from pathlib import Path
from scipy.signal import spectrogram as scipy_spectrogram
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Comprehensive audio processing class for feature extraction and augmentation.
    Provides methods for extracting both MFCC features and spectrograms.
    """

    def __init__(self, sample_rate: int = 22050, duration: float = 30.0):
        """
        Initialize audio processor.

        Args:
            sample_rate: Target sample rate for audio
            duration: Duration of audio segments to process
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)

        # Initialize torchaudio transforms
        self.resampler = None  # Will be created when needed

        # Spectrogram parameters
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128  # Number of mel bands
        self.spectrogram_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=True,
            normalized=True,
            power=2.0,  # Power spectrogram
        )

        # Time masking for augmentation (applied to spectrograms)
        self.time_masking = T.TimeMasking(
            time_mask_param=80
        )  # Mask up to 80 time steps

        # Frequency masking for augmentation (applied to spectrograms)
        self.freq_masking = T.FrequencyMasking(
            freq_mask_param=20
        )  # Mask up to 20 frequency bands

    def load_audio(
        self, audio_path: Union[str, Path], normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file and preprocess.

        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(
                audio_path, sr=self.sample_rate, duration=self.duration, mono=True
            )

            # Normalize if requested
            if normalize:
                audio = librosa.util.normalize(audio)

            # Pad or truncate to fixed length
            if len(audio) < self.n_samples:
                # Pad with zeros
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode="constant")
            elif len(audio) > self.n_samples:
                # Truncate
                audio = audio[: self.n_samples]

            return audio, sr

        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise

    def extract_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ) -> np.ndarray:
        """
        Extract MFCC features from audio.

        Args:
            audio: Audio time series
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filter banks

        Returns:
            MFCC features
        """
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )

            return mfccs

        except Exception as e:
            logger.error(f"Error extracting MFCC: {e}")
            raise

    def extract_features(
        self,
        audio_input: Union[str, Path, np.ndarray],
        feature_type: str = "mfcc_stats",
    ) -> np.ndarray:
        """
        Extract various types of features from audio.

        Args:
            audio_input: Audio file path or audio array
            feature_type: Type of features to extract
                - 'mfcc_stats': MFCC mean and std
                - 'mfcc_sequence': Full MFCC sequence
                - 'spectral': Spectral features
                - 'combined': Combination of features

        Returns:
            Extracted features
        """
        # Load audio if it's a file path
        if isinstance(audio_input, (str, Path)):
            audio, _ = self.load_audio(audio_input)
        else:
            audio = audio_input

        if feature_type == "mfcc_stats":
            return self._extract_mfcc_stats(audio)
        elif feature_type == "mfcc_sequence":
            return self._extract_mfcc_sequence(audio)
        elif feature_type == "spectral":
            return self._extract_spectral_features(audio)
        elif feature_type == "combined":
            return self._extract_combined_features(audio)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def _extract_mfcc_stats(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC mean and standard deviation."""
        mfccs = self.extract_mfcc(audio)

        # Calculate statistics across time
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Combine mean and std
        features = np.concatenate([mfcc_mean, mfcc_std])

        return features

    def _extract_mfcc_sequence(self, audio: np.ndarray) -> np.ndarray:
        """Extract full MFCC sequence."""
        mfccs = self.extract_mfcc(audio)
        return mfccs.T  # Return as [time, features]

    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral features from audio.

        Args:
            audio: Audio time series

        Returns:
            Array of spectral features
        """
        try:
            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate
            )
            zcr = librosa.feature.zero_crossing_rate(audio)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate
            )

            # Combine features
            features = np.concatenate(
                [
                    [np.mean(spectral_centroid), np.std(spectral_centroid)],  # 2
                    [np.mean(spectral_rolloff), np.std(spectral_rolloff)],  # 2
                    [np.mean(zcr), np.std(zcr)],  # 2
                    [np.mean(spectral_bandwidth)],  # 1
                ]
            )

            return features

        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            return np.zeros(7)  # Return zeros in case of error

    def extract_chroma_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract chromagram and rhythm features from audio.

        Args:
            audio: Audio time series

        Returns:
            Array of chroma and rhythm features
        """
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)

            # Extract tempo and onset strength
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            onset = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            rms = librosa.feature.rms(y=audio)

            # Shapes should be: chroma: (12, n_frames), onset: (n_frames,), rms: (1, n_frames)

            # Make sure all values are properly shaped for concatenation (all 1D arrays)
            chroma_mean = (
                np.mean(chroma, axis=1) if chroma.ndim > 1 else chroma
            )  # Handle both 1D and 2D

            # Ensure we have scalars for these values
            tempo_scalar = (
                float(tempo.item()) if hasattr(tempo, "item") else float(tempo)
            )
            onset_scalar = (
                float(np.mean(onset).item())
                if onset.size > 0 and hasattr(np.mean(onset), "item")
                else float(np.mean(onset)) if onset.size > 0 else 0.0
            )

            # Handle RMS - it might be 2D array
            rms_scalar = (
                float(np.mean(rms))
                if rms.ndim > 1
                else float(np.mean(rms[0])) if rms.size > 0 else 0.0
            )

            # Create 1D arrays for concatenation
            tempo_val = np.array([tempo_scalar])
            onset_mean = np.array([onset_scalar])
            rms_mean = np.array([rms_scalar])

            # Combine features
            features = np.concatenate(
                [
                    chroma_mean.ravel(),  # 12 - ensure flat array
                    tempo_val.ravel(),  # 1 - ensure flat array
                    onset_mean.ravel(),  # 1 - ensure flat array
                    rms_mean.ravel(),  # 1 - ensure flat array
                ]
            )

            return features

        except Exception as e:
            logger.error(f"Error extracting chroma features: {e}")
            return np.zeros(15)  # Return zeros in case of error

    def extract_spectrogram_analytical(
        self, audio: np.ndarray, mode: str = "magnitude"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract spectrogram from audio for analytical purposes (e.g., visualization).

        Args:
            audio: Audio time series
            mode: Type of spectrogram to extract ('magnitude' or 'mel')

        Returns:
            Tuple of (frequencies, times, spectrogram)
        """
        try:
            if mode == "magnitude":
                # Compute STFT
                stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)

                # Get magnitude
                magnitude = np.abs(stft)

                # Convert to dB
                spectrogram_db = librosa.amplitude_to_db(magnitude, ref=np.max)

                # Get frequency and time bins
                freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
                times = librosa.times_like(
                    spectrogram_db, sr=self.sample_rate, hop_length=self.hop_length
                )

                return freqs, times, spectrogram_db

            elif mode == "mel":
                # Extract mel-spectrogram using torchaudio
                mel_spec = self.spectrogram_transform(torch.tensor(audio).float())

                # Convert to numpy array and squeeze
                mel_spec = mel_spec.detach().numpy().squeeze()

                # Get mel frequencies and times
                mel_frequencies = np.linspace(0, 1, mel_spec.shape[0])
                times = librosa.times_like(
                    mel_spec, sr=self.sample_rate, hop_length=self.hop_length
                )

                return mel_frequencies, times, mel_spec

            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as e:
            logger.error(f"Error extracting analytical spectrogram: {e}")
            raise

    def extract_spectrogram(
        self, audio_input: Union[str, Path, np.ndarray], augment: bool = False
    ) -> torch.Tensor:
        """
        Extract mel spectrogram suitable for CNN model input.

        Args:
            audio_input: Path to audio file or audio array
            augment: Whether to apply augmentation to spectrogram

        Returns:
            Mel spectrogram as a tensor of shape [1, n_mels, time]
        """
        try:
            # Load audio if path is provided
            if isinstance(audio_input, (str, Path)):
                audio, sr = self.load_audio(audio_input)
            else:
                audio = audio_input

            # Convert to tensor
            waveform = torch.from_numpy(audio).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension if not present

            # Generate mel spectrogram
            mel_spectrogram = self.spectrogram_transform(waveform)

            # Apply log scaling (common practice for spectrograms)
            # Add a small value to avoid log(0)
            log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)

            # Normalize to range [0, 1] for easier model training
            log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.min()) / (
                log_mel_spectrogram.max() - log_mel_spectrogram.min() + 1e-9
            )

            # Apply augmentations if requested
            if augment:
                # Apply time masking
                if np.random.random() > 0.5:
                    log_mel_spectrogram = self.time_masking(log_mel_spectrogram)

                # Apply frequency masking
                if np.random.random() > 0.5:
                    log_mel_spectrogram = self.freq_masking(log_mel_spectrogram)

                # Apply random gain (volume augmentation in spectrogram domain)
                if np.random.random() > 0.5:
                    gain = torch.FloatTensor(1).uniform_(0.8, 1.2)
                    log_mel_spectrogram = log_mel_spectrogram * gain
                    # Re-normalize after gain adjustment
                    log_mel_spectrogram = (
                        log_mel_spectrogram - log_mel_spectrogram.min()
                    ) / (log_mel_spectrogram.max() - log_mel_spectrogram.min() + 1e-9)

                # Apply pitch shift via spectrogram manipulation (vertical shift)
                if np.random.random() > 0.5:
                    shift = int(np.random.uniform(-5, 5))
                    if shift > 0:
                        log_mel_spectrogram = torch.cat(
                            [
                                log_mel_spectrogram[:, shift:, :],
                                torch.zeros(1, shift, log_mel_spectrogram.shape[2]),
                            ],
                            dim=1,
                        )
                    elif shift < 0:
                        shift = abs(shift)
                        log_mel_spectrogram = torch.cat(
                            [
                                torch.zeros(1, shift, log_mel_spectrogram.shape[2]),
                                log_mel_spectrogram[:, :-shift, :],
                            ],
                            dim=1,
                        )

            return log_mel_spectrogram

        except Exception as e:
            logger.error(f"Error extracting spectrogram: {e}")
            # Return an empty spectrogram in case of error
            return torch.zeros(1, self.n_mels, self.n_fft // self.hop_length + 1)

    def batch_extract_spectrograms(
        self, audio_paths: List[Union[str, Path]], augment: bool = False
    ) -> torch.Tensor:
        """
        Extract spectrograms from a batch of audio files.

        Args:
            audio_paths: List of paths to audio files
            augment: Whether to apply augmentation

        Returns:
            Batch of spectrograms as a tensor of shape [batch_size, 1, n_mels, time]
        """
        spectrograms = []

        for path in audio_paths:
            spec = self.extract_spectrogram(path, augment)
            spectrograms.append(spec)

        # Stack spectrograms
        if spectrograms:
            # Pad or crop spectrograms to the same size
            max_length = max(s.shape[-1] for s in spectrograms)
            padded_specs = []

            for spec in spectrograms:
                if spec.shape[-1] < max_length:
                    # Pad
                    padding = torch.zeros(1, self.n_mels, max_length - spec.shape[-1])
                    padded_spec = torch.cat([spec, padding], dim=-1)
                else:
                    # Crop
                    padded_spec = spec[:, :, :max_length]

                padded_specs.append(padded_spec)

            return torch.stack(padded_specs)
        else:
            return torch.zeros(0, 1, self.n_mels, 1)

    def _extract_combined_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract combination of MFCC and spectral features."""
        mfcc_features = self._extract_mfcc_stats(audio)
        spectral_features = self._extract_spectral_features(audio)

        return np.concatenate([mfcc_features, spectral_features])

    def get_melspectrogram(
        self,
        audio: np.ndarray,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """
        Extract mel-spectrogram from audio.

        Args:
            audio: Audio time series
            n_mels: Number of mel filter banks
            n_fft: FFT window size
            hop_length: Hop length for STFT

        Returns:
            Mel-spectrogram
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
            )

            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            return mel_spec_db

        except Exception as e:
            logger.error(f"Error extracting mel-spectrogram: {e}")
            raise

    def plot_spectrogram(
        self,
        spectrogram: np.ndarray,
        sr: int = 22050,
        hop_length: int = 512,
        title: str = "Spectrogram",
        xlabel: str = "Time (s)",
        ylabel: str = "Frequency (Hz)",
        cmap: str = "inferno",
        save_path: Optional[Union[str, Path]] = None,
    ):
        """
        Plot spectrogram.

        Args:
            spectrogram: Spectrogram to plot
            sr: Sample rate of audio
            hop_length: Hop length used in STFT
            title: Title of the plot
            xlabel: X-axis label
            ylabel: Y-axis label
            cmap: Colormap to use
            save_path: Optional path to save the plot
        """
        try:
            # Convert to dB if not already
            if np.max(spectrogram) <= 1:  # Heuristic check
                spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

            # Time axis in seconds
            times = np.arange(spectrogram.shape[1]) * hop_length / sr

            # Plot
            plt.figure(figsize=(10, 4))
            plt.imshow(
                spectrogram,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                extent=[times.min(), times.max(), 0, sr / 2],
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tight_layout()

            # Save or show
            if save_path is not None:
                plt.savefig(save_path)
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Error plotting spectrogram: {e}")
            raise


class AudioDataset(Dataset):
    """
    Dataset for loading audio spectrograms from files on demand.
    This is more memory-efficient than loading all spectrograms at once.
    """

    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        audio_processor: AudioProcessor,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            file_paths: List of audio file paths
            labels: List of integer labels corresponding to file_paths
            audio_processor: AudioProcessor instance for feature extraction
            augment: Whether to apply augmentation
            max_samples: Maximum number of samples to use (useful for debugging)
        """
        if max_samples is not None and max_samples < len(file_paths):
            # Randomly sample files while preserving class distribution
            unique_labels = np.unique(labels)
            sampled_indices = []

            for label in unique_labels:
                label_indices = np.where(np.array(labels) == label)[0]
                n_samples = min(max_samples // len(unique_labels), len(label_indices))
                sampled_indices.extend(
                    np.random.choice(label_indices, size=n_samples, replace=False)
                )

            self.file_paths = [file_paths[i] for i in sampled_indices]
            self.labels = [labels[i] for i in sampled_indices]
        else:
            self.file_paths = file_paths
            self.labels = labels

        self.audio_processor = audio_processor
        self.augment = augment

        # Check for empty dataset
        if not self.file_paths:
            raise ValueError("No audio files provided to dataset")

        logger.info(
            f"Created AudioDataset with {len(self.file_paths)} files, augment={augment}"
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            # Get file path and label
            file_path = self.file_paths[idx]
            label = self.labels[idx]

            # Extract spectrogram
            spectrogram = self.audio_processor.extract_spectrogram(
                file_path, augment=self.augment
            )

            return spectrogram, label

        except Exception as e:
            logger.error(f"Error loading file {self.file_paths[idx]}: {e}")
            # Return empty spectrogram and original label on error
            empty_spec = torch.zeros(
                1,
                self.audio_processor.n_mels,
                self.audio_processor.n_fft // self.audio_processor.hop_length + 1,
            )
            return empty_spec, self.labels[idx]


class AudioAugmentor:
    """Applies augmentations to audio features or spectrograms."""

    def __init__(self):
        """Initialize audio augmentor."""
        pass

    def augment_features(
        self, features: np.ndarray, augment_type: str = "random"
    ) -> np.ndarray:
        """
        Apply augmentation to MFCC or other features.

        Args:
            features: Feature vector or matrix
            augment_type: Type of augmentation to apply

        Returns:
            Augmented features
        """
        if augment_type == "noise":
            # Add Gaussian noise
            noise_level = np.random.uniform(0.01, 0.04)
            return features + np.random.normal(0, noise_level, features.shape)

        elif augment_type == "emphasis":
            # Emphasize certain features
            mask = np.ones_like(features)
            mask_size = np.random.randint(1, max(2, features.shape[0] // 3))
            start = np.random.randint(0, features.shape[0] - mask_size)
            emphasis = np.random.uniform(1.2, 1.8)
            mask[start : start + mask_size] = emphasis
            return features * mask

        elif augment_type == "smoothing":
            # Smooth features
            kernel_size = np.random.choice([2, 3])
            kernel = np.ones(kernel_size) / kernel_size
            padded = np.pad(features, (kernel_size - 1, 0), mode="edge")
            return np.convolve(padded, kernel, mode="valid")[: features.shape[0]]

        elif augment_type == "scaling":
            # Scale features
            scale = np.random.uniform(0.9, 1.1)
            return features * scale

        elif augment_type == "random":
            # Choose a random augmentation
            aug_type = np.random.choice(["noise", "emphasis", "smoothing", "scaling"])
            return self.augment_features(features, aug_type)

        else:
            raise ValueError(f"Unknown augmentation type: {augment_type}")

    def augment_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to spectrogram.

        Args:
            spectrogram: Spectrogram tensor [channels, freq_bins, time_frames]

        Returns:
            Augmented spectrogram
        """
        # Create spectrogram-specific augmentations here
        aug_spec = spectrogram.clone()

        # Time masking
        time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=spectrogram.shape[2] // 8
        )
        aug_spec = time_mask(aug_spec)

        # Frequency masking
        freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=spectrogram.shape[1] // 8
        )
        aug_spec = freq_mask(aug_spec)

        return aug_spec


def apply_augmentation(
    audio: np.ndarray, augmentation_params: Dict[str, float], sample_rate: int = 22050
) -> np.ndarray:
    """
    Apply augmentation to audio based on parameters.

    Args:
        audio: Input audio array
        augmentation_params: Dictionary with augmentation parameters
        sample_rate: Sample rate of audio

    Returns:
        Augmented audio
    """
    augmentor = AudioAugmentor(sample_rate)

    return augmentor.augment_audio(
        audio,
        pitch_shift_semitones=augmentation_params.get("pitch_shift", 0),
        time_stretch_rate=augmentation_params.get("time_stretch", 1.0),
        noise_factor=augmentation_params.get("noise_factor", 0),
    )


def preprocess_for_model(
    audio: np.ndarray, target_length: int, sample_rate: int = 22050
) -> np.ndarray:
    """
    Preprocess audio for model input.

    Args:
        audio: Input audio
        target_length: Target length in samples
        sample_rate: Sample rate

    Returns:
        Preprocessed audio
    """
    # Normalize
    audio = librosa.util.normalize(audio)

    # Pad or truncate to target length
    if len(audio) < target_length:
        # Pad with zeros
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    elif len(audio) > target_length:
        # Truncate
        audio = audio[:target_length]

    return audio


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute spectrogram from audio.

    Args:
        audio: Input audio
        sample_rate: Sample rate
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        Spectrogram magnitude
    """
    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # Get magnitude
    magnitude = np.abs(stft)

    # Convert to dB
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    return magnitude_db


def extract_features_from_audio(audio_file):
    """
    Extract comprehensive feature set from audio file.

    Extracts 52 features including MFCCs, spectral features, chroma, tempo,
    and other audio characteristics for music genre classification.

    Args:
        audio_file: Path to audio file or file-like object

    Returns:
        tuple: (features, audio_data, sample_rate) where features is a numpy
               array of 52 extracted features, or (None, None, None) on error
    """
    try:
        audio_data, sr = librosa.load(audio_file, sr=22050, duration=30.0)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        # Additional features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)

        # Ensure all arrays have proper dimensions before concatenation
        mfccs_mean = np.mean(mfccs, axis=1)  # 1D array of 13 elements
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)  # 1D array of 13 elements
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)  # 1D array of 13 elements

        # Make sure these are all 1D arrays
        spectral_stats = np.array(
            [np.mean(spectral_centroids), np.std(spectral_centroids)]
        )  # 1D array length 2
        rolloff_stats = np.array(
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)]
        )  # 1D array length 2
        zcr_stats = np.array([np.mean(zcr), np.std(zcr)])  # 1D array length 2

        chroma_mean = np.mean(chroma, axis=1)  # 1D array length 12
        tempo_val = np.array([tempo])  # 1D array length 1
        onset_mean = np.array(
            [np.mean(librosa.onset.onset_strength(y=audio_data, sr=sr))]
        )  # 1D array length 1
        rms_mean = np.array(
            [np.mean(librosa.feature.rms(y=audio_data))]
        )  # 1D array length 1
        bandwidth_mean = np.array(
            [np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))]
        )  # 1D array length 1

        # Combine features (52 total)
        features = np.concatenate(
            [
                mfccs_mean,  # 13
                mfcc_delta_mean,  # 13
                mfcc_delta2_mean,  # 13
                spectral_stats,  # 2
                rolloff_stats,  # 2
                zcr_stats,  # 2
                chroma_mean,  # 12
                tempo_val,  # 1
                onset_mean,  # 1
                rms_mean,  # 1
                bandwidth_mean,  # 1
            ]
        )

        # Ensure exactly 52 features
        if len(features) < 52:
            features = np.pad(features, (0, 52 - len(features)), "constant")
        elif len(features) > 52:
            features = features[:52]

        return features, audio_data, sr

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return None, None, None
