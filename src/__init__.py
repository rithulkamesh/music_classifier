"""Music Genre Classifier

A high-performance deep learning system for classifying music genres
from audio files using MFCC features and residual neural networks.

Author: [Your Name]
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

from .audio_processing import extract_features_from_audio
from .data_loader import FMADataLoader
from .models import EnhancedCNN, create_model, ModelType

__all__ = [
    "EnhancedCNN",
    "create_model",
    "ModelType",
    "extract_features_from_audio",
    "FMADataLoader",
]
