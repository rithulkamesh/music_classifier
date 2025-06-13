#!/usr/bin/env python3
"""
Music Genre Classifier Model Definitions

This module contains the CNN models used for music genre classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import enum


class ModelType(enum.Enum):
    """Enumeration of available model types."""

    SIMPLE = "simple"
    ENHANCED = "enhanced"
    FAST = "fast"


def create_model(model_type: str, input_size: int = 52, num_classes: int = 13):
    """
    Factory function to create model based on type.

    Args:
        model_type: Type of model to create (from ModelType enum)
        input_size: Size of input features
        num_classes: Number of output classes

    Returns:
        Instantiated model
    """
    if model_type == ModelType.SIMPLE.value:
        return SimpleCNN(input_size, num_classes)
    elif model_type == ModelType.ENHANCED.value:
        return EnhancedCNN(input_size, num_classes)
    elif model_type == ModelType.FAST.value:
        return FastCNN(input_size, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class EnhancedCNN(nn.Module):
    """
    Enhanced CNN model for music genre classification.

    Structure:  52 → 512 → 512 → 256 → 128 → 13

    Features:
    - Linear Layers: Extract and combine patterns
    - BatchNorm: Normalize, speed up learning
    - Dropout: Prevent overfitting
    - Final Layer: 13 outputs = genre scores
    """

    def __init__(
        self, input_size: int = 52, num_classes: int = 13, dropout_rate: float = 0.3
    ):
        """
        Initialize the model.

        Args:
            input_size: Size of input features (default: 52)
            num_classes: Number of output classes (default: 13)
            dropout_rate: Dropout probability (default: 0.3)
        """
        super(EnhancedCNN, self).__init__()

        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(input_size)

        # Layer 1: Input -> 512
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Layer 2: 512 -> 512 with residual connection
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Layer 3: 512 -> 256
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Layer 4: 256 -> 128
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Output layer: 128 -> num_classes
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        """Forward pass for the model."""
        # Input normalization
        x = self.input_norm(x)

        # Layer 1
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout1(x1)

        # Layer 2 with residual connection
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout2(x2)
        x2 = x2 + x1  # Residual connection

        # Layer 3
        x3 = self.fc3(x2)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.dropout3(x3)

        # Layer 4
        x4 = self.fc4(x3)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)
        x4 = self.dropout4(x4)

        # Output layer
        out = self.fc_out(x4)

        return out


class SimpleCNN(nn.Module):
    """
    Simplified CNN for quick training and demos.

    Structure:  52 → 512 → 512 → 256 → 128 → 13

    Features:
    - Linear Layers: Extract and combine patterns
    - BatchNorm: Normalize, speed up learning
    - Dropout: Prevent overfitting
    - Final Layer: 13 outputs = genre scores
    - ~70% expected accuracy
    """

    def __init__(
        self, input_size: int = 52, num_classes: int = 13, dropout_rate: float = 0.2
    ):
        """
        Initialize the model.

        Args:
            input_size: Size of input features (default: 52)
            num_classes: Number of output classes (default: 13)
            dropout_rate: Dropout probability (default: 0.2)
        """
        super(SimpleCNN, self).__init__()

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)

        # Layer 1: Input -> 512
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Layer 2: 512 -> 512
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Layer 3: 512 -> 256
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Layer 4: 256 -> 128
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Output layer: 128 -> num_classes
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        """Forward pass for the model."""
        # Input normalization
        x = self.input_norm(x)

        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Layer 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        # Output layer
        out = self.fc_out(x)

        return out


class FastCNN(nn.Module):
    """
    Fast CNN model for near real-time inference.

    Structure: 52 → 128 → 64 → 13

    Features:
    - Compact architecture for fast inference
    - Minimal regularization
    - Best for demo applications
    """

    def __init__(self, input_size: int = 52, num_classes: int = 13):
        """
        Initialize the model.

        Args:
            input_size: Size of input features (default: 52)
            num_classes: Number of output classes (default: 13)
        """
        super(FastCNN, self).__init__()

        # Layer 1: Input -> 128
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)

        # Layer 2: 128 -> 64
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        # Output layer: 64 -> num_classes
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        """Forward pass for the model."""
        # Layer 1
        x = F.relu(self.bn1(self.fc1(x)))

        # Layer 2
        x = F.relu(self.bn2(self.fc2(x)))

        # Output layer
        out = self.fc_out(x)

        return out
