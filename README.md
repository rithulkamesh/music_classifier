# 🎵 Music Genre Classifier

A Streamlit-based music genre classifier using the FMA dataset with multiple model architectures.

## ✨ Features

- **🎯 Real-time Classification**: Upload audio files and get instant genre predictions
- **🗺️ Interactive Visualization**: UMAP embeddings showing music genre landscape
- **📊 Comprehensive Analysis**: Feature extraction and model performance insights
- **🔄 Multiple Models**: Choose between Simple CNN demo model and high-performance models
- **🎧 Multi-format Support**: WAV, MP3, FLAC, M4A audio files

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/rithulkamesh/music_classifier
cd music_classifier

# Set up environment with uv
uv venv
source .venv/bin/activate
uv pip install -e .

# Train a quick demo model (~1-2 minutes)
chmod +x scripts/train_simple_cnn.sh
./scripts/train_simple_cnn.sh

# Run the Streamlit app
uv run streamlit run src/app.py
```

### 🐧 NixOS Users

If you're using NixOS and encounter issues with audio libraries, use the provided Nix shell:

```bash
# Use the nix shell environment
nix-shell shell.nix

# Inside the nix-shell, run the app
uv run streamlit run src/app.py
```

## 📂 Project Structure

```
music_classifier/
├── scripts/                  # Training scripts
│   ├── train_simple_cnn.py   # Simple CNN model training script
│   └── train_simple_cnn.sh   # Training shell script
├── src/                      # Source code
│   ├── app.py                # Main Streamlit application
│   ├── audio_processing.py   # Audio feature extraction
│   ├── data_loader.py        # FMA dataset loading & preprocessing
│   ├── models.py             # Neural network model architectures
│   ├── preprocess.py         # Data preprocessing script
│   └── train.py              # Model training core functionality
├── data/                     # Data directory
│   ├── fma_medium/           # FMA Medium audio files
│   ├── fma_metadata/         # FMA metadata files
│   ├── embeddings/           # UMAP embeddings
│   │   ├── umap_embeddings.pkl        # UMAP embeddings for visualization
│   │   └── umap_visualization.png     # Pre-generated UMAP plot
│   └── preprocessed/         # Preprocessed features
├── checkpoints/              # Trained model checkpoints
└── pyproject.toml            # Dependencies and project metadata
```

## 📊 Data Directory Structure

```
data/
├── fma_medium/           # Free Music Archive dataset (audio files)
├── fma_metadata/         # FMA metadata files
├── preprocessed/         # Processed audio features
├── embeddings/           # UMAP visualization data
│   ├── umap_embeddings.pkl        # UMAP embeddings for visualization
│   └── umap_visualization.png     # Pre-generated UMAP plot
├── samples/              # Sample audio files for testing
└── genre_names.json      # Genre name mappings
```

## 🎵 Models

### Simple CNN Demo Model

- **Architecture**: 52 → 512 → 512 → 256 → 128 → 13
- **Training Time**: 1-2 minutes
- **Accuracy**: Loads dynamically from model checkpoint
- **Best For**: Quick demos and testing

### High-Performance Model (Advanced)

- **Architecture**: Deep residual connections with attention
- **Features**: Spectrogram-based with transfer learning
- **Training Time**: ~10-15 minutes with optimizations
- **Accuracy**: 98.6% target
- **Best For**: Production use and highest accuracy

## 🔧 Usage

### Training Models

```bash
# Train a quick demo model
./scripts/train_simple_cnn.sh

# Train the high-performance model (takes longer)
uv run python src/run_training.py
```

### Running the App

```bash
# Start the Streamlit app
uv run streamlit run src/app.py
```

## 🎵 Audio Features (52 total)

- **MFCC (39)**: 13 coefficients + 13 delta + 13 delta²
- **Spectral (8)**: Centroid, rolloff, ZCR, bandwidth (mean/std)
- **Harmonic (4)**: 12 chroma features + tempo
- **Temporal (1)**: Onset strength + RMS energy

## 🎯 Supported Genres

The model classifies 13 genres including Blues, Classical, Country, Electronic, Folk, Hip-Hop, Jazz, Pop, Rock, Soul-RnB, and others.

## �️ Requirements

- Python 3.11+
- PyTorch 2.0+
- Streamlit
- Librosa
- UMAP-learn
- Plotly

All dependencies are managed with `uv` for fast, reliable dependency management.

---

**Built with ❤️ using modern ML techniques for audio processing**
