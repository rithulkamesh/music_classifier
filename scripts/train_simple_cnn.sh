#!/bin/bash
# Simple CNN training script for quick demo
# Expected to train in 1-2 minutes with ~70% accuracy

echo "🎵 Training Simple CNN model for music genre classification..."
echo "This will be quick, optimized for demo purposes."

# Make script executable
chmod +x scripts/train_simple_cnn.py

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Run the training script
uv run python scripts/train_simple_cnn.py \
    --data-file data/preprocessed/features.pkl \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.001

echo "✅ Training complete! Model saved to checkpoints/simple_cnn_best.pt"
echo "📊 You can now run the demo app: uv run streamlit run src/app.py"
