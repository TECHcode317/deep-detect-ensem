#!/bin/bash
set -e  # Exit on error

echo "=== Downloading Deepfake Dataset ==="

# Install Kaggle API if not present
pip install kaggle

# Setup Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Please place your kaggle.json in ~/.kaggle/"
    echo "Download from: https://www.kaggle.com/settings/account"
    exit 1
fi

chmod 600 ~/.kaggle/kaggle.json

# Download dataset
echo "Downloading dataset..."
kaggle datasets download -d nanduncs/1000-deepfake-videos

# Extract
echo "Extracting files..."
unzip -q 1000-deepfake-videos.zip -d data/

# Verify
echo "Verifying download..."
FAKE_COUNT=$(ls -1 data/1000_videos/Fake/*.png 2>/dev/null | wc -l)
REAL_COUNT=$(ls -1 data/1000_videos/Real/*.png 2>/dev/null | wc -l)

echo "Fake frames: $FAKE_COUNT"
echo "Real frames: $REAL_COUNT"

if [ "$FAKE_COUNT" -lt 1000 ] || [ "$REAL_COUNT" -lt 1000 ]; then
    echo "ERROR: Dataset incomplete!"
    exit 1
fi

echo "✓ Dataset ready: data/1000_videos/"