#!/bin/bash

# CIHP_PGN Checkpoint Download Script
# Based on the official repository: https://github.com/Engineering-Course/CIHP_PGN

echo "🤖 Downloading CIHP_PGN Pre-trained Model Checkpoint..."

# Create checkpoint directory
mkdir -p checkpoint
cd checkpoint

# Google Drive file ID for CIHP_PGN pre-trained model
# From the official repository: https://drive.google.com/open?id=1Mqpse5Gen4V4403wFEpv3w3JAsWw2uhk
FILE_ID="1Mqpse5Gen4V4403wFEpv3w3JAsWw2uhk"
CHECKPOINT_DIR="CIHP_pgn"

echo "📁 Creating checkpoint directory: $CHECKPOINT_DIR"
mkdir -p $CHECKPOINT_DIR

# Download using wget with Google Drive direct link
echo "⬇️  Downloading from Google Drive..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILE_ID" -O CIHP_pgn.zip && rm -rf /tmp/cookies.txt

# Alternative method using gdown (if available)
if command -v gdown &> /dev/null; then
    echo "📥 Using gdown to download..."
    gdown https://drive.google.com/uc?id=$FILE_ID -O CIHP_pgn.zip
else
    echo "ℹ️  gdown not found, using wget method"
fi

# Check if download was successful
if [ -f "CIHP_pgn.zip" ]; then
    echo "✅ Download successful! Extracting..."
    
    # Extract the checkpoint
    unzip -q CIHP_pgn.zip -d $CHECKPOINT_DIR
    
    # Clean up zip file
    rm CIHP_pgn.zip
    
    echo "🎉 CIHP_PGN checkpoint downloaded and extracted successfully!"
    echo "📁 Checkpoint location: ./checkpoint/$CHECKPOINT_DIR"
    
    # List contents
    echo "📋 Checkpoint contents:"
    ls -la $CHECKPOINT_DIR/
    
else
    echo "❌ Download failed. Please try manual download from:"
    echo "🔗 https://drive.google.com/open?id=1Mqpse5Gen4V4403wFEpv3w3JAsWw2uhk"
    echo ""
    echo "Manual steps:"
    echo "1. Download the file from the Google Drive link above"
    echo "2. Extract it to: ./checkpoint/CIHP_pgn/"
    echo "3. Ensure the checkpoint files are in the correct location"
    exit 1
fi

cd ..
echo "✨ Setup complete! You can now run human parsing inference." 