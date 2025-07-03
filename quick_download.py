#!/usr/bin/env python3
"""
Quick CIHP_PGN checkpoint download for API environment
"""

import os
import subprocess
import sys

def download_checkpoint():
    """Download CIHP_PGN checkpoint to the correct location"""
    
    # Determine checkpoint directory based on environment
    if os.path.exists('/home/paperspace'):
        # Paperspace/API environment
        checkpoint_dir = '/home/paperspace/checkpoint/CIHP_pgn'
        base_dir = '/home/paperspace/CIHP_PGN'
    else:
        # Local environment
        checkpoint_dir = './checkpoint/CIHP_pgn'
        base_dir = '.'
    
    print(f"🔍 Checking checkpoint at: {checkpoint_dir}")
    
    # Check if checkpoint already exists
    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
        print(f"✅ Checkpoint already exists at: {checkpoint_dir}")
        return True
    
    print(f"📥 Downloading checkpoint to: {checkpoint_dir}")
    
    # Create directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Install gdown if not available
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], 
                      check=False, capture_output=True)
        
        # Download checkpoint
        import gdown
        
        file_id = "1Mqpse5Gen4V4403wFEpv3w3JAsWw2uhk"
        zip_path = os.path.join(os.path.dirname(checkpoint_dir), "CIHP_pgn.zip")
        
        print(f"⬇️  Downloading from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)
        
        if os.path.exists(zip_path):
            print(f"📂 Extracting checkpoint...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(checkpoint_dir)
            
            os.remove(zip_path)
            print(f"✅ Checkpoint downloaded successfully!")
            return True
        else:
            print(f"❌ Download failed")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading checkpoint: {e}")
        return False

if __name__ == "__main__":
    success = download_checkpoint()
    sys.exit(0 if success else 1) 