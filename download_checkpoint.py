#!/usr/bin/env python3
"""
CIHP_PGN Checkpoint Download Script
Downloads the official pre-trained model from Google Drive
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

# https://drive.google.com/file/d//view?usp=sharing
# Google Drive file ID from official repository
GDRIVE_FILE_ID = "1pnZKaMlxzNWId78YLBoj7iFniradsBmi"
CHECKPOINT_DIR = "./checkpoint/CIHP_pgn"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

def install_gdown():
    """Install gdown if not available"""
    try:
        import gdown
        return True
    except ImportError:
        print("📦 Installing gdown for Google Drive downloads...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
            return True
        except Exception as e:
            print(f"❌ Failed to install gdown: {e}")
            return False

def download_checkpoint():
    """Download and extract CIHP_PGN checkpoint"""
    
    print("🤖 CIHP_PGN Checkpoint Downloader")
    print("=" * 50)
    
    # Check if checkpoint already exists
    if os.path.exists(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR):
        print(f"✅ Checkpoint already exists at: {CHECKPOINT_DIR}")
        print("📋 Contents:")
        for item in os.listdir(CHECKPOINT_DIR):
            print(f"  - {item}")
        
        response = input("\n🤔 Do you want to re-download? (y/N): ").lower().strip()
        if response not in ['y', 'yes']:
            print("✨ Using existing checkpoint.")
            return True
    
    # Install gdown if needed
    if not install_gdown():
        print("❌ Cannot install gdown. Please install manually: pip install gdown")
        return False
    
    import gdown
    
    # Create directories
    os.makedirs("./checkpoint", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    try:
        print(f"⬇️  Downloading checkpoint from Google Drive...")
        print(f"🔗 URL: {DOWNLOAD_URL}")
        
        # Download to temporary zip file
        zip_path = "./checkpoint/CIHP_pgn.zip"
        gdown.download(DOWNLOAD_URL, zip_path, quiet=False)
        
        if not os.path.exists(zip_path):
            raise Exception("Download failed - zip file not found")
        
        print(f"✅ Download completed: {zip_path}")
        print(f"📊 File size: {os.path.getsize(zip_path) / (1024*1024):.1f} MB")
        
        # Extract the checkpoint
        print("📂 Extracting checkpoint...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(CHECKPOINT_DIR)
        
        # Clean up zip file
        os.remove(zip_path)
        
        print("🎉 CIHP_PGN checkpoint downloaded and extracted successfully!")
        print(f"📁 Checkpoint location: {os.path.abspath(CHECKPOINT_DIR)}")
        
        # List contents
        print("📋 Checkpoint contents:")
        for root, dirs, files in os.walk(CHECKPOINT_DIR):
            level = root.replace(CHECKPOINT_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024*1024)
                print(f"{subindent}{file} ({file_size:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        print("\n🔧 Manual download instructions:")
        print(f"1. Go to: {DOWNLOAD_URL}")
        print("2. Download the file manually")
        print(f"3. Extract it to: {os.path.abspath(CHECKPOINT_DIR)}")
        print("4. Ensure the checkpoint files are in the correct location")
        return False

def verify_checkpoint():
    """Verify that the checkpoint is properly installed"""
    
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Checkpoint directory not found: {CHECKPOINT_DIR}")
        return False
    
    # Look for common checkpoint file patterns
    checkpoint_files = []
    for root, dirs, files in os.walk(CHECKPOINT_DIR):
        for file in files:
            if any(ext in file.lower() for ext in ['.ckpt', '.pth', '.pkl', '.pb', '.index', '.data']):
                checkpoint_files.append(os.path.join(root, file))
    
    if checkpoint_files:
        print(f"✅ Checkpoint verification passed!")
        print(f"📄 Found {len(checkpoint_files)} checkpoint files")
        return True
    else:
        print(f"⚠️  Warning: No checkpoint files found in {CHECKPOINT_DIR}")
        print("📋 Directory contents:")
        for item in os.listdir(CHECKPOINT_DIR):
            print(f"  - {item}")
        return False

def main():
    """Main function"""
    
    # Download checkpoint
    if download_checkpoint():
        # Verify installation
        if verify_checkpoint():
            print("\n✨ Setup complete! You can now run CIHP_PGN human parsing.")
            print("\n🚀 Quick test:")
            print("python single_image_inference.py --input /path/to/image.jpg --output /path/to/output.png")
        else:
            sys.exit(1)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 