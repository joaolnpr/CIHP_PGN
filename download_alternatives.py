#!/usr/bin/env python3
"""
Alternative CIHP_PGN checkpoint download methods
"""

import os
import sys
import subprocess
import time

def try_wget_download():
    """Try downloading with wget using Google Drive direct link"""
    print("🔄 Trying wget method...")
    
    file_id = "1Mqpse5Gen4V4403wFEpv3w3JAsWw2uhk"
    checkpoint_dir = "/home/paperspace/checkpoint/CIHP_pgn"
    zip_path = "/home/paperspace/checkpoint/CIHP_pgn.zip"
    
    # Create directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Try wget with cookies
    wget_cmd = [
        "wget", "--load-cookies", "/tmp/cookies.txt",
        f"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={file_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={file_id}",
        "-O", zip_path
    ]
    
    try:
        result = subprocess.run(wget_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(zip_path):
            print("✅ wget download successful!")
            return zip_path
        else:
            print(f"❌ wget failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ wget error: {e}")
        return None

def try_curl_download():
    """Try downloading with curl"""
    print("🔄 Trying curl method...")
    
    file_id = "1Mqpse5Gen4V4403wFEpv3w3JAsWw2uhk"
    checkpoint_dir = "/home/paperspace/checkpoint/CIHP_pgn"
    zip_path = "/home/paperspace/checkpoint/CIHP_pgn.zip"
    
    curl_cmd = [
        "curl", "-L", "-o", zip_path,
        f"https://drive.google.com/uc?export=download&id={file_id}"
    ]
    
    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(zip_path):
            # Check if it's actually a zip file (not an error page)
            with open(zip_path, 'rb') as f:
                header = f.read(4)
                if header == b'PK\x03\x04':  # ZIP file header
                    print("✅ curl download successful!")
                    return zip_path
                else:
                    print("❌ curl downloaded error page, not actual file")
                    os.remove(zip_path)
                    return None
        else:
            print(f"❌ curl failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ curl error: {e}")
        return None

def try_alternative_sources():
    """Check for alternative download sources"""
    print("🔍 Checking alternative sources...")
    
    # List of potential alternative sources
    alternatives = [
        "https://huggingface.co/models?search=CIHP_PGN",
        "https://github.com/Engineering-Course/CIHP_PGN/releases",
        "https://paperswithcode.com/paper/instance-level-human-parsing-via-part"
    ]
    
    print("📋 Alternative sources to check manually:")
    for i, url in enumerate(alternatives, 1):
        print(f"  {i}. {url}")
    
    return None

def extract_checkpoint(zip_path):
    """Extract the downloaded checkpoint"""
    try:
        import zipfile
        checkpoint_dir = "/home/paperspace/checkpoint/CIHP_pgn"
        
        print(f"📂 Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(checkpoint_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        
        # List contents
        print("📋 Extracted files:")
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"  - {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def main():
    print("🤖 CIHP_PGN Alternative Checkpoint Downloader")
    print("=" * 60)
    
    # Check if checkpoint already exists and has files
    checkpoint_dir = "/home/paperspace/checkpoint/CIHP_pgn"
    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
        print("✅ Checkpoint already exists!")
        return True
    
    print("📥 Attempting alternative download methods...")
    
    # Method 1: Try wget
    zip_path = try_wget_download()
    if zip_path and extract_checkpoint(zip_path):
        return True
    
    # Wait a bit between attempts
    print("⏳ Waiting 5 seconds before next attempt...")
    time.sleep(5)
    
    # Method 2: Try curl
    zip_path = try_curl_download()
    if zip_path and extract_checkpoint(zip_path):
        return True
    
    # Method 3: Show alternatives
    try_alternative_sources()
    
    print("\n❌ All automatic download methods failed.")
    print("🔧 Manual solutions:")
    print("1. Wait 24 hours for Google Drive quota to reset")
    print("2. Use a different IP/VPN to access Google Drive")
    print("3. Check the alternative sources listed above")
    print("4. Contact the CIHP_PGN authors for direct download")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 