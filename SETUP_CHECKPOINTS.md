# 🤖 CIHP_PGN Checkpoint Setup Guide

This guide will help you download the pre-trained model checkpoints for CIHP_PGN human parsing.

## 📥 Quick Setup (Recommended)

### Option 1: Python Script (Recommended)
```bash
# Navigate to CIHP_PGN directory
cd CIHP_PGN

# Run the Python download script
python download_checkpoint.py
```

### Option 2: Bash Script
```bash
# Navigate to CIHP_PGN directory  
cd CIHP_PGN

# Make script executable and run
chmod +x download_checkpoint.sh
./download_checkpoint.sh
```

### Option 3: Download All Checkpoints (TryOn-Adapter + CIHP_PGN)
```bash
# From project root, run the comprehensive download script
cd FRONTEND/01\ -\ FILES/
chmod +x download_checkpoints.sh
./download_checkpoints.sh
```

## 🔗 Manual Download

If automated download fails, you can download manually:

1. **Go to Google Drive**: https://drive.google.com/open?id=1Mqpse5Gen4V4403wFEpv3w3JAsWw2uhk
2. **Download the checkpoint file**
3. **Create directory**: `mkdir -p CIHP_PGN/checkpoint/CIHP_pgn`
4. **Extract** the downloaded file to `CIHP_PGN/checkpoint/CIHP_pgn/`

## 📁 Expected Directory Structure

After successful download, you should have:

```
CIHP_PGN/
├── checkpoint/
│   └── CIHP_pgn/
│       ├── [checkpoint files]
│       └── [model weights]
├── utils/
├── single_image_inference.py
├── test_pgn.py
├── inf_pgn.py
└── download_checkpoint.py
```

## 🧪 Testing the Setup

Once checkpoints are downloaded, test the setup:

```bash
# Test single image inference
python single_image_inference.py --input /path/to/test_image.jpg --output /path/to/output.png

# Test batch inference  
python inf_pgn.py --directory /path/to/images --output /path/to/results
```

## 🔧 Troubleshooting

### Issue: "Checkpoint not found"
**Solution**: Ensure checkpoints are in `./checkpoint/CIHP_pgn/` directory

### Issue: "gdown not installed"
**Solution**: Install gdown with `pip install gdown`

### Issue: "Download fails from Google Drive"
**Solution**: Use manual download method above

### Issue: "TensorFlow errors"
**Solution**: Ensure TensorFlow 2.13+ is installed: `pip install tensorflow>=2.13.0`

## ⚙️ Dependencies

Make sure you have the required dependencies:

```bash
# Install Python dependencies
pip install -r requirements.pip

# Or install manually:
pip install tensorflow>=2.13.0 opencv-python>=4.2.0 Pillow>=9.0.0 scipy>=1.7.0 numpy>=1.21.0 gdown
```

## 📊 Model Information

- **Model**: CIHP_PGN (Part Grouping Network)
- **Paper**: "Instance-level Human Parsing via Part Grouping Network", ECCV 2018
- **Dataset**: CIHP (Crowd Instance-level Human Parsing)
- **Classes**: 20 human body parts
- **Input Size**: 512x512 pixels

## 🎯 Next Steps

After successful setup:
1. ✅ CIHP_PGN checkpoints downloaded
2. ✅ Test single image inference
3. ✅ Integrate with your AI clothes person pipeline
4. ✅ Run full TryOn-Adapter + CIHP_PGN workflow

For issues, check the [original repository](https://github.com/Engineering-Course/CIHP_PGN) or create an issue. 