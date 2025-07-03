from __future__ import print_function
import argparse
import os
import sys
import cv2
from glob import glob
import numpy as np
import tensorflow as tf
from utils.image_reade_inf import create_inference_dataset
from utils.pgn_keras import PGNKeras

def main():
    argp = argparse.ArgumentParser(description="Inference pipeline")
    argp.add_argument('-i', '--directory', type=str, help='Path of the input dir', default='./datasets/images')
    argp.add_argument('-o', '--output', type=str, help='Path of the output dir', default='./datasets/output')
    argp.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    argp.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process (default: all)')
    args = argp.parse_args()

    # Check if input directory exists
    if not os.path.exists(args.directory):
        print(f"[ERROR] Input directory does not exist: {args.directory}")
        sys.exit(1)

    # Get list of image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_list_inp = []
    
    for ext in image_extensions:
        image_list_inp.extend(glob(os.path.join(args.directory, f'**/{ext}'), recursive=True))
        image_list_inp.extend(glob(os.path.join(args.directory, f'**/{ext.upper()}'), recursive=True))
    
    # Filter to only actual files
    image_list_inp = [i for i in image_list_inp if os.path.isfile(i)]
    
    if not image_list_inp:
        print(f"[ERROR] No image files found in directory: {args.directory}")
        print(f"[INFO] Supported formats: {', '.join(image_extensions)}")
        sys.exit(1)
    
    # Limit images if specified
    if args.max_images is not None:
        image_list_inp = image_list_inp[:args.max_images]
        print(f"[INFO] Processing {len(image_list_inp)} images (limited by max_images)")
    else:
        print(f"[INFO] Found {len(image_list_inp)} images to process")

    N_CLASSES = 20
    RESTORE_FROM = './checkpoint/CIHP_pgn'

    # Check if checkpoint exists
    if not os.path.exists(RESTORE_FROM):
        print(f"[ERROR] Checkpoint directory not found: {RESTORE_FROM}")
        print(f"[INFO] Please ensure the model checkpoint is available")
        sys.exit(1)

    try:
        # Create dataset
        dataset = create_inference_dataset(
            image_list_inp, 
            input_size=(512, 512), 
            random_scale=False, 
            random_mirror=False, 
            batch_size=args.batch_size, 
            shuffle=False
        )

        # Load model
        print(f"[INFO] Loading model...")
        model = PGNKeras(n_classes=N_CLASSES, checkpoint_path=RESTORE_FROM)

        # Create output directories
        parsing_dir = os.path.join(args.output, 'cihp_parsing_maps')
        os.makedirs(parsing_dir, exist_ok=True)
        edge_dir = os.path.join(args.output, 'cihp_edge_maps')
        os.makedirs(edge_dir, exist_ok=True)

        print(f"[INFO] Starting inference...")
        processed_count = 0

        for step, batch in enumerate(dataset):
            try:
                parsing_fc, parsing_rf_fc, edge_rf_fc = model(batch)
                parsing_out = tf.argmax(parsing_fc, axis=-1)
                edge_out = tf.sigmoid(edge_rf_fc)
                parsing_np = parsing_out.numpy().astype(np.uint8)
                edge_np = (edge_out.numpy() > 0.5).astype(np.uint8) * 255
                
                for i in range(parsing_np.shape[0]):
                    img_idx = step * args.batch_size + i
                    if img_idx < len(image_list_inp):
                        img_id = os.path.splitext(os.path.basename(image_list_inp[img_idx]))[0]
                        cv2.imwrite(f'{parsing_dir}/{img_id}.png', parsing_np[i])
                        cv2.imwrite(f'{edge_dir}/{img_id}.png', edge_np[i])
                        print(f"Saved: {img_id}")
                        processed_count += 1
                        
            except Exception as e:
                print(f"[ERROR] Failed to process batch {step}: {str(e)}")
                continue

        print(f"[INFO] Inference completed. Processed {processed_count} images.")
        print(f"[INFO] Results saved to:")
        print(f"  - Parsing maps: {parsing_dir}")
        print(f"  - Edge maps: {edge_dir}")
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()


##############################################################333
