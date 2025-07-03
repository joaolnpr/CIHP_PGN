#!/usr/bin/env python3
"""
Minimal CIHP_PGN test that works without checkpoint files
This creates a basic segmentation mask for testing the API pipeline
"""

import os
import sys
import cv2
import numpy as np

def create_basic_human_mask(input_image_path, output_path):
    """Create a basic human segmentation mask using simple computer vision"""
    
    try:
        print(f"[INFO] Creating basic human mask for: {input_image_path}")
        
        # Read image
        img = cv2.imread(input_image_path)
        if img is None:
            raise ValueError(f"Could not read image: {input_image_path}")
        
        h, w = img.shape[:2]
        
        # Create a simple human-like mask
        # This is a placeholder that creates a basic person silhouette
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Simple approach: assume person is in center area of image
        center_x, center_y = w // 2, h // 2
        
        # Create basic body parts regions (very simplified)
        # Head region (class 1)
        head_region = (slice(max(0, center_y - h//4), min(h, center_y - h//8)),
                      slice(max(0, center_x - w//8), min(w, center_x + w//8)))
        mask[head_region] = 1
        
        # Torso region (class 5) 
        torso_region = (slice(max(0, center_y - h//8), min(h, center_y + h//4)),
                       slice(max(0, center_x - w//6), min(w, center_x + w//6)))
        mask[torso_region] = 5
        
        # Arms regions (class 14, 15)
        left_arm_region = (slice(max(0, center_y - h//12), min(h, center_y + h//6)),
                          slice(max(0, center_x - w//3), min(w, center_x - w//8)))
        mask[left_arm_region] = 14
        
        right_arm_region = (slice(max(0, center_y - h//12), min(h, center_y + h//6)),
                           slice(max(0, center_x + w//8), min(w, center_x + w//3)))
        mask[right_arm_region] = 15
        
        # Legs regions (class 16, 17)
        left_leg_region = (slice(max(0, center_y + h//6), min(h, center_y + h//2)),
                          slice(max(0, center_x - w//12), min(w, center_x + w//24)))
        mask[left_leg_region] = 16
        
        right_leg_region = (slice(max(0, center_y + h//6), min(h, center_y + h//2)),
                           slice(max(0, center_x - w//24), min(w, center_x + w//12)))
        mask[right_leg_region] = 17
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save mask
        cv2.imwrite(output_path, mask)
        
        print(f"[INFO] Basic human mask saved to: {output_path}")
        print(f"[INFO] Mask contains classes: {np.unique(mask)}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create basic mask: {e}")
        return False

def main():
    """Main function that mimics the CIHP_PGN interface"""
    
    print("[INFO] CIHP_PGN Minimal Mode (No Checkpoint Required)")
    print("[WARNING] This is a fallback mode with basic segmentation")
    
    # Expected input/output paths for API integration
    input_path = "/home/paperspace/datasets/input.jpg"
    output_path = "/home/paperspace/output/input.png"
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"[ERROR] Input image not found: {input_path}")
        sys.exit(1)
    
    # Create basic mask
    success = create_basic_human_mask(input_path, output_path)
    
    if success:
        print("[SUCCESS] Basic human parsing completed!")
        print("Saved: input -> input.png")
        sys.exit(0)
    else:
        print("[ERROR] Basic human parsing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 