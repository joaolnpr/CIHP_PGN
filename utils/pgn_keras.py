import tensorflow as tf
import numpy as np
import os
from .model_pgn import PGNModel

class PGNKeras(tf.keras.Model):
    def __init__(self, n_classes=20, checkpoint_path=None):
        super().__init__()
        self.n_classes = n_classes
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._build_model()
        if checkpoint_path is not None:
            self.load_weights_from_checkpoint(checkpoint_path)

    def _build_model(self):
        # Build a dummy model to initialize variables
        dummy_input = tf.zeros([1, 512, 512, 3], dtype=tf.float32)
        self.model = PGNModel({'data': dummy_input}, is_training=False, n_classes=self.n_classes)

    def call(self, images):
        # images: [batch, H, W, 3]
        # Rebuild the model with the correct input if needed
        if images.shape[0] != 1:
            self.model = PGNModel({'data': images}, is_training=False, n_classes=self.n_classes)
        else:
            self.model.layers['data'] = images
        parsing_fc = self.model.layers['parsing_fc']
        parsing_rf_fc = self.model.layers['parsing_rf_fc']
        edge_rf_fc = self.model.layers['edge_rf_fc']
        return parsing_fc, parsing_rf_fc, edge_rf_fc

    def load_weights_from_checkpoint(self, checkpoint_path):
        """Load weights from TensorFlow checkpoint"""
        try:
            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path + '.index'):
                # Try without extension
                if not os.path.exists(checkpoint_path):
                    print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
                    print(f"[ERROR] Please ensure the checkpoint files exist")
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            print(f"[INFO] Loading weights from: {checkpoint_path}")
            
            # Create a checkpoint object
            checkpoint = tf.train.Checkpoint(model=self.model)
            
            # Restore the checkpoint
            status = checkpoint.restore(checkpoint_path)
            
            # Wait for the restore to complete
            try:
                status.expect_partial()  # Some variables might not be in checkpoint
                print(f"[INFO] Successfully loaded checkpoint (partial)")
            except:
                try:
                    status.assert_consumed()
                    print(f"[INFO] Successfully loaded checkpoint (complete)")
                except:
                    print(f"[WARNING] Checkpoint loaded but some variables may not match")
            
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {str(e)}")
            print(f"[INFO] Model will run with random weights")
            # Don't exit here, let the model run with random weights for debugging 