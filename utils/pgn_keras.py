import os
import glob
import tensorflow as tf
import numpy as np
from utils.model_pgn import PGNModel

class PGNKeras:
    """TensorFlow 1.x compatible wrapper for PGN model with proper weight loading"""
    
    def __init__(self, n_classes=20, checkpoint_path=None):
        self.n_classes = n_classes
        self.checkpoint_path = checkpoint_path
        self.sess = None
        
        # Disable eager execution for graph mode
        tf.compat.v1.disable_eager_execution()
        
        print("[INFO] Building model using original PGN architecture...")
        self.build_model()
        
        if checkpoint_path:
            self.load_weights_from_checkpoint(checkpoint_path)

    def build_model(self):
        """Build the model using the original PGN architecture"""
        # Create input placeholder
        self.input_placeholder = tf.compat.v1.placeholder(
            tf.float32, 
            shape=[None, None, None, 3], 
            name='data'
        )
        
        # Create the model in inference mode
        self.model = PGNModel(
            {'data': self.input_placeholder}, 
            trainable=False, 
            is_training=False, 
            n_classes=self.n_classes, 
            keep_prob=1.0
        )
        
        # Get outputs
        self.parsing_fc = self.model.layers['parsing_fc']
        self.parsing_rf_fc = self.model.layers['parsing_rf_fc'] 
        self.edge_rf_fc = self.model.layers['edge_rf_fc']
        
        print("[INFO] Model built successfully using original PGN architecture")

    def load_weights_from_checkpoint(self, checkpoint_path):
        """Load weights from checkpoint using the original approach"""
        try:
            # Create session
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=config)
            
            # Find checkpoint file
            if checkpoint_path.endswith('.ckpt'):
                ckpt_file = checkpoint_path
            else:
                # Find the latest checkpoint
                import glob
                ckpt_files = glob.glob(f"{checkpoint_path}/*.ckpt-*")
                if not ckpt_files:
                    ckpt_files = glob.glob(f"{checkpoint_path}/model.ckpt-*")
                if ckpt_files:
                    # Get the latest checkpoint
                    ckpt_file = max(ckpt_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
                    # Remove the .data-* or .index suffix if present
                    if '.data-' in ckpt_file:
                        ckpt_file = ckpt_file.split('.data-')[0]
                    elif '.index' in ckpt_file:
                        ckpt_file = ckpt_file.replace('.index', '')
                else:
                    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
            
            print(f"[INFO] Loading checkpoint: {ckpt_file}")
            
            # Initialize variables first
            self.sess.run(tf.compat.v1.global_variables_initializer())
            
            # Get variables that exist in checkpoint
            checkpoint_reader = tf.train.load_checkpoint(ckpt_file)
            checkpoint_vars = set(checkpoint_reader.get_variable_to_shape_map().keys())
            
            # Get variables in current graph (excluding momentum variables)
            graph_vars = {}
            for var in tf.compat.v1.global_variables():
                var_name = var.name.split(':')[0]  # Remove :0 suffix
                if '/Momentum' not in var_name:  # Skip momentum variables
                    graph_vars[var_name] = var
            
            # Find intersection - variables that exist in both
            vars_to_restore = {}
            for var_name, var in graph_vars.items():
                if var_name in checkpoint_vars:
                    vars_to_restore[var_name] = var
            
            print(f"[INFO] Found {len(vars_to_restore)} variables to restore out of {len(graph_vars)} graph variables")
            print(f"[INFO] Checkpoint contains {len(checkpoint_vars)} variables")
            
            if vars_to_restore:
                # Create saver with only restorable variables
                saver = tf.compat.v1.train.Saver(vars_to_restore)
                saver.restore(self.sess, ckpt_file)
                print(f"[INFO] Successfully restored {len(vars_to_restore)} variables from checkpoint!")
            else:
                print("[WARNING] No matching variables found between graph and checkpoint!")
            
            print("[INFO] Checkpoint loaded successfully!")
            
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            print("[WARNING] Model will run with random weights - results will be poor")
            if self.sess is None:
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.compat.v1.Session(config=config)
            self.sess.run(tf.compat.v1.global_variables_initializer())

    def __call__(self, image_batch):
        """Run inference on the model"""
        if self.sess is None:
            raise RuntimeError("Session not initialized. Call load_weights_from_checkpoint first.")
        
        # Run inference
        parsing_fc, parsing_rf_fc, edge_rf_fc = self.sess.run(
            [self.parsing_fc, self.parsing_rf_fc, self.edge_rf_fc],
            feed_dict={self.input_placeholder: image_batch}
        )
        
        # Return numpy arrays directly (more compatible with both TF1.x and TF2.x)
        return parsing_fc, parsing_rf_fc, edge_rf_fc

    def __del__(self):
        """Clean up session"""
        if self.sess is not None:
            self.sess.close()

    def summary(self):
        """Print model summary"""
        print("PGN Model Summary:")
        print(f"Classes: {self.n_classes}")
        print(f"Input shape: (batch_size, 512, 512, 3)")
        print(f"Output: parsing_fc, parsing_rf_fc, edge_rf_fc")
        print(f"Session: {'Active' if self.sess else 'None'}")

    def get_config(self):
        """Get model configuration"""
        return {
            'n_classes': self.n_classes,
            'checkpoint_path': self.checkpoint_path
        } 