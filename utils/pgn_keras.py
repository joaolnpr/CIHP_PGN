import os
import glob
import tensorflow as tf
from .model_pgn import PGNModel

class PGNKeras:
    """TensorFlow 1.x compatible wrapper for PGN model with proper weight loading"""
    
    def __init__(self, n_classes=20, checkpoint_path=None):
        self.n_classes = n_classes
        self.checkpoint_path = checkpoint_path
        self.sess = None
        self.model = None
        self.input_placeholder = None
        self.output_tensors = None
        
        # Disable eager execution for TF1.x compatibility
        tf.compat.v1.disable_eager_execution()
        
        # Build model and load weights
        self._build_model()
        
        if checkpoint_path is not None:
            self.load_weights_from_checkpoint(checkpoint_path)

    def _build_model(self):
        """Build the PGN model using TF1.x graph mode"""
        # Clear any existing graph
        tf.compat.v1.reset_default_graph()
        
        # Create input placeholder
        self.input_placeholder = tf.compat.v1.placeholder(
            tf.float32, 
            shape=[None, 512, 512, 3], 
            name='input_images'
        )
        
        # Build the model
        self.model = PGNModel(
            {'data': self.input_placeholder}, 
            is_training=False, 
            n_classes=self.n_classes
        )
        
        # Get output tensors
        self.output_tensors = {
            'parsing_fc': self.model.layers['parsing_fc'],
            'parsing_rf_fc': self.model.layers['parsing_rf_fc'], 
            'edge_rf_fc': self.model.layers['edge_rf_fc']
        }
        
        # Create session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        
        # Initialize variables
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
        print("[INFO] Model built successfully in TF1.x compatibility mode")

    def load_weights_from_checkpoint(self, checkpoint_path):
        """Load weights from TensorFlow 1.x checkpoint"""
        try:
            print(f"[INFO] Loading weights from: {checkpoint_path}")
            
            # Find the correct checkpoint file
            checkpoint_to_load = None
            if os.path.isdir(checkpoint_path):
                # Look for checkpoint file to get the latest checkpoint path
                checkpoint_file = os.path.join(checkpoint_path, "checkpoint")
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file, 'r') as f:
                        content = f.read()
                        if 'model_checkpoint_path:' in content:
                            import re
                            match = re.search(r'model_checkpoint_path:\s*"([^"]+)"', content)
                            if match:
                                checkpoint_to_load = os.path.join(checkpoint_path, match.group(1))
                
                # If no checkpoint file, look for .ckpt files
                if not checkpoint_to_load:
                    ckpt_files = glob.glob(os.path.join(checkpoint_path, "*.ckpt*.index"))
                    if ckpt_files:
                        # Use the first found checkpoint (remove .index extension)
                        checkpoint_to_load = ckpt_files[0].replace('.index', '')
                    else:
                        # Try other patterns
                        ckpt_files = glob.glob(os.path.join(checkpoint_path, "model.ckpt-*"))
                        if ckpt_files:
                            checkpoint_to_load = ckpt_files[0]
            else:
                checkpoint_to_load = checkpoint_path
            
            if not checkpoint_to_load:
                raise Exception("No valid checkpoint files found")
            
            print(f"[INFO] Loading checkpoint: {checkpoint_to_load}")
            
            # Load using TF1.x Saver
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.sess, checkpoint_to_load)
            
            print("[INFO] ✅ Weights loaded successfully!")
            print("[INFO] 🎯 Model is ready for inference!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {str(e)}")
            print("[WARNING] Model will run with random weights - results will be poor")
            return False

    def __call__(self, inputs):
        """Forward pass through the model"""
        try:
            if self.sess is None or self.output_tensors is None:
                raise Exception("Model not properly initialized")
            
            # Convert inputs to numpy if needed
            if hasattr(inputs, 'numpy'):
                inputs_np = inputs.numpy()
            else:
                inputs_np = inputs
            
            # Run inference
            feed_dict = {self.input_placeholder: inputs_np}
            
            parsing_fc, parsing_rf_fc, edge_rf_fc = self.sess.run([
                self.output_tensors['parsing_fc'],
                self.output_tensors['parsing_rf_fc'],
                self.output_tensors['edge_rf_fc']
            ], feed_dict=feed_dict)
            
            # Convert back to tensors for compatibility
            parsing_fc = tf.convert_to_tensor(parsing_fc)
            parsing_rf_fc = tf.convert_to_tensor(parsing_rf_fc)
            edge_rf_fc = tf.convert_to_tensor(edge_rf_fc)
            
            return parsing_fc, parsing_rf_fc, edge_rf_fc
            
        except Exception as e:
            print(f"[ERROR] Forward pass failed: {e}")
            # Return dummy outputs for graceful degradation
            batch_size = inputs.shape[0] if hasattr(inputs, 'shape') else 1
            h, w = 512, 512
            dummy_parsing = tf.zeros([batch_size, h, w, self.n_classes])
            dummy_edge = tf.zeros([batch_size, h, w, 1])
            return dummy_parsing, dummy_parsing, dummy_edge

    def __del__(self):
        """Clean up session when object is destroyed"""
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