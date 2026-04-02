import tensorflow as tf
import numpy as np
import time
import os
from core.base_model import create_athena_model
from core.online_augmentation import OnlineAugmenter
from core.early_detection_loss import EarlyDetectionLoss

class AthenaTrainer:
    """
    Manages the training lifecycle for a single A-THENA model variant.
    
    Responsibilities:
    1. Construct tf.data Pipelines (including Online Augmentation).
    2. Initialize Model and Optimizer.
    3. Execute Custom Training Loop (GradientTape).
    4. Handle Early Stopping and Checkpointing.
    """
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary containing:
                - batch_size (int)
                - learning_rate (float)
                - max_epochs (int)
                - patience (int)
                - d_model, num_heads, etc. (Model Hyperparams)
                - encoding_type (str): 'sinusoidal', 'fourier', or 'rope'
                - output_dir (str): Path to save best weights
        """
        self.cfg = config
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.online_aug = OnlineAugmenter(sequence_length=config.get('seq_len', 30))
        
        # Ensure output directory exists
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])

    def _build_pipeline(self, X, y, is_training=False):
        """
        Creates a tf.data.Dataset optimized for performance.
        """
        flows, timestamps, masks = X
        
        # Create dataset from tensor slices
        ds = tf.data.Dataset.from_tensor_slices(
            ({'flows': flows, 'timestamps': timestamps, 'masks': masks}, y)
        )
        
        if is_training:
            # 1. Shuffle
            ds = ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
            
            # 2. Batch (Online Augmenter works on batches)
            ds = ds.batch(self.cfg['batch_size'])
            
            # 3. Apply Online Augmentation (Wrapper)
            # The augmenter expects (f, t, m) and returns (f, t, m)
            # We must wrap it to handle the label
            def aug_wrapper(inputs, label):
                f = inputs['flows']
                t = inputs['timestamps']
                m = inputs['masks']
                f_aug, t_aug, m_aug = self.online_aug.augment_batch(f, t, m)
                return {'flows': f_aug, 'timestamps': t_aug, 'masks': m_aug}, label

            ds = ds.map(aug_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
            
        else:
            # Validation: Just batch
            ds = ds.batch(self.cfg['batch_size'])
            
        # Prefetch for GPU efficiency
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    @tf.function
    def train_step(self, inputs, labels):
        """
        Single training step using GradientTape.
        """
        masks = inputs['masks']
        
        with tf.GradientTape() as tape:
            # Forward Pass (training=True for Dropout)
            # Input list matches model signature: [bytes, time, mask]
            logits = self.model(inputs, training=True)
            
            # Compute Early Detection Loss
            # We pass 'masks' explicitly to calculate flow length weights
            loss_value = self.loss_fn(labels, logits, masks)
            
        # Compute Gradients
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        
        # Apply Gradients
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss_value

    @tf.function
    def val_step(self, inputs, labels):
        """
        Single validation step.
        """
        masks = inputs['masks']
        
        # Forward Pass (training=False)
        logits = self.model(inputs, training=False)
        
        # Compute Loss
        loss_value = self.loss_fn(labels, logits, masks)
        
        return loss_value

    def train(self, X_train, y_train, X_val, y_val):
        """
        Main training loop.
        """
        # 1. Prepare Data Pipelines
        train_ds = self._build_pipeline(X_train, y_train, is_training=True)
        val_ds = self._build_pipeline(X_val, y_val, is_training=False)
        
        # 2. Initialize Model
        self.model = create_athena_model(
            d_input=self.cfg['packet_dim'],
            d_model=self.cfg['d_model'],
            num_blocks=self.cfg['num_layers'],
            num_heads=self.cfg['num_heads'],
            d_head=self.cfg['d_head'],
            d_ff=self.cfg['d_ff'],
            num_classes=self.cfg['num_classes'],
            encoding_type=self.cfg['encoding_type'],
            dropout_rate=self.cfg['dropout']
        )
        
        # 3. Setup Optimizer and Loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg['learning_rate'])
        self.loss_fn = EarlyDetectionLoss(alpha=0.1, from_logits=False) # Softmax is in model
        
        # 4. Training Loop Variables
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"\n[Trainer] Starting training for {self.cfg['encoding_type']} model...")
        
        start_time = time.time()
        
        for epoch in range(self.cfg['max_epochs']):
            # --- Training Phase ---
            epoch_loss_sum = 0.0
            num_batches = 0
            
            for inputs, labels in train_ds:
                loss = self.train_step(inputs, labels)
                epoch_loss_sum += loss
                num_batches += 1
            
            train_loss = epoch_loss_sum / num_batches
            
            # --- Validation Phase ---
            val_loss_sum = 0.0
            val_batches = 0
            
            for inputs, labels in val_ds:
                loss = self.val_step(inputs, labels)
                val_loss_sum += loss
                val_batches += 1
                
            val_loss = val_loss_sum / val_batches
            
            # --- Logging ---
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            
            print(f"Epoch {epoch+1}/{self.cfg['max_epochs']} | "
                  f"Train Loss: {train_loss:.9f} | Val Loss: {val_loss:.9f}", end='')
            
            # --- Early Stopping & Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save Best Model Weights
                save_path = os.path.join(self.cfg['output_dir'], f"best_model_{self.cfg['encoding_type']}.weights.h5")
                self.model.save_weights(save_path)
                print(" * Saved")
            else:
                patience_counter += 1
                print(f" | Patience {patience_counter}/{self.cfg['patience']}")
                
            if patience_counter >= self.cfg['patience']:
                print(f"\n[Trainer] Early stopping triggered at epoch {epoch+1}.")
                break
                
        total_time = time.time() - start_time
        print(f"[Trainer] Training finished in {total_time:.2f}s. Best Val Loss: {best_val_loss:.4f}")
        
        # Load best weights before returning
        best_weights_path = os.path.join(self.cfg['output_dir'], f"best_model_{self.cfg['encoding_type']}.weights.h5")
        if os.path.exists(best_weights_path):
            self.model.load_weights(best_weights_path)
            
        return history, best_val_loss