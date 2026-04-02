import os
import numpy as np
import tensorflow as tf
import copy
from core.data_splitting import DataSplitter
from core.offline_augmentation import OfflineAugmenter
from core.trainer import AthenaTrainer
from core.base_model import create_athena_model

class AthenaHybridSystem:
    """
    The main A-THENA orchestrator.
    
    It manages the entire lifecycle:
    1. Holding the data and managing splits (via DataSplitter).
    2. Model Selection (Time-Aware Hybrid Encoding).
    3. Final Model Training.
    """
    def __init__(self, flows, times, masks, labels, config):
        """
        Args:
            flows, times, masks, labels: The complete original dataset.
            config (dict): Hyperparameter configuration.
        """
        self.base_config = config
        self.variants = ['sinusoidal', 'fourier', 'rope']
        self.best_variant = None
        self.best_loss = float('inf')
        self.cv_results = {}
        self.final_model = None

        # Initialize DataSplitter with the full dataset
        print("[A-THENA System] Initializing Data Splitter...")
        self.splitter = DataSplitter(flows, times, masks, labels)
        
        # Perform the Initial Split (Dev vs Test) immediately
        # This sets self.splitter.dev_indices
        self.splitter.stratified_split(initial=True, split_size=0.1)

    def _get_model_param_count(self, encoding_type):
        """
        Helper: Builds a temporary model to count exact trainable parameters.
        """
        # Build Model
        temp_model = create_athena_model(
            d_input=self.base_config['packet_dim'],
            d_model=self.base_config['d_model'],
            num_blocks=self.base_config['num_layers'],
            num_heads=self.base_config['num_heads'],
            d_head=self.base_config['d_head'],
            d_ff=self.base_config['d_ff'],
            num_classes=self.base_config['num_classes'],
            encoding_type=encoding_type,
            dropout_rate=self.base_config['dropout']
        )

        # Count Params
        count = temp_model.count_params()
        
        # Cleanup
        del temp_model
        tf.keras.backend.clear_session()
        
        return count
    
    def run_model_selection(self):
        """
        Runs κ-Fold CV on the Development Set to find the best encoding.
        """
        n_folds = self.base_config.get('n_folds', 5)
        print(f"\n[A-THENA System] Starting Model Selection ({n_folds}-Fold CV)...")

        # Store fold indices to ensure all variants use EXACTLY the same splits
        # Generator yields (fold_id, (X_train, y_train), (X_val, y_val))
        folds_data = list(self.splitter.get_cross_validation_folds(n_folds=n_folds))
        
        for variant in self.variants:
            print(f"\n=== Evaluating Variant: {variant} ===")

            # Calculate P dynamically for this variant ---
            current_p = self._get_model_param_count(variant)
            print(f"  > Model Parameters (P): {current_p}")
            
            fold_losses = []
            
            for fold_id, (X_train, y_train), (X_val, y_val) in folds_data:
                print(f"  > Fold {fold_id}/{n_folds}")
                
                # 1. Unpack Data
                tr_f, tr_t, tr_m = X_train
                val_f, val_t, val_m = X_val
                
                # 2. Offline Augmentation
                # We must fit the augmenter on THIS fold's training data
                aug = OfflineAugmenter(model_param_count=current_p)
                aug.fit(y_train)
                
                # Transform Train (Subflow + Oversampling)
                (tr_f_aug, tr_t_aug, tr_m_aug), y_train_aug = aug.transform(
                    tr_f, tr_t, tr_m, y_train, is_validation=False
                )
                
                # Transform Val (Subflow Only)
                (val_f_aug, val_t_aug, val_m_aug), y_val_aug = aug.transform(
                    val_f, val_t, val_m, y_val, is_validation=True
                )
                
                # 3. Configure Trainer
                # Create a specific config for this variant run
                run_config = copy.deepcopy(self.base_config)
                run_config['encoding_type'] = variant
                # Unique output dir for this fold/variant to avoid overwriting
                run_config['output_dir'] = f"{self.base_config['output_dir']}/{variant}/fold_{fold_id}"
                
                trainer = AthenaTrainer(run_config)
                
                # 4. Train
                _, best_val_loss = trainer.train(
                    (tr_f_aug, tr_t_aug, tr_m_aug), y_train_aug,
                    (val_f_aug, val_t_aug, val_m_aug), y_val_aug,
                )
                
                fold_losses.append(best_val_loss)
                
                # Clean up memory
                tf.keras.backend.clear_session()
                del trainer
            
            # Aggregate Results for Variant
            avg_loss = np.mean(fold_losses)
            std_loss = np.std(fold_losses)
            self.cv_results[variant] = {"avg_loss": avg_loss, "std_loss": std_loss}
            
            print(f"--- Variant {variant}: Avg Val Loss = {avg_loss:.10f} (+/- {std_loss:.10f}) ---")
            
            # Check if best
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_variant = variant
        
        print(f"\n[A-THENA System] Best Variant Selected: {self.best_variant} (Loss: {self.best_loss:.6f})")
        return self.best_variant

    def train_final_model(self):
        """
        Trains the selected best variant on the Development set.
        Uses DataSplitter to generate an internal 90/10 split for Early Stopping.
        """
        if self.best_variant is None:
            raise ValueError("Run run_model_selection() first to determine the best variant.")
        
        print(f"\n[A-THENA System] Training Final Model ({self.best_variant}) on Dev Set...")

        # Calculate P dynamically for the best variant
        final_p = self._get_model_param_count(self.best_variant)
        print(f"  > Final Model Parameters (P): {final_p}")

        # 1. Create Internal Validation Split (initial=False)
        # This splits the current Dev indices into Train/Val
        (X_train_final, y_train_final), (X_val_int, y_val_int) = self.splitter.stratified_split(
            initial=False, split_size=0.1
        )

        # 2. Offline Augmentation (Fit on Final Train portion)
        aug = OfflineAugmenter(model_param_count=final_p)
        aug.fit(y_train_final)

        # Transform Final Train (Oversampling)
        tr_f, tr_t, tr_m = X_train_final
        (tr_f_aug, tr_t_aug, tr_m_aug), y_train_aug = aug.transform(
            tr_f, tr_t, tr_m, y_train_final, is_validation=False
        )
        
        # Transform Internal Val (Subflow only)
        val_f, val_t, val_m = X_val_int
        (val_f_aug, val_t_aug, val_m_aug), y_val_aug = aug.transform(
            val_f, val_t, val_m, y_val_int, is_validation=True
        )

        # 3. Configure Trainer
        final_config = copy.deepcopy(self.base_config)
        final_config['encoding_type'] = self.best_variant
        final_config['output_dir'] = f"{self.base_config['output_dir']}/final_model"
        
        trainer = AthenaTrainer(final_config)        
        
        # 4. Train
        history, _ = trainer.train(
            (tr_f_aug, tr_t_aug, tr_m_aug), y_train_aug,
            (val_f_aug, val_t_aug, val_m_aug), y_val_aug,
        )

        self.final_model = trainer.model
        
        return trainer.model, history

    def export_model(self):
        """
        Saves the final model artifacts:
        1. Keras .keras file (for resuming/research).
        2. TFLite .tflite file (for deployment/evaluation).
        """
        if self.final_model is None:
            raise ValueError("No final model trained yet. Run train_final_model() first.")
            
        base_dir = f"{self.base_config['output_dir']}/final_model"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        # 1. Save Keras Model
        keras_path = f"{base_dir}/athena_model.keras"
        self.final_model.save(keras_path)
        print(f"[A-THENA System] Keras model saved to: {keras_path}")
        
        # 2. Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.final_model)
        
        # Handling custom ops (like Select/TensorList) usually not needed for this simple Transformer,
        # but good to track. Standard ops should suffice.
        
        tflite_model = converter.convert()
        
        tflite_path = f"{base_dir}/athena_model.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"[A-THENA System] TFLite model saved to: {tflite_path}")
        return tflite_path
    
    def get_test_set(self):
        """
        Retrieves the untouched Hold-out Test Set for final evaluation.
        """
        if self.splitter.test_indices is None:
            raise ValueError("Test set not initialized.")
            
        # Manually pack the test data based on stored indices
        idx = self.splitter.test_indices
        X_test = (
            self.splitter.flows[idx],
            self.splitter.timestamps[idx],
            self.splitter.masks[idx]
        )
        y_test = self.splitter.labels[idx]
        
        return X_test, y_test