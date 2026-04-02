import numpy as np
import tensorflow as tf
import time

class AthenaEvaluator:
    """
    Handles evaluation using the TFLite runtime, mimicking edge deployment.
    Combines inference simulation (predict_early) and metric calculation.
    """
    def __init__(self, tflite_model_path, num_classes, benign_class_idx=0):
        self.tflite_path = tflite_model_path
        self.num_classes = num_classes
        self.benign_idx = benign_class_idx
        
        # Initialize Interpreter
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Dynamically map inputs to semantic roles (flows, timestamps, masks)
        self.input_indices = self._map_inputs()
        
    def _map_inputs(self):
        """
        Identifies input indices based on shape and name keywords.
        Returns a dict: {'flows': idx, 'timestamps': idx, 'masks': idx}
        """
        indices = {}
        for detail in self.input_details:
            name = detail['name'].lower()
            index = detail['index']
            
            if 'flows' in name:
                indices['flows'] = index
            elif 'timestamps' in name:
                    indices['timestamps'] = index
            else:
                indices['masks'] = index
        
        # Validation
        required = ['flows', 'timestamps', 'masks']
        missing = [r for r in required if r not in indices]
        if missing:
            raise ValueError(
                f"Could not identify {missing} inputs in TFLite model.\n"
                f"Found inputs: {[d['name'] for d in self.input_details]}\n"
                "Ensure the Keras model was built with Input names: 'flows', 'timestamps', 'masks'."
            )
        
        return indices

    def run_evaluation(self, X_test, y_test, threshold=0.95):
        """
        Runs confidence-based early detection on the test set using TFLite.
        Batch size is 1 to simulate real-time packet arrival.
        The incremental packet arrival is simulated by masking indices > k.
        
        Args:
            X_test: Tuple (flows, times, masks), all numpy arrays.
            y_test: Numpy array of labels.
            threshold: Confidence threshold (tau).
        
        Returns:
            metrics_dict: Dictionary containing A, E, FAR, FNR, ERDE.
        """
        # Unpack inputs
        flows, timestamps, orig_masks = X_test
        num_samples = flows.shape[0]
        fixed_seq_len = flows.shape[1]
        
        all_preds = []
        all_earliness = []
        
        print(f"[Evaluator] Starting TFLite Inference on {num_samples} samples...")
        start_time = time.time()

        # 1. Resize Interpreter Input Tensors to Batch Size = 1 and Flow Length = N
        for detail in self.input_details:
            new_shape = list(detail['shape'])
            new_shape[0] = 1
            new_shape[1] = fixed_seq_len
            self.interpreter.resize_tensor_input(detail['index'], new_shape)
        self.interpreter.allocate_tensors()
        
        # Iterate over samples one by one
        for i in range(num_samples):
            # Slice Single Sample (keep batch dim = 1)
            sample_flow = flows[i:i+1]       # Shape: [1, N, d]
            sample_time = timestamps[i:i+1]       # Shape: [1, N]
            sample_mask = orig_masks[i:i+1]  # Shape: [1, N]

            # Determine actual flow length 'n' (ignore padding)
            n_flow = int(np.sum(sample_mask))
            
            decision = -1
            earliness = n_flow
            
            # Incremental Inference Loop (k=1 to N)
            for k in range(1, n_flow + 1):
                # Update Mask: hide future packets (indexes >= k)
                curr_step_mask = sample_mask.copy()
                curr_step_mask[:, k:] = 0
                
                # Set Inputs
                self.interpreter.set_tensor(self.input_indices['flows'], sample_flow)
                self.interpreter.set_tensor(self.input_indices['timestamps'], sample_time)
                self.interpreter.set_tensor(self.input_indices['masks'], curr_step_mask)
                
                # Run Inference
                self.interpreter.invoke()
                
                # Get Output Probabilities
                probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Shape: [num_classes]
                
                pred_class = np.argmax(probs)
                conf = probs[pred_class]
                
                # Decision Condition: Confidence >= threshold
                if conf >= threshold:
                    decision = pred_class
                    earliness = k
                    break
            
            # If loop finishes without threshold met (k reached seq_len), take final prediction
            if decision == -1:
                decision = pred_class # Last calculated prediction
                earliness = n_flow
            
            all_preds.append(decision)
            all_earliness.append(earliness)
            
        print(f"[Evaluator] Inference finished in {time.time() - start_time:.2f}s")
        
        # Compute Metrics
        return self._compute_metrics(
            np.array(all_preds), 
            np.array(all_earliness), 
            y_test
        )
    
    def _compute_metrics(self, y_pred, earliness, y_true, erde_o=5):
        """
        Internal method to calculate A, E, FAR, FNR, ERDE.
        """
        # 1. Accuracy
        accuracy = np.mean(y_pred == y_true)
        
        # 2. Earliness
        max_earliness = np.max(earliness)
        
        # 3. FAR / FNR
        pred_benign = (y_pred == self.benign_idx)
        true_benign = (y_true == self.benign_idx)
        
        # FAR: Benign misclassified as Attack
        if np.sum(true_benign) > 0:
            far = np.sum((~pred_benign) & true_benign) / np.sum(true_benign)
        else:
            far = 0.0
            
        # FNR: Attack misclassified as Benign
        if np.sum(~true_benign) > 0:
            fnr = np.sum(pred_benign & (~true_benign)) / np.sum(~true_benign)
        else:
            fnr = 0.0
            
        # 4. ERDE
        erde = self._compute_erde(y_pred, earliness, y_true, o=erde_o)
        
        return {
            "Accuracy (%)": accuracy * 100,
            "Earliness (pkts)": max_earliness,
            "FAR (%)": far * 100,
            "FNR (%)": fnr * 100,
            f"ERDE_{erde_o}": erde
        }

    def _compute_erde(self, y_pred, earliness, y_true, o=5):
        """
        Calculates Early Risk Detection Error (ERDE) with parameter o.
        Cost logic:
            - FP: Cost = TP / |F|
            - FN: Cost = 1
            - TP: Cost = 1 - (1 / (1 + exp(td - o)))
            - TN: Cost = 0
        """
        is_attack_true = (y_true != self.benign_idx)
        is_attack_pred = (y_pred != self.benign_idx)
        
        # Masks
        tp_mask = is_attack_true & is_attack_pred
        fn_mask = is_attack_true & (~is_attack_pred)
        fp_mask = (~is_attack_true) & is_attack_pred
        
        # FP Cost: TP / |F|
        total = len(y_true)
        tp_count = np.sum(tp_mask)
        fp_cost_val = tp_count / total if total > 0 else 0.0
        
        costs = np.zeros_like(y_true, dtype=np.float64)
        
        # Apply costs
        costs[fp_mask] = fp_cost_val
        costs[fn_mask] = 1.0
        
        if np.any(tp_mask):
            k = earliness[tp_mask]
            costs[tp_mask] = 1.0 - (1.0 / (1.0 + np.exp(k - o)))
        
        # TN Cost remains 0.0

        return np.mean(costs)