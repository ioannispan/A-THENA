import tensorflow as tf

class EarlyDetectionLoss:
    """
    Implements the Early Detection Loss (EDL) function.
    
    Formula:
        L = Mean( w_i * CE(y_i, p_i) )
        w_i = exp(-alpha * n_i)
        
    Where:
        n_i is the length of the i-th flow (derived from the mask).
        alpha is the decay factor (default: 0.1).
    """
    def __init__(self, alpha=0.1, from_logits=False):
        """
        Args:
            alpha (float): The decay rate for the weight. Default: 0.1.
            from_logits (bool): Whether predictions are logits or probabilities. 
                                Default False (softmax is in the model).
        """
        self.alpha = alpha
        # We use Reduction.NONE to get per-sample loss, which we manually weight
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, 
            reduction=tf.keras.losses.Reduction.NONE
        )

    def __call__(self, y_true, y_pred, masks):
        """
        Computes the weighted loss.

        Args:
            y_true: Ground truth labels [batch_size].
            y_pred: Model predictions [batch_size, num_classes].
            masks:  Input masks indicating valid packets [batch_size, seq_len].
        
        Returns:
            Scalar loss value (mean of weighted per-sample losses).
        """
        # 1. Calculate flow lengths (n_i) from masks
        # masks is [Batch, N], sum across time axis -> [Batch]
        lengths = tf.reduce_sum(masks, axis=1)
        
        # 2. Calculate weights: w_i = e^(-0.1 * n_i)
        # We cast lengths to float32 for exp calculation
        weights = tf.exp(-self.alpha * tf.cast(lengths, tf.float32))
        
        # 3. Calculate standard Cross-Entropy per sample
        loss_per_sample = self.ce(y_true, y_pred) # [Batch]
        
        # 4. Apply weights
        weighted_loss = loss_per_sample * weights
        
        # 5. Return overall batch loss (Mean)
        return tf.reduce_mean(weighted_loss)