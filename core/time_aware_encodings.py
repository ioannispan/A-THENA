"""
Time-Aware Positional Encodings for A-THENA

Implements three time-aware positional encoding mechanisms that replace
discrete position indices with continuous timestamps:
    1. TA-Sinusoidal: Time-aware sinusoidal encoding
    2. TA-Fourier: Time-aware Fourier encoding with learnable frequencies
    3. TA-RoPE: Time-aware Rotary Position Embedding

Reference: Paper Section 3.3.2 (lines 780-832)
"""

import tensorflow as tf
import numpy as np


class TASinusoidalEncoding(tf.keras.layers.Layer):
    """
    Time-Aware Sinusoidal Positional Encoding.
    
    Replaces discrete position indices with continuous timestamps in the
    standard sinusoidal positional encoding formula.
    
    Formula:
        PE(t_i, 2j) = sin(t_i / 10000^(2j/d_m))
        PE(t_i, 2j+1) = cos(t_i / 10000^(2j/d_m))
    
    where:
        - t_i is the continuous timestamp of packet i
        - j indexes the sine/cosine pairs (j = 0, 1, ..., d_m/2 - 1)
        - d_m is the model dimension
    
    Args:
        d_model: Hidden dimension (default: 8)
        name: Layer name
    """
    
    def __init__(self, d_model: int = 8, **kwargs):
        super(TASinusoidalEncoding, self).__init__(**kwargs)
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        
        self.d_model = d_model
        self.div_term = None  # created in build()

    def build(self, input_shape):
        # input_shape: (batch, seq_len)

        # Precompute denominators: 10000^(2j/d_m) for j = 0, 1, ..., d_m/2 - 1
        self.div_term = tf.pow(
            10000.0,
            tf.range(0, self.d_model, 2, dtype=tf.float32) / self.d_model
        )
        self.built = True
    
    def call(self, timestamps):
        """
        Apply time-aware sinusoidal encoding.
        
        Args:
            timestamps: (batch_size, seq_len) tensor of continuous timestamps
        
        Returns:
            (batch_size, seq_len, d_model) encoded tensor
        """
        # timestamps shape: (batch, seq_len)
        # Expand dims for broadcasting: (batch, seq_len, 1)
        timestamps = tf.expand_dims(timestamps, axis=-1)

        # Divide timestamps by denominators
        # (batch, seq_len, 1) / (d_model/2,) -> (batch, seq_len, d_model/2)
        angles = timestamps / self.div_term
        
        # Apply sin to even indices, cos to odd indices
        sin_encodings = tf.sin(angles)  # (batch, seq_len, d_model/2)
        cos_encodings = tf.cos(angles)  # (batch, seq_len, d_model/2)
        
        # Interleave sin and cos: [sin, cos, sin, cos, ...]
        # Stack along last dimension and reshape
        encodings = tf.stack([sin_encodings, cos_encodings], axis=-1)
        encodings = tf.reshape(encodings, 
                              [tf.shape(timestamps)[0], 
                               tf.shape(timestamps)[1], 
                               self.d_model])
        
        return encodings
    
    def get_config(self):
        config = super(TASinusoidalEncoding, self).get_config()
        config.update({'d_model': self.d_model})
        return config


class TAFourierEncoding(tf.keras.layers.Layer):
    """
    Time-Aware Fourier Positional Encoding.
    
    Extends sinusoidal encoding with learnable frequency parameters,
    enabling richer and more flexible positional representations.
    
    Formula:
        PE(t_i, 2j) = sin(2π * f_j * t_i)
        PE(t_i, 2j+1) = cos(2π * f_j * t_i)
    
    where f_j are learnable frequency parameters.
    
    Args:
        d_model: Hidden dimension (default: 8)
        name: Layer name
    """
    
    def __init__(self, d_model: int = 8, **kwargs):
        super(TAFourierEncoding, self).__init__(**kwargs)
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        
        self.d_model = d_model
        self.frequencies = None

    def build(self, input_shape):
        # Initialize learnable frequencies (d_model/2 parameters)
        # Initialize with log-spaced values similar to sinusoidal
        init_freqs = 1.0 / (10000.0 ** (2 * np.arange(self.d_model // 2) / self.d_model))

        self.frequencies = self.add_weight(
            name="frequencies",
            shape=(self.d_model // 2,),
            initializer=tf.keras.initializers.Constant(init_freqs),
            trainable=True,
            dtype=tf.float32
        )

        self.built = True
    
    def call(self, timestamps):
        """
        Apply time-aware Fourier encoding.
        
        Args:
            timestamps: (batch_size, seq_len) tensor of continuous timestamps
        
        Returns:
            (batch_size, seq_len, d_model) encoded tensor
        """
        # timestamps shape: (batch, seq_len)
        # Expand dims: (batch, seq_len, 1)
        timestamps = tf.expand_dims(timestamps, axis=-1)
        
        # Compute angles: 2π * f_j * t_i
        # (batch, seq_len, 1) * (d_model/2,) -> (batch, seq_len, d_model/2)
        angles = 2 * np.pi * self.frequencies * timestamps
        
        # Apply sin and cos
        sin_encodings = tf.sin(angles)
        cos_encodings = tf.cos(angles)
        
        # Interleave sin and cos
        encodings = tf.stack([sin_encodings, cos_encodings], axis=-1)
        encodings = tf.reshape(encodings,
                              [tf.shape(timestamps)[0],
                               tf.shape(timestamps)[1],
                               self.d_model])
        
        return encodings
    
    def get_config(self):
        config = super(TAFourierEncoding, self).get_config()
        config.update({'d_model': self.d_model})
        return config


class TARoPE(tf.keras.layers.Layer):
    """
    Time-Aware Rotary Position Embedding.
    
    Applies rotation to query/key vectors based on timestamps rather than
    discrete positions. The rotation is applied per attention head after
    projecting Q and K into multi-head space.
    
    Formula:
        For each pair (x_2j, x_2j+1) in each head:
        [x_rot_2j  ]   [cos(t_i*θ_j)  -sin(t_i*θ_j)] [x_2j    ]
        [x_rot_2j+1] = [sin(t_i*θ_j)   cos(t_i*θ_j)] [x_2j+1  ]
        
        where θ_j = 10000^(-2j/d_h)

    Args:
        d_head: Dimension per attention head (default: 8)
        name: Layer name
    """
    
    def __init__(self, d_head: int = 8, **kwargs):
        super(TARoPE, self).__init__(**kwargs)
        
        if d_head % 2 != 0:
            raise ValueError(f"d_head must be even, got {d_head}")
        
        self.d_head = d_head
        self.theta = None
    
    def build(self, input_shape):
        # input_shape: (batch, seq_len)

        # Precompute rotation angles θ_j = 10000^(-2j/d_m)
        j = np.arange(0, self.d_head // 2, dtype=np.float32)
        theta = 10000.0 ** (-2 * j / self.d_head)
        self.theta = tf.constant(theta, dtype=tf.float32)

        self.built = True

    def call(self, x, timestamps):
        """
        Apply rotary position embedding to multi-head Q or K tensor.
        
        Args:
            x: (batch_size, seq_len, num_heads, d_head) tensor
            timestamps: (batch_size, seq_len) tensor of continuous timestamps
        
        Returns:
            (batch_size, seq_len, num_heads, d_head) rotated tensor
        """
        # x shape: (batch, seq_len, num_heads, d_head)
        # timestamps shape: (batch, seq_len)
        
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        num_heads = tf.shape(x)[2]
        
        # Expand timestamps for broadcasting: (batch, seq_len, 1, 1)
        timestamps = tf.expand_dims(timestamps, axis=-1)
        timestamps = tf.expand_dims(timestamps, axis=-1)
        
        # Compute rotation angles: t_i * θ_j
        # (batch, seq_len, 1, 1) * (d_head/2,) -> (batch, seq_len, 1, d_head/2)
        angles = timestamps * self.theta
        
        # Expand for num_heads: (batch, seq_len, num_heads, d_head/2)
        angles = tf.broadcast_to(angles, [batch_size, seq_len, num_heads, self.d_head // 2])
        
        # Compute cos and sin of angles
        cos_angles = tf.cos(angles)  # (batch, seq_len, num_heads, d_head/2)
        sin_angles = tf.sin(angles)  # (batch, seq_len, num_heads, d_head/2)
        
        # Split x into pairs: (x_0, x_1), (x_2, x_3), ...
        # Reshape x: (batch, seq_len, num_heads, d_head/2, 2)
        x_reshaped = tf.reshape(x, [batch_size, seq_len, num_heads, self.d_head // 2, 2])
        
        x_even = x_reshaped[..., 0]  # (batch, seq_len, num_heads, d_head/2)
        x_odd = x_reshaped[..., 1]   # (batch, seq_len, num_heads, d_head/2)
        
        # Apply rotation matrix:
        # x_rot_even = x_even * cos - x_odd * sin
        # x_rot_odd = x_even * sin + x_odd * cos
        x_rot_even = x_even * cos_angles - x_odd * sin_angles
        x_rot_odd = x_even * sin_angles + x_odd * cos_angles
        
        # Stack and reshape back to original shape
        x_rotated = tf.stack([x_rot_even, x_rot_odd], axis=-1)
        x_rotated = tf.reshape(x_rotated, 
                              [batch_size, seq_len, num_heads, self.d_head])
        
        return x_rotated
    
    def get_config(self):
        config = super(TARoPE, self).get_config()
        config.update({'d_head': self.d_head})
        return config


def apply_rotary_pos_emb(x, timestamps, d_head):
    """
    Implements Time-Aware Rotary Positional Encoding (TA-RoPE).
    Rotates Query and Key vectors based on timestamps.
    
    x: [batch, seq_len, num_heads, head_dim] (Query or Key)
    timestamps: [batch, seq_len]
    d_head: Dimension of the attention head
    """
    # Create the rotation angle theta
    # theta_j = 10000^(-2j/d)
    inv_freq = 1.0 / (10000 ** (tf.range(0, d_head, 2, dtype=tf.float32) / d_head))
    
    # Outer product: timestamps * frequencies
    # timestamps: [batch, seq, 1]
    # inv_freq: [1, 1, d_head/2]
    t_expanded = tf.expand_dims(timestamps, -1)
    inv_freq_expanded = tf.reshape(inv_freq, [1, 1, -1])
    
    freqs = t_expanded * inv_freq_expanded # [batch, seq, d_head/2]
    
    # Create sine and cosine for rotation
    # Repeat along the last axis to match d_head (since we process pairs)
    sin_emb = tf.sin(freqs)
    cos_emb = tf.cos(freqs)
    
    # We need to broadcast these to [batch, seq, num_heads, d_head/2]
    # Assuming x is [batch, seq, heads, dim], we expand dims for heads
    sin_emb = tf.expand_dims(sin_emb, 2)
    cos_emb = tf.expand_dims(cos_emb, 2)
    
    # Repeat to match the pairing structure of RoPE (x1, x2) -> (x1, x2)
    sin_emb = tf.repeat(sin_emb, 2, axis=-1)
    cos_emb = tf.repeat(cos_emb, 2, axis=-1)

    # Apply Rotation
    # x = [x1, x2, x3, x4...]
    # x_rotated = [-x2, x1, -x4, x3...]
    
    # Separate even and odd indices
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    # Reassemble in the rotated order: [-x2, x1]
    # We construct the rotated vector by interleaving
    x_rotated = tf.stack([-x2, x1], axis=-1)
    x_rotated = tf.reshape(x_rotated, tf.shape(x)) # Flatten the stack
    
    # Formula: x' = (x * cos) + (x_rotated * sin)
    return (x * cos_emb) + (x_rotated * sin_emb)


# Utility function for easy instantiation
def get_time_aware_encoding(encoding_type: str, d_model: int = 8, d_head: int = 8):
    """
    Factory function to create time-aware encoding layers.
    
    Args:
        encoding_type: One of 'sinusoidal', 'fourier', 'rope'
        d_model: Hidden dimension (for sinusoidal and fourier)
        d_head: Head dimension (for rope)
    
    Returns:
        Instantiated encoding layer
    
    Example:
        >>> # For input encodings
        >>> encoding = get_time_aware_encoding('sinusoidal', d_model=8)
        >>> timestamps = tf.constant([[0.0, 1.0, 2.0]])
        >>> encoded = encoding(timestamps)
        
        >>> # For RoPE (applied to Q/K after projection)
        >>> rope = get_time_aware_encoding('rope', d_head=8)
        >>> q = tf.random.normal((2, 5, 4, 8))  # (batch, seq, heads, d_head)
        >>> timestamps = tf.constant([[0.0, 1.0, 2.0, 3.0, 4.0]] * 2)
        >>> q_rotated = rope(q, timestamps)
    """
    encoding_type = encoding_type.lower()
    
    if encoding_type == 'sinusoidal':
        return TASinusoidalEncoding(d_model=d_model)
    elif encoding_type == 'fourier':
        return TAFourierEncoding(d_model=d_model)
    elif encoding_type == 'rope':
        return TARoPE(d_head=d_head)
    else:
        raise ValueError(
            f"Unknown encoding type: {encoding_type}. "
            f"Choose from: 'sinusoidal', 'fourier', 'rope'"
        )