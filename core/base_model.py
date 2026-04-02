"""
Base Transformer Model for A-THENA

Implements a lightweight Transformer encoder architecture optimized for
early intrusion detection in IoT environments.

Architecture details:
- Input projection: d=448 -> d_m=8
- L=1 Transformer block
- Multi-head attention: h=4 heads, d_h=8 per head
- Feed-forward network: d_ff=16
- Global average pooling
- Output layer with softmax

Total parameters: ~5,100
"""

import tensorflow as tf
from core.time_aware_encodings import TASinusoidalEncoding, TAFourierEncoding, TARoPE, apply_rotary_pos_emb


class MultiHeadAttentionWithRoPE(tf.keras.layers.Layer):
    """
    Multi-Head Attention layer with optional Time-Aware RoPE support.
    
    This custom implementation allows applying RoPE to Q and K matrices
    after projection but before computing attention scores.
    
    Args:
        d_model: Model dimension (default: 8)
        num_heads: Number of attention heads (default: 4)
        d_head: Dimension per head (default: 8)
        use_rope: Whether to apply TARoPE (default: False)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 8,
        num_heads: int = 4,
        d_head: int = 8,
        use_rope: bool = False,
        dropout: float = 0.1,
        **kwargs
    ):
        super(MultiHeadAttentionWithRoPE, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.use_rope = use_rope
        self.dropout_rate = dropout
        
        # Q, K, V projection layers
        self.wq = tf.keras.layers.Dense(num_heads * d_head, name='query')
        self.wk = tf.keras.layers.Dense(num_heads * d_head, name='key')
        self.wv = tf.keras.layers.Dense(num_heads * d_head, name='value')
        
        # Output projection
        self.wo = tf.keras.layers.Dense(d_model, name='output')
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout)
        
        # RoPE layer
        # if use_rope:
        #     self.rope = TARoPE(d_head=d_head)
        # else:
        #     self.rope = None
    
    def build(self, input_shape):
        """
        Build the layer by creating projection weights.
        
        Args:
            input_shape: Shape of input tensor (batch, seq_len, d_model)
        """
        # input_shape is (batch, seq_len, d_model)

        # Build RoPE for completeness
        # if self.rope:
        #     timestamps_shape = (input_shape[0], input_shape[1])
        #     self.rope.build(timestamps_shape)
        
        # Build Q, K, V projections
        self.wq.build(input_shape)
        self.wk.build(input_shape)
        self.wv.build(input_shape)
        
        # Build output projection
        # Input to output_dense is (batch, seq_len, num_heads * d_head)
        output_input_shape = (input_shape[0], input_shape[1], self.num_heads * self.d_head)
        self.wo.build(output_input_shape)
        
        super(MultiHeadAttentionWithRoPE, self).build(input_shape)
    

    def call(self, x, timestamps=None, mask=None, training=None):
        """
        Apply multi-head attention.
        
        Args:
            x: (batch, seq_len, d_model) input tensor
            timestamps: (batch, seq_len) timestamps (required if use_rope=True)
            mask: (batch, seq_len) attention mask (1=valid, 0=padding)
            training: Boolean for dropout
        
        Returns:
            (batch, seq_len, d_model) output tensor
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Linear projections: (batch, seq_len, num_heads * d_head)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Split into multiple heads: (batch, seq_len, num_heads, d_head)
        q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.d_head))
        k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.d_head))
        v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.d_head))
        
        # Apply RoPE to Q and K if enabled
        if self.use_rope:
            if timestamps is None:
                raise ValueError("timestamps must be provided when use_rope=True")
            # q = self.rope(q, timestamps)
            # k = self.rope(k, timestamps)
            q = apply_rotary_pos_emb(q, timestamps, self.d_head)
            k = apply_rotary_pos_emb(k, timestamps, self.d_head)
        
        # Transpose for attention calculation: (batch, num_heads, seq_len, d_head)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        # Q @ K^T / sqrt(d_head)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale
        dk = tf.cast(self.d_head, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply mask if provided (mask padding positions)
        if mask is not None:
            # Reshape mask: (batch, 1, 1, seq_len)
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
            
            # Add large negative value to masked positions
            scaled_attention_logits += (1.0 - mask) * -1e9
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Weighted sum of values: (batch, num_heads, seq_len, d_head)
        attention_output = tf.matmul(attention_weights, v)
        
        # Transpose back: (batch, seq_len, num_heads, d_head)
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        
        # Concatenate heads: (batch, seq_len, num_heads * d_head)
        concat_attention = tf.reshape(
            attention_output,
            [batch_size, seq_len, self.num_heads * self.d_head]
        )
        
        # Final linear projection: (batch, seq_len, d_model)
        output = self.wo(concat_attention)
        
        return output
    
    def get_config(self):
        config = super(MultiHeadAttentionWithRoPE, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_head': self.d_head,
            'use_rope': self.use_rope,
            'dropout': self.dropout_rate
        })
        return config


class TransformerBlock(tf.keras.layers.Layer):
    """
    Single Transformer encoder block.
    
    Architecture:
    1. Multi-head attention (with optional RoPE)
    2. Add & Layer Norm
    3. Feed-forward network (d_model -> d_ff -> d_model)
    4. Add & Layer Norm
    
    Uses ReLU activation instead of GELU for efficiency.
    
    Args:
        d_model: Model dimension (default: 8)
        num_heads: Number of attention heads (default: 4)
        d_head: Dimension per head (default: 8)
        d_ff: Feed-forward intermediate dimension (default: 16)
        use_rope: Whether to use TARoPE in attention (default: False)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 8,
        num_heads: int = 4,
        d_head: int = 8,
        d_ff: int = 16,
        use_rope: bool = False,
        dropout: float = 0.1,
        **kwargs
    ):
        super(TransformerBlock, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.use_rope = use_rope
        self.dropout_rate = dropout
        
        # Multi-head attention
        self.attention = MultiHeadAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            d_head=d_head,
            use_rope=use_rope,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        # Layer normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
    
    def build(self, input_shape):
        """
        Build the Transformer block.
        
        Args:
            input_shape: Shape of input tensor (batch, seq_len, d_model)
        """
        # Build attention layer
        self.attention.build(input_shape)
        
        # Build feed-forward network
        self.ffn.build(input_shape)
        
        # Build layer normalization layers
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        
        super(TransformerBlock, self).build(input_shape)
    
    def call(self, x, timestamps=None, mask=None, training=None):
        """
        Apply Transformer block.
        
        Args:
            x: (batch, seq_len, d_model) input tensor
            timestamps: (batch, seq_len) timestamps (required if use_rope=True)
            mask: (batch, seq_len) attention mask
            training: Boolean for dropout
        
        Returns:
            (batch, seq_len, d_model) output tensor
        """
        # Multi-head attention with residual connection
        attn_output = self.attention(
            x,
            timestamps=timestamps,
            mask=mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_head': self.d_head,
            'd_ff': self.d_ff,
            'use_rope': self.use_rope,
            'dropout': self.dropout_rate
        })
        return config

# class AthenaModel(tf.keras.Model):
#     """
#     Constructs the A-THENA model using the selected time-aware 
#     positional encoding variant (TA Sinusoidal, TA Fourier, or 
#     TA RoPE). The model is built with explicit input tensors 
#     for flows, timestamps, and masks.
    
#     Architecture:
#     1. Input projection: d=448 -> d_m=8
#     2. L=1 Transformer encoder block
#     3. Global average pooling (over sequence dimension)
#     4. Output layer: d_m=8 -> num_classes
#     5. Softmax activation
    
#     Args:
#         d_input: Input dimension (packet bytes, default: 448)
#         d_model: Model hidden dimension (default: 8)
#         num_blocks: Number of Transformer blocks (default: 1)
#         num_heads: Number of attention heads (default: 4)
#         d_head: Dimension per head (default: 8)
#         d_ff: Feed-forward intermediate dimension (default: 16)
#         num_classes: Number of output classes (dataset-specific)
#         encoding_type: Time-aware encoding (default: 'sinusoidal')
#         dropout: Dropout rate (default: 0.1)
#     """
    
#     def __init__(
#         self,
#         d_input: int = 448,
#         d_model: int = 8,
#         num_blocks: int = 1,
#         num_heads: int = 4,
#         d_head: int = 8,
#         d_ff: int = 16,
#         num_classes: int = 5,
#         encoding_type='sinusoidal',
#         dropout: float = 0.1,
#         **kwargs
#     ):
#         model_name = f"athena_{encoding_type}"
#         super().__init__(name=model_name, **kwargs)
        
#         self.d_input = d_input
#         self.d_model = d_model
#         self.num_blocks = num_blocks
#         self.num_heads = num_heads
#         self.d_head = d_head
#         self.d_ff = d_ff
#         self.num_classes = num_classes
#         self.encoding_type = encoding_type
#         self.dropout_rate = dropout
        
#         # Input projection layer: d_input -> d_model
#         self.input_projection = tf.keras.layers.Dense(
#             d_model,
#             name='input_projection'
#         )

#         # Input-based Time-Aware Positional Encoding
#         self.encoding = None
#         if encoding_type == "sinusoidal":
#             self.encoding = TASinusoidalEncoding(d_model)
#         elif encoding_type == "fourier":
#             self.encoding = TAFourierEncoding(d_model)

#         # Stack of Transformer blocks
#         self.transformer_blocks = [
#             TransformerBlock(
#                 d_model=d_model,
#                 num_heads=num_heads,
#                 d_head=d_head,
#                 d_ff=d_ff,
#                 use_rope=encoding_type=='rope',
#                 dropout=dropout,
#                 name=f'transformer_block_{i}'
#             )
#             for i in range(num_blocks)
#         ]
        
#         # Global average pooling (over sequence dimension)
#         self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        
#         # Output classification layer
#         self.output_layer = tf.keras.layers.Dense(
#             num_classes,
#             name='output_layer'
#         )
    
#     def build(self, input_shape):
#         """
#         Build the model by creating weights.
        
#         Args:
#             input_shape for flows + timestamps + mask:
#                 - [(batch, seq_len, d_input), (batch, seq_len), (batch, seq_len)] 
#         """
#         # Handle different input formats
#         if isinstance(input_shape, list):
#             flows_shape = input_shape[0]
        
#         # Build input projection
#         self.input_projection.build(flows_shape)

#         # Build positional encoding for completeness
#         if self.encoding:
#             timestamps_shape = input_shape[1]
#             self.encoding.build(timestamps_shape)
        
#         # Build transformer blocks
#         # After input projection, shape is (batch, seq_len, d_model)
#         transformer_input_shape = (flows_shape[0], flows_shape[1], self.d_model)
#         for block in self.transformer_blocks:
#             block.build(transformer_input_shape)
        
#         # Build output layer
#         # After global pooling, shape is (batch, d_model)
#         self.output_layer.build((flows_shape[0], self.d_model))
        
#         super(AthenaModel, self).build(input_shape)

#     def call(self, inputs, training=None):
#         """
#         Forward pass through the model.
        
#         Args:
#             inputs: Tuple of (packets, timestamps, mask)
#                 - packets: (batch, seq_len, d_input) normalized packet bytes
#                 - timestamps: (batch, seq_len) continuous timestamps
#                 - mask: (batch, seq_len) attention mask (1=valid, 0=padding)
#             training: Boolean for dropout
        
#         Returns:
#             (batch, num_classes) logits
#         """
#         # Unpack inputs
#         if not isinstance(inputs, (list, tuple)):
#             raise ValueError("A-THENA requires three inputs: (packets, timestamps, mask)")

#         if len(inputs) != 3:
#             raise ValueError(
#                 f"A-THENA requires exactly three inputs (packets, timestamps, mask), "
#                 f"but received {len(inputs)} inputs."
#             )

#         flows, timestamps, masks = inputs
        
#         # Input projection: (batch, seq_len, d_input) -> (batch, seq_len, d_model)
#         x = self.input_projection(flows)

#         # Apply input-based positional encoding (Sinusoidal or Fourier)
#         x = x + self.encoding(timestamps)
    
#         # Apply Transformer blocks
#         for block in self.transformer_blocks:
#             x = block(x, timestamps=timestamps, mask=masks, training=training)
        
#         # Global average pooling with mask
#         if masks is not None:
#             # Mask out padding positions before pooling
#             mask_expanded = tf.expand_dims(tf.cast(masks, tf.float32), axis=-1)
#             x = x * mask_expanded
            
#             # Compute sum and divide by number of valid positions
#             x_sum = tf.reduce_sum(x, axis=1)
#             valid_counts = tf.reduce_sum(mask_expanded, axis=1)
#             x = x_sum / tf.maximum(valid_counts, 1.0)  # Avoid division by zero
#         else:
#             x = self.global_pool(x)
        
#         # Output layer: (batch, d_model) -> (batch, num_classes)
#         logits = self.output_layer(x)
        
#         return logits
    
#     def get_config(self):
#         config = super(AthenaModel, self).get_config()
#         config.update({
#             'd_input': self.d_input,
#             'd_model': self.d_model,
#             'num_blocks': self.num_blocks,
#             'num_heads': self.num_heads,
#             'd_head': self.d_head,
#             'd_ff': self.d_ff,
#             'num_classes': self.num_classes,
#             'use_rope': self.use_rope,
#             'dropout': self.dropout_rate
#         })
#         return config
    
#     def compute_parameter_count(self):
#         """
#         Compute total number of trainable parameters.
#         Should be approximately 5,100 for default hyperparameters.
#         """
#         return sum([tf.size(v).numpy() for v in self.trainable_variables])



# class LightweightTransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, d_model, num_heads, d_head, d_ff, dropout_rate=0.1, encoding_type='none', **kwargs):
#         super(LightweightTransformerBlock, self).__init__(**kwargs)
#         self.num_heads = num_heads
#         self.d_head = d_head
#         self.d_model = d_model
#         self.encoding_type = encoding_type

#         # Attention Projections
#         # We project to (num_heads * d_head) which is NOT necessarily equal to d_model
#         self.wq = tf.keras.layers.Dense(num_heads * d_head)
#         self.wk = tf.keras.layers.Dense(num_heads * d_head)
#         self.wv = tf.keras.layers.Dense(num_heads * d_head)
        
#         # Output projection back to d_model size
#         self.wo = tf.keras.layers.Dense(d_model)
        
#         # Feed Forward Network (ReLU is used per paper Section 3.3.1)
#         self.ffn = tf.keras.Sequential([
#             tf.keras.layers.Dense(d_ff, activation='relu'),
#             tf.keras.layers.Dense(d_model)
#         ])
        
#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
#         self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

#     def call(self, x, timestamps=None, mask=None):
#         # x: [batch, seq_len, d_model]
#         batch_size = tf.shape(x)[0]
#         seq_len = tf.shape(x)[1]

#         # --- Multi-Head Attention Logic ---
        
#         # Project Q, K, V
#         q = self.wq(x) # [batch, seq, heads*d_head]
#         k = self.wk(x)
#         v = self.wv(x)
        
#         # Split heads: [batch, seq, heads, d_head]
#         q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.d_head))
#         k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.d_head))
#         v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.d_head))

#         # Apply RoPE if selected
#         if self.encoding_type == 'rope' and timestamps is not None:
#             q = apply_rotary_pos_emb(q, timestamps, self.d_head)
#             k = apply_rotary_pos_emb(k, timestamps, self.d_head)

#         # Transpose for dot product: [batch, seq, heads, d_head] -> [batch, heads, seq, d_head]
#         q = tf.transpose(q, perm=[0, 2, 1, 3])
#         k = tf.transpose(k, perm=[0, 2, 1, 3])
#         v = tf.transpose(v, perm=[0, 2, 1, 3])

#         # Scaled Dot-Product Attention
#         # Scores: [batch, heads, seq, seq]
#         matmul_qk = tf.matmul(q, k, transpose_b=True)
#         scores = matmul_qk / tf.math.sqrt(tf.cast(self.d_head, tf.float32))

#         # Apply Masking
#         # Mask input is [batch, seq]. We need to broadcast to [batch, 1, 1, seq]
#         # (Note: we mask the Key dimension, which is the last dimension of scores)
#         if mask is not None:
#             # Expand mask dims: [batch, 1, 1, seq]
#             seq_mask = tf.cast(mask, tf.float32)
#             seq_mask = tf.expand_dims(tf.expand_dims(seq_mask, 1), 1)
            
#             # (1 - mask) * -1e9 ensures padded positions get roughly -infinity score
#             scores += (1.0 - seq_mask) * -1e9

#         attn_weights = tf.nn.softmax(scores, axis=-1)

#         # Weighted sum of V
#         attn_output = tf.matmul(attn_weights, v) # [batch, heads, seq, d_head]
        
#         # Merge heads
#         attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3]) # [batch, seq, heads, d_head]
#         attn_output = tf.reshape(attn_output, (batch_size, seq_len, self.num_heads * self.d_head))
        
#         # Final linear projection to restore d_model dimension
#         output = self.wo(attn_output)
        
#         # Residual + Norm
#         x = self.layernorm1(x + self.dropout1(output))
        
#         # --- Feed Forward Logic ---
#         ffn_output = self.ffn(x)
#         x = self.layernorm2(x + self.dropout2(ffn_output))

#         return x

def create_athena_model(
        d_input=448,
        d_model=8,
        num_blocks=1,
        num_heads=4,
        d_head=8,
        d_ff=16,
        num_classes=5,
        encoding_type='sinusoidal',
        dropout_rate=0.1
    ):
    """
    Constructs the A-THENA model using the selected time-aware 
    positional encoding variant (TA Sinusoidal, TA Fourier, or 
    TA RoPE). The model is built with explicit input tensors 
    for flows, timestamps, and masks.

    Architecture:
    1. Input projection: d=448 -> d_m=8
    2. L=1 Transformer encoder block
    3. Global average pooling (over sequence dimension)
    4. Output layer: d_m=8 -> num_classes
    5. Softmax activation
    
    Args:
        d_input: Input dimension (packet bytes, default: 448)
        d_model: Model hidden dimension (default: 8)
        num_blocks: Number of Transformer blocks (default: 1)
        num_heads: Number of attention heads (default: 4)
        d_head: Dimension per head (default: 8)
        d_ff: Feed-forward intermediate dimension (default: 16)
        num_classes: Number of output classes (dataset-specific)
        encoding_type: Time-aware encoding (default: 'sinusoidal')
        dropout: Dropout rate (default: 0.1)
    """
    
    # Inputs
    input_flows = tf.keras.Input(shape=(None, d_input), name="flows") # [B, n, d]
    input_timestamps = tf.keras.Input(shape=(None,), name="timestamps") # [B, n]
    input_masks = tf.keras.Input(shape=(None,), name="masks") # [B, n]
    
    # Input Projection (map packet size d to d_model)
    x = tf.keras.layers.Dense(d_model)(input_flows)
    
    # Apply Input-based Time-Aware Encoding (Sinusoidal or Fourier)
    # RoPE is skipped here as it is applied inside the block
    if encoding_type == 'sinusoidal':
        pe = TASinusoidalEncoding(d_model=d_model)(input_timestamps)
        x = x + pe
    elif encoding_type == 'fourier':
        pe = TAFourierEncoding(d_model=d_model)(input_timestamps)
        x = x + pe

    # Stack Transformer Blocks
    for i in range(num_blocks):
        x = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_head=d_head,
            d_ff=d_ff,
            use_rope=encoding_type=='rope',
            dropout=dropout_rate,
            name=f'transformer_block_{i+1}'
        )(x, timestamps=input_timestamps, mask=input_masks)
    
    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Classification Head
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(
        inputs=[input_flows, input_timestamps, input_masks], 
        outputs=output, 
        name=f"athena_{encoding_type}"
    )
    return model