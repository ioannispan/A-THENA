import tensorflow as tf
import numpy as np

class OnlineAugmenter:
    """
    Implements Online Augmentation.
    Applied stochastically during training batches.
    """
    def __init__(self, sequence_length=30):
        self.N = sequence_length

    @tf.function
    def augment_batch(self, flows, timestamps, masks):
        """
        TF Graph wrapper for the numpy augmentation logic.
        Inputs:
            flows: [Batch, N, d]
            timestamps: [Batch, N]
            masks: [Batch, N]
        """
        # We use py_function to allow numpy operations for complex logic like
        # dropping packets (shifting arrays) which is verbose in pure TF.
        aug_flows, aug_times, aug_masks = tf.numpy_function(
            func=self._numpy_augment_batch,
            inp=[flows, timestamps, masks],
            Tout=[tf.float32, tf.float32, tf.float32]
        )
        
        # Explicit shape setting is required after py_function
        aug_flows.set_shape(flows.shape)
        aug_times.set_shape(timestamps.shape)
        aug_masks.set_shape(masks.shape)
        
        return aug_flows, aug_times, aug_masks

    def _numpy_augment_batch(self, flows, timestamps, masks):
        """
        Iterates over the batch and applies augmentations randomly.
        """
        batch_size = flows.shape[0]
        # Make copies to avoid modifying original data
        aug_flows = np.copy(flows)
        aug_times = np.copy(timestamps)
        aug_masks = np.copy(masks)

        for i in range(batch_size):
            # Apply pipeline per sample
            f, t, m = aug_flows[i], aug_times[i], aug_masks[i]
            n = int(np.sum(m))
            if n == 0: continue

            #print(f"Flow length: {n}")

            # 1. Jitter Injection (Timestamps)
            # U(-0.7*tmin, 0.7*tmin)
            if n > 1:
                # Differences between consecutive timestamps: [d0, d1, ..., dn-2]
                # Assumption: timestamps are strictly increasing or equal.
                diffs = np.diff(t[:n])
                
                # To vectorize finding the min neighbor distance:
                # For index 0: neighbor is diffs[0] (next)
                # For index n-1: neighbor is diffs[-1] (prev)
                # For indices 1..n-2: min(diffs[i-1], diffs[i]) (prev, next)
                
                # Construct "distance to previous" array (pad start with diffs[0] for the first element)
                dists_to_prev = np.concatenate(([diffs[0]], diffs))
                
                # Construct "distance to next" array (pad end with diffs[-1] for the last element)
                dists_to_next = np.concatenate((diffs, [diffs[-1]]))
                
                # Compute local t_min per packet
                local_t_mins = np.minimum(dists_to_prev, dists_to_next)
                
                # Apply random perturbation proportional to local gap
                # U(-0.7 * t_min, 0.7 * t_min)
                jitter = np.random.uniform(-0.7, 0.7, size=n) * local_t_mins

                # Constraint: The first timestamp must always remain 0
                jitter[0] = 0.0
                #print(f"1. Jitter: {jitter}")
                
                t[:n] += jitter
            
            # 2. Traffic Scaling (Timestamps)
            # Factors {0.5, 0.75, 1.0, 1.25, 1.5}
            scale = np.random.choice([0.5, 0.75, 1.0, 1.25, 1.5])
            #print(f"2. Scale: {scale}")
            if scale != 1.0:
                t[:n] *= scale

            # 3. Packet Drop (Packet Level)
            # Drops random packets and shifts left
            max_drop = int(0.25 * n - 0.5)
            #print(f"3. max_drop: {max_drop}")
            if max_drop > 0:
                n_drop = np.random.randint(0, max_drop + 1)
                #print(f"3. n_drop: {n_drop}")
                if n_drop > 0:
                    # Choose indices to keep
                    drop_indices = np.random.choice(n, n_drop, replace=False)
                    #print(f"3. drop_indices: {drop_indices}")
                    keep_mask = np.ones(n, dtype=bool)
                    keep_mask[drop_indices] = False
                    
                    # Shift logic
                    new_len = n - n_drop
                    # Compact flow
                    f[:new_len] = f[:n][keep_mask]
                    # Zero out rest
                    f[new_len:] = 0
                    
                    # Compact time
                    t[:new_len] = t[:n][keep_mask]
                    t[new_len:] = 0
                    
                    # Update mask
                    m[:new_len] = 1
                    m[new_len:] = 0
                    
                    n = new_len # Update length for subsequent steps

            # 4. Packet Insertion (Packet Level)
            # Insert zero-byte packets
            max_insert = int(0.15 * n - 0.5)
            #print(f"4. max_insert: {max_insert}")
            if max_insert > 0:
                n_insert = np.random.randint(0, max_insert + 1)

                # Cap n_insert to ensure we don't exceed sequence length N
                space_available = self.N - n
                n_insert = min(n_insert, space_available)
                #print(f"4. n_insert: {n_insert}")

                if n_insert > 0:
                    # Choose positions to insert (before index i)
                    # We can insert at any position from 0 to n (inclusive)
                    insert_indices = np.random.choice(n + 1, n_insert, replace=True)
                    insert_indices.sort()
                    #print(f"4. insert_indices: {insert_indices}")
                    
                    temp_f = []
                    temp_t = []
                    
                    current_f = f[:n]
                    current_t = t[:n]
                    
                    last_idx = 0
                    for ins_idx in insert_indices:
                        # Append segment from original flow
                        temp_f.append(current_f[last_idx:ins_idx])
                        temp_t.append(current_t[last_idx:ins_idx])
                        
                        # Append zero packet (Bytes)
                        temp_f.append(np.zeros((1, f.shape[1]), dtype=np.float32))
                        
                        # Calculate Time: Average of previous and next original timestamps
                        # 1. Identify Previous Time
                        if ins_idx > 0:
                            t_prev = current_t[ins_idx - 1]
                        else:
                            t_prev = 0.0
                        
                        # 2. Identify Next Time
                        if ins_idx < n:
                            t_next = current_t[ins_idx]
                            # Average
                            new_time_val = (t_prev + t_next) / 2.0
                        else:
                            # Edge Case: Inserting at the end of the flow (no 'next' packet)
                            # Fallback to duplicating the previous time
                            new_time_val = t_prev

                        temp_t.append(np.array([new_time_val], dtype=np.float32))
                        
                        last_idx = ins_idx
                        
                    # Append remaining original flow
                    temp_f.append(current_f[last_idx:])
                    temp_t.append(current_t[last_idx:])
                    
                    # Concatenate and Update
                    new_f_seq = np.concatenate(temp_f, axis=0)
                    new_t_seq = np.concatenate(temp_t, axis=0)
                    
                    new_len = len(new_f_seq)
                    
                    f[:new_len] = new_f_seq
                    f[new_len:] = 0 # Zero out potential remainder
                    
                    t[:new_len] = new_t_seq
                    t[new_len:] = 0
                    
                    m[:new_len] = 1
                    m[new_len:] = 0

                    n = new_len

            # 5. Noise Injection (Bytes)
            # Max floor(n/3) packets, max floor(d/100) bytes per packet
            max_mod_pkts = int(n / 3)
            if max_mod_pkts > 0:
                n_mod = np.random.randint(0, max_mod_pkts + 1)
                #print(f"5. n_mod: {n_mod}")
                if n_mod > 0:
                    mod_indices = np.random.choice(n, n_mod, replace=False)
                    #print(f"5. mod_indices: {mod_indices}")
                    d = f.shape[1]
                    max_bytes = int(d / 100)
                    if max_bytes > 0:
                        for p_idx in mod_indices:
                            n_bytes = np.random.randint(1, max_bytes + 1)
                            byte_indices = np.random.choice(d, n_bytes, replace=False)
                            noise = np.random.normal(0, 0.1, size=n_bytes)
                            f[p_idx, byte_indices] += noise
                            # Clip to [0, 1]
                            f[p_idx, byte_indices] = np.clip(f[p_idx, byte_indices], 0.0, 1.0)

            # Store back
            aug_flows[i] = f
            aug_times[i] = t
            aug_masks[i] = m

            #print(f"New flow length: {n}")

        return aug_flows.astype(np.float32), aug_times.astype(np.float32), aug_masks.astype(np.float32)