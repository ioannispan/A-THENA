"""
Offline Augmentation Module for A-THENA

Implements two offline augmentation techniques applied to the training set:
1. Subflow Generation: Creates partial flows to train early detection
2. Hybrid Oversampling: Balances class distribution using deterministic + stochastic approach
"""

import numpy as np
import math
from collections import Counter

class OfflineAugmenter:
    """
    Implements the Offline Augmentation pipeline (Section 3.2.2):
    1. Subflow Generation: Creates partial flows to simulate early detection scenarios.
    2. Hybrid Oversampling: Balances the dataset using deterministic + stochastic copying.
    """
    def __init__(self, model_param_count=5100, target_density_factor=2.0):
        """
        Args:
            model_param_count (int): P in the paper (approx 5100 for base model).
            target_density_factor (float): Factor to calc md = (factor * P) / C.
        """
        self.P = model_param_count
        self.factor = target_density_factor
        self.stats = {} # Stores md, ac factors per class

    def fit(self, y_train):
        """
        Calculates augmentation statistics based on the training set.
        """
        class_counts = Counter(y_train)
        num_classes = len(class_counts)
        unique_classes = sorted(class_counts.keys())
        
        # Calculate target sample count per class: md approx (2 * P) / C
        md = round((self.factor * self.P) / num_classes)

        self.stats['md'] = md
        self.stats['class_factors'] = {}
        
        print(f"[OfflineAugmenter] Fitting Stats: P={self.P}, C={num_classes}, Target md={md}")

        for c in unique_classes:
            mc = class_counts[c]
            
            if mc < md:
                # Minority Class: Augment to reach md
                # factor ac = (md - mc) / mc
                ac = round((md - mc) / mc, 2)
                # Bounded by paper logic (though paper formula is min((md-mc)/mc, nc-1))
                # We assume flow length might limit it, but we use the calculated factor directly here
                # and handle the 'nc-1' constraint during actual generation if needed.
                self.stats['class_factors'][c] = {'type': 'minority', 'ac': ac}
                print(f"  Class {c} (Minority): count={mc}, factor ac={ac}")
            else:
                # Majority Class: Fixed augmentation for earliness
                self.stats['class_factors'][c] = {'type': 'majority', 'ac': 0.2}
                print(f"  Class {c} (Majority): count={mc}, factor ac=0.2")

    def transform(self, flows, timestamps, masks, labels, is_validation=False):
        """
        Applies Subflow Generation and (if training) Hybrid Oversampling.
        """
        out_flows, out_times, out_masks, out_labels = [], [], [], []
        
        # Group indices by class
        class_indices = {c: np.where(labels == c)[0] for c in self.stats['class_factors']}
        
        # --- Step 1: Subflow Generation ---
        # "The validation set undergoes subflow generation using the same class-specific factors"
        
        generated_counts = Counter()
        
        for c, indices in class_indices.items():
            factor_info = self.stats['class_factors'].get(c, {'type': 'majority', 'ac': 0.0})
            ac = factor_info['ac']
            is_minority = factor_info['type'] == 'minority'

            for idx in indices:
                # Add original sample
                out_flows.append(flows[idx])
                out_times.append(timestamps[idx])
                out_masks.append(masks[idx])
                out_labels.append(labels[idx])
                generated_counts[c] += 1
                
                # Determine number of subflows to generate
                if is_minority:
                    # ac = integer part + decimal part probability
                    n_subflows = int(np.floor(ac))
                    if np.random.rand() < (ac - np.floor(ac)):
                        n_subflows += 1
                else:
                    # Majority: randomly select 20% (ac=0.2) to generate 1 subflow
                    n_subflows = 1 if np.random.rand() < ac else 0

                if n_subflows > 0:
                    # Determine flow length n (based on mask)
                    n = int(np.sum(masks[idx]))
                    if n <= 1: continue # Cannot cut single packet flows

                    # Generate subflows
                    new_f, new_t, new_m = self._generate_subflows(
                        flows[idx], timestamps[idx], n, n_subflows, is_minority
                    )
                    
                    out_flows.extend(new_f)
                    out_times.extend(new_t)
                    out_masks.extend(new_m)
                    out_labels.extend([c] * len(new_f))
                    generated_counts[c] += len(new_f)

        dset = "Validation" if is_validation else "Training"
        total_samples = sum(generated_counts.values())
        print(f"[OfflineAugmenter] {dset} Class Distribution (after Subflow Generation): {dict(generated_counts)}, Total: {total_samples}")

        # --- Step 2: Hybrid Oversampling (Training Only) ---
        if not is_validation:
            # Calculate m_max (reference size)
            m_max = max(generated_counts.values())
            
            # Temporary lists for oversampling
            os_flows, os_times, os_masks, os_labels = [], [], [], []
            
            # Re-group the currently generated data (original + subflows)
            # Optimization: We just iterate the lists we just built.
            # Converting to arrays first is necessary for indexing.
            curr_flows = np.array(out_flows)
            curr_times = np.array(out_times)
            curr_masks = np.array(out_masks)
            curr_labels = np.array(out_labels)
            
            for c in self.stats['class_factors']:
                c_idxs = np.where(curr_labels == c)[0]
                m_prime_c = len(c_idxs)
                
                if m_prime_c < m_max:
                    # Calculate factor zc
                    zc = round((m_max - m_prime_c) / m_prime_c, 3)
                    r = int(np.floor(zc))
                    p = zc - r
                    
                    # 1. Deterministic Step: Duplicate r times
                    if r > 0:
                        os_flows.append(np.repeat(curr_flows[c_idxs], r, axis=0))
                        os_times.append(np.repeat(curr_times[c_idxs], r, axis=0))
                        os_masks.append(np.repeat(curr_masks[c_idxs], r, axis=0))
                        os_labels.append(np.repeat(curr_labels[c_idxs], r, axis=0))
                    
                    # 2. Stochastic Step: Sample p * m'c
                    n_stochastic = int(np.round(p * m_prime_c))
                    if n_stochastic > 0:
                        # Sample without replacement within the subset
                        chosen = np.random.choice(c_idxs, n_stochastic, replace=False)
                        os_flows.append(curr_flows[chosen])
                        os_times.append(curr_times[chosen])
                        os_masks.append(curr_masks[chosen])
                        os_labels.append(curr_labels[chosen])

            # Append oversampled data
            if os_flows:
                out_flows.extend(np.concatenate(os_flows))
                out_times.extend(np.concatenate(os_times))
                out_masks.extend(np.concatenate(os_masks))
                out_labels.extend(np.concatenate(os_labels))

            final_counts = Counter(out_labels)
            total_samples = sum(final_counts.values())
            print(f"[OfflineAugmenter] Training Class Distribution (after Hybrid Oversampling): {dict(final_counts)}, Total: {total_samples}")
            
        # Convert to numpy arrays
        X_out = (np.array(out_flows), np.array(out_times), np.array(out_masks))
        y_out = np.array(out_labels)
        
        # Shuffle
        perm = np.random.permutation(len(y_out))
        return (X_out[0][perm], X_out[1][perm], X_out[2][perm]), y_out[perm]

    def _generate_subflows(self, flow, time, n, count, is_minority):
        """
        Generates 'count' subflows by sampling a cut-point k and zeroing out positions >= k.

        flow, time: np arrays of length n
        """
        sub_flows, sub_times, sub_masks = [], [], []
        
        for _ in range(count):
            if is_minority:
                # Logarithmic sampling: Dense coverage of early packets
                # k in [1, n-1]
                log_min = np.log(1)
                log_max = np.log(n) 
                k_log = np.random.uniform(log_min, log_max)
                k = int(np.exp(k_log))
                k = max(1, min(k, n - 1)) # Ensure bounds
            else:
                # Majority: Restricted to [1, 5]
                limit = min(5, n - 1)
                if limit < 1: limit = 1
                k = np.random.randint(1, limit + 1)

            # --- Create new arrays with same length ---
            new_flow = np.copy(flow)
            new_time = np.copy(time)

            # Zero out everything >= k
            new_flow[k:] = 0
            new_time[k:] = 0

            # Mask: 1 for [0, k-1], 0 otherwise
            new_mask = np.copy(time)
            new_mask[k:] = 0
            new_mask[:k] = 1

            sub_flows.append(new_flow)
            sub_times.append(new_time)
            sub_masks.append(new_mask)

        return sub_flows, sub_times, sub_masks