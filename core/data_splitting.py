import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

class DataSplitter:
    """
    Handles data splitting for Hold-out Test sets, Cross-Validation, 
    and Internal Validation splits.
    """
    def __init__(self, flows, timestamps, masks, labels):
        """
        Args:
            flows: np.ndarray (num_samples, N, d)
            timestamps: np.ndarray (num_samples, N)
            masks: np.ndarray (num_samples, N)
            labels: np.ndarray (num_samples,)
        """

        # Validate matching first dimension
        n_flows = flows.shape[0]
        assert timestamps.shape[0] == n_flows, "timestamps must match flows in the first dimension"
        assert masks.shape[0] == n_flows, "masks must match flows in the first dimension"
        assert labels.shape[0] == n_flows, "labels must match flows in the first dimension"

        self.flows = flows
        self.timestamps = timestamps
        self.masks = masks
        self.labels = labels
        self.dev_indices = None
        self.test_indices = None

    def stratified_split(self, initial=True, split_size=0.1, random_state=42):
        """
        Performs a single stratified split.
        
        Args:
            initial (bool): 
                If True: Splits the FULL dataset -> Dev / Test. Updates self.dev_indices.
                If False: Splits the EXISTING Dev set -> Train / Val. Uses self.dev_indices.
            split_size (float): Proportion of the dataset to include in the split (Test or Val).
            random_state (int): Random seed.

        Returns:
            (X_main, y_main), (X_split, y_split)
            where X is (flows, timestamps, masks)
        """
        # Determine pool of indices to split
        if initial:
            indices = np.arange(len(self.labels))
            stratify_labels = self.labels
        else:
            if self.dev_indices is None:
                raise ValueError("Cannot perform secondary split: Development set not defined (run initial=True first).")
            indices = self.dev_indices
            stratify_labels = self.labels[self.dev_indices]
        
        # Perform Stratified Split
        # Note: train_test_split returns subsets of the input 'indices' array.
        main_idx, split_idx = train_test_split(
            indices, 
            test_size=split_size, 
            stratify=stratify_labels, 
            random_state=random_state,
            shuffle=True
        )

        # If this was the initial split, we store the 'main' part as the Development set
        if initial:
            self.dev_indices = main_idx
            self.test_indices = split_idx

        # Helper to retrieve data arrays
        def pack(idx):
            return (
                self.flows[idx],
                self.timestamps[idx],
                self.masks[idx]
            ), self.labels[idx]

        # Logging
        if initial:
            print(f"[DataSplitter] Initial Split: Dev={len(main_idx)}, Test={len(split_idx)}")
        else:
            print(f"[DataSplitter] Internal Split: Train={len(main_idx)}, Val={len(split_idx)}")

        return pack(main_idx), pack(split_idx)
    
    def get_cross_validation_folds(self, n_folds=5, random_state=42):
        """
        Generator that yields (Training, Validation) pairs for k-fold cross-validation.
        Operates ONLY on the Development set created by stratified_split with initial=True.
        
        Yields:
            (X_train, y_train), (X_val, y_val)
        """
        if self.dev_indices is None:
            raise ValueError("The development set needs to be generated before generating CV folds.")

        # Get the actual labels for the dev set to stratify the folds
        dev_labels = self.labels[self.dev_indices]
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        for fold_i, (train_idx_rel, val_idx_rel) in enumerate(skf.split(self.dev_indices, dev_labels)):
            # Map relative indices (0..len(dev)) back to absolute indices (0..len(total))
            train_idx_abs = self.dev_indices[train_idx_rel]
            val_idx_abs = self.dev_indices[val_idx_rel]
            
            X_train = (
                self.flows[train_idx_abs],
                self.timestamps[train_idx_abs],
                self.masks[train_idx_abs]
            )
            y_train = self.labels[train_idx_abs]
            
            X_val = (
                self.flows[val_idx_abs],
                self.timestamps[val_idx_abs],
                self.masks[val_idx_abs]
            )
            y_val = self.labels[val_idx_abs]
            
            yield fold_i + 1, (X_train, y_train), (X_val, y_val)