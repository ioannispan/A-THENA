import argparse
import numpy as np
import os
import shutil
import sys
from collections import defaultdict

# Import A-THENA modules
from core.data_preparation import DataPreparation
from core.athena_system import AthenaHybridSystem
from core.evaluator import AthenaEvaluator

def generate_synthetic_data(num_samples, max_len, packet_dim, num_classes):
    """
    Generates synthetic demonstration data in memory,
    allowing the code to run out-of-the-box for demonstration.
    """
    print(f"[Main] Generating {num_samples} synthetic samples for demonstration...")
    
    # Random flows [0, 1]
    F = np.random.rand(num_samples, max_len, packet_dim).astype(np.float32)
    
    # Timestamps (strictly increasing)
    intervals = np.random.exponential(scale=0.1, size=(num_samples, max_len))
    T = np.cumsum(intervals, axis=1).astype(np.float32)
    T -= T[:, 0:1] # t0 = 0
    
    # Masks
    lengths = np.random.randint(1, max_len + 1, size=(num_samples,))
    Masks = np.zeros((num_samples, max_len), dtype=np.float32)
    for i, l in enumerate(lengths):
        Masks[i, :l] = 1.0
        F[i, l:] = 0
        T[i, l:] = 0
        
    y = np.random.randint(0, num_classes, size=(num_samples,))
    
    return F, T, Masks, y

def build_full_dataset(dataset_config, max_len, packet_dim, max_per_class=None, save_path=None):
    """
    Aggregates multiple PCAP sources into a single dataset using DataPreparation.
    
    Args:
        dataset_config (dict): Mapping of { 'path/to/pcap_or_dir': label_int }
        max_len (int): Maximum flow length (N).
        packet_dim (int): Packet feature dimension (d).
        max_per_class (int): Maximum number of samples per class to manage memory.
        save_path (str): Optional path to save the .npz file.
        
    Returns:
        Full combined F, T, Masks, y arrays.
    """
    
    # Lists to hold the parts from each class
    all_F = []
    all_T = []
    all_Masks = []
    all_y = []

    # Track how many samples we have collected for each label
    counts_per_class = defaultdict(int)
    
    print(f"Building dataset from {len(dataset_config)} sources...")
    if max_per_class:
        print(f"Enforcing limit: {max_per_class} flows per class.")
    
    for source_path, label in dataset_config.items():
        if not os.path.exists(source_path):
            print(f"Skipping missing path: {source_path}")
            continue

        # Check Quota
        if max_per_class is not None:
            current_count = counts_per_class[label]
            if current_count >= max_per_class:
                print(f"Skipping {source_path}: Quota reached for Class {label} ({current_count}/{max_per_class})")
                continue

            # Calculate remaining quota for this batch
            remaining_quota = max_per_class - current_count
        else:
            remaining_quota = None
            
        print(f"\n--- Processing Label {label}: {source_path} ---")
        if remaining_quota is not None:
            print(f"    Target to collect: up to {remaining_quota} flows")
        
        # Initialize the pipeline for this specific class
        # Note: You can tune ports/thresholds here if classes need different settings
        prep = DataPreparation(
            source_path=source_path,
            d=packet_dim,
            N=max_len,
            target_ports=None, # Example: [1883, 8883] for MQTT
            active_flow_threshold=1000
        )
        
        # Run pipeline and get labeled data
        F, T, Masks, y = prep.run_pipeline(label=label, limit=remaining_quota)
        
        if len(y) > 0:
            # Update global counts
            counts_per_class[label] += len(y)

            all_F.append(F)
            all_T.append(T)
            all_Masks.append(Masks)
            all_y.append(y)

    # Check if we have data
    if not all_F:
        print("Error: No data extracted from any source.")
        return None, None, None, None

    # Concatenate all parts into single arrays
    print("\nAggregating final dataset...")
    final_F = np.concatenate(all_F, axis=0)
    final_T = np.concatenate(all_T, axis=0)
    final_Masks = np.concatenate(all_Masks, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    
    print(f"Final Dataset Shape: F={final_F.shape}, y={final_y.shape}")
    
    # Save to disk (Compressed NumPy format)
    if save_path:
        # Ensure directory exists before saving
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        print(f"Saving to {save_path}...")
        np.savez_compressed(
            save_path, 
            flows=final_F, 
            timestamps=final_T, 
            masks=final_Masks, 
            labels=final_y
        )
        print("Save complete.")
        
    return final_F, final_T, final_Masks, final_y

def main():
    parser = argparse.ArgumentParser(description="A-THENA: Early Intrusion Detection System")
    
    # Data Mode Selection (Mutually Exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data_save_path', type=str, help="Build dataset from PCAPs and save to this .npz path.")
    group.add_argument('--data_load_path', type=str, help="Load preprocessed data from this .npz path.")
    
    # Data params
    parser.add_argument('--num_samples', type=int, default=1000, help="Number of synthetic samples (if generating demo data)")
    parser.add_argument('--num_classes', type=int, default=5, help="Number of classes")
    parser.add_argument('--max_len', type=int, default=30, help="Maximum flow length (N)")
    parser.add_argument('--packet_dim', type=int, default=448, help="Packet feature dimension (d)")
    parser.add_argument('--max_flows_per_class', type=int, default=None, help="Maximum number of flows to extract per class when building from PCAP.")
    
    # Training params
    parser.add_argument('--epochs', type=int, default=200, help="Maximum training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.0002, help="Learning rate")
    parser.add_argument('--patience', type=int, default=7, help="Early stopping patience")
    parser.add_argument('--n_folds', type=int, default=5, help="Number of CV folds")
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./athena_output', help="Directory for logs and models")
    
    args = parser.parse_args()

    # 0. Setup
    if os.path.exists(args.output_dir):
        print(f"[Main] Cleaning up previous output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # -------------------------------------------------------------------------
    # 1. GET DATA
    # -------------------------------------------------------------------------
    
    flows, timestamps, masks, labels = None, None, None, None

    if args.data_save_path:
        # MODE A: BUILD FROM PCAP AND SAVE
        print("\n[Main] Mode: Building Dataset from PCAP files...")
        print("IMPORTANT:")
        print("  1. Set 'target_ports' if whitelist filtering is required.")
        print("  2. Ensure you have configured 'src/data_preparation.py' by updating 'packet_filtering(pkt)' to match your network protocols (e.g., HTTP, MQTT).")
        
        # TODO: Define your dataset mapping here: { 'path/to/pcap': label_int }
        dataset_map = {
            "path/to/benign.pcap": 0,
            "path/to/bruteforce.pcap": 1,
            # Add more files/directories as needed...
        }
        
        flows, timestamps, masks, labels = build_full_dataset(
            dataset_map,
            max_len=args.max_len,
            packet_dim=args.packet_dim,
            max_per_class=args.max_flows_per_class,
            save_path=args.data_save_path # Save the result to avoiding rebuilding next time
        )
        
        if flows is None:
            print("[Main] Failed to build dataset. Exiting.")
            sys.exit(1)

    elif args.data_load_path:

        # MODE B: LOAD PREPROCESSED DATA
        print(f"\n[Main] Mode: Loading Preprocessed Data from {args.data_load_path}...")
        
        if not os.path.exists(args.data_load_path):
            print(f"[Error] File not found: {args.data_load_path}")
            sys.exit(1)
            
        data = np.load(args.data_load_path)
        flows = data['flows']
        timestamps = data['timestamps']
        masks = data['masks']
        labels = data['labels']
        
        print(f"Loaded Data Shape: F={flows.shape}, y={labels.shape}")

    else:
        # MODE C: DEMO / SYNTHETIC
        print("\n[Main] No data path provided. Using Synthetic Data for Demonstration.")
        flows, timestamps, masks, labels = generate_synthetic_data(
            args.num_samples, args.max_len, args.packet_dim, args.num_classes
        )

    # -------------------------------------------------------------------------
    # 2. CONFIGURE SYSTEM
    # -------------------------------------------------------------------------
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_epochs': args.epochs,
        'patience': args.patience,
        'n_folds': args.n_folds,
        
        # Model Architecture
        'max_len': args.max_len,
        'packet_dim': args.packet_dim,
        'num_classes': args.num_classes,
        'd_model': 8,
        'num_heads': 4,
        'd_head': 8,
        'd_ff': 16,
        'num_layers': 1,
        'dropout': 0.1,
        
        'output_dir': args.output_dir
    }

    # -------------------------------------------------------------------------
    # 3. INITIALIZE A-THENA SYSTEM
    # -------------------------------------------------------------------------
    # This automatically performs the Initial Split (Dev vs Test)
    system = AthenaHybridSystem(flows, timestamps, masks, labels, config)

    # -------------------------------------------------------------------------
    # 4. MODEL SELECTION
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("PHASE 1: Model Selection (Time-Aware Hybrid Encoding)")
    print("="*50)
    best_variant = system.run_model_selection()

    # -------------------------------------------------------------------------
    # 5. FINAL MODEL TRAINING
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print(f"PHASE 2: Training Final Model ({best_variant})")
    print("="*50)
    system.train_final_model()

    # -------------------------------------------------------------------------
    # 6. EXPORT MODEL
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("PHASE 3: Exporting Model to TFLite")
    print("="*50)
    tflite_path = system.export_model()

    # -------------------------------------------------------------------------
    # 7. EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("PHASE 4: Final Evaluation on Test Set")
    print("="*50)
    
    # Get Test Data
    X_test, y_test = system.get_test_set()
    
    # Initialize Evaluator with TFLite model
    evaluator = AthenaEvaluator(
        tflite_model_path=tflite_path, 
        num_classes=config['num_classes'],
        benign_class_idx=0 # TODO: specify the benign class
    )
    
    # Run Evaluation
    # Threshold = 0.95 (High confidence requirement for earliness)
    metrics = evaluator.run_evaluation(X_test, y_test, threshold=0.95)
    
    # -------------------------------------------------------------------------
    # 8. REPORT RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Selected Encoding: {best_variant}")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()