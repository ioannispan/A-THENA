# A-THENA: Early Intrusion Detection for IoT with Time-Aware Hybrid Encoding

This repository contains the official source code for the paper **"A-THENA: Early Intrusion Detection for IoT with Time-Aware Hybrid Encoding and Network-Specific Augmentation"** (currently in peer review).

A-THENA is a lightweight, Transformer-based Early Intrusion Detection System (EIDS) tailored for resource-constrained IoT devices. Unlike traditional models that treat network flows as uniform sequences, A-THENA utilizes packet timestamps to capture fine-grained temporal dynamics.

## 🌟 Key Contributions

*   **Time-Aware Hybrid Encoding (THE)**: A mechanism that dynamically selects the optimal temporal representation (Sinusoidal, Fourier, or Rotary) for a specific network environment via cross-validation.
*   **Network-Specific Augmentation**:
    *   **Offline**: Subflow generation and hybrid oversampling to simulate early detection scenarios and handle class imbalance.
    *   **Online**: Real-time jitter injection, traffic scaling, and packet manipulation during training.
*   **Early Detection Loss (EDL)**: A custom loss function that penalizes misclassifications in shorter flow prefixes, forcing the model to decide as early as possible.
*   **Edge-Optimized**: The final model is exported to **TensorFlow Lite**, enabling inference with minimal latency on devices like the Raspberry Pi Zero 2 W.

## 📂 Project Structure

```text
athena/
├── main.py                         # Main entry point
├── data/                           # Directory for PCAP or .npz files
├── core/
│   ├── athena_system.py            # System Orchestrator (Model Selection & Training)
│   ├── base_model.py               # Transformer Architecture
│   ├── time_aware_encodings.py     # Custom Keras Layers (Sin/Four/RoPE)
│   ├── offline_augmentation.py     # Subflow Generation & Hybrid Oversampling
│   ├── online_augmentation.py      # TF-based Jitter, Drop, Noise
│   ├── data_preparation.py         # Scapy parsing & Flow Aggregation
│   ├── evaluator.py                # TFLite Inference & Metric Calculation (ERDE)
│   └── ...
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ioannispan/A-THENA.git
    cd A-THENA
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

The system supports three modes of operation:

### 1. Demo Mode (Synthetic Data)
If no data arguments are provided, the script generates synthetic network flows to demonstrate the full pipeline (Selection $\rightarrow$ Training $\rightarrow$ Export $\rightarrow$ Evaluation).

```bash
python main.py --epochs 5 --n_folds 3
```

### 2. Load Preprocessed Sample Data
To experiment using an existing processed dataset:

```bash
python main.py \
    --load_path ./data/sample_mqtt.npz \
    --output_dir ./results \
    --batch_size 8
    --num_classes 5
```

### 3. Real Data (Build from PCAP)
To process raw network traffic, provide the paths to your PCAP files inside `main.py` (dictionary `dataset_map`) and use the `--save_path` argument. This will parse packets, extract flows, filter protocols, and save the processed tensors to a `.npz` file.

```bash
# Ensure 'dataset_map' in main.py points to your PCAP files
python main.py \
    --save_path ./data/dataset.npz \
```

## 📊 Workflow & Evaluation

The system executes the following phases automatically:

1.  **Model Selection**: Performs $k$-Fold Cross-Validation to evaluate Time-Aware Sinusoidal, Fourier, and Rotary encodings. The best variant is selected based on validation loss.
2.  **Final Training**: Retrains the selected variant on the development set.
3.  **Export**: Converts the model to `.tflite` format.
4.  **Evaluation**: Simulates real-time packet arrival on the Hold-out Test Set using the TFLite interpreter.

**Metrics Reported:**
*   **Accuracy**: Top-1 Classification Accuracy.
*   **Earliness ($E$)**: Maximum number of packets observed before a confident decision ($> \tau$).
*   **FAR / FNR**: False Alarm Rate and False Negative Rate.
*   **ERDE**: Early Risk Detection Error (penalizes late detection of attacks).


## ⚙️ Configuration

Key hyperparameters can be adjusted via CLI arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--max_len` | 30 | Maximum flow length ($N$). |
| `--packet_dim` | 448 | Packet feature dimension (bytes). |
| `--n_folds` | 5 | Number of CV folds for Model Selection. |
| `--learning_rate` | 0.0002 | Adam learning rate. |
| `--patience` | 7 | Early stopping patience. |
| `--batch_size` | 8 | Batch size. |

## 📚 Citation

If you utilize this code in your research, please cite:

```bibtex
@article{athena2026,
  title={Anonymous submission},
  author={Anonymous Authors},
  journal={Under review},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License.