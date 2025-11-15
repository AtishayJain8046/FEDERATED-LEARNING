# 🔒 Federated Learning with Privacy-Preserving Techniques

A comprehensive demonstration of federated learning enhanced with **Differential Privacy**, **Secure Multi-Party Computation (SMPC)**, and **Homomorphic Encryption (HE)**.

## 🎯 Project Overview

This project implements a federated learning system where multiple clients collaboratively train a machine learning model without sharing raw data. Privacy is protected through three complementary techniques:

1. **Differential Privacy (DP)**: Adds calibrated noise to gradients to prevent inference attacks
2. **Secure Multi-Party Computation (SMPC)**: Uses secret sharing to enable secure aggregation
3. **Homomorphic Encryption (HE)**: Allows computation on encrypted data

## 📁 Project Structure

```
federated-privacy-demo/
├── data/                    # Dataset generators
│   ├── __init__.py
│   └── generator.py        # Synthetic data generation
├── models/                  # ML models
│   ├── __init__.py
│   └── mlp.py              # SimpleMLP and LogisticRegression
├── federated/              # Core federated learning
│   ├── __init__.py
│   ├── client.py          # FederatedClient implementation
│   ├── server.py          # FederatedServer with FedAvg
│   └── protocols.py       # DP, SMPC, HE implementations
├── experiments/
│   └── run_demo.py        # Main demo script
├── templates/             # Web frontend
│   └── index.html         # Main HTML template
├── static/                 # Static web assets
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── app.js         # Frontend logic
├── app.py                 # Flask web application
├── start_frontend.py      # Quick start script
├── requirements.txt        # Python dependencies
├── CONTRIBUTING.md         # Work division and guide
├── FRONTEND_README.md      # Frontend user guide
├── IMPROVEMENT_IDEAS.md    # Enhancement suggestions
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd federated-privacy-demo
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** TenSEAL (for Homomorphic Encryption) may take a few minutes to install. If it fails, try:
   ```bash
   pip install tenseal --prefer-binary
   ```

### Running the Demo

**Option 1: Command Line Demo**
```bash
python experiments/run_demo.py
```

This will run four demonstrations:
1. **Basic Federated Learning** - FedAvg algorithm
2. **Differential Privacy** - Gradient clipping and noise addition
3. **Secure Multi-Party Computation** - Secret sharing demonstration
4. **Homomorphic Encryption** - Encrypted computation (requires TenSEAL)

**Option 2: Interactive Web Frontend** 🌐
```bash
# Quick start
python start_frontend.py

# Or run directly
python app.py
```

Then open `http://localhost:5000` in your browser to:
- 🎛️ **Configure experiments** with interactive controls
- 📊 **Visualize** how noise affects model accuracy
- 🔍 **Compare** different privacy levels (ε values)
- 📈 **See real-time** training progress and metrics

See [FRONTEND_README.md](FRONTEND_README.md) for detailed frontend documentation.

## 📚 Components Explained

### 1. Federated Learning Core

**FederatedClient** (`federated/client.py`):
- Handles local model training on client data
- Manages model parameters and updates
- Supports evaluation on test data

**FederatedServer** (`federated/server.py`):
- Coordinates training across multiple clients
- Implements FedAvg aggregation algorithm
- Manages client selection and model distribution

### 2. Differential Privacy

**DifferentialPrivacy** (`federated/protocols.py`):
- **Gradient Clipping**: Bounds gradient sensitivity
- **Noise Addition**: Adds calibrated Gaussian noise
- **Privacy Budget**: Configurable (ε, δ) parameters

**Usage:**
```python
from federated import DifferentialPrivacy

dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
clipped_grads = dp.clip_gradients(gradients)
noisy_grads = dp.add_noise(clipped_grads)
```

### 3. Secure Multi-Party Computation

**SecureMultiPartyComputation** (`federated/protocols.py`):
- **Secret Sharing**: Splits values into shares
- **Secure Aggregation**: Aggregates without revealing individual values
- **Reconstruction**: Reconstructs original from shares

**Usage:**
```python
from federated import SecureMultiPartyComputation

smpc = SecureMultiPartyComputation(num_parties=3)
shares = smpc.secret_share(value)
reconstructed = smpc.reconstruct(shares)
```

### 4. Homomorphic Encryption

**HomomorphicEncryption** (`federated/protocols.py`):
- **CKKS Scheme**: Supports approximate arithmetic on encrypted data
- **Encrypted Operations**: Addition and scalar multiplication
- **TenSEAL Integration**: Uses Microsoft SEAL library

**Usage:**
```python
from federated import HomomorphicEncryption

he = HomomorphicEncryption()
encrypted = he.encrypt(tensor)
encrypted_sum = he.encrypted_add(encrypted1, encrypted2)
decrypted = he.decrypt(encrypted_sum)
```

## 🧪 Example Usage

### Basic Federated Learning

```python
from models import SimpleMLP
from data import generate_synthetic_data, split_data_among_clients
from federated import FederatedClient, FederatedServer

# Generate and split data
X, y = generate_synthetic_data(n_samples=1000, n_features=784, n_classes=10)
client_data = split_data_among_clients(X, y, num_clients=5)

# Create server and clients
model = SimpleMLP(input_size=784, num_classes=10)
server = FederatedServer(global_model=model)

for i, (X_client, y_client) in enumerate(client_data):
    client_model = SimpleMLP(input_size=784, num_classes=10)
    client = FederatedClient(i, client_model, train_data=(X_client, y_client))
    server.register_client(client)

# Run federated learning
for round_num in range(10):
    server.run_round(num_clients=3, local_epochs=1)
```

### With Differential Privacy

```python
from federated import DifferentialPrivacy

dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

# In client training, apply DP to gradients
gradients = client.get_gradients()
dp_gradients = dp.apply_dp(gradients)  # Clip + add noise
```

## 📊 Features

- ✅ **FedAvg Algorithm**: Standard federated averaging
- ✅ **IID/Non-IID Data**: Support for different data distributions
- ✅ **Differential Privacy**: Configurable privacy-accuracy trade-off
- ✅ **Secret Sharing**: Secure aggregation via SMPC
- ✅ **Homomorphic Encryption**: Encrypted computation support
- ✅ **Synthetic Data**: Easy-to-use data generators
- ✅ **Interactive Web Frontend**: Visualize privacy-accuracy trade-offs
- ✅ **Real-time Visualizations**: Charts showing training progress
- ✅ **Modular Design**: Easy to extend and customize

## 🛠️ Dependencies

- **PyTorch** (≥2.0.0): Deep learning framework
- **NumPy** (≥1.24.0): Numerical computations
- **Opacus** (≥1.4.0): Differential privacy utilities
- **TenSEAL** (≥0.3.14): Homomorphic encryption (optional but recommended)
- **scikit-learn** (≥1.3.0): Data generation utilities
- **Flask** (≥2.3.0): Web framework for frontend
- **flask-cors** (≥4.0.0): CORS support for API
- **matplotlib** (≥3.7.0): Visualization (optional)
- **tqdm** (≥4.65.0): Progress bars (optional)

## 🎓 Understanding the Trade-offs

### Privacy vs. Accuracy
- **Differential Privacy**: Higher ε = better accuracy, less privacy
- **SMPC**: No accuracy loss, but requires multiple parties
- **HE**: No accuracy loss, but computationally expensive

### Computation vs. Privacy
- **DP**: Minimal overhead, good privacy
- **SMPC**: Moderate overhead, strong privacy
- **HE**: High overhead, strongest privacy

## 🐛 Troubleshooting

### TenSEAL Installation Issues
```bash
# Try pre-built wheels
pip install tenseal --prefer-binary

# Or install from conda
conda install -c conda-forge tenseal
```

### Import Errors
```bash
# Make sure you're in the project root
cd federated-privacy-demo

# Add to PYTHONPATH (Linux/Mac)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or (Windows PowerShell)
$env:PYTHONPATH="$(pwd)"
```

### CUDA/GPU Issues
All code defaults to CPU. To use GPU:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 📖 Further Reading

- [Federated Learning Paper](https://arxiv.org/abs/1602.05629)
- [Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy)
- [Secure Multi-Party Computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation)
- [Homomorphic Encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption)
- [TenSEAL Documentation](https://github.com/OpenMined/TenSEAL)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for work division plan and development guidelines.

## 📝 License

This project is created for educational and hackathon purposes.

## 🎯 Hackathon Tips

1. **Start Simple**: Get basic FL working first, then add privacy
2. **Test Incrementally**: Don't wait until the end to test
3. **Document Well**: Clear README and comments help judges understand
4. **Show Trade-offs**: Discuss privacy vs. accuracy vs. computation
5. **Visualize**: Use graphs to show training progress and privacy metrics

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- OpenMined for TenSEAL
- Facebook Research for Opacus
- All contributors to the open-source privacy-preserving ML community

---

**Built for Hackathon 2024** 🚀

For questions or issues, please open a GitHub issue.
