# 📚 Comprehensive Code Documentation

## 🏗️ Architecture Overview

This project implements a **Federated Learning System with Privacy-Preserving Techniques** using a Flask web backend and interactive JavaScript frontend.

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Web Frontend                          │
│  (HTML/CSS/JavaScript with Chart.js)                    │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/WebSocket
┌──────────────────▼──────────────────────────────────────┐
│              Flask Backend (app.py)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   API Routes │  │ WebSocket   │  │  Experiment  │  │
│  │              │  │  Handlers    │  │   Runner      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│         Federated Learning Core                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Client     │  │   Server     │  │  Protocols   │  │
│  │              │  │  (FedAvg)    │  │  (DP/SMPC/HE)│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure & Documentation

### Backend Files

#### `app.py` - Main Flask Application

**Purpose:** Web server and API for federated learning experiments.

**Key Functions:**

1. **`run_federated_experiment()`** (Lines 114-320)
   - Main experiment runner
   - Handles dataset loading (synthetic, MNIST, uploaded)
   - Applies privacy techniques (DP, SMPC, HE)
   - Returns experiment results with metrics
   
   **Parameters:**
   - `num_clients`: Number of federated clients
   - `num_rounds`: Federated learning rounds
   - `use_dp/use_smpc/use_he`: Privacy technique flags
   - `dataset_name`: "synthetic", "mnist", or "uploaded:filename"
   
   **Returns:**
   - Dictionary with `success`, `round_metrics`, `final_accuracy`, `config`

2. **`aggregate_with_he()`** (Lines 48-111)
   - Lightweight homomorphic encryption aggregation
   - Only encrypts first layer parameters for performance
   - Falls back to standard aggregation for other layers
   
3. **`load_custom_dataset()`** (Lines 554-617)
   - Loads custom datasets from uploaded files
   - Supports: .npz, .pkl, .pth, .pt formats
   - Validates data structure (X, y format)
   
4. **`upload_dataset()`** (Lines 495-551)
   - Handles file uploads via POST
   - Validates file type and structure
   - Returns dataset metadata

**API Endpoints:**

- `GET /` - Serve main page
- `POST /api/run_experiment` - Run single experiment
- `POST /api/compare_noise` - Compare multiple epsilon values
- `POST /api/upload_dataset` - Upload custom dataset
- `GET /api/history` - Get experiment history
- `POST /api/clear_history` - Clear history

**WebSocket Events:**

- `connect` - Device connection
- `disconnect` - Device disconnection
- `join_experiment` - Join experiment room
- `experiment_started` - Experiment begins
- `round_completed` - Round finishes with metrics

---

#### `federated/client.py` - Federated Client

**Class: `FederatedClient`**

**Purpose:** Represents a client in federated learning.

**Key Methods:**

- `train(epochs)` - Local training on client data
- `get_model_parameters()` - Extract current model state
- `set_model_parameters(params)` - Update model from server
- `evaluate(test_data)` - Evaluate model on test set

**Usage:**
```python
client = FederatedClient(
    client_id=0,
    model=SimpleMLP(),
    train_data=(X_train, y_train),
    lr=0.01
)
metrics = client.train(epochs=1)
```

---

#### `federated/server.py` - Federated Server

**Class: `FederatedServer`**

**Purpose:** Coordinates federated learning across clients.

**Key Methods:**

- `register_client(client)` - Add client to federation
- `run_round(num_clients, local_epochs)` - Execute one FL round
- `aggregate_updates(client_updates)` - FedAvg aggregation
- `broadcast_model()` - Distribute global model
- `evaluate_global_model(test_data)` - Test global model

**Algorithm:** Federated Averaging (FedAvg)

**Usage:**
```python
server = FederatedServer(global_model=model)
server.register_client(client1)
server.register_client(client2)
round_info = server.run_round(num_clients=2, local_epochs=1)
```

---

#### `federated/protocols.py` - Privacy Protocols

**Classes:**

1. **`DifferentialPrivacy`**
   - Implements (ε, δ)-differential privacy
   - Gaussian mechanism for DP-SGD
   - Methods: `clip_gradients()`, `add_noise()`, `apply_dp()`
   
2. **`SecureMultiPartyComputation`**
   - Secret sharing for secure aggregation
   - Methods: `secret_share()`, `reconstruct()`, `secure_aggregation()`
   
3. **`HomomorphicEncryption`**
   - CKKS scheme using TenSEAL
   - Methods: `encrypt()`, `decrypt()`, `encrypted_add()`, `encrypted_multiply()`

---

#### `data/generator.py` - Dataset Handling

**Functions:**

- `generate_synthetic_data()` - Create synthetic classification data
- `split_data_among_clients()` - Distribute data (IID/non-IID)
- `load_mnist()` - Load MNIST dataset
- `load_dataset()` - Generic dataset loader

**Supported Formats:**
- Synthetic: Generated on-the-fly
- MNIST: Via torchvision (auto-download)
- Custom: .npz, .pkl, .pth, .pt files

---

### Frontend Files

#### `templates/index.html` - Main UI

**Structure:**
- Control Panel: Experiment configuration
- Results Panel: Charts and metrics
- Info Panel: Educational content

**Key Elements:**
- Dataset selector (synthetic/MNIST/upload)
- Privacy technique checkboxes
- Real-time charts (Chart.js)
- Error display area

---

#### `static/js/app.js` - Frontend Logic

**Global Variables:**
- `trainingChart` - Main training progress chart
- `comparisonChart` - Privacy-accuracy trade-off chart
- `uploadedDatasetInfo` - Uploaded dataset metadata

**Key Functions:**

1. **`runExperiment()`** (Lines 89-128)
   - Collects form data
   - Sends API request
   - Handles response/errors
   - Updates UI

2. **`displayResults(result)`** (Lines 171-324)
   - Validates result structure
   - Updates metrics display
   - Creates/updates Chart.js charts
   - Error handling for chart failures

3. **`uploadDataset()`** (Lines 491-544)
   - Handles file upload
   - Validates file
   - Updates dataset selector
   - Shows upload status

4. **`showError(message)`** (Lines 451-475)
   - Displays error messages
   - Auto-hides after 10 seconds
   - Logs to console

**Error Handling:**
- Try-catch blocks around all async operations
- Validation of API responses
- Chart.js availability checks
- DOM element existence checks

---

#### `static/css/style.css` - Styling

**Key Features:**
- Responsive grid layout
- Gradient color scheme
- Smooth animations
- Mobile-friendly design

**Classes:**
- `.privacy-controls` - Privacy technique panels
- `.upload-status` - File upload feedback
- `.warning-box` - Important notices
- `.metric-card` - Result display cards

---

## 🔍 Debugging Guide

### Common Issues & Solutions

#### 1. Charts Not Showing

**Symptoms:** Blank chart area, no errors visible

**Debug Steps:**
```javascript
// Open browser console (F12)
// Check for:
console.log(typeof Chart);  // Should be "function"
console.log(document.getElementById('training-chart'));  // Should not be null

// Check Chart.js loading:
// Network tab → Look for chart.js CDN request
```

**Fixes:**
- Ensure Chart.js CDN is loading (check Network tab)
- Verify canvas element exists in HTML
- Check for JavaScript errors in console

#### 2. Experiments Failing

**Symptoms:** Error message, no results

**Debug Steps:**
```python
# In app.py, add logging:
import logging
logging.basicConfig(level=logging.DEBUG)

# Check terminal output for:
# - Import errors
# - Dataset loading errors
# - Model creation errors
```

**Common Causes:**
- Missing dependencies
- Invalid dataset format
- Out of memory
- Invalid parameters

#### 3. File Upload Not Working

**Symptoms:** Upload button does nothing, no feedback

**Debug Steps:**
```javascript
// Check file input:
const fileInput = document.getElementById('dataset_file');
console.log(fileInput.files);  // Should show selected file

// Check API response:
// Network tab → upload_dataset → Response
```

**Fixes:**
- Verify file format is supported (.npz, .pkl, etc.)
- Check file size (max 100MB)
- Ensure file contains 'X' and 'y' keys/arrays

#### 4. WebSocket Connection Issues

**Symptoms:** Real-time updates not working

**Debug Steps:**
```javascript
// Check Socket.IO connection:
// Browser console:
socket = io();
socket.on('connect', () => console.log('Connected'));
```

**Fixes:**
- Check server is running
- Verify WebSocket port is accessible
- Check firewall settings

---

## 🧪 Testing Checklist

### Backend Testing

```python
# Test dataset loading
from data import load_dataset
X, y = load_dataset("synthetic", n_samples=100)
assert len(X) == 100

# Test privacy protocols
from federated import DifferentialPrivacy
dp = DifferentialPrivacy(epsilon=1.0)
grads = {"weight": torch.randn(10, 5)}
noisy = dp.apply_dp(grads)
assert noisy["weight"].shape == grads["weight"].shape

# Test experiment runner
result = run_federated_experiment(
    num_clients=3,
    num_rounds=2,
    use_dp=True
)
assert result["success"] == True
assert "round_metrics" in result
```

### Frontend Testing

```javascript
// Test chart creation
const canvas = document.getElementById('training-chart');
const ctx = canvas.getContext('2d');
const chart = new Chart(ctx, {type: 'line', data: {labels: [1,2,3], datasets: []}});
// Should create chart without errors

// Test API call
fetch('/api/run_experiment', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({num_clients: 3, num_rounds: 2})
})
.then(r => r.json())
.then(d => console.log(d));
```

---

## 📊 Data Flow

### Experiment Execution Flow

```
1. User clicks "Run Experiment"
   ↓
2. JavaScript collects form data
   ↓
3. POST /api/run_experiment
   ↓
4. Backend: run_federated_experiment()
   ├─ Load dataset (synthetic/MNIST/uploaded)
   ├─ Split among clients
   ├─ Initialize privacy mechanisms
   └─ Run FL rounds:
      ├─ Broadcast model to clients
      ├─ Clients train locally
      ├─ Apply privacy (DP/SMPC/HE)
      ├─ Aggregate updates
      └─ Evaluate on test set
   ↓
5. Return results JSON
   ↓
6. JavaScript: displayResults()
   ├─ Update metrics
   ├─ Create/update charts
   └─ Show results panel
```

### Privacy Technique Application Order

```
1. Client Training
   ↓
2. Get Updated Parameters
   ↓
3. Apply DP (if enabled)
   ├─ Compute parameter differences
   ├─ Clip gradients
   └─ Add noise
   ↓
4. Aggregate (if SMPC/HE enabled)
   ├─ SMPC: Secret share → Aggregate → Reconstruct
   └─ HE: Encrypt → Aggregate → Decrypt
   ↓
5. Update Global Model
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Set upload directory
export UPLOAD_FOLDER=/path/to/uploads

# Optional: Set max file size
export MAX_UPLOAD_SIZE=100000000  # 100MB
```

### Flask Configuration

```python
# In app.py:
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'npz', 'pkl', 'pt', 'pth', 'npy'}
```

---

## 📝 Code Style & Best Practices

### Python

- **Type Hints:** Used where applicable
- **Docstrings:** All functions documented
- **Error Handling:** Try-except blocks with informative messages
- **Logging:** Print statements for debugging (can be upgraded to logging)

### JavaScript

- **Error Handling:** Try-catch around all async operations
- **Validation:** Check DOM elements exist before use
- **Console Logging:** Error messages logged for debugging
- **Comments:** Key functions documented

---

## 🐛 Known Issues & Limitations

1. **HE Performance:** Only encrypts subset of parameters (lightweight mode)
2. **SMPC Security:** Simplified implementation (not production-grade)
3. **Memory:** Large datasets may cause memory issues
4. **Concurrency:** Only one experiment at a time (by design)

---

## 🚀 Performance Tips

1. **Reduce Samples:** Use fewer samples for faster experiments
2. **Fewer Rounds:** Start with 5-10 rounds for testing
3. **Disable HE:** HE is slow, use only for demonstration
4. **Use Synthetic Data:** Faster than loading real datasets

---

## 📖 Additional Resources

- **Federated Learning:** [FedAvg Paper](https://arxiv.org/abs/1602.05629)
- **Differential Privacy:** [DP-SGD Paper](https://arxiv.org/abs/1607.00133)
- **Chart.js Docs:** [chartjs.org](https://www.chartjs.org/)
- **Flask Docs:** [flask.palletsprojects.com](https://flask.palletsprojects.com/)

---

**Last Updated:** 2024
**Version:** 1.0

