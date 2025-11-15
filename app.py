"""
Flask Web Application for Federated Learning Privacy Demo
Demonstrates how noise affects model accuracy in federated learning.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import traceback
import threading
import time
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models import SimpleMLP
from data import generate_synthetic_data, split_data_among_clients, load_dataset
from federated import FederatedClient, FederatedServer, DifferentialPrivacy, SecureMultiPartyComputation, HomomorphicEncryption

app = Flask(__name__)
app.config['SECRET_KEY'] = 'federated-learning-demo-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'npz', 'pkl', 'pt', 'pth', 'npy'}
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global state for experiments
experiment_history = []

# Real-time multi-device state
connected_devices = {}  # {device_id: {name, connected_at, status}}
active_experiments = {}  # {experiment_id: {devices, status, progress}}
device_rooms = defaultdict(set)  # {device_id: set of room_ids}


def aggregate_with_he(he, client_updates, global_params):
    """
    Lightweight HE aggregation: Only encrypt/aggregate a subset of parameters.
    This makes it practical for demonstration while showing the concept.
    """
    import time
    start_time = time.time()
    
    # For lightweight demo: only aggregate first layer weights
    # This significantly reduces computation time
    param_name = "fc1.weight"  # First layer weights
    
    if param_name not in client_updates[0]["parameters"]:
        # Fallback: use first available parameter
        param_name = list(client_updates[0]["parameters"].keys())[0]
    
    # Get parameter shape
    param_shape = client_updates[0]["parameters"][param_name].shape
    param_size = param_shape[0] * param_shape[1] if len(param_shape) == 2 else param_shape[0]
    
    # Limit to small parameters for performance (max 1000 elements)
    if param_size > 1000:
        # Sample a subset for demonstration
        sample_size = min(100, param_size)
        # For simplicity, just aggregate first layer bias if available
        if "fc1.bias" in client_updates[0]["parameters"]:
            param_name = "fc1.bias"
            param_shape = client_updates[0]["parameters"][param_name].shape
    
    # Initialize aggregated parameters with global model
    aggregated_params = global_params.copy()
    
    # Encrypt and aggregate the selected parameter
    encrypted_sum = None
    total_samples = sum(update["num_samples"] for update in client_updates)
    
    for update in client_updates:
        param_value = update["parameters"][param_name]
        weight = update["num_samples"] / total_samples
        
        # Flatten and encrypt
        param_flat = param_value.flatten()
        
        # Encrypt
        encrypted = he.encrypt(param_flat)
        
        # Scale by weight
        encrypted_scaled = he.encrypted_multiply(encrypted, weight)
        
        # Accumulate
        if encrypted_sum is None:
            encrypted_sum = encrypted_scaled
        else:
            encrypted_sum = he.encrypted_add(encrypted_sum, encrypted_scaled)
    
    # Decrypt aggregated result
    decrypted_flat = he.decrypt(encrypted_sum, shape=param_shape)
    aggregated_params[param_name] = decrypted_flat.reshape(param_shape)
    
    # For other parameters, use standard aggregation (much faster)
    for key in aggregated_params.keys():
        if key != param_name:
            total_samples = sum(update["num_samples"] for update in client_updates)
            aggregated_params[key] = torch.zeros_like(global_params[key])
            for update in client_updates:
                weight = update["num_samples"] / total_samples
                aggregated_params[key] += weight * update["parameters"][key]
    
    elapsed_time = time.time() - start_time
    print(f"HE aggregation completed in {elapsed_time:.2f}s (lightweight mode)")
    
    return aggregated_params


def run_federated_experiment(
    num_clients=5,
    num_rounds=10,
    local_epochs=1,
    epsilon=1.0,
    delta=1e-5,
    clip_norm=1.0,
    use_dp=False,
    use_smpc=False,
    use_he=False,
    num_samples=1000,
    num_features=784,
    num_classes=10,
    dataset_name="synthetic"
):
    """
    Run a federated learning experiment with optional differential privacy.
    
    Returns:
        Dictionary with experiment results and metrics
    """
    try:
        # Load or generate data based on dataset_name
        if dataset_name.lower().startswith("uploaded:"):
            # Custom uploaded dataset
            filename = dataset_name.replace("uploaded:", "")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return {
                    "success": False,
                    "error": f"Uploaded dataset file not found: {filename}"
                }
            try:
                X, y = load_custom_dataset(filepath)
                # Limit samples if needed
                if len(X) > num_samples:
                    indices = torch.randperm(len(X))[:num_samples]
                    X = X[indices]
                    y = y[indices]
                num_features = X.shape[1]
                num_classes = len(torch.unique(y))
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to load uploaded dataset: {str(e)}"
                }
        elif dataset_name.lower() == "mnist":
            try:
                X, y = load_dataset("mnist", train=True, download=True)
                # Limit samples for faster training
                if len(X) > num_samples:
                    indices = torch.randperm(len(X))[:num_samples]
                    X = X[indices]
                    y = y[indices]
                num_features = 784  # MNIST is 28x28 = 784
                num_classes = 10
            except ImportError as e:
                return {
                    "success": False,
                    "error": f"MNIST requires torchvision: {str(e)}. Install with: pip install torchvision"
                }
        else:
            # Use synthetic data
            X, y = generate_synthetic_data(
                n_samples=num_samples,
                n_features=num_features,
                n_classes=num_classes,
                random_state=42
            )
        
        # Split among clients
        client_data = split_data_among_clients(X, y, num_clients, iid=True)
        
        # Create model
        model = SimpleMLP(input_size=num_features, hidden_size=128, num_classes=num_classes)
        
        # Initialize server
        server = FederatedServer(global_model=model)
        
        # Initialize privacy mechanisms if requested
        dp = None
        smpc = None
        he = None
        
        if use_dp:
            dp = DifferentialPrivacy(epsilon=epsilon, delta=delta, clip_norm=clip_norm)
        
        if use_smpc:
            smpc = SecureMultiPartyComputation(num_parties=num_clients)
        
        if use_he:
            try:
                he = HomomorphicEncryption()
            except RuntimeError as e:
                # If TenSEAL not available, disable HE
                use_he = False
                print(f"Warning: HE disabled - {e}")
        
        # Create and register clients
        clients = []
        for i, (X_client, y_client) in enumerate(client_data):
            client_model = SimpleMLP(input_size=num_features, hidden_size=128, num_classes=num_classes)
            client = FederatedClient(
                client_id=i,
                model=client_model,
                train_data=(X_client, y_client),
                lr=0.01
            )
            clients.append(client)
            server.register_client(client)
        
        # Run federated learning rounds
        round_metrics = []
        for round_num in range(num_rounds):
            # Broadcast global model to all clients
            server.broadcast_model()
            
            # Select clients for this round
            selected_indices = server.select_clients(min(3, num_clients))
            selected_clients = [server.clients[i] for i in selected_indices]
            
            # Collect updates from selected clients
            client_updates = []
            client_metrics = []
            
            for client in selected_clients:
                # Local training
                metrics = client.train(epochs=local_epochs)
                client_metrics.append(metrics)
                
                # Get updated parameters
                params = client.get_model_parameters()
                
                # Apply DP to parameter updates if enabled
                if use_dp and dp:
                    # Convert parameters to gradients (difference from global model)
                    global_params = server.global_model.state_dict()
                    param_diff = {}
                    for key in params.keys():
                        param_diff[key] = params[key] - global_params[key]
                    
                    # Apply DP to the parameter differences (treating them as gradients)
                    dp_params = dp.apply_dp(param_diff)
                    
                    # Add back to global model to get noisy parameters
                    noisy_params = {}
                    for key in params.keys():
                        noisy_params[key] = global_params[key] + dp_params[key]
                    params = noisy_params
                
                num_samples = len(client.data_loader.dataset) if client.data_loader else 1
                
                client_updates.append({
                    "parameters": params,
                    "num_samples": num_samples,
                    "client_id": client.client_id
                })
            
            # Aggregate updates using selected privacy mechanisms
            # Apply in order: DP (on individual updates) -> SMPC/HE (on aggregation)
            
            # Step 1: Apply DP to individual client updates if enabled
            if use_dp and dp:
                for update in client_updates:
                    # Apply DP to parameter differences
                    global_params = server.global_model.state_dict()
                    param_diff = {}
                    for key in update["parameters"].keys():
                        param_diff[key] = update["parameters"][key] - global_params[key]
                    
                    # Apply DP
                    dp_params = dp.apply_dp(param_diff)
                    
                    # Add back to global model
                    for key in update["parameters"].keys():
                        update["parameters"][key] = global_params[key] + dp_params[key]
            
            # Step 2: Aggregate using SMPC or HE if enabled
            if use_he and he:
                # Lightweight HE: Only encrypt/aggregate first layer parameters for demo
                try:
                    aggregated_params = aggregate_with_he(he, client_updates, server.global_model.state_dict())
                except Exception as e:
                    # Fallback to SMPC or regular aggregation if HE fails
                    print(f"HE aggregation failed, falling back: {e}")
                    if use_smpc and smpc:
                        total_samples = sum(update["num_samples"] for update in client_updates)
                        weighted_updates = []
                        for update in client_updates:
                            weight = update["num_samples"] / total_samples
                            weighted_params = {k: v * weight for k, v in update["parameters"].items()}
                            weighted_updates.append(weighted_params)
                        aggregated_params = smpc.secure_aggregation(weighted_updates)
                    else:
                        aggregated_params = server.aggregate_updates(client_updates)
            elif use_smpc and smpc:
                # Use SMPC for secure aggregation
                total_samples = sum(update["num_samples"] for update in client_updates)
                weighted_updates = []
                for update in client_updates:
                    weight = update["num_samples"] / total_samples
                    weighted_params = {k: v * weight for k, v in update["parameters"].items()}
                    weighted_updates.append(weighted_params)
                aggregated_params = smpc.secure_aggregation(weighted_updates)
            else:
                # Standard FedAvg aggregation
                aggregated_params = server.aggregate_updates(client_updates)
            
            # Update global model
            server.global_model.load_state_dict(aggregated_params)
            
            # Calculate average metrics
            avg_loss = sum(m["loss"] for m in client_metrics) / len(client_metrics) if client_metrics else 0.0
            
            # Evaluate on test set
            if dataset_name.lower() == "mnist":
                try:
                    test_X, test_y = load_dataset("mnist", train=False, download=True)
                    # Limit test samples
                    if len(test_X) > 200:
                        indices = torch.randperm(len(test_X))[:200]
                        test_X = test_X[indices]
                        test_y = test_y[indices]
                except ImportError:
                    # Fallback to synthetic test data
                    test_X, test_y = generate_synthetic_data(
                        n_samples=200,
                        n_features=num_features,
                        n_classes=num_classes,
                        random_state=999
                    )
            else:
                test_X, test_y = generate_synthetic_data(
                    n_samples=200,
                    n_features=num_features,
                    n_classes=num_classes,
                    random_state=999
                )
            metrics = server.evaluate_global_model((test_X, test_y))
            
            round_metrics.append({
                "round": round_num + 1,
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "avg_client_loss": avg_loss
            })
        
        return {
            "success": True,
            "round_metrics": round_metrics,
            "final_accuracy": round_metrics[-1]["accuracy"],
            "final_loss": round_metrics[-1]["loss"],
            "config": {
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "epsilon": epsilon if use_dp else None,
                "delta": delta if use_dp else None,
                "clip_norm": clip_norm if use_dp else None,
                "use_dp": use_dp,
                "use_smpc": use_smpc,
                "use_he": use_he
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/run_experiment', methods=['POST'])
def run_experiment():
    """Run a federated learning experiment."""
    try:
        data = request.json
        
        num_clients = int(data.get('num_clients', 5))
        num_rounds = int(data.get('num_rounds', 10))
        local_epochs = int(data.get('local_epochs', 1))
        epsilon = float(data.get('epsilon', 1.0))
        delta = float(data.get('delta', 1e-5))
        clip_norm = float(data.get('clip_norm', 1.0))
        use_dp = bool(data.get('use_dp', False))
        use_smpc = bool(data.get('use_smpc', False))
        use_he = bool(data.get('use_he', False))
        num_samples = int(data.get('num_samples', 1000))
        dataset_name = data.get('dataset_name', 'synthetic')
        
        # Allow multiple privacy techniques
        # They will be applied in order: DP -> SMPC -> HE (if applicable)
        
        result = run_federated_experiment(
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            epsilon=epsilon,
            delta=delta,
            clip_norm=clip_norm,
            use_dp=use_dp,
            use_smpc=use_smpc,
            use_he=use_he,
            num_samples=num_samples,
            dataset_name=dataset_name
        )
        
        # Store in history
        if result["success"]:
            experiment_history.append(result)
            result["experiment_id"] = len(experiment_history) - 1
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/compare_noise', methods=['POST'])
def compare_noise():
    """Compare multiple experiments with different noise levels."""
    try:
        data = request.json
        epsilon_values = data.get('epsilon_values', [0.1, 0.5, 1.0, 2.0, 5.0])
        num_rounds = int(data.get('num_rounds', 10))
        num_clients = int(data.get('num_clients', 5))
        
        results = []
        for epsilon in epsilon_values:
            result = run_federated_experiment(
                num_clients=num_clients,
                num_rounds=num_rounds,
                epsilon=epsilon,
                delta=1e-5,
                clip_norm=1.0,
                use_dp=True,
                num_samples=1000
            )
            if result["success"]:
                results.append({
                    "epsilon": epsilon,
                    "final_accuracy": result["final_accuracy"],
                    "final_loss": result["final_loss"],
                    "round_metrics": result["round_metrics"]
                })
        
        # Also run baseline (no DP)
        baseline = run_federated_experiment(
            num_clients=num_clients,
            num_rounds=num_rounds,
            use_dp=False,
            num_samples=1000
        )
        
        return jsonify({
            "success": True,
            "results": results,
            "baseline": baseline if baseline["success"] else None
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get experiment history."""
    return jsonify({
        "success": True,
        "history": experiment_history
    })


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear experiment history."""
    global experiment_history
    experiment_history = []
    return jsonify({"success": True})


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    """
    Upload a custom dataset file.
    Supports: .npz (numpy), .pkl/.pth/.pt (PyTorch), .npy (numpy array)
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Try to load and validate the dataset
            try:
                X, y = load_custom_dataset(filepath)
                
                return jsonify({
                    "success": True,
                    "filename": filename,
                    "filepath": filepath,
                    "num_samples": len(X),
                    "num_features": X.shape[1] if len(X.shape) > 1 else X.shape[0],
                    "num_classes": len(torch.unique(y)) if y is not None else None,
                    "message": f"Dataset uploaded successfully: {len(X)} samples"
                })
            except Exception as e:
                # Remove invalid file
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    "success": False,
                    "error": f"Invalid dataset file: {str(e)}"
                }), 400
        else:
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Allowed: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"
            }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Upload failed: {str(e)}"
        }), 500


def load_custom_dataset(filepath):
    """
    Load a custom dataset from file.
    
    Supported formats:
    - .npz: numpy compressed file with 'X' and 'y' keys
    - .pkl/.pth/.pt: PyTorch pickle with (X, y) tuple or dict
    - .npy: numpy array (single array, will need manual splitting)
    
    Args:
        filepath: Path to the dataset file
        
    Returns:
        Tuple of (X, y) tensors
    """
    import numpy as np
    
    ext = filepath.rsplit('.', 1)[1].lower()
    
    if ext == 'npz':
        # NumPy compressed format
        data = np.load(filepath)
        if 'X' in data and 'y' in data:
            X = torch.FloatTensor(data['X'])
            y = torch.LongTensor(data['y'])
        else:
            raise ValueError("NPZ file must contain 'X' and 'y' keys")
        data.close()
        
    elif ext in ['pkl', 'pth', 'pt']:
        # PyTorch pickle format
        data = torch.load(filepath, map_location='cpu')
        if isinstance(data, tuple) and len(data) == 2:
            X, y = data
            X = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X.float()
            y = torch.LongTensor(y) if not isinstance(y, torch.Tensor) else y.long()
        elif isinstance(data, dict):
            if 'X' in data and 'y' in data:
                X = torch.FloatTensor(data['X']) if not isinstance(data['X'], torch.Tensor) else data['X'].float()
                y = torch.LongTensor(data['y']) if not isinstance(data['y'], torch.Tensor) else data['y'].long()
            else:
                raise ValueError("Dict must contain 'X' and 'y' keys")
        else:
            raise ValueError("File must contain (X, y) tuple or dict with 'X' and 'y' keys")
            
    elif ext == 'npy':
        # Single numpy array - assume it's features only
        data = np.load(filepath)
        X = torch.FloatTensor(data)
        # Create dummy labels (user should provide proper dataset)
        y = torch.zeros(len(X), dtype=torch.long)
        raise ValueError("Numpy array format detected. Please provide dataset with both features (X) and labels (y) in .npz or .pkl format")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Validate shapes
    if len(X.shape) != 2:
        raise ValueError(f"Features X must be 2D (samples, features), got shape {X.shape}")
    if len(y.shape) != 1:
        raise ValueError(f"Labels y must be 1D (samples,), got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
    
    return X, y


# WebSocket event handlers for real-time multi-device support
@socketio.on('connect')
def handle_connect():
    """Handle device connection."""
    device_id = request.sid
    device_name = request.args.get('device_name', f'Device-{device_id[:8]}')
    
    connected_devices[device_id] = {
        'name': device_name,
        'connected_at': time.time(),
        'status': 'idle'
    }
    
    emit('connected', {
        'device_id': device_id,
        'device_name': device_name,
        'connected_devices': list(connected_devices.keys())
    })
    
    # Broadcast to all clients
    socketio.emit('device_connected', {
        'device_id': device_id,
        'device_name': device_name,
        'total_devices': len(connected_devices)
    }, broadcast=True)
    
    print(f"Device connected: {device_name} ({device_id})")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle device disconnection."""
    device_id = request.sid
    
    if device_id in connected_devices:
        device_name = connected_devices[device_id]['name']
        del connected_devices[device_id]
        
        # Leave all rooms
        if device_id in device_rooms:
            for room_id in device_rooms[device_id]:
                leave_room(room_id)
            del device_rooms[device_id]
        
        # Broadcast to all clients
        socketio.emit('device_disconnected', {
            'device_id': device_id,
            'device_name': device_name,
            'total_devices': len(connected_devices)
        }, broadcast=True)
        
        print(f"Device disconnected: {device_name} ({device_id})")


@socketio.on('join_experiment')
def handle_join_experiment(data):
    """Device joins an experiment room."""
    device_id = request.sid
    experiment_id = data.get('experiment_id')
    
    if experiment_id:
        join_room(experiment_id)
        if device_id not in device_rooms:
            device_rooms[device_id] = set()
        device_rooms[device_id].add(experiment_id)
        
        emit('joined_experiment', {'experiment_id': experiment_id})


@socketio.on('request_device_list')
def handle_device_list():
    """Send list of connected devices."""
    device_id = request.sid
    devices = [
        {
            'device_id': did,
            'name': info['name'],
            'status': info['status']
        }
        for did, info in connected_devices.items()
    ]
    emit('device_list', {'devices': devices})


@app.route('/api/run_realtime_experiment', methods=['POST'])
def run_realtime_experiment():
    """Run federated learning experiment with real-time updates via WebSocket."""
    try:
        data = request.json
        
        num_clients = int(data.get('num_clients', len(connected_devices)))
        num_rounds = int(data.get('num_rounds', 10))
        local_epochs = int(data.get('local_epochs', 1))
        epsilon = float(data.get('epsilon', 1.0))
        delta = float(data.get('delta', 1e-5))
        clip_norm = float(data.get('clip_norm', 1.0))
        use_dp = bool(data.get('use_dp', False))
        use_smpc = bool(data.get('use_smpc', False))
        use_he = bool(data.get('use_he', False))
        num_samples = int(data.get('num_samples', 1000))
        dataset_name = data.get('dataset_name', 'synthetic')
        
        experiment_id = f"exp_{int(time.time())}"
        
        # Start experiment in background thread
        def run_experiment_thread():
            try:
                socketio.emit('experiment_started', {
                    'experiment_id': experiment_id,
                    'config': {
                        'num_clients': num_clients,
                        'num_rounds': num_rounds,
                        'use_dp': use_dp,
                        'use_smpc': use_smpc,
                        'use_he': use_he
                    }
                }, room=experiment_id, broadcast=True)
                
                # Run experiment with progress updates
                result = run_federated_experiment_with_progress(
                    experiment_id=experiment_id,
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    local_epochs=local_epochs,
                    epsilon=epsilon,
                    delta=delta,
                    clip_norm=clip_norm,
                    use_dp=use_dp,
                    use_smpc=use_smpc,
                    use_he=use_he,
                    num_samples=num_samples,
                    dataset_name=dataset_name
                )
                
                socketio.emit('experiment_completed', {
                    'experiment_id': experiment_id,
                    'result': result
                }, room=experiment_id, broadcast=True)
                
            except Exception as e:
                socketio.emit('experiment_error', {
                    'experiment_id': experiment_id,
                    'error': str(e)
                }, room=experiment_id, broadcast=True)
        
        thread = threading.Thread(target=run_experiment_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "experiment_id": experiment_id,
            "message": "Experiment started in real-time mode"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def run_federated_experiment_with_progress(
    experiment_id,
    num_clients=5,
    num_rounds=10,
    local_epochs=1,
    epsilon=1.0,
    delta=1e-5,
    clip_norm=1.0,
    use_dp=False,
    use_smpc=False,
    use_he=False,
    num_samples=1000,
    num_features=784,
    num_classes=10,
    dataset_name="synthetic"
):
    """Run federated learning experiment with real-time progress updates."""
    try:
        # Load or generate data based on dataset_name
        if dataset_name.lower() == "mnist":
            try:
                X, y = load_dataset("mnist", train=True, download=True)
                if len(X) > num_samples:
                    indices = torch.randperm(len(X))[:num_samples]
                    X = X[indices]
                    y = y[indices]
                num_features = 784
                num_classes = 10
            except ImportError as e:
                socketio.emit('experiment_error', {
                    'experiment_id': experiment_id,
                    'error': f"MNIST requires torchvision: {str(e)}"
                }, room=experiment_id, broadcast=True)
                return {"success": False, "error": str(e)}
        else:
            X, y = generate_synthetic_data(
                n_samples=num_samples,
                n_features=num_features,
                n_classes=num_classes,
                random_state=42
            )
        
        # Split among clients
        client_data = split_data_among_clients(X, y, num_clients, iid=True)
        
        # Create model
        model = SimpleMLP(input_size=num_features, hidden_size=128, num_classes=num_classes)
        
        # Initialize server
        server = FederatedServer(global_model=model)
        
        # Initialize privacy mechanisms
        dp = None
        smpc = None
        he = None
        
        if use_dp:
            dp = DifferentialPrivacy(epsilon=epsilon, delta=delta, clip_norm=clip_norm)
        if use_smpc:
            smpc = SecureMultiPartyComputation(num_parties=num_clients)
        if use_he:
            try:
                he = HomomorphicEncryption()
            except RuntimeError as e:
                use_he = False
                socketio.emit('experiment_warning', {
                    'experiment_id': experiment_id,
                    'message': f'HE disabled: {e}'
                }, room=experiment_id, broadcast=True)
        
        # Create and register clients
        clients = []
        for i, (X_client, y_client) in enumerate(client_data):
            client_model = SimpleMLP(input_size=num_features, hidden_size=128, num_classes=num_classes)
            client = FederatedClient(
                client_id=i,
                model=client_model,
                train_data=(X_client, y_client),
                lr=0.01
            )
            clients.append(client)
            server.register_client(client)
        
        # Run federated learning rounds with progress updates
        round_metrics = []
        for round_num in range(num_rounds):
            # Emit round start
            socketio.emit('round_started', {
                'experiment_id': experiment_id,
                'round': round_num + 1,
                'total_rounds': num_rounds
            }, room=experiment_id, broadcast=True)
            
            # Broadcast global model
            server.broadcast_model()
            
            # Select clients
            selected_indices = server.select_clients(min(3, num_clients))
            selected_clients = [server.clients[i] for i in selected_indices]
            
            # Collect updates
            client_updates = []
            client_metrics = []
            
            for idx, client in enumerate(selected_clients):
                # Emit client training start
                socketio.emit('client_training', {
                    'experiment_id': experiment_id,
                    'round': round_num + 1,
                    'client_id': client.client_id,
                    'progress': (idx + 1) / len(selected_clients)
                }, room=experiment_id, broadcast=True)
                
                # Local training
                metrics = client.train(epochs=local_epochs)
                client_metrics.append(metrics)
                
                # Get updated parameters
                params = client.get_model_parameters()
                
                # Apply DP if enabled
                if use_dp and dp:
                    global_params = server.global_model.state_dict()
                    param_diff = {k: params[k] - global_params[k] for k in params.keys()}
                    dp_params = dp.apply_dp(param_diff)
                    params = {k: global_params[k] + dp_params[k] for k in params.keys()}
                
                num_samples = len(client.data_loader.dataset) if client.data_loader else 1
                client_updates.append({
                    "parameters": params,
                    "num_samples": num_samples,
                    "client_id": client.client_id
                })
            
            # Aggregate with progress update
            socketio.emit('aggregating', {
                'experiment_id': experiment_id,
                'round': round_num + 1
            }, room=experiment_id, broadcast=True)
            
            # Apply aggregation with privacy techniques
            if use_he and he:
                try:
                    aggregated_params = aggregate_with_he(he, client_updates, server.global_model.state_dict())
                except Exception as e:
                    if use_smpc and smpc:
                        total_samples = sum(u["num_samples"] for u in client_updates)
                        weighted_updates = [{k: v * (u["num_samples"] / total_samples) 
                                            for k, v in u["parameters"].items()} 
                                           for u in client_updates]
                        aggregated_params = smpc.secure_aggregation(weighted_updates)
                    else:
                        aggregated_params = server.aggregate_updates(client_updates)
            elif use_smpc and smpc:
                total_samples = sum(u["num_samples"] for u in client_updates)
                weighted_updates = [{k: v * (u["num_samples"] / total_samples) 
                                    for k, v in u["parameters"].items()} 
                                   for u in client_updates]
                aggregated_params = smpc.secure_aggregation(weighted_updates)
            else:
                aggregated_params = server.aggregate_updates(client_updates)
            
            # Update global model
            server.global_model.load_state_dict(aggregated_params)
            
            # Evaluate
            if dataset_name.lower() == "mnist":
                try:
                    test_X, test_y = load_dataset("mnist", train=False, download=True)
                    if len(test_X) > 200:
                        indices = torch.randperm(len(test_X))[:200]
                        test_X = test_X[indices]
                        test_y = test_y[indices]
                except ImportError:
                    test_X, test_y = generate_synthetic_data(
                        n_samples=200, n_features=num_features, n_classes=num_classes, random_state=999
                    )
            else:
                test_X, test_y = generate_synthetic_data(
                    n_samples=200, n_features=num_features, n_classes=num_classes, random_state=999
                )
            metrics = server.evaluate_global_model((test_X, test_y))
            
            round_metrics.append({
                "round": round_num + 1,
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "avg_client_loss": sum(m["loss"] for m in client_metrics) / len(client_metrics) if client_metrics else 0.0
            })
            
            # Emit round complete
            socketio.emit('round_completed', {
                'experiment_id': experiment_id,
                'round': round_num + 1,
                'metrics': round_metrics[-1],
                'progress': (round_num + 1) / num_rounds
            }, room=experiment_id, broadcast=True)
        
        return {
            "success": True,
            "round_metrics": round_metrics,
            "final_accuracy": round_metrics[-1]["accuracy"],
            "final_loss": round_metrics[-1]["loss"],
            "config": {
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "epsilon": epsilon if use_dp else None,
                "use_dp": use_dp,
                "use_smpc": use_smpc,
                "use_he": use_he
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == '__main__':
    print("Starting Federated Learning Privacy Demo...")
    print("Open http://localhost:5000 in your browser")
    print("Real-time multi-device support enabled!")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

