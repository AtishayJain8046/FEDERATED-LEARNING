"""
Federated Learning Demo Runner
Main script to run federated learning experiments with privacy-preserving techniques.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import SimpleMLP
from data import generate_synthetic_data, split_data_among_clients
from federated import FederatedClient, FederatedServer, DifferentialPrivacy, SecureMultiPartyComputation, HomomorphicEncryption


def create_model(model_type: str = "mlp", input_size: int = 784, num_classes: int = 10) -> nn.Module:
    """
    Create a model instance.
    
    Args:
        model_type: Type of model ("mlp" or "logistic")
        input_size: Input feature size
        num_classes: Number of output classes
        
    Returns:
        Model instance
    """
    if model_type == "mlp":
        return SimpleMLP(input_size=input_size, hidden_size=128, num_classes=num_classes)
    else:
        from models import LogisticRegression
        return LogisticRegression(input_size=input_size, num_classes=num_classes)


def demo_basic_federated_learning():
    """Demo: Basic federated learning without privacy."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Federated Learning")
    print("="*60)
    
    # Generate data
    print("\nGenerating synthetic data...")
    X, y = generate_synthetic_data(n_samples=1000, n_features=784, n_classes=10)
    
    # Split among clients
    num_clients = 5
    client_data = split_data_among_clients(X, y, num_clients, iid=True)
    print(f"Split data among {num_clients} clients")
    
    # Create model
    model = create_model("mlp", input_size=784, num_classes=10)
    
    # Initialize server
    server = FederatedServer(global_model=model)
    
    # Create and register clients
    clients = []
    for i, (X_client, y_client) in enumerate(client_data):
        client_model = create_model("mlp", input_size=784, num_classes=10)
        client = FederatedClient(
            client_id=i,
            model=client_model,
            train_data=(X_client, y_client),
            lr=0.01
        )
        clients.append(client)
        server.register_client(client)
    
    # Run federated learning rounds
    num_rounds = 5
    print(f"\nRunning {num_rounds} federated learning rounds...")
    
    for round_num in range(num_rounds):
        round_info = server.run_round(num_clients=3, local_epochs=1)
        print(f"Round {round_info['round']}: Avg Loss = {round_info['avg_loss']:.4f}")
    
    # Evaluate
    test_X, test_y = generate_synthetic_data(n_samples=200, n_features=784, n_classes=10, random_state=999)
    metrics = server.evaluate_global_model((test_X, test_y))
    print(f"\nGlobal Model Test Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Global Model Test Loss: {metrics['loss']:.4f}")


def demo_differential_privacy():
    """Demo: Federated learning with differential privacy."""
    print("\n" + "="*60)
    print("DEMO 2: Federated Learning with Differential Privacy")
    print("="*60)
    
    # Initialize DP mechanism
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
    print(f"\nDP Parameters: ε={dp.epsilon}, δ={dp.delta}, Clip Norm={dp.clip_norm}")
    
    # Generate sample gradients
    print("\nDemonstrating DP on gradients...")
    sample_gradients = {
        "fc1.weight": torch.randn(128, 784) * 0.1,
        "fc1.bias": torch.randn(128) * 0.1,
    }
    
    # Apply DP
    clipped = dp.clip_gradients(sample_gradients)
    noisy = dp.add_noise(clipped)
    
    print(f"Original gradient norm: {sum(g.norm().item() for g in sample_gradients.values()):.4f}")
    print(f"Clipped gradient norm: {sum(g.norm().item() for g in clipped.values()):.4f}")
    print(f"Noisy gradient norm: {sum(g.norm().item() for g in noisy.values()):.4f}")
    print("\n✓ Differential Privacy mechanism applied successfully!")


def demo_smpc():
    """Demo: Secure Multi-Party Computation."""
    print("\n" + "="*60)
    print("DEMO 3: Secure Multi-Party Computation (Secret Sharing)")
    print("="*60)
    
    # Initialize SMPC
    smpc = SecureMultiPartyComputation(num_parties=3)
    print(f"\nSMPC initialized with {smpc.num_parties} parties")
    
    # Create secret value
    secret_value = torch.tensor([[1.5, 2.3, 4.1], [0.8, 1.2, 3.5]])
    print(f"\nOriginal secret value:\n{secret_value}")
    
    # Secret share
    shares = smpc.secret_share(secret_value)
    print(f"\nCreated {len(shares)} secret shares")
    print(f"Share 1 (first few values): {shares[0][0, :3]}")
    print(f"Share 2 (first few values): {shares[1][0, :3]}")
    print(f"Share 3 (first few values): {shares[2][0, :3]}")
    
    # Reconstruct
    reconstructed = smpc.reconstruct(shares)
    print(f"\nReconstructed value:\n{reconstructed}")
    print(f"Reconstruction error: {(secret_value - reconstructed).abs().max().item():.6f}")
    print("\n✓ Secret sharing and reconstruction successful!")


def demo_fl_with_dp():
    """Demo: Compare Federated Learning with and without Differential Privacy."""
    print("\n" + "="*60)
    print("DEMO 5: Federated Learning: With vs Without Differential Privacy")
    print("="*60)
    
    # Generate data
    print("\nGenerating synthetic data...")
    X, y = generate_synthetic_data(n_samples=1000, n_features=784, n_classes=10)
    test_X, test_y = generate_synthetic_data(n_samples=200, n_features=784, n_classes=10, random_state=999)
    
    num_clients = 5
    num_rounds = 10
    
    # Split data
    client_data = split_data_among_clients(X, y, num_clients, iid=True)
    print(f"Split data among {num_clients} clients")
    
    results = {}
    
    # Run FL WITHOUT DP
    print("\n" + "-"*60)
    print("Running Federated Learning WITHOUT Differential Privacy...")
    print("-"*60)
    
    model_no_dp = create_model("mlp", input_size=784, num_classes=10)
    server_no_dp = FederatedServer(global_model=model_no_dp)
    
    for i, (X_client, y_client) in enumerate(client_data):
        client_model = create_model("mlp", input_size=784, num_classes=10)
        client = FederatedClient(
            client_id=i,
            model=client_model,
            train_data=(X_client, y_client),
            lr=0.01
        )
        server_no_dp.register_client(client)
    
    print(f"Running {num_rounds} federated learning rounds...")
    for round_num in range(num_rounds):
        round_info = server_no_dp.run_round(num_clients=3, local_epochs=1)
        if (round_num + 1) % 2 == 0:
            print(f"  Round {round_info['round']}: Avg Loss = {round_info['avg_loss']:.4f}")
    
    metrics_no_dp = server_no_dp.evaluate_global_model((test_X, test_y))
    results['no_dp'] = {
        'accuracy': metrics_no_dp['accuracy'],
        'loss': metrics_no_dp['loss']
    }
    
    print(f"\n✓ Without DP - Final Accuracy: {metrics_no_dp['accuracy']:.2f}%")
    print(f"  Final Loss: {metrics_no_dp['loss']:.4f}")
    
    # Run FL WITH DP (different epsilon values)
    epsilon_values = [0.5, 1.0, 2.0, 5.0]
    
    for epsilon in epsilon_values:
        print("\n" + "-"*60)
        print(f"Running Federated Learning WITH Differential Privacy (ε={epsilon})...")
        print("-"*60)
        
        dp = DifferentialPrivacy(epsilon=epsilon, delta=1e-5, clip_norm=1.0)
        print(f"DP Parameters: ε={dp.epsilon}, δ={dp.delta}, Clip Norm={dp.clip_norm}")
        
        model_dp = create_model("mlp", input_size=784, num_classes=10)
        server_dp = FederatedServer(global_model=model_dp)
        
        for i, (X_client, y_client) in enumerate(client_data):
            client_model = create_model("mlp", input_size=784, num_classes=10)
            client = FederatedClient(
                client_id=i,
                model=client_model,
                train_data=(X_client, y_client),
                lr=0.01
            )
            server_dp.register_client(client)
        
        print(f"Running {num_rounds} federated learning rounds...")
        for round_num in range(num_rounds):
            # Broadcast model
            server_dp.broadcast_model()
            
            # Select clients
            selected_indices = server_dp.select_clients(3)
            selected_clients = [server_dp.clients[i] for i in selected_indices]
            
            # Collect updates with DP
            client_updates = []
            client_metrics = []
            
            for client in selected_clients:
                metrics = client.train(epochs=1)
                client_metrics.append(metrics)
                
                params = client.get_model_parameters()
                
                # Apply DP
                global_params = server_dp.global_model.state_dict()
                param_diff = {k: params[k] - global_params[k] for k in params.keys()}
                dp_params = dp.apply_dp(param_diff)
                params = {k: global_params[k] + dp_params[k] for k in params.keys()}
                
                num_samples = len(client.data_loader.dataset) if client.data_loader else 1
                client_updates.append({
                    "parameters": params,
                    "num_samples": num_samples,
                    "client_id": client.client_id
                })
            
            # Aggregate
            aggregated_params = server_dp.aggregate_updates(client_updates)
            server_dp.global_model.load_state_dict(aggregated_params)
            
            if (round_num + 1) % 2 == 0:
                avg_loss = sum(m["loss"] for m in client_metrics) / len(client_metrics)
                print(f"  Round {round_num + 1}: Avg Loss = {avg_loss:.4f}")
        
        metrics_dp = server_dp.evaluate_global_model((test_X, test_y))
        results[f'dp_eps_{epsilon}'] = {
            'accuracy': metrics_dp['accuracy'],
            'loss': metrics_dp['loss'],
            'epsilon': epsilon
        }
        
        print(f"\n✓ With DP (ε={epsilon}) - Final Accuracy: {metrics_dp['accuracy']:.2f}%")
        print(f"  Final Loss: {metrics_dp['loss']:.4f}")
        print(f"  Accuracy Drop: {results['no_dp']['accuracy'] - metrics_dp['accuracy']:.2f}%")
    
    # Summary comparison
    print("\n" + "="*60)
    print("PRIVACY-ACCURACY TRADE-OFF SUMMARY")
    print("="*60)
    print(f"\nBaseline (No DP):")
    print(f"  Accuracy: {results['no_dp']['accuracy']:.2f}%")
    print(f"  Loss: {results['no_dp']['loss']:.4f}")
    
    print(f"\nWith Differential Privacy:")
    for key in sorted([k for k in results.keys() if k.startswith('dp_eps_')]):
        eps = results[key]['epsilon']
        acc = results[key]['accuracy']
        loss = results[key]['loss']
        drop = results['no_dp']['accuracy'] - acc
        print(f"  ε={eps:4.1f}: Accuracy={acc:6.2f}% (Drop: {drop:5.2f}%), Loss={loss:.4f}")
    
    print("\n" + "="*60)
    print("KEY INSIGHT: Lower ε (more privacy) = Higher accuracy drop")
    print("="*60)


def demo_homomorphic_encryption():
    """Demo: Homomorphic Encryption."""
    print("\n" + "="*60)
    print("DEMO 4: Homomorphic Encryption")
    print("="*60)
    
    try:
        # Initialize HE
        he = HomomorphicEncryption()
        print("\nHomomorphic Encryption initialized (CKKS scheme)")
        
        # Encrypt values
        value1 = torch.tensor([1.5, 2.3, 4.1, 0.8])
        value2 = torch.tensor([0.5, 1.2, 2.0, 0.3])
        
        print(f"\nPlaintext 1: {value1}")
        print(f"Plaintext 2: {value2}")
        
        encrypted1 = he.encrypt(value1)
        encrypted2 = he.encrypt(value2)
        print("\n✓ Values encrypted")
        
        # Encrypted addition
        encrypted_sum = he.encrypted_add(encrypted1, encrypted2)
        decrypted_sum = he.decrypt(encrypted_sum, shape=value1.shape)
        
        print(f"\nEncrypted addition result: {decrypted_sum}")
        print(f"Expected sum: {value1 + value2}")
        print(f"Error: {(decrypted_sum - (value1 + value2)).abs().max().item():.6f}")
        
        # Encrypted scalar multiplication
        encrypted_scaled = he.encrypted_multiply(encrypted1, 2.0)
        decrypted_scaled = he.decrypt(encrypted_scaled, shape=value1.shape)
        
        print(f"\nEncrypted scalar multiplication (×2) result: {decrypted_scaled}")
        print(f"Expected: {value1 * 2.0}")
        print(f"Error: {(decrypted_scaled - (value1 * 2.0)).abs().max().item():.6f}")
        
        print("\n✓ Homomorphic operations successful!")
        
    except RuntimeError as e:
        print(f"\n⚠ Warning: {e}")
        print("Install TenSEAL: pip install tenseal")


def main():
    """Main function to run all demos."""
    print("\n" + "="*60)
    print("FEDERATED LEARNING WITH PRIVACY-PRESERVING TECHNIQUES")
    print("="*60)
    print("\nThis demo showcases:")
    print("  1. Basic Federated Learning (FedAvg)")
    print("  2. Differential Privacy")
    print("  3. Secure Multi-Party Computation")
    print("  4. Homomorphic Encryption")
    print("  5. FL with vs without DP (Privacy-Accuracy Trade-off)")
    
    # Run demos
    try:
        demo_basic_federated_learning()
        demo_differential_privacy()
        demo_smpc()
        demo_homomorphic_encryption()
        demo_fl_with_dp()
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
