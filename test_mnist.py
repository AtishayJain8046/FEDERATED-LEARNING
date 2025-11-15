#!/usr/bin/env python
"""
Quick test script to verify MNIST loading and training works.
"""

import torch
from data import load_mnist, split_data_among_clients
from models import SimpleMLP
from federated import FederatedClient, FederatedServer

def test_mnist():
    print("Testing MNIST dataset loading...")
    
    try:
        # Load MNIST
        X, y = load_mnist(train=True, download=False)
        print(f"[OK] Loaded MNIST: {X.shape}, {y.shape}")
        print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  y unique: {torch.unique(y).tolist()}")
        
        # Test splitting
        client_data = split_data_among_clients(X, y, num_clients=3, iid=True)
        print(f"[OK] Split into {len(client_data)} clients")
        for i, (X_c, y_c) in enumerate(client_data):
            print(f"  Client {i}: {X_c.shape}, {y_c.shape}")
        
        # Test model creation
        model = SimpleMLP(input_size=784, hidden_size=128, num_classes=10)
        print(f"[OK] Created model: {model}")
        
        # Test client training
        server = FederatedServer(global_model=model)
        client = FederatedClient(
            client_id=0,
            model=SimpleMLP(input_size=784, hidden_size=128, num_classes=10),
            train_data=(client_data[0][0][:100], client_data[0][1][:100]),  # Use 100 samples
            lr=0.01
        )
        server.register_client(client)
        
        # Test one training round
        print("\nTesting training...")
        metrics = client.train(epochs=1)
        print(f"[OK] Training completed: {metrics}")
        
        # Test evaluation
        test_X, test_y = load_mnist(train=False, download=False)
        test_X = test_X[:100]
        test_y = test_y[:100]
        
        eval_metrics = client.evaluate((test_X, test_y))
        print(f"[OK] Evaluation: {eval_metrics}")
        
        print("\n[SUCCESS] All tests passed! MNIST is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mnist()

