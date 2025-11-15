"""
Synthetic dataset generator for federated learning demo.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
from sklearn.datasets import make_classification, make_blobs

try:
    from torchvision import datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available. MNIST dataset will not be available.")


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for federated learning.
    """
    
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        Initialize dataset.
        
        Args:
            X: Features tensor
            y: Labels tensor
        """
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 784,
    n_classes: int = 10,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic classification data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        random_state: Random seed
        
    Returns:
        Tuple of (features, labels) tensors
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        random_state=random_state
    )
    
    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    return torch.FloatTensor(X), torch.LongTensor(y)


def split_data_among_clients(
    X: torch.Tensor,
    y: torch.Tensor,
    num_clients: int,
    iid: bool = True,
    alpha: float = 0.5
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split data among multiple clients (IID or non-IID).
    
    Args:
        X: Features tensor
        y: Labels tensor
        num_clients: Number of clients
        iid: If True, use IID distribution; if False, use Dirichlet distribution
        alpha: Dirichlet distribution parameter (lower = more non-IID)
        
    Returns:
        List of (X_client, y_client) tuples for each client
    """
    n_samples = len(y)
    samples_per_client = n_samples // num_clients
    
    if iid:
        # IID: Random shuffle and split
        indices = torch.randperm(n_samples)
        client_data = []
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else n_samples
            client_indices = indices[start_idx:end_idx]
            client_data.append((X[client_indices], y[client_indices]))
        return client_data
    else:
        # Non-IID: Use Dirichlet distribution
        # Assign samples to clients based on label distribution
        n_classes = len(torch.unique(y))
        client_data = [([], []) for _ in range(num_clients)]
        
        for class_idx in range(n_classes):
            class_indices = (y == class_idx).nonzero(as_tuple=True)[0]
            n_class_samples = len(class_indices)
            
            # Generate Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = proportions / proportions.sum()
            client_counts = (proportions * n_class_samples).astype(int)
            client_counts[-1] += n_class_samples - client_counts.sum()  # Fix rounding
            
            # Shuffle and assign
            shuffled_indices = class_indices[torch.randperm(n_class_samples)]
            start_idx = 0
            for client_idx, count in enumerate(client_counts):
                end_idx = start_idx + count
                client_data[client_idx][0].append(X[shuffled_indices[start_idx:end_idx]])
                client_data[client_idx][1].append(y[shuffled_indices[start_idx:end_idx]])
                start_idx = end_idx
        
        # Concatenate and return
        return [
            (torch.cat(X_list), torch.cat(y_list))
            for X_list, y_list in client_data
        ]


def load_mnist(
    data_dir: str = './data/mnist',
    train: bool = True,
    download: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load MNIST dataset.
    
    Args:
        data_dir: Directory to store/load MNIST data
        train: If True, load training set; if False, load test set
        download: If True, download MNIST if not present
        
    Returns:
        Tuple of (features, labels) tensors
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required for MNIST. Install with: pip install torchvision")
    
    # Transform: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load dataset
    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    # Convert to tensors
    X_list = []
    y_list = []
    
    for i in range(len(dataset)):
        img, label = dataset[i]
        # Flatten image (28x28 -> 784)
        X_list.append(img.view(-1))
        y_list.append(label)
    
    X = torch.stack(X_list)
    y = torch.stack(y_list)
    
    return X, y


def load_dataset(dataset_name: str = "synthetic", **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of dataset ("synthetic" or "mnist")
        **kwargs: Additional arguments for dataset loading
        
    Returns:
        Tuple of (features, labels) tensors
    """
    if dataset_name.lower() == "mnist":
        train = kwargs.get('train', True)
        return load_mnist(train=train, download=kwargs.get('download', True))
    elif dataset_name.lower() == "synthetic":
        return generate_synthetic_data(
            n_samples=kwargs.get('n_samples', 1000),
            n_features=kwargs.get('n_features', 784),
            n_classes=kwargs.get('n_classes', 10),
            random_state=kwargs.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'synthetic', 'mnist'")

