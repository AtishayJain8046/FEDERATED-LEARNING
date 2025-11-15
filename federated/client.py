"""
Federated Learning Client
Handles local training and communication with the server.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional
import copy


class FederatedClient:
    """
    Client for federated learning.
    Handles local model training and parameter updates.
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: Optional[tuple] = None,
        lr: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique identifier for this client
            model: Local model instance
            train_data: Tuple of (X, y) training data
            lr: Learning rate
            device: Device to run on ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.train_data = train_data
        self.train_losses = []
        
        # Create data loader if data provided
        if train_data is not None:
            X, y = train_data
            dataset = TensorDataset(X, y)
            self.data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        else:
            self.data_loader = None
        
    def train(self, epochs: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Perform local training on client data.
        
        Args:
            epochs: Number of training epochs
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics and updated model parameters
        """
        if self.data_loader is None:
            raise ValueError("No training data provided for client")
        
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / len(self.data_loader)
        
        avg_loss = total_loss / epochs
        self.train_losses.append(avg_loss)
        
        return {
            "client_id": self.client_id,
            "loss": avg_loss,
            "num_samples": len(self.data_loader.dataset)
        }
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Extract current model parameters.
        
        Returns:
            Dictionary of model parameter tensors
        """
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Update local model with received parameters.
        
        Args:
            parameters: Model parameters from server
        """
        self.model.load_state_dict(parameters)
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Get current model gradients.
        
        Returns:
            Dictionary of gradient tensors
        """
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients
    
    def evaluate(self, test_data: Optional[tuple] = None) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Tuple of (X, y) test data
            
        Returns:
            Dictionary with accuracy and loss
        """
        if test_data is None:
            return {"accuracy": 0.0, "loss": 0.0}
        
        self.model.eval()
        X_test, y_test = test_data
        dataset = TensorDataset(X_test, y_test)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(data_loader)
        
        return {"accuracy": accuracy, "loss": avg_loss}
