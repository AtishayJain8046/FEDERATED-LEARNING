"""
Federated Learning Server
Coordinates training across multiple clients.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import random
import copy


class FederatedServer:
    """
    Server for federated learning.
    Aggregates client updates and distributes global model.
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        aggregation_method: str = "fedavg",
        device: str = "cpu"
    ):
        """
        Initialize federated server.
        
        Args:
            global_model: Global model instance
            aggregation_method: Aggregation method ('fedavg' or 'weighted_avg')
            device: Device to run on ('cpu' or 'cuda')
        """
        self.global_model = global_model.to(device)
        self.device = device
        self.aggregation_method = aggregation_method
        self.clients = []
        self.round_history = []
        
    def register_client(self, client) -> None:
        """
        Register a client with the server.
        
        Args:
            client: FederatedClient instance
        """
        self.clients.append(client)
    
    def aggregate_updates(
        self,
        client_updates: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates from multiple clients using FedAvg.
        
        Args:
            client_updates: List of dicts with 'parameters' and 'num_samples'
            **kwargs: Additional aggregation parameters
            
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            return self.global_model.state_dict()
        
        # Get total number of samples
        total_samples = sum(update.get("num_samples", 1) for update in client_updates)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        for key in client_updates[0]["parameters"].keys():
            aggregated_params[key] = torch.zeros_like(
                client_updates[0]["parameters"][key]
            )
        
        # Weighted average aggregation (FedAvg)
        for update in client_updates:
            params = update["parameters"]
            weight = update.get("num_samples", 1) / total_samples
            
            for key in aggregated_params.keys():
                aggregated_params[key] += weight * params[key]
        
        return aggregated_params
    
    def broadcast_model(self, parameters: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        Broadcast global model parameters to all clients.
        
        Args:
            parameters: Model parameters to distribute (if None, use current global model)
        """
        if parameters is None:
            parameters = self.global_model.state_dict()
        
        for client in self.clients:
            client.set_model_parameters(parameters)
    
    def select_clients(
        self,
        num_clients: int,
        strategy: str = "random",
        **kwargs
    ) -> List[int]:
        """
        Select subset of clients for current round.
        
        Args:
            num_clients: Number of clients to select
            strategy: Selection strategy ('random' or 'all')
            **kwargs: Additional selection criteria
            
        Returns:
            List of selected client indices
        """
        if strategy == "all" or num_clients >= len(self.clients):
            return list(range(len(self.clients)))
        
        if strategy == "random":
            return random.sample(range(len(self.clients)), num_clients)
        
        return list(range(min(num_clients, len(self.clients))))
    
    def run_round(
        self,
        num_clients: int = 3,
        local_epochs: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute one round of federated learning.
        
        Args:
            num_clients: Number of clients to participate
            local_epochs: Number of local training epochs per client
            **kwargs: Round-specific parameters
            
        Returns:
            Round metrics and statistics
        """
        # Select clients
        selected_indices = self.select_clients(num_clients)
        selected_clients = [self.clients[i] for i in selected_indices]
        
        # Broadcast global model
        self.broadcast_model()
        
        # Collect updates from selected clients
        client_updates = []
        client_metrics = []
        
        for client in selected_clients:
            # Local training
            metrics = client.train(epochs=local_epochs)
            client_metrics.append(metrics)
            
            # Get updated parameters
            params = client.get_model_parameters()
            num_samples = len(client.data_loader.dataset) if client.data_loader else 1
            
            client_updates.append({
                "parameters": params,
                "num_samples": num_samples,
                "client_id": client.client_id
            })
        
        # Aggregate updates
        aggregated_params = self.aggregate_updates(client_updates)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_params)
        
        # Calculate average metrics
        avg_loss = sum(m["loss"] for m in client_metrics) / len(client_metrics)
        
        round_info = {
            "round": len(self.round_history) + 1,
            "num_clients": len(selected_clients),
            "avg_loss": avg_loss,
            "client_metrics": client_metrics
        }
        
        self.round_history.append(round_info)
        
        return round_info
    
    def evaluate_global_model(self, test_data: tuple) -> Dict[str, float]:
        """
        Evaluate global model on test data.
        
        Args:
            test_data: Tuple of (X, y) test data
            
        Returns:
            Dictionary with accuracy and loss
        """
        self.global_model.eval()
        X_test, y_test = test_data
        dataset = torch.utils.data.TensorDataset(X_test, y_test)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(data_loader)
        
        return {"accuracy": accuracy, "loss": avg_loss}
