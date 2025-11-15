"""
Simple Multi-Layer Perceptron (MLP) for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    Simple MLP for classification tasks.
    """
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        """
        Initialize MLP model.
        
        Args:
            input_size: Size of input features (e.g., 784 for 28x28 images)
            hidden_size: Size of hidden layer
            num_classes: Number of output classes
        """
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class LogisticRegression(nn.Module):
    """
    Simple Logistic Regression model.
    """
    
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        """
        Initialize Logistic Regression model.
        
        Args:
            input_size: Size of input features
            num_classes: Number of output classes
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        x = x.view(x.size(0), -1)  # Flatten
        return self.linear(x)

