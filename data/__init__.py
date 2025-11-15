"""
Dataset loaders and generators for federated learning.
"""

from .generator import SyntheticDataset, generate_synthetic_data, split_data_among_clients

__all__ = ["SyntheticDataset", "generate_synthetic_data", "split_data_among_clients"]
