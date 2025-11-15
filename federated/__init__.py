"""
Federated learning components.
"""

from .client import FederatedClient
from .server import FederatedServer
from .protocols import DifferentialPrivacy, SecureMultiPartyComputation, HomomorphicEncryption

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "DifferentialPrivacy",
    "SecureMultiPartyComputation",
    "HomomorphicEncryption",
]

