"""
Privacy-Preserving Protocols
Utilities for Differential Privacy, Secure Multi-Party Computation, and Homomorphic Encryption.
"""

import torch
from typing import Dict, Any, List, Optional
import math

try:
    import numpy as np
except ImportError:
    np = None


class DifferentialPrivacy:
    """
    Differential Privacy utilities for federated learning.
    Implements Gaussian mechanism for DP-SGD.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        noise_multiplier: Optional[float] = None
    ):
        """
        Initialize differential privacy mechanism.
        
        Args:
            epsilon: Privacy budget (epsilon)
            delta: Failure probability (delta)
            clip_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise multiplier (if None, computed from epsilon/delta)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        
        # Compute noise multiplier if not provided
        if noise_multiplier is None:
            # Approximate: sigma = sqrt(2 * ln(1.25/delta)) / epsilon
            self.noise_multiplier = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        else:
            self.noise_multiplier = noise_multiplier
        
    def clip_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        clip_norm: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Clip gradients to bound sensitivity.
        
        Args:
            gradients: Model gradients
            clip_norm: Maximum gradient norm (if None, use self.clip_norm)
            
        Returns:
            Clipped gradients
        """
        if clip_norm is None:
            clip_norm = self.clip_norm
        
        # Compute total norm
        total_norm = 0.0
        for grad in gradients.values():
            param_norm = grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip if norm exceeds threshold
        clip_coef = min(1.0, clip_norm / (total_norm + 1e-6))
        
        clipped_gradients = {}
        for name, grad in gradients.items():
            clipped_gradients[name] = grad * clip_coef
        
        return clipped_gradients
    
    def add_noise(
        self,
        gradients: Dict[str, torch.Tensor],
        noise_multiplier: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Add calibrated Gaussian noise to gradients for DP.
        
        Args:
            gradients: Model gradients (should be clipped first)
            noise_multiplier: Noise multiplier (if None, use self.noise_multiplier)
            
        Returns:
            Noisy gradients
        """
        if noise_multiplier is None:
            noise_multiplier = self.noise_multiplier
        
        noisy_gradients = {}
        for name, grad in gradients.items():
            # Add Gaussian noise: N(0, (sigma * clip_norm)^2)
            noise_scale = noise_multiplier * self.clip_norm
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=grad.shape,
                device=grad.device,
                dtype=grad.dtype
            )
            noisy_gradients[name] = grad + noise
        
        return noisy_gradients
    
    def apply_dp(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply full DP mechanism: clip then add noise.
        
        Args:
            gradients: Model gradients
            
        Returns:
            DP-protected gradients
        """
        clipped = self.clip_gradients(gradients)
        noisy = self.add_noise(clipped)
        return noisy


class SecureMultiPartyComputation:
    """
    Secure Multi-Party Computation utilities.
    Implements simple secret sharing for demonstration.
    """
    
    def __init__(self, num_parties: int = 3, field_size: int = 2**31 - 1):
        """
        Initialize SMPC protocol.
        
        Args:
            num_parties: Number of participating parties
            field_size: Size of finite field for secret sharing
        """
        self.num_parties = num_parties
        self.field_size = field_size
    
    def secret_share(
        self,
        value: torch.Tensor,
        num_shares: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Create secret shares of a value using additive secret sharing.
        
        Args:
            value: Value to share
            num_shares: Number of shares (if None, use self.num_parties)
            
        Returns:
            List of secret shares
        """
        if num_shares is None:
            num_shares = self.num_parties
        
        # Generate random shares that sum to the original value
        shares = []
        cumulative = torch.zeros_like(value)
        
        # Generate n-1 random shares
        for i in range(num_shares - 1):
            share = torch.randint(
                low=-self.field_size // 2,
                high=self.field_size // 2,
                size=value.shape,
                dtype=value.dtype
            ).float()
            shares.append(share)
            cumulative += share
        
        # Last share is computed to make sum equal to original value
        last_share = value - cumulative
        shares.append(last_share)
        
        return shares
    
    def reconstruct(
        self,
        shares: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Reconstruct value from secret shares.
        
        Args:
            shares: Secret shares
            **kwargs: Additional reconstruction parameters
            
        Returns:
            Reconstructed value
        """
        if not shares:
            raise ValueError("No shares provided")
        
        # Sum all shares to reconstruct
        reconstructed = shares[0].clone()
        for share in shares[1:]:
            reconstructed += share
        
        return reconstructed
    
    def secure_aggregation(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Perform secure aggregation using secret sharing.
        
        Args:
            client_updates: Updates from multiple clients
            **kwargs: Additional aggregation parameters
            
        Returns:
            Securely aggregated result
        """
        if not client_updates:
            return {}
        
        # Get all parameter names
        param_names = client_updates[0].keys()
        
        # Secret share each client's update
        all_shares = []
        for update in client_updates:
            client_shares = {}
            for name, param in update.items():
                shares = self.secret_share(param)
                client_shares[name] = shares
            all_shares.append(client_shares)
        
        # Aggregate shares (sum corresponding shares)
        aggregated = {}
        for name in param_names:
            # Sum first share from each client
            aggregated_param = all_shares[0][name][0].clone()
            for client_shares in all_shares[1:]:
                aggregated_param += client_shares[name][0]
            
            # Reconstruct from aggregated shares
            aggregated_shares = [aggregated_param]
            for i in range(1, len(all_shares[0][name])):
                share_sum = all_shares[0][name][i].clone()
                for client_shares in all_shares[1:]:
                    share_sum += client_shares[name][i]
                aggregated_shares.append(share_sum)
            
            aggregated[name] = self.reconstruct(aggregated_shares)
        
        return aggregated


class HomomorphicEncryption:
    """
    Homomorphic Encryption utilities for encrypted computation.
    Wrapper around TenSEAL (Microsoft SEAL) for CKKS scheme.
    """
    
    def __init__(
        self,
        scheme: str = "CKKS",
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60]
    ):
        """
        Initialize homomorphic encryption scheme.
        
        Args:
            scheme: Encryption scheme ("CKKS" supported)
            poly_modulus_degree: Polynomial modulus degree
            coeff_mod_bit_sizes: Coefficient modulus bit sizes
        """
        self.scheme = scheme
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.context = None
        self.public_key = None
        self.secret_key = None
        self._initialize_context()
    
    def _initialize_context(self):
        """Initialize TenSEAL context and keys."""
        try:
            import tenseal as ts
            
            # Create context
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
            )
            self.context.generate_galois_keys()
            self.context.global_scale = 2**40
            
            # Generate keys
            self.secret_key = self.context.secret_key()
            self.public_key = self.context.public_key()
            
        except ImportError:
            print("Warning: TenSEAL not installed. HE features will be limited.")
            self.context = None
    
    def encrypt(
        self,
        plaintext: torch.Tensor,
        **kwargs
    ) -> Any:
        """
        Encrypt plaintext tensor.
        
        Args:
            plaintext: Plaintext tensor to encrypt
            **kwargs: Additional encryption parameters
            
        Returns:
            Encrypted tensor (TenSEAL CKKSTensor)
        """
        if self.context is None:
            raise RuntimeError("TenSEAL not available. Install with: pip install tenseal")
        
        import tenseal as ts
        
        # Convert to numpy and flatten
        plaintext_np = plaintext.detach().cpu().numpy().flatten()
        
        # Encrypt
        encrypted = ts.ckks_vector(self.context, plaintext_np.tolist())
        
        return encrypted
    
    def decrypt(
        self,
        ciphertext: Any,
        shape: Optional[tuple] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Decrypt ciphertext tensor.
        
        Args:
            ciphertext: Encrypted tensor (TenSEAL CKKSTensor)
            shape: Original shape to reshape to
            **kwargs: Additional decryption parameters
            
        Returns:
            Decrypted plaintext tensor
        """
        if self.context is None:
            raise RuntimeError("TenSEAL not available")
        
        # Decrypt
        decrypted_list = ciphertext.decrypt()
        
        # Convert to tensor
        if np is not None:
            decrypted_np = np.array(decrypted_list)
            if shape is not None:
                decrypted_np = decrypted_np.reshape(shape)
            return torch.FloatTensor(decrypted_np)
        else:
            decrypted_tensor = torch.tensor(decrypted_list, dtype=torch.float32)
            if shape is not None:
                decrypted_tensor = decrypted_tensor.reshape(shape)
            return decrypted_tensor
    
    def encrypted_add(
        self,
        ciphertext1: Any,
        ciphertext2: Any,
        **kwargs
    ) -> Any:
        """
        Perform encrypted addition.
        
        Args:
            ciphertext1: First encrypted tensor
            ciphertext2: Second encrypted tensor
            **kwargs: Additional operation parameters
            
        Returns:
            Encrypted sum
        """
        if self.context is None:
            raise RuntimeError("TenSEAL not available")
        
        return ciphertext1 + ciphertext2
    
    def encrypted_multiply(
        self,
        ciphertext1: Any,
        scalar: float,
        **kwargs
    ) -> Any:
        """
        Perform encrypted scalar multiplication.
        
        Args:
            ciphertext1: Encrypted tensor
            scalar: Scalar multiplier
            **kwargs: Additional operation parameters
            
        Returns:
            Encrypted product
        """
        if self.context is None:
            raise RuntimeError("TenSEAL not available")
        
        return ciphertext1 * scalar
