"""
Normalizing flow models for astronomical data analysis using nflows library.
This module replaces the custom implementation in normflows.py with tested, robust implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.nn.nets import ResidualNet

class Flow5D(nn.Module):
    """
    5D normalizing flow for complete Galactic analysis.
    Jointly models [age, [Fe/H], [Mg/Fe], sqrt(Jz), Lz]
    
    Parameters:
    -----------
    n_transforms : int
        Number of transforms in the flow
    hidden_dims : list
        Dimensions of hidden layers in transform networks
    num_bins : int
        Number of bins in spline transforms
    tail_bound : float
        Bound on spline tails
    """
    
    def __init__(self, n_transforms=16, hidden_dims=None, num_bins=16, tail_bound=5.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        # Base distribution (5D standard normal)
        base_dist = StandardNormal(shape=[5])
        
        # Build a sequence of transforms
        transforms = []
        for i in range(n_transforms):
            # Add alternating permutation and autoregressive transforms
            transforms.append(ReversePermutation(features=5))
            
            # Use masked autoregressive transform with rational quadratic splines
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=5,
                    hidden_features=hidden_dims[0],
                    context_features=None,
                    num_bins=num_bins,
                    tails='linear',
                    tail_bound=tail_bound,
                    num_blocks=2,
                    use_residual_blocks=True,
                    random_mask=False,
                    activation=torch.nn.ReLU(),
                    dropout_probability=0.0,
                    use_batch_norm=False
                )
            )
        
        # Create the flow model
        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=base_dist
        )
    
    def log_prob(self, x):
        """Compute log probability of x"""
        return self.flow.log_prob(x)
    
    def sample(self, n):
        """Sample n points from the flow"""
        return self.flow.sample(n)


class Flow3D(nn.Module):
    """
    3D normalizing flow for analyzing [age, [Fe/H], [Mg/Fe]] jointly.
    
    Parameters:
    -----------
    n_transforms : int
        Number of transforms in the flow
    hidden_dims : list
        Dimensions of hidden layers in transform networks
    num_bins : int
        Number of bins in spline transforms
    tail_bound : float
        Bound on spline tails
    """
    
    def __init__(self, n_transforms=16, hidden_dims=None, num_bins=16, tail_bound=5.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        # Base distribution (3D standard normal)
        base_dist = StandardNormal(shape=[3])
        
        # Build a sequence of transforms
        transforms = []
        for i in range(n_transforms):
            # Add alternating permutation and coupling transforms
            transforms.append(ReversePermutation(features=3))
            
            # Use piecewise rational quadratic coupling transform
            transforms.append(
                PiecewiseRationalQuadraticCouplingTransform(
                    mask=create_alternating_binary_mask(features=3, even=(i % 2 == 0)),
                    transform_net_create_fn=lambda in_features, out_features: 
                        ResidualNet(
                            in_features=in_features,
                            out_features=out_features,
                            hidden_features=hidden_dims[0],
                            num_blocks=2,
                            activation=F.relu,
                            dropout_probability=0.0,
                            use_batch_norm=False
                        ),
                    num_bins=num_bins,
                    tails='linear',
                    tail_bound=tail_bound,
                    apply_unconditional_transform=False
                )
            )
        
        # Create the flow model
        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=base_dist
        )
    
    def log_prob(self, x):
        """Compute log probability of x"""
        return self.flow.log_prob(x)
    
    def sample(self, n):
        """Sample n points from the flow"""
        return self.flow.sample(n)


def create_alternating_binary_mask(features, even=True):
    """
    Creates a binary mask of a given dimension which alternates between 0 and 1.
    
    Parameters:
    -----------
    features : int
        Number of features
    even : bool
        If True, even bits are 1, odd bits are 0, and vice versa
        
    Returns:
    --------
    torch.Tensor
        Binary mask
    """
    mask = torch.zeros(features)
    start = 0 if even else 1
    mask[start::2] = 1
    return mask