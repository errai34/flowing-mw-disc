#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradient analysis functions for normalizing flow models.
Analyzes how probability density changes with respect to parameters.
"""

import numpy as np
import torch


def compute_density_gradients(flow, data_grid, param_idx=1, requires_grad=True):
    """
    Compute gradients of log-probability w.r.t. specified parameter.

    Parameters:
    -----------
    flow : Flow5D
        Trained normalizing flow model
    data_grid : torch.Tensor
        Grid of points to evaluate gradients (N, D)
    param_idx : int
        Index of parameter to compute gradient for (1=[Fe/H])
    requires_grad : bool
        Whether to require gradients for the input data

    Returns:
    --------
    torch.Tensor
        Gradients at each point in data_grid (N,)
    """
    # Ensure data requires gradients
    if requires_grad:
        data_grid = data_grid.detach().clone().requires_grad_(True)

    # Compute log probability
    log_prob = flow.log_prob(data_grid)

    # Compute gradients
    gradients = torch.autograd.grad(log_prob.sum(), data_grid, create_graph=True)[0]

    # Extract gradients w.r.t. specific parameter
    param_gradients = gradients[:, param_idx]

    return param_gradients


def analyze_metallicity_gradients(
    flow, scaler, age_values, feh_range, mgfe=0.0, jz=0.0, lz=1000.0
):
    """
    Analyze how log-probability gradients w.r.t. [Fe/H] change across different ages.

    Parameters:
    -----------
    flow : Flow5D
        Trained normalizing flow model
    scaler : StandardScaler
        Scaler used to normalize the data
    age_values : list
        List of age values to evaluate
    feh_range : tuple
        (min, max, steps) for [Fe/H] range
    mgfe, jz, lz : float
        Fixed values for other parameters

    Returns:
    --------
    dict
        Dictionary containing gradient analysis results
    """
    device = next(flow.parameters()).device
    flow.eval()

    # Create [Fe/H] grid
    feh_min, feh_max, feh_steps = feh_range
    feh_grid = np.linspace(feh_min, feh_max, feh_steps)

    results = {}

    for age in age_values:
        # Create grid for this age
        grid_points = []
        for feh in feh_grid:
            # Create data point [age, feh, mgfe, jz, lz]
            point = np.array([[np.log10(age), feh, mgfe, jz, lz]])
            grid_points.append(point)

        grid_points = np.vstack(grid_points)

        # Scale the points
        scaled_points = scaler.transform(grid_points)
        scaled_points_tensor = torch.tensor(
            scaled_points, dtype=torch.float32, device=device
        )

        # Compute gradients
        with torch.no_grad():
            log_probs = flow.log_prob(scaled_points_tensor).cpu().numpy()

        gradients = (
            compute_density_gradients(flow, scaled_points_tensor, param_idx=1)
            .detach()
            .cpu()
            .numpy()
        )

        results[age] = {
            "feh_grid": feh_grid,
            "log_probs": log_probs,
            "gradients": gradients,
        }

    return results


def find_stable_points(gradients, feh_grid):
    """
    Find stable points (zero crossings of gradients) in the metallicity distribution.

    Parameters:
    -----------
    gradients : np.ndarray
        Gradients w.r.t. [Fe/H]
    feh_grid : np.ndarray
        Grid of [Fe/H] values

    Returns:
    --------
    list
        List of [Fe/H] values where gradient is zero (stable points)
    """
    stable_points = []

    # Find zero crossings
    for i in range(len(gradients) - 1):
        if gradients[i] * gradients[i + 1] <= 0:
            # Linear interpolation to find zero crossing
            t = -gradients[i] / (gradients[i + 1] - gradients[i] + 1e-10)
            zero_point = feh_grid[i] + t * (feh_grid[i + 1] - feh_grid[i])
            stable_points.append(zero_point)

    return stable_points
