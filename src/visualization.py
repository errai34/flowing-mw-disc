#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization functions for normalizing flow analysis of Galactic evolution.
Simplified for minimal design and direct plotting.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_bin_chemical_distribution(
    flow,
    scaler,
    bin_name,
    save_dir=None,
    age_range=(0, 20),
    feh_range=(-1.0, 0.5),
    mgfe_range=(0.0, 0.4),
    n_samples=10000,
):
    """
    Simple scatter plots of Age vs. [Fe/H] and Age vs. [Mg/Fe].

    Parameters:
    -----------
    flow : Flow5D
        Trained normalizing flow model
    scaler : StandardScaler
        Scaler used to normalize the data
    bin_name : str
        Name of the radial bin
    save_dir : str
        Directory to save figure
    age_range, feh_range, mgfe_range : tuple
        (min, max) for each parameter range
    n_samples : int
        Number of samples to draw
    """
    device = next(flow.parameters()).device
    flow.eval()

    # Sample from the flow
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy()

    # Inverse transform to get original scale
    samples_original = scaler.inverse_transform(samples)

    # Extract parameters
    log_ages = samples_original[:, 0]  # First dimension is log(age)
    fehs = samples_original[:, 1]  # Second dimension is [Fe/H]
    mgfes = samples_original[:, 2]  # Third dimension is [Mg/Fe]

    # Convert log age to linear age for plotting
    ages = 10**log_ages

    # Filter points within the specified ranges
    mask = (
        (ages >= age_range[0])
        & (ages <= age_range[1])
        & (fehs >= feh_range[0])
        & (fehs <= feh_range[1])
        & (mgfes >= mgfe_range[0])
        & (mgfes <= mgfe_range[1])
    )

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Age vs [Fe/H]
    ax1.scatter(ages[mask], fehs[mask], s=3, alpha=0.6, color="blue")
    ax1.set_xlabel("Age (Gyr)")
    ax1.set_ylabel("[Fe/H]")
    ax1.set_title(f"{bin_name}: Age vs. [Fe/H]")
    ax1.set_xlim(age_range[1], age_range[0])  # Reversed to show oldest stars on left
    ax1.set_ylim(feh_range)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Age vs [Mg/Fe]
    ax2.scatter(ages[mask], mgfes[mask], s=3, alpha=0.6, color="green")
    ax2.set_xlabel("Age (Gyr)")
    ax2.set_ylabel("[Mg/Fe]")
    ax2.set_title(f"{bin_name}: Age vs. [Mg/Fe]")
    ax2.set_xlim(age_range[1], age_range[0])  # Reversed to show oldest stars on left
    ax2.set_ylim(mgfe_range)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{bin_name}_chemical_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


def compute_2d_density(flow, scaler, grid1, grid2, idx1, idx2, fixed_values, device):
    """
    Compute 2D density for a pair of parameters.

    Parameters:
    -----------
    flow : Flow5D
        Trained normalizing flow model
    scaler : StandardScaler
        Scaler used to normalize the data
    grid1, grid2 : np.ndarray
        Grids for the two parameters
    idx1, idx2 : int
        Indices of the two parameters
    fixed_values : dict
        Fixed values for other parameters
    device : torch.device
        Device to use for computation

    Returns:
    --------
    np.ndarray
        2D density array
    """
    # Create parameter mapping
    param_names = ["age", "feh", "mgfe", "jz", "lz"]
    param_idx = {name: i for i, name in enumerate(param_names)}

    # Create meshgrid for the two parameters
    X, Y = np.meshgrid(grid1, grid2)

    # Reshape to get all combinations
    points = np.zeros((len(grid1) * len(grid2), 5))

    # Set fixed values for all dimensions first
    for name, value in fixed_values.items():
        idx = param_idx[name]
        points[:, idx] = value

    # Set the grid values for the two dimensions we're plotting
    points[:, idx1] = X.flatten()
    points[:, idx2] = Y.flatten()

    # Scale points using the provided scaler
    scaled_points = scaler.transform(points)
    tensor_points = torch.tensor(scaled_points, dtype=torch.float32, device=device)

    # Compute log probability
    with torch.no_grad():
        log_probs = flow.log_prob(tensor_points).cpu().numpy()

    # Convert to probability density (use exp but handle numerical issues)
    # Subtract max for numerical stability
    log_probs_shifted = log_probs - np.max(log_probs)
    probs = np.exp(log_probs_shifted)

    # Reshape to match the original 2D grid dimensions
    # Note: reshaping with the correct dimensions - Y corresponds to rows (grid2)
    density = probs.reshape(len(grid2), len(grid1))

    # Apply a small amount of smoothing for better visualization
    from scipy.ndimage import gaussian_filter

    density = gaussian_filter(density, sigma=0.7)

    # Normalize density to [0, 1] range for visualization
    if np.max(density) - np.min(density) > 0:
        density = (density - np.min(density)) / (np.max(density) - np.min(density))

    return density


def plot_radial_bin_comparison(flows_dict, scalers_dict, save_path=None):
    """
    Create a master figure showing age-[Fe/H] distributions across different radial bins.
    Uses scatter plots instead of density plots.

    Parameters:
    -----------
    flows_dict : dict
        Dictionary mapping bin names to flow models
    scalers_dict : dict
        Dictionary mapping bin names to scalers
    save_path : str
        Path to save the figure
    """
    # Set up figure
    n_bins = len(flows_dict)
    fig, axes = plt.subplots(
        1, n_bins, figsize=(5 * n_bins, 5), sharex=True, sharey=True
    )
    if n_bins == 1:
        axes = [axes]

    # Plot each bin
    for i, (bin_name, flow) in enumerate(flows_dict.items()):
        device = next(flow.parameters()).device
        flow.eval()
        scaler = scalers_dict[bin_name]

        # Sample from the flow
        with torch.no_grad():
            samples = flow.sample(3000).cpu().numpy()

        # Inverse transform to get original scale
        samples_original = scaler.inverse_transform(samples)

        # Extract age and [Fe/H]
        log_ages = samples_original[:, 0]
        fehs = samples_original[:, 1]

        # Convert log age to linear age
        ages = 10**log_ages

        # Filter points within ranges
        mask = (ages >= 0) & (ages <= 20) & (fehs >= -1.0) & (fehs <= 0.5)

        # Plot
        axes[i].scatter(ages[mask], fehs[mask], s=3, alpha=0.6, color="blue")
        axes[i].set_title(f"Bin: {bin_name}")
        axes[i].set_xlabel("Age (Gyr)")
        if i == 0:
            axes[i].set_ylabel("[Fe/H]")

        # Set axis limits - age from 20 to 0 (reversed)
        axes[i].set_xlim(20, 0)
        axes[i].set_ylim(-1.0, 0.5)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Age-Metallicity Relation Across Radial Bins", fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_gradient_analysis(gradient_results, bin_name, save_path=None):
    """
    Plot gradient analysis results.

    Parameters:
    -----------
    gradient_results : dict
        Results from gradient analysis
    bin_name : str
        Name of the radial bin
    save_path : str
        Path to save the figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Define colors for different ages
    age_values = list(gradient_results.keys())
    age_values.sort()
    colors = plt.cm.viridis(np.linspace(0, 1, len(age_values)))

    # Plot log probabilities
    for i, age in enumerate(age_values):
        result = gradient_results[age]
        ax1.plot(
            result["feh_grid"],
            result["log_probs"],
            label=f"Age = {age} Gyr",
            color=colors[i],
        )

    ax1.set_title(f"{bin_name}: Log Probability vs. [Fe/H]")
    ax1.set_xlabel("[Fe/H]")
    ax1.set_ylabel("Log Probability")
    ax1.legend()

    # Plot gradients
    for i, age in enumerate(age_values):
        result = gradient_results[age]
        ax2.plot(
            result["feh_grid"],
            result["gradients"],
            label=f"Age = {age} Gyr",
            color=colors[i],
        )

        # Mark zero crossings
        stable_points = find_stable_points(result["gradients"], result["feh_grid"])
        for point in stable_points:
            ax2.axvline(x=point, color=colors[i], linestyle="--", alpha=0.3)

    ax2.set_title(f"{bin_name}: Gradient of Log Probability w.r.t. [Fe/H]")
    ax2.set_xlabel("[Fe/H]")
    ax2.set_ylabel("Gradient")
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_master_gradient_plot(gradient_results_dict, save_path=None):
    """
    Create a master plot showing gradient analysis across all radial bins.

    Parameters:
    -----------
    gradient_results_dict : dict
        Dictionary mapping bin names to gradient results
    save_path : str
        Path to save the figure
    """
    # Get all age values and bin names
    all_ages = set()
    bin_names = list(gradient_results_dict.keys())

    for bin_results in gradient_results_dict.values():
        for age in bin_results.keys():
            all_ages.add(age)

    all_ages = sorted(list(all_ages))

    # Create figure
    n_ages = len(all_ages)
    n_bins = len(bin_names)

    fig, axes = plt.subplots(
        n_ages, n_bins, figsize=(4 * n_bins, 3 * n_ages), sharex=True
    )

    # Plot gradients for each age and bin
    for i, age in enumerate(all_ages):
        for j, bin_name in enumerate(bin_names):
            ax = axes[i, j] if n_ages > 1 else axes[j]

            if age in gradient_results_dict[bin_name]:
                result = gradient_results_dict[bin_name][age]
                ax.plot(result["feh_grid"], result["gradients"], "b-")
                ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

                # Mark stable points
                stable_points = find_stable_points(
                    result["gradients"], result["feh_grid"]
                )
                for point in stable_points:
                    ax.axvline(x=point, color="red", linestyle="--", alpha=0.5)

            # Add labels
            if i == 0:
                ax.set_title(bin_name)
            if i == n_ages - 1:
                ax.set_xlabel("[Fe/H]")
            if j == 0:
                ax.set_ylabel(f"Age = {age} Gyr\nGradient")

    plt.suptitle(
        "Rate of Change in Probability Density with Respect to [Fe/H]", fontsize=16
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


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


def plot_age_metallicity_scatter(
    flow,
    scaler,
    n_samples=10000,
    save_path=None,
    age_range=(0, 20),
    feh_range=(-1.5, 0.5),
    figsize=(10, 8),
    flip_age_axis=True,
):
    """
    Create a scatter plot of Age vs. [Fe/H] using samples from the flow model.

    Parameters:
    -----------
    flow : Flow5D
        Trained normalizing flow model
    scaler : StandardScaler
        Scaler used to normalize the data
    n_samples : int
        Number of samples to draw
    save_path : str
        Path to save the figure
    age_range : tuple
        (min, max) for age range in plot
    feh_range : tuple
        (min, max) for [Fe/H] range in plot
    figsize : tuple
        Figure size
    flip_age_axis : bool
        If True, plot age from high to low (oldest to youngest)

    Returns:
    --------
    tuple
        (fig, ax) - Figure and axis objects
    """
    # Set device and evaluation mode
    device = next(flow.parameters()).device
    flow.eval()

    # Sample from the flow
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy()

    # Inverse transform to get original scale
    samples_original = scaler.inverse_transform(samples)

    # Extract age and [Fe/H]
    log_ages = samples_original[:, 0]  # First dimension is log(age)
    fehs = samples_original[:, 1]  # Second dimension is [Fe/H]

    # Convert log age to linear age
    ages = 10**log_ages

    # Create scatter plot
    fig, ax = plt.subplots(figsize=figsize)

    # Filter points within the specified ranges
    mask = (
        (ages >= age_range[0])
        & (ages <= age_range[1])
        & (fehs >= feh_range[0])
        & (fehs <= feh_range[1])
    )

    # Simple scatter plot
    ax.scatter(
        ages[mask],
        fehs[mask],
        alpha=0.6,
        s=3,
        color="blue",
    )

    # Set labels and title
    ax.set_ylabel("[Fe/H]")
    ax.set_xlabel("Age (Gyr)")
    ax.set_title("Age-Metallicity Relation from Flow Model Samples")

    # Set axis ranges
    ax.set_ylim(feh_range)
    ax.set_xlim(age_range[1], age_range[0])  # Reversed to show oldest stars on left

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig, ax


def plot_multiple_bin_age_metallicity(
    flows_dict,
    scalers_dict,
    n_samples=5000,
    save_path=None,
    age_range=(0, 20),
    feh_range=(-1.5, 0.5),
    flip_age_axis=True,
):
    """
    Create scatter plots of Age vs. [Fe/H] for multiple radial bins.

    Parameters:
    -----------
    flows_dict : dict
        Dictionary mapping bin names to flow models
    scalers_dict : dict
        Dictionary mapping bin names to scalers
    n_samples : int
        Number of samples per bin
    save_path : str
        Path to save the figure
    age_range : tuple
        (min, max) for age range in plot
    feh_range : tuple
        (min, max) for [Fe/H] range in plot
    flip_age_axis : bool
        If True, plot age from high to low (oldest to youngest)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Set up figure
    n_bins = len(flows_dict)
    fig, axes = plt.subplots(
        1, n_bins, figsize=(5 * n_bins, 5), sharey=True, sharex=True
    )

    # Handle the case of a single bin
    if n_bins == 1:
        axes = [axes]

    # Plot each bin
    for i, (bin_name, flow) in enumerate(flows_dict.items()):
        device = next(flow.parameters()).device
        flow.eval()
        scaler = scalers_dict[bin_name]

        # Sample from the flow
        with torch.no_grad():
            samples = flow.sample(n_samples).cpu().numpy()

        # Inverse transform to get original scale
        samples_original = scaler.inverse_transform(samples)

        # Extract age and [Fe/H]
        log_ages = samples_original[:, 0]
        fehs = samples_original[:, 1]

        # Convert log age to linear age
        ages = 10**log_ages

        # Filter points within the specified ranges
        mask = (
            (ages >= age_range[0])
            & (ages <= age_range[1])
            & (fehs >= feh_range[0])
            & (fehs <= feh_range[1])
        )

        # Create scatter plot
        axes[i].scatter(
            ages[mask],
            fehs[mask],
            alpha=0.6,
            s=3,
            color="blue",
        )

        # Set title and labels
        axes[i].set_title(f"Bin: {bin_name}")
        if i == 0:
            axes[i].set_ylabel("[Fe/H] (dex)")
        axes[i].set_xlabel("Age (Gyr)")

        # Set axis ranges
        axes[i].set_ylim(feh_range)
        axes[i].set_xlim(
            age_range[1], age_range[0]
        )  # Reversed to show oldest stars on left

        # Add grid
        axes[i].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Age-Metallicity Relation Across Radial Bins", fontsize=16)
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig
