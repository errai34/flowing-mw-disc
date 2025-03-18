#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply normalizing flow deconvolution to APOGEE data to recover the underlying
age-metallicity-abundance distribution for the Milky Way disc.

This script focuses on the inner disc (0-6 kpc) radial bin.
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data_handler import StellarDataHandler, prepare_data_for_radial_bins
from src.uncertainty import (
    RecognitionNetwork,
    compute_diagnostics,
    importance_weighted_elbo,
    uncertainty_aware_elbo,
)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------ Model Definitions ------


class Flow3D(nn.Module):
    """
    3D normalizing flow for analyzing [age, [Fe/H], [Mg/Fe]] jointly.
    This is a simplified version for the APOGEE analysis.
    """

    def __init__(
        self,
        n_transforms=16,
        hidden_dim=128,
        num_bins=16,
        tail_bound=5.0,
        use_residual_blocks=True,
        dropout_probability=0.0,
    ):
        super().__init__()
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.transforms.autoregressive import (
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        )
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import ReversePermutation

        # Base distribution (3D standard normal)
        base_dist = StandardNormal(shape=[3])

        # Build a sequence of transforms
        transforms = []
        for i in range(n_transforms):
            # Add alternating permutation and autoregressive transforms
            transforms.append(ReversePermutation(features=3))
            
            # Use masked autoregressive transform with rational quadratic splines
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=3,
                    hidden_features=hidden_dim,
                    context_features=None,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=tail_bound,
                    num_blocks=2,
                    use_residual_blocks=use_residual_blocks,
                    activation=F.relu,
                    dropout_probability=dropout_probability,
                )
            )

        # Create the flow model
        self.flow = Flow(
            transform=CompositeTransform(transforms), distribution=base_dist
        )

    def log_prob(self, x):
        """Compute log probability of x"""
        return self.flow.log_prob(x)

    def sample(self, n):
        """Sample n points from the flow"""
        return self.flow.sample(n)


# ------ Data Loading Functions ------


def load_apogee_data(config_path="config.yaml", r_min=0, r_max=6):
    """
    Load APOGEE data for a specific radial bin.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    r_min : float
        Minimum radius for the bin in kpc
    r_max : float
        Maximum radius for the bin in kpc
        
    Returns:
    --------
    tuple
        (data, errors) arrays for age, [Fe/H], [Mg/Fe]
    """
    print(f"\n=== Loading APOGEE data for {r_min}-{r_max} kpc bin ===")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize data handler
    data_handler = StellarDataHandler(
        config["apogee_path"], 
        config["galah_path"]
    )
    
    try:
        # Try loading data from fits file first
        mw_data = data_handler.load_apogee_fits()
    except FileNotFoundError:
        # Fall back to CSV if fits not found
        print("FITS file not found, trying CSV...")
        mw_data = data_handler.load_apogee_csv()
    
    # Filter data based on quality criteria
    conditions = [
        (mw_data["pred_logAge_std"] < 0.2),  # Age uncertainty
        (mw_data["FE_H_ERR"] < 0.1),         # [Fe/H] uncertainty
        (mw_data["MG_FE_ERR"] < 0.1),        # [Mg/Fe] uncertainty
        (mw_data["age"] > 0),                # Valid age
        (mw_data["age"] < 20),               # Reasonable age (< 20 Gyr)
        (~np.isnan(mw_data["rmean"])),       # Valid mean radius
    ]
    
    # Apply quality filters
    from src.data_handler import filter_data
    filtered_data = filter_data(mw_data, conditions)
    
    # Prepare data for radial bins
    radial_bins = [(r_min, r_max)]
    bin_data = prepare_data_for_radial_bins(filtered_data, radial_bins)
    
    # Get data for our specific bin
    bin_name = f"R{r_min}-{r_max}"
    if bin_name not in bin_data:
        raise ValueError(f"No data available for bin {bin_name}")
    
    # Extract only age, [Fe/H], [Mg/Fe] (first 3 columns)
    data_3d = bin_data[bin_name]["data"][:, :3]
    err_3d = bin_data[bin_name]["err"][:, :3]
    
    print(f"Loaded {len(data_3d)} stars for {bin_name} bin")
    print(f"Data shape: {data_3d.shape}, Errors shape: {err_3d.shape}")
    
    return data_3d, err_3d


# ------ Training Functions ------


def pretrain_flow(data, args):
    """
    Pretrain flow model using maximum likelihood on data.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array of shape (N, 3) with age, [Fe/H], [Mg/Fe]
    args : argparse.Namespace
        Command-line arguments
        
    Returns:
    --------
    tuple
        (trained flow model, scaler, training stats)
    """
    print(
        f"\n=== Stage 1: Pretraining 3D flow model for {args.pretraining_epochs} epochs ==="
    )

    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Convert to tensors
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)

    # Create dataset and loader
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize flow model
    flow = Flow3D(
        n_transforms=args.n_transforms,
        hidden_dim=args.hidden_dim,
        num_bins=args.num_bins,
        tail_bound=args.tail_bound,
        dropout_probability=args.dropout,
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(
        flow.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Training loop
    train_stats = {"log_likelihood": [], "time": []}
    start_time = time.time()
    
    pbar = tqdm(range(args.pretraining_epochs), desc="Pretraining flow")

    for epoch in pbar:
        flow.train()
        epoch_lls = []

        for (batch_data,) in loader:
            optimizer.zero_grad()

            # Compute log likelihood
            log_likelihood = flow.log_prob(batch_data)
            loss = -torch.mean(log_likelihood)

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            optimizer.step()

            epoch_lls.append(-loss.item())

        # Track stats
        avg_ll = np.mean(epoch_lls)
        total_time = time.time() - start_time
        
        train_stats["log_likelihood"].append(avg_ll)
        train_stats["time"].append(total_time)
        
        pbar.set_postfix({"LL": f"{avg_ll:.4f}"})

    print(f"Pretraining complete. Final Log-Likelihood: {avg_ll:.4f}")
    return flow, scaler, train_stats


def train_flow_with_uncertainty(data, errors, args):
    """
    Train flow model with uncertainty-aware approach.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array of shape (N, 3) with age, [Fe/H], [Mg/Fe]
    errors : np.ndarray
        Error array of shape (N, 3)
    args : argparse.Namespace
        Command-line arguments
        
    Returns:
    --------
    tuple
        (flow, recognition_net, scaler, (pretrain_stats, train_stats))
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1: Pretrain the flow model
    flow, scaler, pretrain_stats = pretrain_flow(data=data, args=args)

    # Scale data and errors for main training
    data_scaled = scaler.transform(data)
    errors_scaled = errors / scaler.scale_

    # Convert to tensors
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)
    errors_tensor = torch.tensor(errors_scaled, dtype=torch.float32).to(device)

    # Create dataset and loader
    dataset = TensorDataset(data_tensor, errors_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize recognition network
    recognition_net = RecognitionNetwork(
        input_dim=3,
        n_transforms=args.recognition_n_transforms,
        hidden_dims=[args.hidden_dim, args.hidden_dim],
    ).to(device)

    # Stage 2: Train with uncertainty-aware ELBO
    print(
        f"\n=== Stage 2: Training with uncertainty-aware objective for {args.epochs} epochs ==="
    )

    # Optimizer
    optimizer = optim.Adam(
        list(flow.parameters()) + list(recognition_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 ** ((epoch - warmup_epochs) / 20)  # Halve the LR every 20 epochs

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training stats
    train_stats = {
        "objective": [], 
        "lr": [],
        "diagnostics": [],
    }
    
    best_objective = -float("inf")
    best_state = {"flow": None, "recognition": None}

    # Main training loop
    objective_name = "IWAE" if args.use_iwae else "ELBO"
    pbar = tqdm(range(args.epochs), desc=f"Training with {objective_name}")

    for epoch in pbar:
        flow.train()
        recognition_net.train()

        epoch_objectives = []
        curr_lr = optimizer.param_groups[0]["lr"]

        for batch_data, batch_errors in loader:
            optimizer.zero_grad()

            # Calculate objective based on chosen method
            if args.use_iwae:
                objective = importance_weighted_elbo(
                    flow, recognition_net, batch_data, batch_errors, K=args.mc_samples
                )
            else:
                objective = uncertainty_aware_elbo(
                    flow, recognition_net, batch_data, batch_errors, K=args.mc_samples
                )

            batch_obj = objective.item()
            epoch_objectives.append(batch_obj)

            # Backward pass
            loss = -objective
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(flow.parameters()) + list(recognition_net.parameters()), 10.0
            )
            optimizer.step()

        # Update learning rate
        scheduler.step()

        # Record epoch stats
        epoch_avg_obj = np.mean(epoch_objectives)
        train_stats["objective"].append(epoch_avg_obj)
        train_stats["lr"].append(curr_lr)
        
        # Compute diagnostics every 5 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                # Compute on a subset for efficiency
                subset_size = min(1000, len(data_tensor))
                indices = torch.randperm(len(data_tensor))[:subset_size]
                subset_data = data_tensor[indices]
                subset_errors = errors_tensor[indices]
                
                diag = compute_diagnostics(
                    flow, recognition_net, subset_data, subset_errors, n_samples=20
                )
                train_stats["diagnostics"].append(diag)
                
                pbar.set_postfix({
                    objective_name: f"{epoch_avg_obj:.4f}",
                    "LR": f"{curr_lr:.6f}",
                    "KL": f"{diag['kl_term']:.2f}",
                })
        else:
            pbar.set_postfix({
                objective_name: f"{epoch_avg_obj:.4f}",
                "LR": f"{curr_lr:.6f}",
            })

        # Save best model
        if epoch_avg_obj > best_objective:
            best_objective = epoch_avg_obj
            best_state = {
                "flow": {k: v.cpu().clone() for k, v in flow.state_dict().items()},
                "recognition": {
                    k: v.cpu().clone() for k, v in recognition_net.state_dict().items()
                },
            }
            print(f"New best {objective_name}: {best_objective:.4f}")
            
        # Create training progress plot every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            plot_training_curves(pretrain_stats, train_stats, args.output_dir, objective_name)

    # Load best model
    flow.load_state_dict(best_state["flow"])
    recognition_net.load_state_dict(best_state["recognition"])
    flow.to(device)
    recognition_net.to(device)

    print(f"Training complete. Best {objective_name}: {best_objective:.4f}")

    # Save models
    save_models(flow, recognition_net, scaler, args, args.output_dir)

    # Generate samples and visualization
    visualize_apogee_results(
        flow,
        recognition_net,
        scaler,
        data_scaled,
        errors_scaled,
        args.output_dir,
        n_samples=args.n_samples,
    )

    return flow, recognition_net, scaler, (pretrain_stats, train_stats)


# ------ Visualization Functions ------


def plot_training_curves(
    pretrain_stats, train_stats, output_dir, objective_name="ELBO"
):
    """
    Plot training curves.
    
    Parameters:
    -----------
    pretrain_stats : dict
        Dictionary with pretraining statistics
    train_stats : dict
        Dictionary with training statistics
    output_dir : str
        Directory to save output plots
    objective_name : str
        Name of the objective function used
    """
    # Create directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot pretraining log likelihood
    ax1.plot(pretrain_stats["log_likelihood"], marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Log Likelihood")
    ax1.set_title("Flow Pretraining")
    ax1.grid(True, alpha=0.3)

    # Plot training objective
    ax2.plot(train_stats["objective"], marker="o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(f"{objective_name} Value")
    ax2.set_title(f"Uncertainty-Aware Training ({objective_name})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=300)
    plt.close()
    
    # If we have diagnostics, plot them too
    if "diagnostics" in train_stats and train_stats["diagnostics"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Extract diagnostics at 5-epoch intervals
        epochs = list(range(0, len(train_stats["objective"]), 5))
        if len(epochs) > len(train_stats["diagnostics"]):
            epochs = epochs[:len(train_stats["diagnostics"])]
            
        recon_terms = [d["reconstruction_term"] for d in train_stats["diagnostics"]]
        kl_terms = [d["kl_term"] for d in train_stats["diagnostics"]]
        ess = [d["effective_sample_size"] for d in train_stats["diagnostics"]]
        
        # Plot KL and reconstruction terms
        ax1.plot(epochs, recon_terms, "b-o", label="Reconstruction Term")
        ax1.plot(epochs, kl_terms, "r-o", label="KL Term")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Value")
        ax1.set_title("ELBO Components")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot effective sample size
        ax2.plot(epochs, ess, "g-o")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("ESS / K")
        ax2.set_title("Effective Sample Size Ratio")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "training_diagnostics.png"), dpi=300)
        plt.close()


def save_models(flow, recognition_net, scaler, args, output_dir):
    """
    Save trained models and configuration.
    
    Parameters:
    -----------
    flow : Flow3D
        Trained flow model
    recognition_net : RecognitionNetwork
        Trained recognition network
    scaler : StandardScaler
        Data scaler
    args : argparse.Namespace
        Command-line arguments
    output_dir : str
        Output directory
    """
    model_path = os.path.join(output_dir, "apogee_deconvolution_model.pt")
    
    # Save as dictionary with all components
    torch.save(
        {
            "flow_state": flow.state_dict(),
            "recognition_state": recognition_net.state_dict(),
            "scaler": scaler,
            "config": vars(args),
        },
        model_path,
    )
    print(f"Model saved to {model_path}")


def visualize_apogee_results(
    flow,
    recognition_net,
    scaler,
    data_scaled,
    errors_scaled,
    output_dir,
    n_samples=10000,
):
    """
    Generate visualizations for APOGEE data results.
    
    Parameters:
    -----------
    flow : Flow3D
        Trained flow model
    recognition_net : RecognitionNetwork
        Trained recognition network
    scaler : StandardScaler
        Data scaler
    data_scaled : np.ndarray
        Scaled data array
    errors_scaled : np.ndarray
        Scaled errors array
    output_dir : str
        Output directory
    n_samples : int
        Number of samples to generate
    """
    # Create directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set evaluation mode
    flow.eval()
    recognition_net.eval()

    # Sample from the prior
    with torch.no_grad():
        # Generate samples from the prior
        prior_samples = flow.sample(n_samples).cpu().numpy()
        
        # Get some observed data points for comparison
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)
        errors_tensor = torch.tensor(errors_scaled, dtype=torch.float32).to(device)
        
        # Use a subset of observed data for efficiency
        subset_size = min(n_samples, len(data_scaled))
        indices = np.random.choice(len(data_scaled), subset_size, replace=False)
        observed_subset = data_tensor[indices]
        errors_subset = errors_tensor[indices]

    # Convert samples back to original scale
    prior_samples_original = scaler.inverse_transform(prior_samples)
    observed_data_original = scaler.inverse_transform(data_scaled[indices])
    
    # Extract features
    log_ages_prior = prior_samples_original[:, 0]
    fehs_prior = prior_samples_original[:, 1]
    mgfes_prior = prior_samples_original[:, 2]
    
    log_ages_obs = observed_data_original[:, 0]
    fehs_obs = observed_data_original[:, 1]
    mgfes_obs = observed_data_original[:, 2]
    
    # Convert log age to linear age
    ages_prior = 10**log_ages_prior
    ages_obs = 10**log_ages_obs
    
    # Create plots comparing model samples with observed data
    
    # 1. Age-[Fe/H] relation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Model samples (left)
    scatter1 = ax1.scatter(
        ages_prior, fehs_prior, s=2, alpha=0.5, c=mgfes_prior, 
        cmap="viridis", vmin=-0.1, vmax=0.4
    )
    cbar1 = fig.colorbar(scatter1, ax=ax1, label="[Mg/Fe]")
    ax1.set_xlabel("Age (Gyr)")
    ax1.set_ylabel("[Fe/H]")
    ax1.set_title("Age-[Fe/H] Relation (Deconvolved Model)")
    ax1.set_xlim(0, 15)
    ax1.set_ylim(-1.0, 0.5)
    ax1.grid(True, alpha=0.3)
    
    # Observed data (right)
    scatter2 = ax2.scatter(
        ages_obs, fehs_obs, s=2, alpha=0.5, c=mgfes_obs, 
        cmap="viridis", vmin=-0.1, vmax=0.4
    )
    cbar2 = fig.colorbar(scatter2, ax=ax2, label="[Mg/Fe]")
    ax2.set_xlabel("Age (Gyr)")
    ax2.set_ylabel("[Fe/H]")
    ax2.set_title("Age-[Fe/H] Relation (Observed Data)")
    ax2.set_xlim(0, 15)
    ax2.set_ylim(-1.0, 0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "age_feh_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Age-[Mg/Fe] relation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Model samples (left)
    scatter1 = ax1.scatter(
        ages_prior, mgfes_prior, s=2, alpha=0.5, c=fehs_prior, 
        cmap="plasma", vmin=-0.8, vmax=0.3
    )
    cbar1 = fig.colorbar(scatter1, ax=ax1, label="[Fe/H]")
    ax1.set_xlabel("Age (Gyr)")
    ax1.set_ylabel("[Mg/Fe]")
    ax1.set_title("Age-[Mg/Fe] Relation (Deconvolved Model)")
    ax1.set_xlim(0, 15)
    ax1.set_ylim(-0.1, 0.4)
    ax1.grid(True, alpha=0.3)
    
    # Observed data (right)
    scatter2 = ax2.scatter(
        ages_obs, mgfes_obs, s=2, alpha=0.5, c=fehs_obs, 
        cmap="plasma", vmin=-0.8, vmax=0.3
    )
    cbar2 = fig.colorbar(scatter2, ax=ax2, label="[Fe/H]")
    ax2.set_xlabel("Age (Gyr)")
    ax2.set_ylabel("[Mg/Fe]")
    ax2.set_title("Age-[Mg/Fe] Relation (Observed Data)")
    ax2.set_xlim(0, 15)
    ax2.set_ylim(-0.1, 0.4)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "age_mgfe_comparison.png"), dpi=300)
    plt.close()
    
    # 3. [Fe/H]-[Mg/Fe] relation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Model samples (left)
    scatter1 = ax1.scatter(
        fehs_prior, mgfes_prior, s=2, alpha=0.5, c=ages_prior, 
        cmap="viridis", vmin=0, vmax=13
    )
    cbar1 = fig.colorbar(scatter1, ax=ax1, label="Age (Gyr)")
    ax1.set_xlabel("[Fe/H]")
    ax1.set_ylabel("[Mg/Fe]")
    ax1.set_title("[Fe/H]-[Mg/Fe] Relation (Deconvolved Model)")
    ax1.set_xlim(-1.0, 0.5)
    ax1.set_ylim(-0.1, 0.4)
    ax1.grid(True, alpha=0.3)
    
    # Observed data (right)
    scatter2 = ax2.scatter(
        fehs_obs, mgfes_obs, s=2, alpha=0.5, c=ages_obs, 
        cmap="viridis", vmin=0, vmax=13
    )
    cbar2 = fig.colorbar(scatter2, ax=ax2, label="Age (Gyr)")
    ax2.set_xlabel("[Fe/H]")
    ax2.set_ylabel("[Mg/Fe]")
    ax2.set_title("[Fe/H]-[Mg/Fe] Relation (Observed Data)")
    ax2.set_xlim(-1.0, 0.5)
    ax2.set_ylim(-0.1, 0.4)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feh_mgfe_comparison.png"), dpi=300)
    plt.close()
    
    # 4. Create density plots for age distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prior model density
    ax1.hist(
        ages_prior, bins=30, alpha=0.7, density=True, 
        color="blue", label="Deconvolved Model"
    )
    ax1.set_xlabel("Age (Gyr)")
    ax1.set_ylabel("Density")
    ax1.set_title("Age Distribution (Deconvolved Model)")
    ax1.set_xlim(0, 15)
    ax1.grid(True, alpha=0.3)
    
    # Observed data density
    ax2.hist(
        ages_obs, bins=30, alpha=0.7, density=True, 
        color="red", label="Observed Data"
    )
    ax2.set_xlabel("Age (Gyr)")
    ax2.set_ylabel("Density")
    ax2.set_title("Age Distribution (Observed Data)")
    ax2.set_xlim(0, 15)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "age_distribution.png"), dpi=300)
    plt.close()
    
    print(f"Generated and saved visualizations to {plots_dir}")


# ------ Command-line Interface ------


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train a normalizing flow model for APOGEE data deconvolution."
    )

    # Dataset options
    parser.add_argument(
        "--r_min",
        type=float,
        default=0.0,
        help="Minimum radius in kpc for the radial bin",
    )
    parser.add_argument(
        "--r_max",
        type=float,
        default=6.0,
        help="Maximum radius in kpc for the radial bin",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of samples for visualization",
    )

    # Model architecture
    parser.add_argument(
        "--n_transforms",
        type=int,
        default=16,
        help="Number of transforms in the flow model",
    )
    parser.add_argument(
        "--recognition_n_transforms",
        type=int,
        default=8,
        help="Number of transforms in the recognition network",
    )
    parser.add_argument(
        "--hidden_dim", 
        type=int, 
        default=128, 
        help="Hidden dimension for networks"
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=16,
        help="Number of bins for piecewise transforms",
    )
    parser.add_argument(
        "--tail_bound",
        type=float,
        default=5.0,
        help="Tail bound for piecewise transforms",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability for regularization",
    )

    # Training parameters
    parser.add_argument(
        "--pretraining_epochs",
        type=int,
        default=30,
        help="Number of epochs for flow pretraining",
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50, 
        help="Number of epochs for full training"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=256, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3, 
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=10,
        help="Number of Monte Carlo samples for objective estimation",
    )

    # Output options
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results_apogee", 
        help="Directory to save outputs"
    )

    # Advanced options
    parser.add_argument(
        "--use_iwae",
        action="store_true",
        help="Use Importance Weighted Autoencoder objective instead of ELBO",
    )
    
    return parser.parse_args()


# ------ Main Function ------


def main():
    """
    Main function to run the APOGEE flow model training.
    """
    # Parse command-line arguments
    args = parse_args()

    # Print configuration
    print("\n=== Configuration ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Create output directory with radial bin info
    bin_output_dir = os.path.join(
        args.output_dir, f"R{int(args.r_min)}-{int(args.r_max)}_kpc"
    )
    args.output_dir = bin_output_dir
    os.makedirs(bin_output_dir, exist_ok=True)

    # Check for nflows package
    try:
        import nflows
        print("Using nflows version:", nflows.__version__)
    except ImportError:
        print("ERROR: This script requires nflows. Install it with:")
        print("pip install nflows")
        exit(1)

    # Load APOGEE data
    data, errors = load_apogee_data(
        config_path=args.config_path, 
        r_min=args.r_min, 
        r_max=args.r_max
    )

    # Train model
    flow, recognition_net, scaler, stats = train_flow_with_uncertainty(
        data, errors, args
    )

    print(f"Training and visualization complete! Results saved to {bin_output_dir}/")


if __name__ == "__main__":
    main()