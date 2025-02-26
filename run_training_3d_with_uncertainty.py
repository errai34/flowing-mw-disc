#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Execution script for 3D normalizing flow model training with uncertainty-aware approach.
Focuses only on age, [Fe/H], and [Mg/Fe] but maintains the two-phase training methodology.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_handler import StellarDataHandler
from flow_model import Flow3D

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Create a 3D RecognitionNetwork (adapted from uncertainty.py)
class RecognitionNetwork3D(torch.nn.Module):
    """
    Recognition network for 3D amortized variational inference.
    Implements q(v|w) for 3D data.
    """

    def __init__(self, n_transforms=8, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        from nflows.distributions.normal import StandardNormal
        from nflows.transforms.autoregressive import (
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        )
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import ReversePermutation

        # Base distribution (3D standard normal)
        self.base_dist = StandardNormal(shape=[3])

        # Create conditioning network
        self.conditioning_net = torch.nn.Sequential(
            torch.nn.Linear(3 * 2, hidden_dims[0]),  # observed data + error info
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[0]),
        )

        # Create transforms
        transforms = []
        for i in range(n_transforms):
            transforms.append(ReversePermutation(features=3))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=3,
                    hidden_features=hidden_dims[0],
                    context_features=hidden_dims[
                        0
                    ],  # context from conditioning network
                    num_bins=16,
                    tails="linear",
                    tail_bound=5.0,
                    use_residual_blocks=True,
                    random_mask=False,
                    activation=torch.nn.functional.relu,
                )
            )

        self.transform = CompositeTransform(transforms)

    def forward(self, observed_data, uncertainties):
        """Process observed data and uncertainties to create context."""
        # Concatenate observed data and uncertainties
        context_input = torch.cat([observed_data, uncertainties], dim=1)
        return self.conditioning_net(context_input)

    def sample(self, observed_data, uncertainties, n_samples=1):
        """Sample from q(v|w)."""
        batch_size = observed_data.shape[0]

        # Get context from conditioning network
        context = self.forward(observed_data, uncertainties)

        # Draw samples from base distribution
        eps = self.base_dist.sample(batch_size * n_samples).reshape(
            batch_size * n_samples, -1
        )

        # Create batched context by repeating for each sample
        batched_context = context.repeat_interleave(n_samples, dim=0)

        # Transform samples using context
        samples, _ = self.transform.inverse(eps, batched_context)

        return samples.reshape(batch_size * n_samples, -1)

    def log_prob(self, latent_samples, observed_data, uncertainties):
        """Compute log q(v|w)."""
        # Get context from conditioning network
        context = self.forward(observed_data, uncertainties)

        # Transform samples to base space
        noise, logabsdet = self.transform(latent_samples, context)

        # Compute log probability
        log_prob = self.base_dist.log_prob(noise) + logabsdet

        return log_prob


def compute_log_noise_pdf(w, v, e):
    """
    Compute log p_noise(w | v) in a numerically stable way.

    Parameters:
    -----------
    w : torch.Tensor
        Observed data points
    v : torch.Tensor
        True latent data points
    e : torch.Tensor
        Measurement uncertainties

    Returns:
    --------
    torch.Tensor
        Log probability of the noise model
    """
    # Clamp the error so that the variance never vanishes
    e_safe = e.clamp(min=1e-8)

    # Compute log probability
    log_prob = -0.5 * torch.sum(
        torch.log(2 * torch.pi * e_safe.pow(2)) + ((w - v) / e_safe).pow(2), dim=-1
    )

    return log_prob


def uncertainty_aware_elbo(flow, recognition_net, observed_data, uncertainties, K=10):
    """
    Compute uncertainty-aware ELBO for 3D models.
    ELBO = E_q(v|w)[log p(w|v) + log p(v) - log q(v|w)]

    Parameters:
    -----------
    flow : Flow3D
        Normalizing flow model
    recognition_net : RecognitionNetwork3D
        Recognition network
    observed_data : torch.Tensor
        Observed data points
    uncertainties : torch.Tensor
        Measurement uncertainties
    K : int
        Number of Monte Carlo samples

    Returns:
    --------
    torch.Tensor
        ELBO value (scalar)
    """
    batch_size = observed_data.shape[0]

    # Sample from recognition network q(v|w)
    samples = recognition_net.sample(observed_data, uncertainties, n_samples=K)

    # Repeat observed data and uncertainties for each sample
    repeated_observed = observed_data.repeat_interleave(K, dim=0)
    repeated_uncertainties = uncertainties.repeat_interleave(K, dim=0)

    # Compute log probabilities
    log_p_v = flow.log_prob(samples)  # Prior
    log_p_w_given_v = compute_log_noise_pdf(
        repeated_observed, samples, repeated_uncertainties
    )  # Likelihood
    log_q_v_given_w = recognition_net.log_prob(
        samples, repeated_observed, repeated_uncertainties
    )  # Posterior

    # Compute ELBO for each sample
    elbo_components = log_p_w_given_v + log_p_v - log_q_v_given_w
    elbo_components = elbo_components.reshape(batch_size, K)

    # Average over MC samples
    elbo = torch.mean(elbo_components, dim=1)

    return torch.mean(elbo)


def format_bin_name(bin_range):
    """
    Format bin name to ensure consistent naming with decimal points.
    Converts '0-6' to 'R0.0-6.0'
    """
    r_min, r_max = map(float, bin_range.split("-"))
    return f"R{r_min:.1f}-{r_max:.1f}"


def prepare_data_for_radial_bins_3d(df, radial_bins=None):
    """
    Prepare 3D data (age, [Fe/H], [Mg/Fe]) for analysis in radial bins.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with necessary stellar parameters
    radial_bins : list
        List of (min_R, max_R) tuples defining radial bins

    Returns:
    --------
    dict
        Dictionary mapping bin names to data and errors
    """
    # Define default radial bins if not provided
    if radial_bins is None:
        radial_bins = [(0, 6), (6, 8), (8, 10), (10, 15)]

    # Create a proper copy to avoid SettingWithCopyWarning
    df = df.copy()

    bin_data = {}

    for r_min, r_max in radial_bins:
        bin_name = f"R{r_min:.1f}-{r_max:.1f}"
        mask = (df["rmean"] >= r_min) & (df["rmean"] < r_max)
        bin_df = df[mask]

        # Skip if too few stars
        if len(bin_df) < 100:
            print(f"Warning: Bin {bin_name} has only {len(bin_df)} stars, skipping")
            continue

        # Extract data and errors for the 3D model only (age, [Fe/H], [Mg/Fe])
        data_3d = bin_df[["pred_logAge", "FE_H", "MG_FE"]].values
        err_3d = bin_df[["pred_logAge_std", "FE_H_ERR", "MG_FE_ERR"]].values

        bin_data[bin_name] = {"data": data_3d, "err": err_3d, "count": len(bin_df)}

        print(f"Bin {bin_name}: {len(bin_df)} stars")

    return bin_data


def pretrain_flow_3d(data, n_epochs=10, batch_size=256, lr=1e-3, weight_decay=1e-5):
    """
    Pretrain 3D flow model using maximum likelihood on data.

    Parameters:
    -----------
    data : np.ndarray
        Data array of shape (N, 3)
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    weight_decay : float
        Weight decay for optimizer

    Returns:
    --------
    tuple
        (trained flow model, scaler, training stats)
    """
    # Ensure n_epochs is an integer
    n_epochs = int(n_epochs)

    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Convert to tensors
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)

    # Create dataset and loader
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize flow model
    flow = Flow3D(
        n_transforms=12,
        hidden_dims=[128, 128],
        num_bins=24,
        tail_bound=5.0,
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(flow.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    train_stats = {"log_likelihood": [], "time": []}

    print(f"Pretraining 3D flow on {len(data)} data points for {n_epochs} epochs")
    for epoch in range(n_epochs):
        flow.train()
        epoch_lls = []

        # Progress bar
        batch_progress = tqdm(
            loader,
            desc=f"Pretraining Epoch {epoch+1}/{n_epochs}",
            leave=False,
            ncols=100,
        )

        for (batch_data,) in batch_progress:
            optimizer.zero_grad()

            # Compute log likelihood
            log_likelihood = flow.log_prob(batch_data)
            loss = -torch.mean(log_likelihood)

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            optimizer.step()

            epoch_lls.append(-loss.item())
            batch_progress.set_postfix({"LL": f"{-loss.item():.4f}"})

        # Track stats
        avg_ll = np.mean(epoch_lls)
        train_stats["log_likelihood"].append(avg_ll)

        print(f"Pretraining Epoch {epoch+1}/{n_epochs}, Log-Likelihood: {avg_ll:.4f}")

    print(
        f"Pretraining complete. Final Log-Likelihood: {train_stats['log_likelihood'][-1]:.4f}"
    )
    return flow, scaler, train_stats


def train_3d_flow_with_uncertainty(
    data,
    errors,
    n_transforms=12,
    hidden_dims=None,
    pretraining_epochs=10,
    n_epochs=50,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    mc_samples=10,
    output_dir=None,
):
    """
    Train 3D flow model with uncertainty-aware approach.

    Parameters:
    -----------
    data : np.ndarray
        Training data of shape (N, 3) for [age, [Fe/H], [Mg/Fe]]
    errors : np.ndarray
        Measurement errors for training data
    n_transforms : int
        Number of transforms in the flow
    hidden_dims : list
        List of hidden layer dimensions
    pretraining_epochs : int
        Number of pretraining epochs
    n_epochs : int
        Number of main training epochs
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    weight_decay : float
        Weight decay for optimizer
    mc_samples : int
        Number of Monte Carlo samples for ELBO estimation
    output_dir : str
        Directory to save training progress

    Returns:
    --------
    tuple
        (trained flow model, recognition network, scaler, training stats)
    """
    print("Starting 3D flow model training with uncertainty...")

    # Ensure integer parameters
    n_epochs = int(n_epochs)
    pretraining_epochs = int(pretraining_epochs)
    mc_samples = int(mc_samples)

    # Use default hidden dimensions if not provided
    if hidden_dims is None:
        hidden_dims = [128, 128]

    # Stage 1: Pretrain the flow model
    print(
        f"\n=== Stage 1: Pretraining 3D flow model for {pretraining_epochs} epochs ==="
    )
    flow, scaler, pretrain_stats = pretrain_flow_3d(
        data=data,
        n_epochs=pretraining_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scale data for main training
    data_scaled = scaler.transform(data)
    errors_scaled = errors / scaler.scale_

    # Convert to tensors
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)
    errors_tensor = torch.tensor(errors_scaled, dtype=torch.float32).to(device)

    # Create dataset and loader
    dataset = TensorDataset(data_tensor, errors_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize recognition network
    recognition_net = RecognitionNetwork3D(
        n_transforms=8,
        hidden_dims=hidden_dims,
    ).to(device)

    # Stage 2: Train with uncertainty-aware ELBO
    print(
        f"\n=== Stage 2: Training with uncertainty-aware ELBO for {n_epochs} epochs ==="
    )

    # Optimizer
    optimizer = optim.AdamW(
        list(flow.parameters()) + list(recognition_net.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 ** (epoch / 20)  # Halve the learning rate every 20 epochs

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Training stats
    train_stats = {
        "elbo": [],
        "lr": [],
    }

    best_elbo = -float("inf")
    best_state = {"flow": None, "recognition": None}

    print(f"Training on {len(data)} data points for {n_epochs} epochs")

    # Main training loop
    for epoch in range(n_epochs):
        flow.train()
        recognition_net.train()

        epoch_elbo_values = []

        # Batch loop with progress bar
        batch_progress = tqdm(
            loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False, ncols=100
        )

        for batch_data, batch_errors in batch_progress:
            optimizer.zero_grad()

            # Calculate objective
            elbo = uncertainty_aware_elbo(
                flow, recognition_net, batch_data, batch_errors, K=mc_samples
            )
            batch_elbo = elbo.item()
            epoch_elbo_values.append(batch_elbo)

            # Backward pass
            loss = -elbo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(flow.parameters()) + list(recognition_net.parameters()), 10.0
            )
            optimizer.step()

            # Update progress bar
            batch_progress.set_postfix({"ELBO": f"{batch_elbo:.4f}"})

        # Update learning rate
        scheduler.step()

        # Record epoch stats
        epoch_avg_elbo = (
            sum(epoch_elbo_values) / len(epoch_elbo_values) if epoch_elbo_values else 0
        )
        train_stats["elbo"].append(epoch_avg_elbo)
        train_stats["lr"].append(optimizer.param_groups[0]["lr"])

        # Save best model
        if epoch_avg_elbo > best_elbo:
            best_elbo = epoch_avg_elbo
            best_state = {
                "flow": {k: v.cpu().clone() for k, v in flow.state_dict().items()},
                "recognition": {
                    k: v.cpu().clone() for k, v in recognition_net.state_dict().items()
                },
            }
            print(f"New best objective: {best_elbo:.4f}")

        # Print status
        print(
            f"Epoch {epoch+1}/{n_epochs}, ELBO: {epoch_avg_elbo:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Create simple visualization every 5 epochs
        if epoch % 5 == 0 and output_dir:
            import matplotlib.pyplot as plt

            bin_name = (
                os.path.basename(output_dir).split("_")[0] if output_dir else "unknown"
            )
            plot_path = os.path.join(plots_dir, f"training_progress_{bin_name}.png")

            plt.figure(figsize=(10, 5))
            plt.plot(range(epoch + 1), train_stats["elbo"], "b-", marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("ELBO")
            plt.title(f"Training Progress for {bin_name}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

    # Load best model
    flow.load_state_dict(best_state["flow"])
    recognition_net.load_state_dict(best_state["recognition"])
    flow.to(device)
    recognition_net.to(device)

    print(f"Training complete. Best objective: {best_elbo:.4f}")
    return flow, recognition_net, scaler, train_stats


def sample_flow_and_visualize(flow, scaler, bin_name, save_dir, n_samples=5000):
    """
    Sample from the flow model and create visualizations.

    Parameters:
    -----------
    flow : Flow3D
        Trained flow model
    scaler : StandardScaler
        Scaler used to normalize the data
    bin_name : str
        Name of the radial bin
    save_dir : str
        Directory to save visualizations
    n_samples : int
        Number of samples to draw from the model
    """
    import matplotlib.pyplot as plt

    # Set evaluation mode
    flow.eval()

    # Sample from the flow
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy()

    # Inverse transform to get original scale
    samples_original = scaler.inverse_transform(samples)

    # Extract age, [Fe/H], and [Mg/Fe]
    log_ages = samples_original[:, 0]
    fehs = samples_original[:, 1]
    mgfes = samples_original[:, 2]

    # Convert log age to linear age
    ages = 10**log_ages

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Age vs [Fe/H]
    ax1.scatter(ages, fehs, s=3, alpha=0.6, color="blue")
    ax1.set_xlabel("Age (Gyr)")
    ax1.set_ylabel("[Fe/H]")
    ax1.set_title(f"{bin_name}: Age vs. [Fe/H]")
    ax1.set_xlim(14, 0)  # Reversed to show oldest stars on left
    ax1.set_ylim(-1.5, 0.5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Age vs [Mg/Fe]
    ax2.scatter(ages, mgfes, s=3, alpha=0.6, color="green")
    ax2.set_xlabel("Age (Gyr)")
    ax2.set_ylabel("[Mg/Fe]")
    ax2.set_title(f"{bin_name}: Age vs. [Mg/Fe]")
    ax2.set_xlim(14, 0)  # Reversed to show oldest stars on left
    ax2.set_ylim(-0.2, 0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, f"{bin_name}_3d_chemical_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create a second figure for [Mg/Fe] vs [Fe/H]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(fehs, mgfes, s=3, alpha=0.6, color="purple")
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("[Mg/Fe]")
    ax.set_title(f"{bin_name}: [Mg/Fe] vs. [Fe/H]")
    ax.set_xlim(-1.5, 0.5)
    ax.set_ylim(-0.2, 0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(
        os.path.join(save_dir, f"{bin_name}_3d_mgfe_feh.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main(args):
    """Main function to run the training."""
    print("Starting 3D flow model training with uncertainty-aware approach...")

    # Create models directory if it doesn't exist
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Create plots directory
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Create run directory for logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_3d_uncertainty_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Format bin name consistently
    formatted_bin_name = args.bin_name
    if "-" in args.bin_name and not args.bin_name.startswith("R"):
        formatted_bin_name = format_bin_name(args.bin_name)
    elif "-" in args.bin_name and args.bin_name.startswith("R"):
        range_part = args.bin_name[1:]
        formatted_bin_name = format_bin_name(range_part)

    # Save configuration
    config = {
        "data_config": {"apogee_path": args.data_path, "n_samples": args.n_samples},
        "model_config": {
            "n_transforms": args.n_transforms,
            "hidden_dims": [args.hidden_dim, args.hidden_dim],
            "num_bins": args.num_bins,
            "model_type": "3D",  # Explicitly note we're using a 3D model
        },
        "training_config": {
            "pretraining_epochs": args.pretraining_epochs,
            "n_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "mc_samples": args.mc_samples,
        },
        "bin_name": formatted_bin_name,
    }

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Load data
    data_handler = StellarDataHandler(args.data_path, "")

    try:
        print("Attempting to load data...")
        if args.data_format == "csv":
            mw_data = data_handler.load_apogee_csv()
        else:
            mw_data = data_handler.load_apogee_fits()

        print(f"Successfully loaded {len(mw_data)} stars")

        # Apply quality filters
        filtered_mw = data_handler.apply_quality_filters(mw_data)
        print(f"After filtering, we have {len(filtered_mw)} stars")

        # If requested, take a subset of the data
        if args.n_samples > 0 and args.n_samples < len(filtered_mw):
            filtered_mw = filtered_mw.sample(args.n_samples, random_state=42)
            print(f"Using subset of {len(filtered_mw)} stars for training")

        # Convert ranges to actual numeric values
        bin_ranges = []
        formatted_bin_ranges = []

        for bin_range in args.bin_ranges:
            formatted = format_bin_name(bin_range)
            formatted_bin_ranges.append(formatted)

            # Extract min and max from formatted name
            r_min, r_max = map(float, formatted[1:].split("-"))
            bin_ranges.append((r_min, r_max))

        # Prepare 3D data for radial bins
        bin_data = prepare_data_for_radial_bins_3d(filtered_mw, bin_ranges)

        # Train model for selected bin
        if formatted_bin_name not in bin_data:
            print(
                f"Bin {formatted_bin_name} not found. Available bins: {list(bin_data.keys())}"
            )
            return

        bin_info = bin_data[formatted_bin_name]
        print(
            f"Training model for bin {formatted_bin_name} with {bin_info['count']} stars"
        )

        # Get data and errors
        data = bin_info["data"]
        errors = bin_info["err"]

        # Train the 3D flow model with uncertainty
        flow, recognition_net, scaler, stats = train_3d_flow_with_uncertainty(
            data=data,
            errors=errors,
            n_transforms=args.n_transforms,
            hidden_dims=[args.hidden_dim, args.hidden_dim],
            pretraining_epochs=args.pretraining_epochs,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            mc_samples=args.mc_samples,
            output_dir=run_dir,
        )

        # Save trained model
        model_path = os.path.join(
            models_dir, f"{formatted_bin_name}_3d_uncertainty_model.pt"
        )
        torch.save(
            {
                "model_state": flow.state_dict(),
                "recognition_state": recognition_net.state_dict(),
                "scaler": scaler,
                "stats": stats,
                "model_config": {
                    "n_transforms": args.n_transforms,
                    "hidden_dims": [args.hidden_dim, args.hidden_dim],
                    "num_bins": args.num_bins,
                    "model_type": "3D with uncertainty",
                },
            },
            model_path,
        )

        print(f"Model saved to {model_path}")

        # Generate visualization
        sample_flow_and_visualize(
            flow=flow,
            scaler=scaler,
            bin_name=formatted_bin_name,
            save_dir=plots_dir,
            n_samples=5000,
        )

        print("Training and visualization complete!")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 3D normalizing flow model training with uncertainty-aware approach"
    )

    # Data parameters
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="fits",
        choices=["fits", "csv"],
        help="Data format (fits or csv)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=-1,
        help="Number of samples to use (-1 for all)",
    )
    parser.add_argument(
        "--bin_name", type=str, default="R8-10", help="Radial bin to train on"
    )
    parser.add_argument(
        "--bin_ranges",
        type=str,
        nargs="+",
        default=["0-6", "6-8", "8-10", "10-15"],
        help="Radial bin ranges",
    )

    # Model parameters
    parser.add_argument(
        "--n_transforms",
        type=int,
        default=4,
        help="Number of transforms in flow",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=24,
        help="Number of bins in spline transforms",
    )

    # Training parameters
    parser.add_argument(
        "--pretraining_epochs",
        type=int,
        default=10,
        help="Number of pretraining epochs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=5,
        help="Number of Monte Carlo samples",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )

    args = parser.parse_args()
    main(args)
