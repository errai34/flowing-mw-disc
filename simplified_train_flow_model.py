import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Set GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------ Model Definitions ------


class Flow3D(nn.Module):
    """
    3D normalizing flow - simplified version with less transforms and simpler structure
    """

    def __init__(self, n_transforms=8, hidden_dim=64, num_bins=12):
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

        # Build transforms
        transforms = []
        for i in range(n_transforms):
            transforms.append(ReversePermutation(features=3))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=3,
                    hidden_features=hidden_dim,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=5.0,
                    num_blocks=2,
                    use_residual_blocks=True,
                    activation=F.relu,
                    dropout_probability=0.0,
                )
            )

        # Create flow model
        self.flow = Flow(
            transform=CompositeTransform(transforms), distribution=base_dist
        )

    def log_prob(self, x):
        """Compute log probability of x"""
        return self.flow.log_prob(x)

    def sample(self, n):
        """Sample n points from the flow"""
        return self.flow.sample(n)


class RecognitionNetwork3D(nn.Module):
    """
    Recognition network for 3D amortized variational inference.
    Implements q(v|w) for 3D data - simplified version.
    """

    def __init__(self, n_transforms=6, hidden_dim=64):
        super().__init__()
        from nflows.distributions.normal import StandardNormal
        from nflows.transforms.autoregressive import (
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        )
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import ReversePermutation

        # Base distribution (3D standard normal)
        self.base_dist = StandardNormal(shape=[3])

        # Create conditioning network
        self.conditioning_net = nn.Sequential(
            nn.Linear(3 * 2, hidden_dim),  # observed data + error info
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Create transforms
        transforms = []
        for i in range(n_transforms):
            transforms.append(ReversePermutation(features=3))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=3,
                    hidden_features=hidden_dim,
                    context_features=hidden_dim,  # context from conditioning network
                    num_bins=12,
                    tails="linear",
                    tail_bound=5.0,
                    use_residual_blocks=True,
                    activation=F.relu,
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

        return samples

    def log_prob(self, latent_samples, observed_data, uncertainties):
        """Compute log q(v|w)."""
        # Get context from conditioning network
        context = self.forward(observed_data, uncertainties)

        # Transform samples to base space
        noise, logabsdet = self.transform(latent_samples, context)

        # Compute log probability
        log_prob = self.base_dist.log_prob(noise) + logabsdet

        return log_prob


# ------ Support Functions ------


def compute_log_noise_pdf(w, v, e):
    """
    Compute log p_noise(w | v) in a numerically stable way.
    """
    # Clamp the error for numerical stability
    e_safe = e.clamp(min=1e-8)

    # Compute log probability
    log_prob = -0.5 * torch.sum(
        torch.log(2 * torch.pi * e_safe.pow(2)) + ((w - v) / e_safe).pow(2), dim=-1
    )

    return log_prob


def uncertainty_aware_elbo(flow, recognition_net, observed_data, uncertainties, K=10):
    """
    Compute uncertainty-aware ELBO.
    ELBO = E_q(v|w)[log p(w|v) + log p(v) - log q(v|w)]
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


# ------ Training Functions ------


def pretrain_flow(data, n_epochs=10, batch_size=128, lr=1e-3, weight_decay=1e-5):
    """Pretrain flow model using maximum likelihood on data."""
    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Convert to tensors
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)

    # Create dataset and loader
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize flow model with simple configuration
    flow = Flow3D(n_transforms=8, hidden_dim=64, num_bins=12).to(device)

    # Optimizer
    optimizer = optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    train_stats = {"log_likelihood": []}
    pbar = tqdm(range(n_epochs), desc="Pretraining flow")

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
            optimizer.step()

            epoch_lls.append(-loss.item())

        # Track stats
        avg_ll = np.mean(epoch_lls)
        train_stats["log_likelihood"].append(avg_ll)
        pbar.set_postfix({"LL": f"{avg_ll:.4f}"})

    print(f"Pretraining complete. Final Log-Likelihood: {avg_ll:.4f}")
    return flow, scaler, train_stats


def train_flow_with_uncertainty(data, errors, output_dir="results", config=None):
    """
    Train 3D flow model with uncertainty-aware approach.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set default configuration if not provided
    if config is None:
        config = {
            "pretraining_epochs": 15,
            "training_epochs": 30,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "flow_n_transforms": 8,
            "recognition_n_transforms": 6,
            "hidden_dim": 64,
            "num_bins": 12,
            "mc_samples": 5,
        }

    print("Starting 3D flow model training with uncertainty...")
    print(f"Configuration: {config}")

    # Stage 1: Pretrain the flow model
    print(
        f"\n=== Stage 1: Pretraining 3D flow model for {config['pretraining_epochs']} epochs ==="
    )
    flow, scaler, pretrain_stats = pretrain_flow(
        data=data,
        n_epochs=config["pretraining_epochs"],
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Scale data for main training
    data_scaled = scaler.transform(data)
    errors_scaled = errors / scaler.scale_

    # Convert to tensors
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)
    errors_tensor = torch.tensor(errors_scaled, dtype=torch.float32).to(device)

    # Create dataset and loader
    dataset = TensorDataset(data_tensor, errors_tensor)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Initialize recognition network
    recognition_net = RecognitionNetwork3D(
        n_transforms=config["recognition_n_transforms"],
        hidden_dim=config["hidden_dim"],
    ).to(device)

    # Stage 2: Train with uncertainty-aware ELBO
    print(
        f"\n=== Stage 2: Training with uncertainty-aware ELBO for {config['training_epochs']} epochs ==="
    )

    # Optimizer
    optimizer = optim.Adam(
        list(flow.parameters()) + list(recognition_net.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Training stats
    train_stats = {"elbo": []}
    best_elbo = -float("inf")
    best_state = {"flow": None, "recognition": None}

    # Main training loop
    pbar = tqdm(range(config["training_epochs"]), desc="Training with ELBO")
    for epoch in pbar:
        flow.train()
        recognition_net.train()

        epoch_elbo_values = []

        for batch_data, batch_errors in loader:
            optimizer.zero_grad()

            # Calculate objective
            elbo = uncertainty_aware_elbo(
                flow, recognition_net, batch_data, batch_errors, K=config["mc_samples"]
            )
            batch_elbo = elbo.item()
            epoch_elbo_values.append(batch_elbo)

            # Backward pass
            loss = -elbo
            loss.backward()
            optimizer.step()

        # Record epoch stats
        epoch_avg_elbo = np.mean(epoch_elbo_values)
        train_stats["elbo"].append(epoch_avg_elbo)
        pbar.set_postfix({"ELBO": f"{epoch_avg_elbo:.4f}"})

        # Save best model
        if epoch_avg_elbo > best_elbo:
            best_elbo = epoch_avg_elbo
            best_state = {
                "flow": {k: v.cpu().clone() for k, v in flow.state_dict().items()},
                "recognition": {
                    k: v.cpu().clone() for k, v in recognition_net.state_dict().items()
                },
            }

    # Load best model
    flow.load_state_dict(best_state["flow"])
    recognition_net.load_state_dict(best_state["recognition"])
    flow.to(device)
    recognition_net.to(device)

    print(f"Training complete. Best objective: {best_elbo:.4f}")

    # Plot training curves
    plot_training_curves(pretrain_stats, train_stats, output_dir)

    # Save models
    save_models(flow, recognition_net, scaler, config, output_dir)

    # Generate samples and plot
    sample_and_visualize(flow, scaler, output_dir)

    return flow, recognition_net, scaler, (pretrain_stats, train_stats)


# ------ Utility Functions ------


def plot_training_curves(pretrain_stats, train_stats, output_dir):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot pretraining log likelihood
    ax1.plot(pretrain_stats["log_likelihood"], marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Log Likelihood")
    ax1.set_title("Flow Pretraining")
    ax1.grid(True, alpha=0.3)

    # Plot training ELBO
    ax2.plot(train_stats["elbo"], marker="o")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ELBO")
    ax2.set_title("Uncertainty-Aware Training")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=200)
    plt.close()


def save_models(flow, recognition_net, scaler, config, output_dir):
    """Save trained models and configuration."""
    model_path = os.path.join(output_dir, "3d_uncertainty_model.pt")
    torch.save(
        {
            "flow_state": flow.state_dict(),
            "recognition_state": recognition_net.state_dict(),
            "scaler": scaler,
            "config": config,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")


def sample_and_visualize(flow, scaler, output_dir, n_samples=5000):
    """Sample from the flow model and create visualizations."""
    # Set evaluation mode
    flow.eval()

    # Sample from the flow
    with torch.no_grad():
        samples = flow.sample(n_samples).cpu().numpy()

    # Inverse transform to get original scale
    samples_original = scaler.inverse_transform(samples)

    # Extract parameters
    log_ages = samples_original[:, 0]
    fehs = samples_original[:, 1]
    mgfes = samples_original[:, 2]

    # Convert log age to linear age
    ages = 10**log_ages

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Age vs [Fe/H]
    scatter1 = ax1.scatter(ages, fehs, s=5, alpha=0.5, c=mgfes, cmap="viridis")
    ax1.set_xlabel("Age (Gyr)")
    ax1.set_ylabel("[Fe/H]")
    ax1.set_title("Age vs [Fe/H] - Model Samples")
    ax1.set_xlim(14, 0)  # Reversed, oldest stars on left
    ax1.set_ylim(-1.5, 0.5)
    ax1.grid(True, alpha=0.3)

    # Plot Age vs [Mg/Fe]
    scatter2 = ax2.scatter(ages, mgfes, s=5, alpha=0.5, c=fehs, cmap="plasma")
    ax2.set_xlabel("Age (Gyr)")
    ax2.set_ylabel("[Mg/Fe]")
    ax2.set_title("Age vs [Mg/Fe] - Model Samples")
    ax2.set_xlim(14, 0)  # Reversed, oldest stars on left
    ax2.set_ylim(-0.2, 0.5)
    ax2.grid(True, alpha=0.3)

    # Plot [Fe/H] vs [Mg/Fe]
    scatter3 = ax3.scatter(fehs, mgfes, s=5, alpha=0.5, c=ages, cmap="viridis")
    fig.colorbar(scatter3, ax=ax3, label="Age (Gyr)")
    ax3.set_xlabel("[Fe/H]")
    ax3.set_ylabel("[Mg/Fe]")
    ax3.set_title("[Fe/H] vs [Mg/Fe] - Model Samples")
    ax3.set_xlim(-1.5, 0.5)
    ax3.set_ylim(-0.2, 0.5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_samples.png"), dpi=300)
    plt.close()


# ------ Main Script ------


def main():
    """Main function to run the 3D flow model training."""
    # Create directories
    os.makedirs("mock_data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load data
    try:
        data = np.load("mock_data/mock_data.npy")
        errors = np.load("mock_data/mock_errors.npy")
        print(f"Loaded mock data: {data.shape}, errors: {errors.shape}")
    except FileNotFoundError:
        print("Mock data not found. Generating new mock data...")
        from generate_mock_data import (
            generate_mock_stellar_data,
            save_mock_data,
            visualize_mock_data,
        )

        # Generate mock data
        data, errors = generate_mock_stellar_data(n_samples=5000)
        save_mock_data(data, errors)
        visualize_mock_data(data, errors)

    # Configuration for 3D flow model with sensible parameters
    config = {
        "pretraining_epochs": 15,  # Reduced from original code
        "training_epochs": 30,  # Reduced from original code
        "batch_size": 128,  # Reduced batch size for better stability
        "learning_rate": 5e-4,  # Lower learning rate
        "weight_decay": 1e-5,
        "flow_n_transforms": 8,  # Reduced transforms for simplicity
        "recognition_n_transforms": 6,  # Reduced transforms for simplicity
        "hidden_dim": 64,  # Smaller network
        "num_bins": 12,  # Fewer bins for smoother representation
        "mc_samples": 5,  # Fewer Monte Carlo samples
    }

    # Train model
    flow, recognition_net, scaler, stats = train_flow_with_uncertainty(
        data, errors, output_dir="results", config=config
    )

    print("Training and visualization complete!")


if __name__ == "__main__":
    # Make sure required packages are installed
    try:
        import nflows
    except ImportError:
        print("ERROR: This script requires nflows. Install it with:")
        print("pip install nflows")
        exit(1)

    main()
