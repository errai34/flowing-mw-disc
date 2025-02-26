import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add the src directory to the path so we can import the generate_high_error_data module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Try to import FITS support
try:
    from astropy.io import fits
except ImportError:
    print("Warning: astropy not installed. FITS file support is disabled.")
    fits = None

# Import the data generation module
try:
    from generate_high_error_data import (
        generate_mock_stellar_data,
        save_mock_data,
        visualize_mock_data,
    )
except ImportError:
    print("Warning: generate_high_error_data.py not found in the src directory.")

    # Define fallback functions
    def generate_mock_stellar_data(n_samples=5000, random_seed=42):
        """Fallback mock data generation function."""
        print("Using fallback data generation function.")
        np.random.seed(random_seed)

        # Simple mock data generation
        data = np.random.normal(size=(n_samples, 3))
        errors = 0.1 * np.ones((n_samples, 3))
        true_data = data.copy()

        return data, errors, true_data

    def save_mock_data(data, errors, true_data=None, output_dir="mock_data"):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "mock_data.npy"), data)
        np.save(os.path.join(output_dir, "mock_errors.npy"), errors)
        if true_data is not None:
            np.save(os.path.join(output_dir, "true_data.npy"), true_data)
        print(f"Saved mock data to {output_dir}/")

    def visualize_mock_data(data, errors, true_data=None, output_dir="mock_data"):
        print("Visualize function not available in fallback mode.")


# Set GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------ Data Loading Functions ------


def load_data_from_fits(data_path, bin_name=None):
    """Load data from FITS files."""
    if fits is None:
        raise ImportError("astropy.io.fits is required to load FITS files.")

    file_path = os.path.join(data_path, f"{bin_name}.fits" if bin_name else "data.fits")
    print(f"Loading FITS file: {file_path}")

    hdul = fits.open(file_path)
    data = hdul[1].data

    # Extract relevant columns - modify this based on your FITS structure
    log_ages = data["log_age"]
    fehs = data["feh"]
    mgfes = data["mgfe"]

    # Extract errors or use default if not available
    try:
        age_errors = data["log_age_err"]
        feh_errors = data["feh_err"]
        mgfe_errors = data["mgfe_err"]
    except KeyError:
        print("Warning: Error columns not found in FITS file. Using defaults.")
        age_errors = np.ones_like(log_ages) * 0.1
        feh_errors = np.ones_like(fehs) * 0.05
        mgfe_errors = np.ones_like(mgfes) * 0.03

    # Stack data and errors
    data_array = np.column_stack([log_ages, fehs, mgfes])
    errors_array = np.column_stack([age_errors, feh_errors, mgfe_errors])

    hdul.close()
    return data_array, errors_array


def load_data_from_npy(data_path, bin_name=None):
    """Load data from NumPy files."""
    data_file = os.path.join(data_path, "mock_data.npy")
    errors_file = os.path.join(data_path, "mock_errors.npy")

    if bin_name:
        data_file = os.path.join(data_path, f"{bin_name}_data.npy")
        errors_file = os.path.join(data_path, f"{bin_name}_errors.npy")

    print(f"Loading data from: {data_file}")
    data = np.load(data_file)
    errors = np.load(errors_file)

    return data, errors


def load_data(args):
    """Load data based on specified format."""
    if args.generate_mock:
        print("Generating mock data...")
        # Call the generate function with the specified number of samples
        data, errors, true_data = generate_mock_stellar_data(n_samples=args.n_samples)
        # Save the generated data to the specified output directory
        save_mock_data(data, errors, true_data, output_dir=args.mock_data_dir)
        # Visualize the generated data
        visualize_mock_data(data, errors, true_data, output_dir=args.mock_data_dir)
        return data, errors

    if args.data_format.lower() == "fits":
        return load_data_from_fits(args.data_path, args.bin_name)
    elif args.data_format.lower() == "npy":
        return load_data_from_npy(args.data_path, args.bin_name)
    else:
        raise ValueError(f"Unsupported data format: {args.data_format}")


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


class NeuralSplineFlow3D(nn.Module):
    """
    3D normalizing flow using neural spline transforms
    Better for capturing multi-modal distributions
    """

    def __init__(self, n_transforms=8, hidden_dim=64, num_bins=12):
        super().__init__()
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.nn.nets import ResidualNet
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.coupling import (
            PiecewiseRationalQuadraticCouplingTransform,
        )
        from nflows.transforms.permutations import ReversePermutation

        # Base distribution
        base_dist = StandardNormal(shape=[3])

        # Build transforms
        transforms = []
        for i in range(n_transforms):
            transforms.append(ReversePermutation(features=3))

            # Neural spline coupling transform
            transform_net = ResidualNet(
                in_features=3 // 2,
                out_features=(3 - 3 // 2) * num_bins * 3 + (3 - 3 // 2),
                hidden_features=hidden_dim,
                num_blocks=2,
                activation=F.relu,
            )

            coupling_transform = PiecewiseRationalQuadraticCouplingTransform(
                mask=torch.cat([torch.zeros(3 // 2), torch.ones(3 - 3 // 2)]),
                transform_net_create_fn=lambda in_features, out_features: transform_net,
                tails="linear",
                tail_bound=5.0,
                num_bins=num_bins,
            )

            transforms.append(coupling_transform)

        # Create flow model
        self.flow = Flow(
            transform=CompositeTransform(transforms), distribution=base_dist
        )

    def log_prob(self, x):
        return self.flow.log_prob(x)

    def sample(self, n):
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


class EnhancedRecognitionNetwork3D(nn.Module):
    """
    Enhanced recognition network with more sophisticated conditioning
    and more expressive transforms.
    """

    def __init__(self, n_transforms=6, hidden_dim=128):
        super().__init__()
        from nflows.distributions.normal import StandardNormal
        from nflows.transforms.autoregressive import (
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        )
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import ReversePermutation

        # Base distribution
        self.base_dist = StandardNormal(shape=[3])

        # Enhanced conditioning network - deeper and with more capacity
        self.conditioning_net = nn.Sequential(
            nn.Linear(3 * 2, hidden_dim),  # observed data + error info
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Create transforms with more expressive architecture
        transforms = []
        for i in range(n_transforms):
            transforms.append(ReversePermutation(features=3))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=3,
                    hidden_features=hidden_dim,
                    context_features=hidden_dim,
                    num_bins=16,  # More bins for more expressivity
                    tails="linear",
                    tail_bound=5.0,
                    use_residual_blocks=True,
                    activation=F.relu,
                )
            )

        self.transform = CompositeTransform(transforms)

    def forward(self, observed_data, uncertainties):
        """Process observed data and uncertainties to create context."""
        context_input = torch.cat([observed_data, uncertainties], dim=1)
        return self.conditioning_net(context_input)

    def sample(self, observed_data, uncertainties, n_samples=1):
        """Sample from q(v|w)."""
        batch_size = observed_data.shape[0]
        context = self.forward(observed_data, uncertainties)
        eps = self.base_dist.sample(batch_size * n_samples).reshape(
            batch_size * n_samples, -1
        )
        batched_context = context.repeat_interleave(n_samples, dim=0)
        samples, _ = self.transform.inverse(eps, batched_context)
        return samples

    def log_prob(self, latent_samples, observed_data, uncertainties):
        """Compute log q(v|w)."""
        context = self.forward(observed_data, uncertainties)
        noise, logabsdet = self.transform(latent_samples, context)
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


def uncertainty_aware_iwae(flow, recognition_net, observed_data, uncertainties, K=10):
    """
    Compute uncertainty-aware objective using importance weighting (IWAE).
    IWAE = log(1/K * sum_k[p(w|v_k) * p(v_k) / q(v_k|w)])
    This provides a tighter bound than standard ELBO.
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

    # Compute importance weights
    log_weights = log_p_w_given_v + log_p_v - log_q_v_given_w
    log_weights = log_weights.reshape(batch_size, K)

    # Compute IWAE objective (stabilized with log-sum-exp)
    max_log_weights = torch.max(log_weights, dim=1, keepdim=True)[0]
    iwae = max_log_weights + torch.log(
        torch.mean(torch.exp(log_weights - max_log_weights), dim=1)
    )

    return torch.mean(iwae)


# ------ Training Functions ------


def pretrain_flow(data, args):
    """Pretrain flow model using maximum likelihood on data."""
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

    # Initialize flow model with customizable configuration
    if args.use_neural_spline:
        flow = NeuralSplineFlow3D(
            n_transforms=args.n_transforms,
            hidden_dim=args.hidden_dim,
            num_bins=args.num_bins,
        ).to(device)
    else:
        flow = Flow3D(
            n_transforms=args.n_transforms,
            hidden_dim=args.hidden_dim,
            num_bins=args.num_bins,
        ).to(device)

    # Optimizer
    optimizer = optim.Adam(
        flow.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Training loop
    train_stats = {"log_likelihood": []}
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
            optimizer.step()

            epoch_lls.append(-loss.item())

        # Track stats
        avg_ll = np.mean(epoch_lls)
        train_stats["log_likelihood"].append(avg_ll)
        pbar.set_postfix({"LL": f"{avg_ll:.4f}"})

    print(f"Pretraining complete. Final Log-Likelihood: {avg_ll:.4f}")
    return flow, scaler, train_stats


def train_flow_with_uncertainty(data, errors, args):
    """
    Train 3D flow model with uncertainty-aware approach using user-specified parameters.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1: Pretrain the flow model
    flow, scaler, pretrain_stats = pretrain_flow(data=data, args=args)

    # Scale data for main training
    data_scaled = scaler.transform(data)
    errors_scaled = errors / scaler.scale_

    # Convert to tensors
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)
    errors_tensor = torch.tensor(errors_scaled, dtype=torch.float32).to(device)

    # Create dataset and loader
    dataset = TensorDataset(data_tensor, errors_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize recognition network with customizable configuration
    if args.use_enhanced_recognition:
        recognition_net = EnhancedRecognitionNetwork3D(
            n_transforms=args.recognition_n_transforms, hidden_dim=args.hidden_dim
        ).to(device)
    else:
        recognition_net = RecognitionNetwork3D(
            n_transforms=args.recognition_n_transforms, hidden_dim=args.hidden_dim
        ).to(device)

    # Stage 2: Train with uncertainty-aware ELBO
    print(
        f"\n=== Stage 2: Training with uncertainty-aware ELBO for {args.epochs} epochs ==="
    )

    # Optimizer
    optimizer = optim.Adam(
        list(flow.parameters()) + list(recognition_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Temperature annealing settings
    initial_temp = 1.0
    final_temp = 0.1 if args.temperature_annealing else 1.0
    temp_scheduler = np.linspace(initial_temp, final_temp, args.epochs)

    # Training stats
    train_stats = {"elbo": []}
    best_elbo = -float("inf")
    best_state = {"flow": None, "recognition": None}

    # Main training loop
    pbar = tqdm(
        range(args.epochs),
        desc="Training with ELBO" if not args.use_iwae else "Training with IWAE",
    )
    for epoch in pbar:
        flow.train()
        recognition_net.train()

        # Current temperature for annealing
        temperature = temp_scheduler[epoch]
        epoch_elbo_values = []

        for batch_data, batch_errors in loader:
            optimizer.zero_grad()

            # Calculate objective based on chosen method
            if args.use_iwae:
                objective = uncertainty_aware_iwae(
                    flow, recognition_net, batch_data, batch_errors, K=args.mc_samples
                )
            else:
                objective = uncertainty_aware_elbo(
                    flow, recognition_net, batch_data, batch_errors, K=args.mc_samples
                )

            # Apply temperature if annealing is enabled
            if args.temperature_annealing:
                objective = objective * temperature

            batch_obj = objective.item()
            epoch_elbo_values.append(batch_obj)

            # Backward pass
            loss = -objective
            loss.backward()
            optimizer.step()

        # Record epoch stats
        epoch_avg_elbo = np.mean(epoch_elbo_values)
        train_stats["elbo"].append(epoch_avg_elbo)
        pbar.set_postfix(
            {"Objective": f"{epoch_avg_elbo:.4f}", "Temp": f"{temperature:.3f}"}
        )

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
    plot_training_curves(pretrain_stats, train_stats, args.output_dir)

    # Save models
    save_models(flow, recognition_net, scaler, vars(args), args.output_dir)

    # Generate samples and plot
    sample_and_visualize(flow, scaler, args.output_dir, n_samples=args.n_samples)

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
    ax2.set_ylabel("Objective")
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
    fig.colorbar(scatter1, ax=ax1, label="[Mg/Fe]")
    ax1.set_xlabel("Age (Gyr)")
    ax1.set_ylabel("[Fe/H]")
    ax1.set_title("Age vs [Fe/H] - Model Samples")
    ax1.set_xlim(20, 0)  # Reversed, oldest stars on left
    ax1.set_ylim(-1.0, 0.5)
    ax1.grid(True, alpha=0.3)

    # Plot Age vs [Mg/Fe]
    scatter2 = ax2.scatter(ages, mgfes, s=5, alpha=0.5, c=fehs, cmap="plasma")
    fig.colorbar(scatter2, ax=ax2, label="[Fe/H]")
    ax2.set_xlabel("Age (Gyr)")
    ax2.set_ylabel("[Mg/Fe]")
    ax2.set_title("Age vs [Mg/Fe] - Model Samples")
    ax2.set_xlim(20, 0)  # Reversed, oldest stars on left
    ax2.set_ylim(-0.2, 0.5)
    ax2.grid(True, alpha=0.3)

    # Plot [Fe/H] vs [Mg/Fe]
    scatter3 = ax3.scatter(fehs, mgfes, s=5, alpha=0.5, c=ages, cmap="viridis")
    fig.colorbar(scatter3, ax=ax3, label="Age (Gyr)")
    ax3.set_xlabel("[Fe/H]")
    ax3.set_ylabel("[Mg/Fe]")
    ax3.set_title("[Fe/H] vs [Mg/Fe] - Model Samples")
    ax3.set_xlim(-1.0, 0.5)
    ax3.set_ylim(-0.2, 0.5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_samples.png"), dpi=300)
    plt.close()

    # NEW: KDE Comparison between model samples, true data, and noisy data
    try:
        # Load true and noisy data for comparison
        mock_data_dir = (
            os.path.dirname(output_dir) if os.path.dirname(output_dir) else "."
        )
        mock_data_dir = os.path.join(mock_data_dir, "mock_data")

        true_data_path = os.path.join(mock_data_dir, "true_data.npy")
        noisy_data_path = os.path.join(mock_data_dir, "mock_data.npy")

        if os.path.exists(true_data_path) and os.path.exists(noisy_data_path):
            # Load data
            true_data = np.load(true_data_path)
            noisy_data = np.load(noisy_data_path)

            # Handle case where true_data has starburst flag (4th column)
            if true_data.shape[1] > 3:
                true_data = true_data[:, :3]

            # Create KDE plot figure
            import seaborn as sns

            plt.figure(figsize=(18, 6))

            # KDE for log age
            plt.subplot(1, 3, 1)
            sns.kdeplot(
                true_data[:, 0], color="blue", label="True Data", fill=True, alpha=0.2
            )
            sns.kdeplot(
                noisy_data[:, 0], color="red", label="Noisy Data", fill=True, alpha=0.2
            )
            sns.kdeplot(
                log_ages, color="green", label="Flow Model", fill=True, alpha=0.2
            )
            plt.xlabel("Log Age")
            plt.ylabel("Density")
            plt.title("KDE: Log Age Comparison")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # KDE for [Fe/H]
            plt.subplot(1, 3, 2)
            sns.kdeplot(
                true_data[:, 1], color="blue", label="True Data", fill=True, alpha=0.2
            )
            sns.kdeplot(
                noisy_data[:, 1], color="red", label="Noisy Data", fill=True, alpha=0.2
            )
            sns.kdeplot(fehs, color="green", label="Flow Model", fill=True, alpha=0.2)
            plt.xlabel("[Fe/H]")
            plt.ylabel("Density")
            plt.title("KDE: [Fe/H] Comparison")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # KDE for [Mg/Fe]
            plt.subplot(1, 3, 3)
            sns.kdeplot(
                true_data[:, 2], color="blue", label="True Data", fill=True, alpha=0.2
            )
            sns.kdeplot(
                noisy_data[:, 2], color="red", label="Noisy Data", fill=True, alpha=0.2
            )
            sns.kdeplot(mgfes, color="green", label="Flow Model", fill=True, alpha=0.2)
            plt.xlabel("[Mg/Fe]")
            plt.ylabel("Density")
            plt.title("KDE: [Mg/Fe] Comparison")
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "kde_model_comparison.png"), dpi=300)
            plt.close()

            # Additional 2D KDE comparisons (Age vs [Fe/H])
            plt.figure(figsize=(18, 5))

            # Convert true and noisy log ages to linear
            true_ages = 10 ** true_data[:, 0]
            noisy_ages = 10 ** noisy_data[:, 0]

            # True data
            plt.subplot(1, 3, 1)
            sns.kdeplot(
                x=true_ages,
                y=true_data[:, 1],
                cmap="Blues",
                fill=True,
                alpha=0.7,
                levels=10,
            )
            plt.xlabel("Age (Gyr)")
            plt.ylabel("[Fe/H]")
            plt.title("True Data")
            plt.xlim(20, 0)  # Reversed x-axis
            plt.ylim(-1.0, 0.5)
            plt.grid(True, alpha=0.3)

            # Noisy data
            plt.subplot(1, 3, 2)
            sns.kdeplot(
                x=noisy_ages,
                y=noisy_data[:, 1],
                cmap="Reds",
                fill=True,
                alpha=0.7,
                levels=10,
            )
            plt.xlabel("Age (Gyr)")
            plt.ylabel("[Fe/H]")
            plt.title("Noisy Data")
            plt.xlim(20, 0)  # Reversed x-axis
            plt.ylim(-1.0, 0.5)
            plt.grid(True, alpha=0.3)

            # Model samples
            plt.subplot(1, 3, 3)
            sns.kdeplot(x=ages, y=fehs, cmap="Greens", fill=True, alpha=0.7, levels=10)
            plt.xlabel("Age (Gyr)")
            plt.ylabel("[Fe/H]")
            plt.title("Flow Model")
            plt.xlim(20, 0)  # Reversed x-axis
            plt.ylim(-1.0, 0.5)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "2d_kde_model_comparison.png"), dpi=300
            )
            plt.close()

            print("KDE comparison visualizations created!")

    except Exception as e:
        print(f"Could not create KDE comparison: {e}")


# ------ Command-line Interface ------


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a normalizing flow model with uncertainty-awareness."
    )

    # Data options
    parser.add_argument(
        "--data_path", type=str, default="mock_data", help="Path to data directory"
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="npy",
        choices=["npy", "fits"],
        help="Format of input data",
    )
    parser.add_argument(
        "--bin_name", type=str, default=None, help="Name of the bin for FITS files"
    )
    parser.add_argument(
        "--generate_mock",
        action="store_true",
        help="Generate mock data instead of loading",
    )
    parser.add_argument(
        "--mock_data_dir",
        type=str,
        default="mock_data",
        help="Directory to save generated mock data",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of samples for mock data generation or visualization",
    )

    # Model architecture
    parser.add_argument(
        "--n_transforms",
        type=int,
        default=8,
        help="Number of transforms in the flow model",
    )
    parser.add_argument(
        "--recognition_n_transforms",
        type=int,
        default=6,
        help="Number of transforms in the recognition network",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension for both networks"
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=12,
        help="Number of bins for piecewise transforms",
    )

    # Training parameters
    parser.add_argument(
        "--pretraining_epochs",
        type=int,
        default=15,
        help="Number of epochs for flow pretraining",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs for full training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=5,
        help="Number of Monte Carlo samples for ELBO estimation",
    )

    # Output options
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save outputs"
    )

    # New arguments based on the paper
    parser.add_argument(
        "--use_iwae",
        action="store_true",
        help="Use Importance Weighted Autoencoder objective",
    )
    parser.add_argument(
        "--use_neural_spline",
        action="store_true",
        help="Use Neural Spline Flows instead of MAF",
    )
    parser.add_argument(
        "--use_enhanced_recognition",
        action="store_true",
        help="Use enhanced recognition network with deeper conditioning",
    )
    parser.add_argument(
        "--temperature_annealing",
        action="store_true",
        help="Use temperature annealing during training",
    )

    return parser.parse_args()


# ------ Main Function ------


def main():
    """Main function to run the 3D flow model training."""
    # Parse command-line arguments
    args = parse_args()

    # Print configuration
    print("\n=== Configuration ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Check for nflows package
    try:
        import nflows

        print("Using nflows version!")
    except ImportError:
        print("ERROR: This script requires nflows. Install it with:")
        print("pip install nflows")
        exit(1)

    # Load or generate data
    data, errors = load_data(args)
    print(f"Data shape: {data.shape}, Errors shape: {errors.shape}")

    # Train model
    flow, recognition_net, scaler, stats = train_flow_with_uncertainty(
        data, errors, args
    )

    print(f"Training and visualization complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
