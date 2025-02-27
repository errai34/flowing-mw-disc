import argparse
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


class FlowModel(nn.Module):
    """
    Normalizing flow model with configurable dimensions and architecture
    """

    def __init__(self, dim=2, n_transforms=8, hidden_dim=64, num_bins=12):
        super().__init__()
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.transforms.autoregressive import (
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        )
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import ReversePermutation

        # Base distribution (standard normal with specified dimension)
        base_dist = StandardNormal(shape=[dim])

        # Build transforms
        transforms = []
        for i in range(n_transforms):
            transforms.append(ReversePermutation(features=dim))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
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


class NeuralSplineFlow(nn.Module):
    """
    Neural spline flow using coupling layers - more suitable for multi-modal distributions
    """

    def __init__(self, dim=2, n_transforms=8, hidden_dim=64, num_bins=12):
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
        base_dist = StandardNormal(shape=[dim])

        # Build transforms
        transforms = []
        for i in range(n_transforms):
            transforms.append(ReversePermutation(features=dim))

            # Neural spline coupling transform
            mask = torch.zeros(dim)
            mask[dim // 2 :] = 1

            transform_net = ResidualNet(
                in_features=dim // 2,
                out_features=(dim - dim // 2) * num_bins * 3 + (dim - dim // 2),
                hidden_features=hidden_dim,
                num_blocks=2,
                activation=F.relu,
            )

            coupling_transform = PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
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


class RecognitionNetwork(nn.Module):
    """
    Recognition network for amortized variational inference.
    Implements q(v|w) for data with configurable dimensions.
    """

    def __init__(self, dim=2, n_transforms=6, hidden_dim=64):
        super().__init__()
        from nflows.distributions.normal import StandardNormal
        from nflows.transforms.autoregressive import (
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        )
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import ReversePermutation

        # Base distribution with specified dimension
        self.base_dist = StandardNormal(shape=[dim])

        # Create conditioning network (takes observed data + error info)
        self.conditioning_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),  # observed data + error info
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Create transforms
        transforms = []
        for i in range(n_transforms):
            transforms.append(ReversePermutation(features=dim))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
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


class EnhancedRecognitionNetwork(nn.Module):
    """
    Enhanced recognition network with more sophisticated conditioning
    and more expressive transforms.
    """

    def __init__(self, dim=2, n_transforms=6, hidden_dim=128):
        super().__init__()
        from nflows.distributions.normal import StandardNormal
        from nflows.transforms.autoregressive import (
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        )
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import ReversePermutation

        # Base distribution
        self.base_dist = StandardNormal(shape=[dim])

        # Enhanced conditioning network - deeper and with more capacity
        self.conditioning_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),  # observed data + error info
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Create transforms with more expressive architecture
        transforms = []
        for i in range(n_transforms):
            transforms.append(ReversePermutation(features=dim))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
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


# ------ Data Loading Functions ------


def load_gaussian_mixture_data(data_path="mixture_gaussian_data"):
    """
    Load the 2D Gaussian mixture data.
    """
    # Try to load train split first (recommended)
    try:
        train_v = np.load(os.path.join(data_path, "train", "v.npy"))
        train_w = np.load(os.path.join(data_path, "train", "w.npy"))
        print(
            f"Loaded training data from splits: clean shape {train_v.shape}, noisy shape {train_w.shape}"
        )
        return train_v, train_w
    except (FileNotFoundError, OSError):
        pass

    # If train split not found, try to load full dataset
    try:
        clean_data = np.load(os.path.join(data_path, "clean_data.npy"))
        noisy_data = np.load(os.path.join(data_path, "noisy_data.npy"))
        print(
            f"Loaded full dataset: clean shape {clean_data.shape}, noisy shape {noisy_data.shape}"
        )
        return clean_data, noisy_data
    except (FileNotFoundError, OSError):
        pass

    # If no data found, raise error
    raise FileNotFoundError(
        f"Could not find Gaussian mixture data in {data_path}. Please run generate_mixture_gaussians.py first."
    )


def load_stellar_data(data_path, data_format="npy", bin_name=None):
    """
    Load stellar data in either npy or fits format.
    """
    if data_format.lower() == "fits":
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError("astropy.io.fits is required to load FITS files.")

        file_path = os.path.join(
            data_path, f"{bin_name}.fits" if bin_name else "data.fits"
        )
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

    elif data_format.lower() == "npy":
        data_file = os.path.join(data_path, "mock_data.npy")
        errors_file = os.path.join(data_path, "mock_errors.npy")

        if bin_name:
            data_file = os.path.join(data_path, f"{bin_name}_data.npy")
            errors_file = os.path.join(data_path, f"{bin_name}_errors.npy")

        print(f"Loading data from: {data_file}")
        data = np.load(data_file)
        errors = np.load(errors_file)

        return data, errors

    else:
        raise ValueError(f"Unsupported data format: {data_format}")


def load_data(args):
    """
    Generic data loading function based on dataset type.
    """
    if args.dataset_type == "gaussian_mixture":
        # Load 2D Gaussian mixture data
        clean_data, noisy_data = load_gaussian_mixture_data(args.data_path)

        # Create uncertainty estimates from the known noise model
        if args.use_known_noise:
            # For Gaussian mixture, we know the noise covariance is [0.1, 0; 0, 1.0]
            uncertainties = np.ones_like(noisy_data)
            uncertainties[:, 0] = np.sqrt(0.1)
            uncertainties[:, 1] = 1.0
            print("Using known noise model for uncertainties")
        else:
            # If we don't want to use the known noise, estimate it
            # (e.g., as standard deviation of w-v, or some fixed proportion)
            uncertainties = np.std(noisy_data - clean_data, axis=0) * np.ones_like(
                noisy_data
            )
            print(f"Estimated uncertainties: {np.mean(uncertainties, axis=0)}")

        return clean_data, noisy_data, uncertainties

    elif args.dataset_type == "stellar":
        # Load 3D stellar data
        data, errors = load_stellar_data(
            args.data_path, args.data_format, args.bin_name
        )
        return (
            data,
            data,
            errors,
        )  # Return data as both clean and noisy (we don't have actual clean)

    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")


# ------ Training Functions ------


def pretrain_flow(data, dim, args):
    """
    Pretrain flow model using maximum likelihood on data.
    """
    print(
        f"\n=== Stage 1: Pretraining {dim}D flow model for {args.pretraining_epochs} epochs ==="
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
        flow = NeuralSplineFlow(
            dim=dim,
            n_transforms=args.n_transforms,
            hidden_dim=args.hidden_dim,
            num_bins=args.num_bins,
        ).to(device)
    else:
        flow = FlowModel(
            dim=dim,
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


def train_flow_with_uncertainty(clean_data, noisy_data, uncertainties, args):
    """
    Train flow model with uncertainty-aware approach.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # Get data dimension
    dim = noisy_data.shape[1]

    # Stage 1: Pretrain the flow model on clean data (if available)
    # For real-world data where clean data is unavailable, we'd use noisy data
    pretrain_data = clean_data if args.use_clean_for_pretraining else noisy_data
    flow, scaler, pretrain_stats = pretrain_flow(data=pretrain_data, dim=dim, args=args)

    # Scale data for main training
    clean_data_scaled = scaler.transform(clean_data)
    noisy_data_scaled = scaler.transform(noisy_data)

    # Scale uncertainties
    if uncertainties is not None:
        uncertainties_scaled = uncertainties / scaler.scale_
    else:
        # If uncertainties not provided, use a default
        uncertainties_scaled = np.ones_like(noisy_data_scaled) * 0.1

    # Convert to tensors
    noisy_data_tensor = torch.tensor(noisy_data_scaled, dtype=torch.float32).to(device)
    uncertainties_tensor = torch.tensor(uncertainties_scaled, dtype=torch.float32).to(
        device
    )

    # Create dataset and loader
    dataset = TensorDataset(noisy_data_tensor, uncertainties_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize recognition network with customizable configuration
    if args.use_enhanced_recognition:
        recognition_net = EnhancedRecognitionNetwork(
            dim=dim,
            n_transforms=args.recognition_n_transforms,
            hidden_dim=args.hidden_dim,
        ).to(device)
    else:
        recognition_net = RecognitionNetwork(
            dim=dim,
            n_transforms=args.recognition_n_transforms,
            hidden_dim=args.hidden_dim,
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

    # Temperature annealing settings
    initial_temp = 1.0
    final_temp = 0.1 if args.temperature_annealing else 1.0
    temp_scheduler = np.linspace(initial_temp, final_temp, args.epochs)

    # Training stats
    train_stats = {"objective": []}
    best_objective = -float("inf")
    best_state = {"flow": None, "recognition": None}

    # Main training loop
    objective_name = "IWAE" if args.use_iwae else "ELBO"
    pbar = tqdm(range(args.epochs), desc=f"Training with {objective_name}")

    for epoch in pbar:
        flow.train()
        recognition_net.train()

        # Current temperature for annealing
        temperature = temp_scheduler[epoch]
        epoch_objectives = []

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
            epoch_objectives.append(batch_obj)

            # Backward pass
            loss = -objective
            loss.backward()
            optimizer.step()

        # Record epoch stats
        epoch_avg_obj = np.mean(epoch_objectives)
        train_stats["objective"].append(epoch_avg_obj)
        pbar.set_postfix(
            {objective_name: f"{epoch_avg_obj:.4f}", "Temp": f"{temperature:.3f}"}
        )

        # Save best model
        if epoch_avg_obj > best_objective:
            best_objective = epoch_avg_obj
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

    print(f"Training complete. Best {objective_name}: {best_objective:.4f}")

    # Plot training curves
    plot_training_curves(pretrain_stats, train_stats, args.output_dir, objective_name)

    # Save models
    save_models(flow, recognition_net, scaler, dim, vars(args), args.output_dir)

    # Generate samples and visualize
    if dim == 2:  # Only create gaussian mixture visualizations for 2D data
        sample_and_visualize_2d_gaussian(
            flow,
            recognition_net,
            scaler,
            clean_data_scaled,
            noisy_data_scaled,
            uncertainties_scaled,
            args.output_dir,
            n_samples=args.n_samples,
        )
    elif dim == 3:  # Create 3D stellar data visualizations
        sample_and_visualize_3d_stellar(
            flow, scaler, args.output_dir, n_samples=args.n_samples
        )

    return flow, recognition_net, scaler, (pretrain_stats, train_stats)


# ------ Visualization Functions ------


def plot_training_curves(
    pretrain_stats, train_stats, output_dir, objective_name="ELBO"
):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot pretraining log likelihood
    ax1.plot(pretrain_stats["log_likelihood"], marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Log Likelihood")
    ax1.set_title("Flow Pretraining")
    ax1.grid(True, alpha=0.3)

    # Plot training objective
    ax2.plot(train_stats["objective"], marker="o")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(f"{objective_name} Value")
    ax2.set_title(f"Uncertainty-Aware Training ({objective_name})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=200)
    plt.close()


def save_models(flow, recognition_net, scaler, dim, config, output_dir):
    """Save trained models and configuration."""
    model_path = os.path.join(output_dir, f"{dim}d_deconvolution_model.pt")
    torch.save(
        {
            "flow_state": flow.state_dict(),
            "recognition_state": recognition_net.state_dict(),
            "scaler": scaler,
            "dim": dim,
            "config": config,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")


def sample_and_visualize_2d_gaussian(
    flow,
    recognition_net,
    scaler,
    clean_data,
    noisy_data,
    uncertainties,
    output_dir,
    n_samples=5000,
):
    """
    Generate visualizations for 2D Gaussian mixture data.
    Uses a similar style to the generate_mixture_gaussians.py file.
    """
    # Import necessary libraries
    import seaborn as sns
    from sklearn.neighbors import KernelDensity

    # Set evaluation mode
    flow.eval()
    recognition_net.eval()

    # Sample from the prior
    with torch.no_grad():
        prior_samples = flow.sample(n_samples).cpu().numpy()

        # Choose some examples from the noisy data to get posterior samples
        n_examples = min(5, len(noisy_data))
        example_indices = np.random.choice(len(noisy_data), n_examples, replace=False)
        example_noisy = torch.tensor(
            noisy_data[example_indices], dtype=torch.float32
        ).to(device)
        example_uncertainties = torch.tensor(
            uncertainties[example_indices], dtype=torch.float32
        ).to(device)

        # Generate posterior samples for each example
        posterior_samples = []
        for i in range(n_examples):
            w_single = example_noisy[i : i + 1]
            e_single = example_uncertainties[i : i + 1]
            # Sample from q(v|w)
            v_samples = (
                recognition_net.sample(w_single, e_single, n_samples=1000).cpu().numpy()
            )
            posterior_samples.append((w_single.cpu().numpy()[0], v_samples))

    # Inverse transform to get original scale
    prior_samples_original = scaler.inverse_transform(prior_samples)

    # Create high-quality visualizations similar to generate_mixture_gaussians.py

    # 1. Prior Distribution
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create 2D histogram with contour lines
    sns.histplot(
        x=prior_samples_original[:, 0],
        y=prior_samples_original[:, 1],
        bins=50,
        pthresh=0.02,
        cmap="Blues",
        ax=ax,
        cbar=True,
        stat="density",
    )

    # Add contour lines
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    values = np.vstack([prior_samples_original[:, 0], prior_samples_original[:, 1]]).T

    # Fit KDE and get contour lines
    kernel = KernelDensity(bandwidth=0.2).fit(values)
    zz = np.exp(kernel.score_samples(positions))
    zz = np.reshape(zz, xx.shape)
    ax.contour(xx, yy, zz, levels=4, colors="black", alpha=0.7)

    ax.set_title("Learned Prior Distribution p(v)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prior_distribution.png"), dpi=300)
    plt.close()

    # 2. Compare deconvolved distribution (prior samples) with noisy data
    try:
        # Load a sample of original noisy data for comparison

        sample_size = min(5000, len(noisy_data))
        random_indices = np.random.choice(len(noisy_data), sample_size, replace=False)
        noisy_sample = noisy_data[random_indices]
        noisy_sample_original = scaler.inverse_transform(noisy_sample)

        # Create comparison plot

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Prior distribution (left)
        sns.histplot(
            x=prior_samples_original[:, 0],
            y=prior_samples_original[:, 1],
            bins=50,
            pthresh=0.02,
            cmap="Blues",
            ax=ax1,
            cbar=True,
            stat="density",
        )

        # Add contour lines to prior
        xmin, xmax = ax1.get_xlim()
        ymin, ymax = ax1.get_ylim()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()]).T
        values = np.vstack(
            [prior_samples_original[:, 0], prior_samples_original[:, 1]]
        ).T
        kernel = KernelDensity(bandwidth=0.2).fit(values)
        zz = np.exp(kernel.score_samples(positions))
        zz = np.reshape(zz, xx.shape)
        ax1.contour(xx, yy, zz, levels=4, colors="black", alpha=0.7)

        ax1.set_title("Deconvolved Distribution p(v)")
        ax1.set_xlabel("Dimension 1")
        ax1.set_ylabel("Dimension 2")

        # Noisy data (right)
        sns.histplot(
            x=noisy_sample_original[:, 0],
            y=noisy_sample_original[:, 1],
            bins=50,
            pthresh=0.02,
            cmap="Reds",
            ax=ax2,
            cbar=True,
            stat="density",
        )

        # Add contour lines to noisy data
        xmin, xmax = ax2.get_xlim()
        ymin, ymax = ax2.get_ylim()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()]).T
        values = np.vstack([noisy_sample_original[:, 0], noisy_sample_original[:, 1]]).T
        kernel = KernelDensity(bandwidth=0.2).fit(values)
        zz = np.exp(kernel.score_samples(positions))
        zz = np.reshape(zz, xx.shape)
        ax2.contour(xx, yy, zz, levels=4, colors="black", alpha=0.7)

        ax2.set_title("Observed Data p(w)")
        ax2.set_xlabel("Dimension 1")
        ax2.set_ylabel("Dimension 2")
        # Ensure both plots have the same axes limits
        xlims = [
            min(ax1.get_xlim()[0], ax2.get_xlim()[0]),
            max(ax1.get_xlim()[1], ax2.get_xlim()[1]),
        ]
        ylims = [
            min(ax1.get_ylim()[0], ax2.get_ylim()[0]),
            max(ax1.get_ylim()[1], ax2.get_ylim()[1]),
        ]
        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims)
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "deconvolution_comparison.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")

    # # 3. Posterior visualizations for specific observations
    # for i, (w, v_samples) in enumerate(posterior_samples):
    #     # Inverse transform to original scale
    #     w_original = scaler.inverse_transform(w.reshape(1, -1))[0]
    #     v_samples_original = scaler.inverse_transform(v_samples)

    #     # Create plot for posterior distribution
    #     fig, ax = plt.subplots(figsize=(10, 8))

    #     # Plot posterior samples histogram
    #     sns.histplot(
    #         x=v_samples_original[:, 0],
    #         y=v_samples_original[:, 1],
    #         bins=40,
    #         pthresh=0.02,
    #         cmap="Greens",
    #         ax=ax,
    #         cbar=True,
    #         stat="density",
    #     )

    #     # Add contour lines
    #     xmin, xmax = ax.get_xlim()
    #     ymin, ymax = ax.get_ylim()
    #     xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    #     positions = np.vstack([xx.ravel(), yy.ravel()]).T
    #     values = np.vstack([v_samples_original[:, 0], v_samples_original[:, 1]]).T

    #     try:
    #         kernel = KernelDensity(bandwidth=0.2).fit(values)
    #         zz = np.exp(kernel.score_samples(positions))
    #         zz = np.reshape(zz, xx.shape)
    #         ax.contour(xx, yy, zz, levels=4, colors="black", alpha=0.7)
    #     except Exception as e:
    #         print(f"Error creating contour plot for posterior {i+1}: {e}")

    #     # Mark the observed point
    #     ax.scatter(
    #         w_original[0],
    #         w_original[1],
    #         s=150,
    #         marker="x",
    #         c="red",
    #         linewidths=2,
    #         label="Observed Data Point",
    #     )

    #     # Add 1-sigma noise level ellipse around the observation
    #     noise_1sigma = scaler.scale_ * np.sqrt(
    #         np.array([0.1, 1.0])
    #     )  # Based on known noise model
    #     ellipse = plt.matplotlib.patches.Ellipse(
    #         xy=w_original,
    #         width=2 * noise_1sigma[0],
    #         height=2 * noise_1sigma[1],
    #         fill=False,
    #         edgecolor="red",
    #         linestyle="--",
    #         label="1-σ Noise Level",
    #     )
    #     ax.add_patch(ellipse)

    #     ax.set_title(f"Posterior Distribution p(v|w) for Observation {i+1}")
    #     ax.set_xlabel("Dimension 1")
    #     ax.set_ylabel("Dimension 2")
    #     ax.legend(loc="upper right")

    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, f"posterior_{i+1}.png"), dpi=300)
    #     plt.close()

    # 3. Try to load original clean data for a three-way comparison if available
    try:
        clean_sample = clean_data[random_indices]
        clean_sample_original = scaler.inverse_transform(clean_sample)

        # Create a three-way comparison: original clean, noisy observed, and deconvolved
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

        # Original clean data (left)
        sns.histplot(
            x=clean_sample_original[:, 0],
            y=clean_sample_original[:, 1],
            bins=50,
            pthresh=0.02,
            cmap="Greens",
            ax=ax1,
            cbar=True,
            stat="density",
        )

        # Add contour lines
        xmin, xmax = ax1.get_xlim()
        ymin, ymax = ax1.get_ylim()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()]).T
        values = np.vstack([clean_sample_original[:, 0], clean_sample_original[:, 1]]).T
        kernel = KernelDensity(bandwidth=0.2).fit(values)
        zz = np.exp(kernel.score_samples(positions))
        zz = np.reshape(zz, xx.shape)
        ax1.contour(xx, yy, zz, levels=4, colors="black", alpha=0.7)

        ax1.set_title("Original Clean Data p(v)")
        ax1.set_xlabel("Dimension 1")
        ax1.set_ylabel("Dimension 2")

        sns.histplot(
            x=noisy_sample_original[:, 0],
            y=noisy_sample_original[:, 1],
            bins=50,
            pthresh=0.02,
            cmap="Reds",
            ax=ax2,
            cbar=True,
            stat="density",
        )

        # Add contour lines
        xmin, xmax = ax2.get_xlim()
        ymin, ymax = ax2.get_ylim()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()]).T
        values = np.vstack([noisy_sample_original[:, 0], noisy_sample_original[:, 1]]).T
        kernel = KernelDensity(bandwidth=0.2).fit(values)
        zz = np.exp(kernel.score_samples(positions))
        zz = np.reshape(zz, xx.shape)
        ax2.contour(xx, yy, zz, levels=4, colors="black", alpha=0.7)

        ax2.set_title("Observed Noisy Data p(w)")
        ax2.set_xlabel("Dimension 1")
        ax2.set_ylabel("Dimension 2")

        # Deconvolved distribution (right)
        sns.histplot(
            x=prior_samples_original[:, 0],
            y=prior_samples_original[:, 1],
            bins=50,
            pthresh=0.02,
            cmap="Blues",
            ax=ax3,
            cbar=True,
            stat="density",
        )

        # Add contour lines
        xmin, xmax = ax3.get_xlim()
        ymin, ymax = ax3.get_ylim()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()]).T
        values = np.vstack(
            [prior_samples_original[:, 0], prior_samples_original[:, 1]]
        ).T
        kernel = KernelDensity(bandwidth=0.2).fit(values)
        zz = np.exp(kernel.score_samples(positions))
        zz = np.reshape(zz, xx.shape)
        ax3.contour(xx, yy, zz, levels=4, colors="black", alpha=0.7)

        xlims = [
            min(ax1.get_xlim()[0], ax2.get_xlim()[0], ax3.get_xlim()[0]),
            max(ax1.get_xlim()[1], ax2.get_xlim()[1], ax3.get_xlim()[1]),
        ]
        ylims = [
            min(ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0]),
            max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1]),
        ]
        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims)
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)
        ax3.set_xlim(xlims)
        ax3.set_ylim(ylims)

        ax3.set_title("Deconvolved Distribution p(v)")
        ax3.set_xlabel("Dimension 1")
        ax3.set_ylabel("Dimension 2")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "three_way_comparison.png"), dpi=300)
        plt.close()

        print("Created three-way comparison with original clean data.")
    except Exception as e:
        print(f"Could not create three-way comparison: {e}")


def sample_and_visualize_3d_stellar(flow, scaler, output_dir, n_samples=5000):
    """
    Generate visualizations for 3D stellar data.
    """
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


# ------ Command-line Interface ------


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a normalizing flow model with uncertainty-awareness for data deconvolution."
    )

    # Dataset options
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="gaussian_mixture",
        choices=["gaussian_mixture", "stellar"],
        help="Type of dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="mixture_gaussian_data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="npy",
        choices=["npy", "fits"],
        help="Format of input data (for stellar data)",
    )
    parser.add_argument(
        "--bin_name",
        type=str,
        default=None,
        help="Name of the bin for FITS files (for stellar data)",
    )
    parser.add_argument(
        "--use_known_noise",
        action="store_true",
        help="Use known noise model for uncertainties (for gaussian mixture data)",
    )
    parser.add_argument(
        "--use_clean_for_pretraining",
        action="store_true",
        help="Use clean data for pretraining the flow (if available)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of samples for visualization",
    )

    # Model architecture
    parser.add_argument(
        "--n_transforms",
        type=int,
        default=4,
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
        "--epochs", type=int, default=10, help="Number of epochs for full training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
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
        help="Number of Monte Carlo samples for objective estimation",
    )

    # Output options
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save outputs"
    )

    # Advanced options
    parser.add_argument(
        "--use_iwae",
        action="store_true",
        help="Use Importance Weighted Autoencoder objective instead of ELBO",
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
    """Main function to run the flow model training."""
    # Parse command-line arguments
    args = parse_args()

    # Print configuration
    print("\n=== Configuration ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Check for nflows package
    try:
        import nflows

        print("Using nflows!")
    except ImportError:
        print("ERROR: This script requires nflows. Install it with:")
        print("pip install nflows")
        exit(1)

    # Load data
    clean_data, noisy_data, uncertainties = load_data(args)
    print(
        f"Clean data shape: {clean_data.shape}, Noisy data shape: {noisy_data.shape}, Uncertainties shape: {uncertainties.shape}"
    )

    # Train model
    flow, recognition_net, scaler, stats = train_flow_with_uncertainty(
        clean_data, noisy_data, uncertainties, args
    )

    print(f"Training and visualization complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
