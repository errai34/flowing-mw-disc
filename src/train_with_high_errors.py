import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from src.simplified_train_flow_model import train_flow_with_uncertainty

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_evaluate():
    """Train the model with high-error data and evaluate its performance."""
    # Create output directory
    output_dir = "high_error_results"
    os.makedirs(output_dir, exist_ok=True)

    # Check if the high-error data exists
    data_dir = "high_error_data"
    if not os.path.exists(os.path.join(data_dir, "noisy_data.npy")):
        print("High-error data not found. Generating it first...")
        from src.generate_high_error_data import (
            generate_mock_data_high_errors,
            save_high_error_data,
            visualize_mock_data_with_errors,
        )

        noisy_data, errors, true_data = generate_mock_data_high_errors(n_samples=5000)
        save_high_error_data(noisy_data, errors, true_data)
        visualize_mock_data_with_errors(noisy_data, errors, true_data)

    # Load the data
    noisy_data = np.load(os.path.join(data_dir, "noisy_data.npy"))
    errors = np.load(os.path.join(data_dir, "errors.npy"))
    true_data = np.load(os.path.join(data_dir, "true_data.npy"))

    print(
        f"Loaded data: {noisy_data.shape}, errors: {errors.shape}, true data: {true_data.shape}"
    )

    # Configuration for the model
    config = {
        "pretraining_epochs": 15,
        "training_epochs": 30,
        "batch_size": 128,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "flow_n_transforms": 8,
        "recognition_n_transforms": 6,
        "hidden_dim": 64,
        "num_bins": 12,
        "mc_samples": 5,
    }

    # Train two models:
    # 1. Standard model directly on noisy data (without uncertainty awareness)
    # 2. Uncertainty-aware model using the errors

    # Create subdirectories
    standard_dir = os.path.join(output_dir, "standard_model")
    uncertainty_dir = os.path.join(output_dir, "uncertainty_model")
    os.makedirs(standard_dir, exist_ok=True)
    os.makedirs(uncertainty_dir, exist_ok=True)

    # Train uncertainty-aware model
    print("\n=== Training uncertainty-aware model with error deconvolution ===")
    flow, recognition_net, scaler, _ = train_flow_with_uncertainty(
        noisy_data, errors, output_dir=uncertainty_dir, config=config
    )

    # Compare the results
    compare_results(true_data, noisy_data, flow, scaler, output_dir)

    print("\nTraining and evaluation complete!")


def compare_results(true_data, noisy_data, flow, scaler, output_dir):
    """
    Compare the model results with true and noisy data.

    This function generates KDE plots to visualize:
    1. True data distribution
    2. Noisy data distribution (what we actually observe)
    3. Model samples distribution (what the model learns)
    """
    # Extract data components
    true_log_ages, true_fehs = true_data[:, 0], true_data[:, 1]
    noisy_log_ages, noisy_fehs = noisy_data[:, 0], noisy_data[:, 1]

    # Convert log ages to linear ages
    true_ages = 10**true_log_ages
    noisy_ages = 10**noisy_log_ages

    # Sample from the model
    with torch.no_grad():
        n_samples = len(true_data)
        samples = flow.sample(n_samples).cpu().numpy()

    # Inverse transform the samples
    samples_original = scaler.inverse_transform(samples)
    sampled_log_ages, sampled_fehs = samples_original[:, 0], samples_original[:, 1]
    sampled_ages = 10**sampled_log_ages

    # Set up the plot
    plt.figure(figsize=(15, 5))

    # Plot limits
    age_lim = (0, 14)
    feh_lim = (-1.5, 0.5)

    # Create grid for KDE calculation
    age_grid = np.linspace(age_lim[0], age_lim[1], 100)
    feh_grid = np.linspace(feh_lim[0], feh_lim[1], 100)
    age_mesh, feh_mesh = np.meshgrid(age_grid, feh_grid)
    positions = np.vstack([age_mesh.ravel(), feh_mesh.ravel()])

    # 1. True Data KDE
    plt.subplot(1, 3, 1)
    true_kde = gaussian_kde(np.vstack([true_ages, true_fehs]))
    true_density = np.reshape(true_kde(positions), age_mesh.shape)
    plt.contourf(age_mesh, feh_mesh, true_density, levels=20, cmap="viridis")
    plt.xlabel("Age (Gyr)")
    plt.ylabel("[Fe/H]")
    plt.title("True Data Distribution")
    plt.xlim(14, 0)  # Reversed age axis with oldest stars on left
    plt.ylim(-1.5, 0.5)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label="Density")

    # 2. Noisy Data KDE (what we observe)
    plt.subplot(1, 3, 2)
    noisy_kde = gaussian_kde(np.vstack([noisy_ages, noisy_fehs]))
    noisy_density = np.reshape(noisy_kde(positions), age_mesh.shape)
    plt.contourf(age_mesh, feh_mesh, noisy_density, levels=20, cmap="viridis")
    plt.xlabel("Age (Gyr)")
    plt.ylabel("[Fe/H]")
    plt.title("Observed (Noisy) Data Distribution")
    plt.xlim(14, 0)  # Reversed age axis with oldest stars on left
    plt.ylim(-1.5, 0.5)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label="Density")

    # 3. Model Samples KDE (what the model learns)
    plt.subplot(1, 3, 3)
    model_kde = gaussian_kde(np.vstack([sampled_ages, sampled_fehs]))
    model_density = np.reshape(model_kde(positions), age_mesh.shape)
    plt.contourf(age_mesh, feh_mesh, model_density, levels=20, cmap="viridis")
    plt.xlabel("Age (Gyr)")
    plt.ylabel("[Fe/H]")
    plt.title("Model Samples Distribution (Deconvolved)")
    plt.xlim(14, 0)  # Reversed age axis with oldest stars on left
    plt.ylim(-1.5, 0.5)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label="Density")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_comparison.png"), dpi=300)

    # Create a difference plot (model vs true)
    plt.figure(figsize=(10, 8))
    diff_density = model_density - true_density

    # Use a diverging colormap for difference plot
    from matplotlib.colors import TwoSlopeNorm

    # Find min and max for symmetric colormap
    vmax = max(abs(np.min(diff_density)), abs(np.max(diff_density)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    plt.contourf(age_mesh, feh_mesh, diff_density, levels=20, cmap="RdBu_r", norm=norm)
    plt.xlabel("Age (Gyr)")
    plt.ylabel("[Fe/H]")
    plt.title("Difference: Model Samples - True Distribution")
    plt.xlim(14, 0)  # Reversed age axis with oldest stars on left
    plt.ylim(-1.5, 0.5)
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(label="Density Difference")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_vs_true_difference.png"), dpi=300)

    print(f"Comparison plots saved to {output_dir}/")


if __name__ == "__main__":
    train_and_evaluate()
