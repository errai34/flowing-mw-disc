import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_mock_data_high_errors(n_samples=5000, random_seed=42):
    """
    Generate mock stellar data with artificially high errors for older stars.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (data, errors) - Arrays of shape (n_samples, 3)
    """
    np.random.seed(random_seed)

    # Generate ages with a skewed distribution (more younger stars)
    # Ages in log10 space (will be converted to Gyr in visualization)
    log_ages = np.random.normal(loc=0.8, scale=0.3, size=n_samples)
    log_ages = np.clip(log_ages, 0, 1.15)  # Clip to ~1-14 Gyr range

    # Generate [Fe/H] with correlation to age
    # Older stars tend to be more metal poor
    feh_means = -0.5 + 0.7 * (1 - log_ages / 1.2)  # Scale to get proper correlation
    feh_scatter = 0.2 + 0.1 * np.random.random(n_samples)
    fehs = np.random.normal(loc=feh_means, scale=feh_scatter, size=n_samples)
    fehs = np.clip(fehs, -1.5, 0.5)

    # Generate [Mg/Fe] with anti-correlation to [Fe/H]
    # Metal-poor stars tend to be more alpha-enhanced
    mgfe_means = 0.3 - 0.3 * (fehs + 1.5) / 2.0  # Scale to get proper anti-correlation
    mgfe_scatter = 0.05 + 0.03 * np.random.random(n_samples)
    mgfes = np.random.normal(loc=mgfe_means, scale=mgfe_scatter, size=n_samples)

    # Add bimodality to simulate thin/thick disk
    bimodal_mask = np.random.random(n_samples) < 0.3
    mgfes[bimodal_mask] += 0.15
    mgfes = np.clip(mgfes, -0.2, 0.5)

    # Stack the data
    data = np.column_stack([log_ages, fehs, mgfes])

    # Generate realistic errors with much higher errors for older stars
    # Convert log age to linear for computing errors
    ages = 10**log_ages

    # Make errors strongly dependent on age: older stars have much higher errors
    # For log age:
    age_errors = 0.05 + 0.25 * (ages / 14.0) ** 2  # Quadratic increase with age

    # For [Fe/H]: also increase with age
    feh_errors = 0.05 + 0.15 * (ages / 14.0) ** 2 + 0.02 * np.abs(fehs)

    # For [Mg/Fe]: moderate increase with age
    mgfe_errors = 0.03 + 0.08 * (ages / 14.0) ** 2 + 0.02 * np.abs(mgfes)

    errors = np.column_stack([age_errors, feh_errors, mgfe_errors])

    # Add noise to the observables based on these errors
    noisy_data = data.copy()
    for i in range(n_samples):
        noisy_data[i, 0] = data[i, 0] + np.random.normal(
            0, age_errors[i]
        )  # Log age with noise
        noisy_data[i, 1] = data[i, 1] + np.random.normal(
            0, feh_errors[i]
        )  # [Fe/H] with noise
        noisy_data[i, 2] = data[i, 2] + np.random.normal(
            0, mgfe_errors[i]
        )  # [Mg/Fe] with noise

    # Clip again to reasonable ranges
    noisy_data[:, 0] = np.clip(noisy_data[:, 0], 0, 1.15)  # Log age
    noisy_data[:, 1] = np.clip(noisy_data[:, 1], -1.5, 0.5)  # [Fe/H]
    noisy_data[:, 2] = np.clip(noisy_data[:, 2], -0.2, 0.5)  # [Mg/Fe]

    return noisy_data, errors, data  # Return noisy data, errors, and true values


def visualize_mock_data_with_errors(
    noisy_data, errors, true_data, save_dir="high_error_data"
):
    """
    Create visualizations of the mock data, highlighting the errors.

    Parameters:
    -----------
    noisy_data : np.ndarray
        Noisy data array of shape (n_samples, 3)
    errors : np.ndarray
        Error array of shape (n_samples, 3)
    true_data : np.ndarray
        True data array of shape (n_samples, 3)
    save_dir : str
        Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract parameters
    noisy_log_ages, noisy_fehs, noisy_mgfes = (
        noisy_data[:, 0],
        noisy_data[:, 1],
        noisy_data[:, 2],
    )
    true_log_ages, true_fehs, true_mgfes = (
        true_data[:, 0],
        true_data[:, 1],
        true_data[:, 2],
    )

    # Convert log age to linear age
    noisy_ages = 10**noisy_log_ages
    true_ages = 10**true_log_ages

    # Create a figure with comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot Age vs [Fe/H] for true and noisy data
    sc1 = axes[0].scatter(
        true_ages, true_fehs, s=5, alpha=0.5, color="blue", label="True Values"
    )
    sc2 = axes[0].scatter(
        noisy_ages,
        noisy_fehs,
        s=5,
        alpha=0.5,
        color="red",
        label="Observed Values (with errors)",
    )
    axes[0].set_xlabel("Age (Gyr)")
    axes[0].set_ylabel("[Fe/H]")
    axes[0].set_title("Age vs [Fe/H]: True vs. Noisy Data")
    axes[0].set_xlim(14, 0)  # Reversed, oldest stars on left
    axes[0].set_ylim(-1.5, 0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot Log Age errors vs. Age
    ages = 10**true_log_ages
    axes[1].scatter(ages, errors[:, 0], s=5, alpha=0.5, c=ages, cmap="viridis")
    axes[1].set_xlabel("Age (Gyr)")
    axes[1].set_ylabel("Log Age Error")
    axes[1].set_title("Age-Dependent Errors in Log Age")
    axes[1].set_xlim(0, 14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "true_vs_noisy_data.png"), dpi=300)
    plt.close()

    # Create error distribution plots
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(ages, errors[:, 0], s=5, alpha=0.7, c=ages, cmap="viridis")
    plt.colorbar(label="Age (Gyr)")
    plt.title("Age vs. Log Age Error")
    plt.xlabel("Age (Gyr)")
    plt.ylabel("Log Age Error")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.scatter(ages, errors[:, 1], s=5, alpha=0.7, c=ages, cmap="viridis")
    plt.colorbar(label="Age (Gyr)")
    plt.title("Age vs. [Fe/H] Error")
    plt.xlabel("Age (Gyr)")
    plt.ylabel("[Fe/H] Error")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.scatter(ages, errors[:, 2], s=5, alpha=0.7, c=ages, cmap="viridis")
    plt.colorbar(label="Age (Gyr)")
    plt.title("Age vs. [Mg/Fe] Error")
    plt.xlabel("Age (Gyr)")
    plt.ylabel("[Mg/Fe] Error")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "age_dependent_errors.png"), dpi=300)
    plt.close()


def save_high_error_data(noisy_data, errors, true_data, save_dir="high_error_data"):
    """
    Save mock data with high errors to files.

    Parameters:
    -----------
    noisy_data : np.ndarray
        Noisy data array of shape (n_samples, 3)
    errors : np.ndarray
        Error array of shape (n_samples, 3)
    true_data : np.ndarray
        True data array of shape (n_samples, 3)
    save_dir : str
        Directory to save data
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create DataFrames
    df_noisy = pd.DataFrame(noisy_data, columns=["log_age", "feh", "mgfe"])
    df_errors = pd.DataFrame(errors, columns=["log_age_err", "feh_err", "mgfe_err"])
    df_true = pd.DataFrame(true_data, columns=["true_log_age", "true_feh", "true_mgfe"])

    # Add linear age
    df_noisy["age"] = 10 ** df_noisy["log_age"]
    df_true["true_age"] = 10 ** df_true["true_log_age"]

    # Combine and save
    df_combined = pd.concat([df_noisy, df_errors, df_true], axis=1)
    df_combined.to_csv(
        os.path.join(save_dir, "mock_stellar_data_high_errors.csv"), index=False
    )

    # Also save as NumPy arrays for direct loading
    np.save(os.path.join(save_dir, "noisy_data.npy"), noisy_data)
    np.save(os.path.join(save_dir, "errors.npy"), errors)
    np.save(os.path.join(save_dir, "true_data.npy"), true_data)

    print(f"Saved high-error mock data to {save_dir}/")


if __name__ == "__main__":
    # Generate mock data with high errors for older stars
    noisy_data, errors, true_data = generate_mock_data_high_errors(n_samples=5000)

    # Save the data
    save_high_error_data(noisy_data, errors, true_data)

    # Visualize the data with errors
    visualize_mock_data_with_errors(noisy_data, errors, true_data)

    print("High-error mock data generated successfully!")
