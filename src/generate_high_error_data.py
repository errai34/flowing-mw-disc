import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_mock_stellar_data(n_samples=5000, random_seed=42):
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
        (noisy_data, errors, true_data) - Arrays of shape (n_samples, 3)
    """
    np.random.seed(random_seed)

    # Generate ages with a skewed distribution (more younger stars)
    # Ages in log10 space (will be converted to Gyr in visualization)
    # Tighter distribution with smaller scale
    log_ages = np.random.normal(loc=0.8, scale=0.25, size=n_samples)
    log_ages = np.clip(log_ages, 0, 1.3)  # Clip to ~1-20 Gyr range

    # Generate [Fe/H] with correlation to age
    # Older stars tend to be more metal poor
    feh_means = -0.5 + 0.7 * (1 - log_ages / 1.2)  # Scale to get proper correlation
    # Tighter scatter for more coherent true data
    feh_scatter = 0.15 + 0.08 * np.random.random(n_samples)
    fehs = np.random.normal(loc=feh_means, scale=feh_scatter, size=n_samples)
    fehs = np.clip(fehs, -1.0, 0.5)

    # Generate [Mg/Fe] with anti-correlation to [Fe/H]
    # Metal-poor stars tend to be more alpha-enhanced
    mgfe_means = 0.3 - 0.3 * (fehs + 1.5) / 2.0  # Scale to get proper anti-correlation
    # Tighter scatter for Mg/Fe
    mgfe_scatter = 0.04 + 0.02 * np.random.random(n_samples)
    mgfes = np.random.normal(loc=mgfe_means, scale=mgfe_scatter, size=n_samples)

    # Add bimodality to simulate thin/thick disk
    bimodal_mask = np.random.random(n_samples) < 0.3
    mgfes[bimodal_mask] += 0.15
    mgfes = np.clip(mgfes, -0.2, 0.5)

    # Add starburst around 10-12 Gyr
    # Convert 10-12 Gyr to log age: log10(10) ≈ 1, log10(12) ≈ 1.08
    starburst_size = int(n_samples * 0.15)  # 15% of the stars in the starburst
    starburst_indices = np.random.choice(n_samples, size=starburst_size, replace=False)

    # Generate ages for starburst centered around 11 Gyr (log10(11) ≈ 1.04)
    starburst_log_ages = np.random.normal(loc=1.04, scale=0.03, size=starburst_size)
    starburst_log_ages = np.clip(starburst_log_ages, 1.0, 1.08)

    # Generate metallicities for starburst - slightly more metal-rich than typical stars of this age
    starburst_fehs = np.random.normal(loc=-0.4, scale=0.1, size=starburst_size)
    starburst_fehs = np.clip(starburst_fehs, -0.6, -0.2)

    # Generate Mg/Fe for starburst - moderately alpha-enhanced
    starburst_mgfes = np.random.normal(loc=0.25, scale=0.05, size=starburst_size)
    starburst_mgfes = np.clip(starburst_mgfes, 0.15, 0.35)

    # Replace values for starburst stars
    log_ages[starburst_indices] = starburst_log_ages
    fehs[starburst_indices] = starburst_fehs
    mgfes[starburst_indices] = starburst_mgfes

    # Stack the data
    true_data = np.column_stack([log_ages, fehs, mgfes])

    # Generate realistic errors with much higher errors for older stars
    # Convert log age to linear for computing errors
    ages = 10**log_ages

    # Make errors EXTREMELY strongly dependent on age: much higher errors for older stars
    # For log age: greatly increased coefficients for much noisier data
    age_errors = 0.1 + 0.6 * (ages / 20.0) ** 2  # Stronger quadratic increase with age

    # For [Fe/H]: greatly increased age dependence and base error
    feh_errors = 0.12 + 0.45 * (ages / 20.0) ** 2 + 0.07 * np.abs(fehs)

    # For [Mg/Fe]: greatly increased age dependence and base error
    mgfe_errors = 0.08 + 0.25 * (ages / 20.0) ** 2 + 0.06 * np.abs(mgfes)

    errors = np.column_stack([age_errors, feh_errors, mgfe_errors])

    # Add noise to the observables based on these errors
    noisy_data = true_data.copy()
    for i in range(n_samples):
        noisy_data[i, 0] = true_data[i, 0] + np.random.normal(
            0, age_errors[i]
        )  # Log age with noise
        noisy_data[i, 1] = true_data[i, 1] + np.random.normal(
            0, feh_errors[i]
        )  # [Fe/H] with noise
        noisy_data[i, 2] = true_data[i, 2] + np.random.normal(
            0, mgfe_errors[i]
        )  # [Mg/Fe] with noise

    # Clip again to reasonable ranges
    noisy_data[:, 0] = np.clip(noisy_data[:, 0], 0, 1.3)  # Log age
    noisy_data[:, 1] = np.clip(noisy_data[:, 1], -1.0, 0.5)  # [Fe/H]
    noisy_data[:, 2] = np.clip(noisy_data[:, 2], -0.2, 0.5)  # [Mg/Fe]

    # Create a metadata array to identify starburst stars (1 for starburst, 0 for regular)
    starburst_flag = np.zeros(n_samples)
    starburst_flag[starburst_indices] = 1

    # Add the starburst flag as a 4th column to true_data for visualization purposes
    true_data_with_flag = np.column_stack([true_data, starburst_flag])

    return noisy_data, errors, true_data_with_flag


def save_mock_data(data, errors, true_data=None, output_dir="mock_data"):
    """
    Save mock data to files.

    Parameters:
    -----------
    data : np.ndarray
        Noisy data array of shape (n_samples, 3)
    errors : np.ndarray
        Error array of shape (n_samples, 3)
    true_data : np.ndarray, optional
        True data array of shape (n_samples, 3) or (n_samples, 4) with starburst flag
    output_dir : str
        Directory to save data
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "mock_data.npy"), data)
    np.save(os.path.join(output_dir, "mock_errors.npy"), errors)

    if true_data is not None:
        np.save(os.path.join(output_dir, "true_data.npy"), true_data)

    print(f"Saved mock data to {output_dir}/")


def visualize_mock_data(data, errors, true_data=None, output_dir="mock_data"):
    """
    Create visualizations of the mock data.

    Parameters:
    -----------
    data : np.ndarray
        Noisy data array of shape (n_samples, 3)
    errors : np.ndarray
        Error array of shape (n_samples, 3)
    true_data : np.ndarray, optional
        True data array of shape (n_samples, 3) or (n_samples, 4) with starburst flag
    output_dir : str
        Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract parameters
    log_ages, fehs, mgfes = data[:, 0], data[:, 1], data[:, 2]

    # Convert log age to linear age
    ages = 10**log_ages

    # Check if true_data has starburst flag (4 columns)
    has_starburst_flag = true_data is not None and true_data.shape[1] > 3

    if has_starburst_flag:
        # Extract starburst flag and basic true data
        starburst_flag = true_data[:, 3].astype(bool)
        true_data_basic = true_data[:, :3]
    else:
        true_data_basic = true_data

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Age vs [Fe/H]
    sc1 = axes[0].scatter(ages, fehs, s=5, alpha=0.5, c=mgfes, cmap="viridis")
    fig.colorbar(sc1, ax=axes[0], label="[Mg/Fe]")
    axes[0].set_xlabel("Age (Gyr)")
    axes[0].set_ylabel("[Fe/H]")
    axes[0].set_title("Age vs [Fe/H]")
    axes[0].set_xlim(20, 0)  # Reversed, oldest stars on left
    axes[0].set_ylim(-1.0, 0.5)
    axes[0].grid(True, alpha=0.3)

    # Plot Age vs [Mg/Fe]
    sc2 = axes[1].scatter(ages, mgfes, s=5, alpha=0.5, c=fehs, cmap="plasma")
    fig.colorbar(sc2, ax=axes[1], label="[Fe/H]")
    axes[1].set_xlabel("Age (Gyr)")
    axes[1].set_ylabel("[Mg/Fe]")
    axes[1].set_title("Age vs [Mg/Fe]")
    axes[1].set_xlim(20, 0)  # Reversed, oldest stars on left
    axes[1].set_ylim(-0.2, 0.5)
    axes[1].grid(True, alpha=0.3)

    # Plot Age vs log age error
    sc3 = axes[2].scatter(ages, errors[:, 0], s=5, alpha=0.5, c=ages, cmap="viridis")
    fig.colorbar(sc3, ax=axes[2], label="Age (Gyr)")
    axes[2].set_xlabel("Age (Gyr)")
    axes[2].set_ylabel("Log Age Error")
    axes[2].set_title("Age-Dependent Errors")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mock_data_viz.png"), dpi=300)
    plt.close()

    # If true data is provided, create comparison plot
    if true_data_basic is not None:
        true_log_ages, true_fehs, true_mgfes = (
            true_data_basic[:, 0],
            true_data_basic[:, 1],
            true_data_basic[:, 2],
        )
        true_ages = 10**true_log_ages

        # Create scatter plot comparison
        plt.figure(figsize=(10, 6))
        plt.scatter(true_ages, true_fehs, s=5, alpha=0.5, color="blue", label="True")
        plt.scatter(ages, fehs, s=5, alpha=0.5, color="red", label="Observed")
        plt.xlabel("Age (Gyr)")
        plt.ylabel("[Fe/H]")
        plt.title("True vs Observed Values")
        plt.xlim(20, 0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "true_vs_observed.png"), dpi=300)
        plt.close()

        # Create KDE plots comparing true vs observed distributions
        plt.figure(figsize=(18, 6))

        # KDE for log age
        plt.subplot(1, 3, 1)
        sns.kdeplot(true_log_ages, color="blue", label="True", fill=True, alpha=0.3)
        sns.kdeplot(log_ages, color="red", label="Observed", fill=True, alpha=0.3)
        plt.xlabel("Log Age")
        plt.ylabel("Density")
        plt.title("KDE: True vs Observed Log Age")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # KDE for [Fe/H]
        plt.subplot(1, 3, 2)
        sns.kdeplot(true_fehs, color="blue", label="True", fill=True, alpha=0.3)
        sns.kdeplot(fehs, color="red", label="Observed", fill=True, alpha=0.3)
        plt.xlabel("[Fe/H]")
        plt.ylabel("Density")
        plt.title("KDE: True vs Observed [Fe/H]")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # KDE for [Mg/Fe]
        plt.subplot(1, 3, 3)
        sns.kdeplot(true_mgfes, color="blue", label="True", fill=True, alpha=0.3)
        sns.kdeplot(mgfes, color="red", label="Observed", fill=True, alpha=0.3)
        plt.xlabel("[Mg/Fe]")
        plt.ylabel("Density")
        plt.title("KDE: True vs Observed [Mg/Fe]")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "kde_comparisons.png"), dpi=300)
        plt.close()

    print(f"Visualizations saved to {output_dir}/")

    # Additional 2D KDE plot for age vs [Fe/H]
    if true_data_basic is not None:
        plt.figure(figsize=(14, 6))

        # 2D KDE for true data
        plt.subplot(1, 2, 1)
        sns.kdeplot(
            x=true_ages,
            y=true_fehs,
            cmap="Blues",
            fill=True,
            alpha=0.7,
            levels=10,
            thresh=0.05,
        )
        plt.xlabel("Age (Gyr)")
        plt.ylabel("[Fe/H]")
        plt.title("True Data Distribution")
        plt.xlim(20, 0)  # Reversed x-axis
        plt.ylim(-1.0, 0.5)
        plt.grid(True, alpha=0.3)

        # 2D KDE for observed data
        plt.subplot(1, 2, 2)
        sns.kdeplot(
            x=ages, y=fehs, cmap="Reds", fill=True, alpha=0.7, levels=10, thresh=0.05
        )
        plt.xlabel("Age (Gyr)")
        plt.ylabel("[Fe/H]")
        plt.title("Observed Data Distribution")
        plt.xlim(20, 0)  # Reversed x-axis
        plt.ylim(-1.0, 0.5)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "2d_kde_comparison.png"), dpi=300)
        plt.close()

        # If we have starburst information, visualize the starburst
        if has_starburst_flag:
            plt.figure(figsize=(14, 6))

            # Plot true data with starburst highlighted
            plt.subplot(1, 2, 1)
            # First plot all data
            plt.scatter(
                true_ages,
                true_fehs,
                s=5,
                alpha=0.3,
                color="blue",
                label="Regular Stars",
            )
            # Then highlight starburst stars
            plt.scatter(
                true_ages[starburst_flag],
                true_fehs[starburst_flag],
                s=15,
                alpha=0.8,
                color="red",
                label="Starburst Stars",
            )
            plt.xlabel("Age (Gyr)")
            plt.ylabel("[Fe/H]")
            plt.title("True Data with Starburst Highlighted")
            plt.xlim(20, 0)  # Reversed x-axis
            plt.ylim(-1.0, 0.5)
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Plot noisy data (starburst should be harder to see)
            plt.subplot(1, 2, 2)
            plt.scatter(ages, fehs, s=5, alpha=0.5, color="gray")
            plt.xlabel("Age (Gyr)")
            plt.ylabel("[Fe/H]")
            plt.title("Observed Data (Starburst Obscured by Noise)")
            plt.xlim(20, 0)  # Reversed x-axis
            plt.ylim(-1.0, 0.5)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "starburst_visualization.png"), dpi=300
            )
            plt.close()


if __name__ == "__main__":
    # Generate mock data
    noisy_data, errors, true_data = generate_mock_stellar_data(n_samples=5000)

    # Save data
    save_mock_data(noisy_data, errors, true_data)

    # Visualize data
    visualize_mock_data(noisy_data, errors, true_data)

    print("Mock data generation complete!")
