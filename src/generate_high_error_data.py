import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_mock_stellar_data(n_samples=5000, random_seed=42):
    """
    Generate mock stellar data with artificially high errors for older stars.
    Creates a bimodal age distribution with a clear starburst feature.

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

    # Create a bimodal age distribution (younger peak + older starburst)
    # First mode: younger stars (majority)
    young_fraction = 0.6
    young_stars = int(n_samples * young_fraction)
    old_stars = n_samples - young_stars

    # Ages in log10 space (will be converted to Gyr in visualization)
    # Younger population centered around 3-5 Gyr (log10 ~ 0.6)
    young_log_ages = np.random.normal(loc=0.6, scale=0.15, size=young_stars)
    young_log_ages = np.clip(young_log_ages, 0.3, 0.9)  # ~2-8 Gyr range

    # Older starburst population centered around 11 Gyr (log10 ~ 1.04)
    old_log_ages = np.random.normal(loc=1.04, scale=0.06, size=old_stars)
    old_log_ages = np.clip(old_log_ages, 0.9, 1.2)  # ~8-16 Gyr range

    # Combine the two populations
    log_ages = np.concatenate([young_log_ages, old_log_ages])

    # Shuffle the combined data to mix young and old stars
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    log_ages = log_ages[indices]

    # Generate [Fe/H] with correlation to age
    # Create a distinct metallicity signature for the starburst
    # First, baseline metallicity trend with age
    feh_means = -0.5 + 0.7 * (
        1 - log_ages / 1.2
    )  # General trend: older = more metal poor

    # Modify trend for starburst region (10-12 Gyr)
    starburst_mask = (log_ages > 1.0) & (log_ages < 1.15)  # log10(10) to log10(12.5)
    feh_means[starburst_mask] = -0.5 + np.random.normal(
        0, 0.05, size=np.sum(starburst_mask)
    )  # Tighter metallicity

    # Add scatter around the trend (tighter for starburst)
    feh_scatter = np.ones(n_samples) * 0.15  # Base scatter
    feh_scatter[starburst_mask] = 0.08  # Reduced scatter for starburst

    fehs = np.array(
        [
            np.random.normal(mean, scatter)
            for mean, scatter in zip(feh_means, feh_scatter)
        ]
    )
    fehs = np.clip(fehs, -1.0, 0.5)  # Reasonable range

    # Generate [Mg/Fe] with anti-correlation to [Fe/H]
    # Metal-poor stars tend to be more alpha-enhanced
    mgfe_means = 0.3 - 0.3 * (fehs + 1.5) / 2.0  # Scale to get proper anti-correlation

    # Distinct alpha enhancement for starburst
    mgfe_means[starburst_mask] += 0.1  # Make starburst stars more alpha-enhanced

    # Add scatter (tighter for starburst)
    mgfe_scatter = np.ones(n_samples) * 0.05  # Base scatter
    mgfe_scatter[starburst_mask] = 0.03  # Reduced scatter for starburst

    mgfes = np.array(
        [
            np.random.normal(mean, scatter)
            for mean, scatter in zip(mgfe_means, mgfe_scatter)
        ]
    )

    # Add thick/thin disk bimodality
    bimodal_mask = (np.random.random(n_samples) < 0.3) & (
        ~starburst_mask
    )  # Only for non-starburst stars
    mgfes[bimodal_mask] += 0.15
    mgfes = np.clip(mgfes, -0.2, 0.5)

    # Stack the data
    true_data = np.column_stack([log_ages, fehs, mgfes])

    # Generate realistic errors with much higher errors for older stars
    # Convert log age to linear for computing errors
    ages = 10**log_ages

    # Make errors moderately dependent on age
    # But reduce errors for starburst stars to help with detectability
    age_errors = 0.05 + 0.2 * (ages / 20.0)  # More moderate increase with age
    age_errors[starburst_mask] *= 0.8  # Less aggressive reduction for starburst stars

    feh_errors = 0.08 + 0.15 * (ages / 20.0) + 0.03 * np.abs(fehs)
    feh_errors[starburst_mask] *= 0.8  # Less aggressive reduction for starburst stars

    mgfe_errors = 0.05 + 0.1 * (ages / 20.0) + 0.03 * np.abs(mgfes)
    mgfe_errors[starburst_mask] *= 0.8  # Less aggressive reduction for starburst stars

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
    starburst_flag[starburst_mask] = 1

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

        # KDE for log age - with clear bimodality
        plt.subplot(1, 3, 1)
        sns.kdeplot(
            true_log_ages,
            color="blue",
            label="True",
            fill=True,
            alpha=0.3,
            bw_adjust=0.5,
        )
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
            levels=15,
            thresh=0.05,
            bw_adjust=0.7,  # Sharper contours to highlight bimodality
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
            plt.title("Observed Data (Starburst Partially Obscured)")
            plt.xlim(20, 0)  # Reversed x-axis
            plt.ylim(-1.0, 0.5)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "starburst_visualization.png"), dpi=300
            )
            plt.close()

            # Add histogram of ages to clearly show bimodality
            plt.figure(figsize=(12, 6))

            # Plot age histograms
            plt.subplot(1, 2, 1)
            plt.hist(true_ages, bins=40, alpha=0.6, color="blue", label="All Stars")
            plt.hist(
                true_ages[starburst_flag],
                bins=15,
                alpha=0.8,
                color="red",
                label="Starburst",
            )
            plt.xlabel("Age (Gyr)")
            plt.ylabel("Count")
            plt.title("True Age Distribution")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot noisy ages histogram
            plt.subplot(1, 2, 2)
            plt.hist(ages, bins=40, alpha=0.6, color="gray")
            plt.xlabel("Age (Gyr)")
            plt.ylabel("Count")
            plt.title("Observed Age Distribution")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "age_histograms.png"), dpi=300)
            plt.close()

            # Add bimodal age-metallicity visualization
            plt.figure(figsize=(10, 8))

            # Hexbin plot to show density in age-metallicity space
            plt.hexbin(true_ages, true_fehs, gridsize=30, cmap="Blues", mincnt=1)
            plt.colorbar(label="Density")

            # Use direct kdeplot instead of custom KDE
            sns.kdeplot(x=true_ages, y=true_fehs, color="red", levels=10, linewidths=1)

            plt.xlabel("Age (Gyr)")
            plt.ylabel("[Fe/H]")
            plt.title("Age-Metallicity Bimodality")
            plt.xlim(20, 0)  # Reversed x-axis
            plt.ylim(-1.0, 0.5)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "age_metallicity_bimodality.png"), dpi=300
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
