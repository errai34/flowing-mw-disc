import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from src.simplified_train_flow_model import Flow3D

# Load the trained model
model_path = "results/3d_uncertainty_model.pt"
checkpoint = torch.load(model_path, map_location="cpu")
scaler = checkpoint["scaler"]

# Initialize and load the flow model
flow = Flow3D(
    n_transforms=checkpoint.get("config", {}).get("flow_n_transforms", 8),
    hidden_dim=checkpoint.get("config", {}).get("hidden_dim", 64),
    num_bins=checkpoint.get("config", {}).get("num_bins", 12),
)
flow.load_state_dict(checkpoint["flow_state"])
flow.eval()

# Load real data
real_data = np.load("mock_data/mock_data.npy")
real_log_ages, real_fehs = real_data[:, 0], real_data[:, 1]
real_ages = 10**real_log_ages  # Convert log age to linear age

# Sample from the model
with torch.no_grad():
    samples = flow.sample(len(real_data)).cpu().numpy()
sampled_data = scaler.inverse_transform(samples)
sampled_log_ages, sampled_fehs = sampled_data[:, 0], sampled_data[:, 1]
sampled_ages = 10**sampled_log_ages  # Convert log age to linear age

# Set up the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Age vs [Fe/H] KDE plot - Real Data
age_grid = np.linspace(0, 14, 100)
feh_grid = np.linspace(-1.5, 0.5, 100)
age_mesh, feh_mesh = np.meshgrid(age_grid, feh_grid)
positions = np.vstack([age_mesh.ravel(), feh_mesh.ravel()])

# Calculate KDE for real data
real_kde = gaussian_kde(np.vstack([real_ages, real_fehs]))
real_density = np.reshape(real_kde(positions), age_mesh.shape)

# Plot real data KDE
contour1 = ax1.contourf(age_mesh, feh_mesh, real_density, levels=20, cmap="viridis")
ax1.set_xlabel("Age (Gyr)")
ax1.set_ylabel("[Fe/H]")
ax1.set_title("Real Data: Age vs [Fe/H]")
ax1.set_xlim(14, 0)  # Reversed age axis with oldest stars on left
ax1.set_ylim(-1.5, 0.5)
plt.colorbar(contour1, ax=ax1, label="Density")
ax1.grid(True, alpha=0.3)

# Calculate KDE for sampled data
sampled_kde = gaussian_kde(np.vstack([sampled_ages, sampled_fehs]))
sampled_density = np.reshape(sampled_kde(positions), age_mesh.shape)

# Plot sampled data KDE
contour2 = ax2.contourf(age_mesh, feh_mesh, sampled_density, levels=20, cmap="viridis")
ax2.set_xlabel("Age (Gyr)")
ax2.set_ylabel("[Fe/H]")
ax2.set_title("Model Samples: Age vs [Fe/H]")
ax2.set_xlim(14, 0)  # Reversed age axis with oldest stars on left
ax2.set_ylim(-1.5, 0.5)
plt.colorbar(contour2, ax=ax2, label="Density")
ax2.grid(True, alpha=0.3)

# Save and show plot
plt.tight_layout()
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "age_feh_kde_simple.png"), dpi=300)
print(f"Plot saved to {save_dir}/age_feh_kde_simple.png")
plt.show()
