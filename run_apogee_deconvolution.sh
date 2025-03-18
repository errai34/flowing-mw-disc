#!/bin/bash

# Make sure output directory exists
if [ ! -d "results_apogee" ]; then
    echo "Creating output directory..."
    mkdir -p results_apogee
fi

# Install required packages if missing
echo "Checking dependencies..."
pip install scikit-learn seaborn nflows tqdm matplotlib pyyaml astropy pandas --quiet

# Define radial bins to analyze
# You can uncomment additional bins if you want to run multiple analyses
RADIAL_BINS=(
    "0 6"    # Inner disc
    # "6 8"    # Inner-middle disc
    # "8 10"   # Solar neighborhood
    # "10 15"  # Outer disc
)

# Loop through each radial bin
for bin in "${RADIAL_BINS[@]}"; do
    read -r r_min r_max <<< "$bin"
    echo "Running deconvolution for radial bin $r_min-$r_max kpc..."
    
    python src/run_apogee_deconvolution.py \
        --r_min $r_min \
        --r_max $r_max \
        --config_path config.yaml \
        --output_dir results_apogee \
        --pretraining_epochs 30 \
        --epochs 50 \
        --batch_size 256 \
        --n_transforms 16 \
        --recognition_n_transforms 8 \
        --hidden_dim 128 \
        --num_bins 16 \
        --tail_bound 5.0 \
        --mc_samples 10 \
        --n_samples 10000 \
        --lr 1e-3 \
        --weight_decay 1e-5
        # Uncomment if you want to use IWAE objective
        # --use_iwae
done

echo "Deconvolution complete! Check results in results_apogee/ directory."