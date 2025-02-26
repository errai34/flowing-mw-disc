#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data handling utilities for processing and preparing stellar data
for normalizing flow analysis of Galactic evolution.
"""

import os

import numpy as np
import pandas as pd
import yaml
from astropy.table import Table


class StellarDataHandler:
    def __init__(self, apogee_path, galah_path):
        """
        Initialize with the base paths to APOGEE and GALAH data.
        """
        # Expand '~' if used
        self.apogee_path = os.path.expanduser(apogee_path)
        self.galah_path = os.path.expanduser(galah_path)

    @classmethod
    def from_config(cls, config_path="config.yaml"):
        """
        Alternate constructor: read 'apogee_path' and 'galah_path' from a YAML config.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        apogee_path = cfg["apogee_path"]
        galah_path = cfg["galah_path"]
        return cls(apogee_path, galah_path)

    def load_apogee_fits(self):
        """
        Load APOGEE data from a fits file in self.apogee_path.
        """
        fits_file = os.path.join(
            self.apogee_path, "APOGEE_DR17_bingoages_conservative_age_cut.fits"
        )
        if not os.path.exists(fits_file):
            raise FileNotFoundError(f"FITS file not found: {fits_file}")

        apdr17 = Table.read(fits_file)
        apdr17 = apdr17[apdr17["age"] < 20]
        mw = apdr17.to_pandas()
        mw = self._preprocess_mw(mw)
        print(f"Loaded {len(mw)} stars from APOGEE FITS")
        return mw

    def load_apogee_csv(self):
        """
        Load APOGEE data from a CSV file in self.apogee_path.
        """
        csv_file = os.path.join(
            self.apogee_path, "apogee_dr17_bingoages_highquality.csv"
        )
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        mw = pd.read_csv(csv_file)
        mw = self._preprocess_mw(mw)
        print(f"Loaded {len(mw)} stars from APOGEE CSV")
        return mw

    def load_galah_data(self):
        """
        Load GALAH data from 'galah_dr4_allstar_240705.fits'.
        """
        galah_file = os.path.join(self.galah_path, "galah_dr4_allstar_240705.fits")
        if not os.path.exists(galah_file):
            print(f"GALAH data file not found: {galah_file}")
            return pd.DataFrame()

        galah_data = Table.read(galah_file, format="fits")
        galah_df = galah_data.to_pandas()
        print(f"Loaded {len(galah_df)} stars from GALAH")
        return galah_df

    def load_galah_dyn(self):
        """
        Load GALAH dynamical data from 'galah_dr4_vac_dynamics_240207.fits'.
        """
        galah_file = os.path.join(self.galah_path, "galah_dr4_vac_dynamics_240207.fits")
        if not os.path.exists(galah_file):
            print(f"GALAH dynamics file not found: {galah_file}")
            return pd.DataFrame()

        galah_data = Table.read(galah_file, format="fits")
        galah_df = galah_data.to_pandas()
        print(f"Loaded {len(galah_df)} stars from GALAH dynamics")
        return galah_df

    def _preprocess_mw(self, mw):
        """
        Preprocess Milky Way data with additional useful columns.
        """
        # Create a copy to avoid SettingWithCopyWarning
        mw = mw.copy()

        if "age_lowess_correct" in mw.columns:
            mw["log_age_ann"] = np.log10(mw["age_lowess_correct"])
            mw["age_ann"] = mw["age_lowess_correct"]

        if "pred_logAge" in mw.columns:
            mw["log_age"] = mw["pred_logAge"]
            mw["log_age_err"] = mw["pred_logAge_std"]

        if "rperi" in mw.columns and "rap" in mw.columns:
            mw["rmean"] = (mw["rperi"] + mw["rap"]) / 2.0

        if "pred_logAge" in mw.columns and "pred_logAge_std" in mw.columns:
            mw["age_err"] = 0.5 * (
                (
                    10 ** (mw["pred_logAge"] + mw["pred_logAge_std"])
                    - 10 ** mw["pred_logAge"]
                )
                + (
                    10 ** mw["pred_logAge"]
                    - 10 ** (mw["pred_logAge"] - mw["pred_logAge_std"])
                )
            )

        if "age_total_error" in mw.columns:
            mw["age_ann_err"] = mw["age_total_error"]
            mw["log_age_ann_err"] = np.log10(mw["age_total_error"])
            
        # Ensure MG_FE exists if we have MG_H and FE_H
        if "MG_FE" not in mw.columns and "MG_H" in mw.columns and "FE_H" in mw.columns:
            mw["MG_FE"] = mw["MG_H"] - mw["FE_H"]
            mw["MG_FE_ERR"] = np.sqrt(
                mw["MG_H_ERR"] ** 2 + mw["FE_H_ERR"] ** 2
            )
            print("Computed [Mg/Fe] from [Mg/H] and [Fe/H]")

        return mw
        
    def apply_quality_filters(self, data):
        """
        Apply standard quality filters to stellar data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with stellar data
            
        Returns:
        --------
        pandas.DataFrame
            Filtered data meeting quality criteria
        """
        # Define standard filtering conditions for high-quality data
        conditions = [
            (data["pred_logAge_std"] < 0.2),
            (data["FE_H_ERR"] < 0.1),
            (data["MG_H_ERR"] < 0.1),
            (data["age"] < 20),
            (data["age"] > 0),
        ]
        
        filtered_data = filter_data(data, conditions)
        return filtered_data


def prepare_data_for_radial_bins(df, radial_bins=None):
    """
    Prepare data for analysis in radial bins.

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

    # Compute sqrt(Jz) and propagate errors
    df["jz_safe"] = np.maximum(df["jz"], 0.0)
    df["sqrt_Jz"] = np.sqrt(df["jz_safe"])

    # Propagate error: σ_sqrt(x) = σ_x / (2 * sqrt(x))
    denominator = 2 * df["sqrt_Jz"]
    denominator = np.maximum(denominator, 1e-8)  # Prevent division by zero
    df["sqrt_Jz_err"] = df["jz_err"] / denominator

    bin_data = {}

    for r_min, r_max in radial_bins:
        bin_name = f"R{r_min}-{r_max}"
        mask = (df["rmean"] >= r_min) & (df["rmean"] < r_max)
        bin_df = df[mask]

        # Skip if too few stars
        if len(bin_df) < 100:
            print(f"Warning: Bin {bin_name} has only {len(bin_df)} stars, skipping")
            continue

        # Extract data and errors for the 5D model
        data_5d = bin_df[["pred_logAge", "FE_H", "MG_FE", "sqrt_Jz", "Lz"]].values
        err_5d = bin_df[
            ["pred_logAge_std", "FE_H_ERR", "MG_FE_ERR", "sqrt_Jz_err", "Lz_err"]
        ].values

        bin_data[bin_name] = {"data": data_5d, "err": err_5d, "count": len(bin_df)}

        print(f"Bin {bin_name}: {len(bin_df)} stars")

    return bin_data


def filter_data(df, conditions):
    """
    Filter DataFrame based on multiple conditions.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to filter
    conditions : list
        List of boolean Series representing filtering conditions

    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame
    """
    if not conditions:
        return df.copy()  # Return a copy even without filtering

    mask = conditions[0]
    for condition in conditions[1:]:
        mask = mask & condition

    filtered_df = df[mask].copy()  # Create a copy of the filtered data
    print(f"Filtered data from {len(df)} to {len(filtered_df)} rows")
    return filtered_df
