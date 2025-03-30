#!/usr/bin/env python3
"""
SCORCH2 Feature Extraction Tool (Refactored for Dask & Parquet Acceleration)

This script extracts features from protein-ligand complexes for use in SCORCH2.
It processes protein-ligand pairs to extract:

1. Extended Connectivity Interaction Features (ECIF)
2. BINANA interaction features
3. Kier flexibility
4. RDKit descriptors

Features are saved in Parquet format for efficiency.

Usage:
    python scorch2_feature_extraction.py --protein-dir /path/to/proteins --ligand-dir /path/to/ligands --output-dir /path/to/output

Authors: Your Name (with Dask/Parquet modifications)
License: MIT
"""

import os
import pandas as pd
import binana
import random
from tqdm import tqdm
import kier
import ecif
from openbabel import pybel
from rdkit import Chem, RDLogger
import json
import dask.dataframe as dd
# Removed multiprocessing imports, replaced by dask
from functools import partial
from binana import PDB
# Removed Pool, TimeoutError from multiprocessing
# Removed signal, wraps (if they were for timeout handling not shown)
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import glob
import argparse
import sys
from dask.distributed import Client, LocalCluster # Use dask.distributed
import dask
import numpy as np # For numeric types if needed
import traceback # For detailed error printing


# --- Configuration & Setup ---
# Mute RDKit/OpenBabel warnings
RDLogger.logger().setLevel(RDLogger.ERROR)
try:
    ob_log_handler = pybel.ob.OBMessageHandler()
    ob_log_handler.SetOutputLevel(0)
    pybel.ob.obErrorLog.StopLogging()
except Exception as e:
    print(f"Warning: Could not configure OpenBabel logging: {e}")


# --- Feature Calculation Functions (Modified for Robustness & Type Consistency) ---

def calculate_ecifs(ligand_pdbqt_block, receptor_path): # Pass path for ECIF
    """
    Calculate Extended Connectivity Interaction Features (ECIF).
    Returns a DataFrame with numeric features (0 on error).
    """
    try:
        # Ensure ECIF library gets path if needed, or adjust call signature
        # Assuming ecif.GetECIF takes receptor path and ligand block
        ecif_data_list = ecif.GetECIF(receptor_path, ligand_pdbqt_block, distance_cutoff=6.0)
        ecif_headers = [header.replace(';', '') for header in ecif.PossibleECIF]

        # Ensure data is numeric, defaulting to 0
        ecif_data_dict = {}
        for header, value in zip(ecif_headers, ecif_data_list):
            try:
                ecif_data_dict[header] = np.float32(value) # Use float32 for memory
            except (ValueError, TypeError):
                ecif_data_dict[header] = np.float32(0.0)

        return pd.DataFrame(ecif_data_dict, index=[0])

    except Exception as e:
        print(f"Error calculating ECIF features: {e}. Returning zeros.")
        # Return DataFrame of zeros with expected columns
        ecif_headers = [header.replace(';', '') for header in ecif.PossibleECIF]
        return pd.DataFrame({header: [np.float32(0.0)] for header in ecif_headers})

def kier_flexibility(ligand_pdbqt_block):
    """
    Calculate Kier flexibility and RDKit descriptors.
    Returns Kier (float) and a DataFrame of numeric RDKit descriptors (0 on error).
    """
    # List of rdkit descriptors to calculate
    # (Assuming these are all numeric)
    invariant_rdkit_descriptors = [
        'HeavyAtomMolWt','HeavyAtomCount','NumRotatableBonds', 'RingCount', 'NumAromaticRings', 'NumAliphaticRings',
        'NumSaturatedRings', 'NumHeterocycles', 'NumAromaticHeterocycles',
        'NumAliphaticHeterocycles', 'NumSaturatedHeterocycles', 'FractionCSP3',
        'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v', 'BalabanJ', 'BertzCT',
        'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3',
        'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
        'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10',
        'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
        'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
        'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SMR_VSA10',
        'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
        'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10',
        'SlogP_VSA11', 'SlogP_VSA12',
        'TPSA'
    ]
    # Create default zero dictionary first
    default_features = {name: np.float32(0.0) for name in invariant_rdkit_descriptors}

    try:
        mol = kier.SmilePrep(ligand_pdbqt_block) # Assuming SmilePrep handles PDBQT block
        if mol is None:
             raise ValueError("RDKit could not parse ligand block")
        mol.GetRingInfo() # Necessary?
        mol_without_H = Chem.RemoveHs(mol)

        descriptor_names = [desc[0] for desc in Descriptors._descList] # All available
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        all_descriptors_values = calculator.CalcDescriptors(mol_without_H)

        # Filter and ensure numeric type
        features = {}
        descriptor_map = dict(zip(descriptor_names, all_descriptors_values))

        for name in invariant_rdkit_descriptors:
            value = descriptor_map.get(name) # Get value if exists
            try:
                # Attempt conversion, default to 0 if fail or missing
                features[name] = np.float32(value) if value is not None else np.float32(0.0)
            except (ValueError, TypeError):
                 features[name] = np.float32(0.0)

        kier_flex = kier.CalculateFlexibility(mol)
        kier_flex = np.float32(kier_flex) if kier_flex is not None else np.float32(0.0)

        return kier_flex, pd.DataFrame([features]) # Return DF here

    except Exception as e:
        print(f"Error calculating Kier/RDKit: {e}. Returning defaults.")
        # Return default Kier and DataFrame of zeros
        return np.float32(0.0), pd.DataFrame([default_features])


def run_binana(ligand_pdbqt_block_lines, protein_object): # Pass lines and object
    """
    Calculate BINANA features.
    Returns a DataFrame with numeric features (0 or NaN on error/missing).
    """
    # Define feature names exactly as they will appear in the DataFrame
    # Using np.float32 for consistency and memory efficiency
    default_value = np.float32(np.nan) # Use NaN for missing interactions

    # Define all expected BINANA feature columns
    # (This list should match the keys used below)
    binana_feature_cols = [
        "closest_2.5_HD_OA", "closest_2.5_HD_HD", "closest_2.5_HD_N", "closest_2.5_C_HD",
        "closest_2.5_OA_ZN", "closest_2.5_HD_ZN", "closest_2.5_A_HD",
        "close_4.0_C_C", "close_4.0_HD_OA", "close_4.0_C_HD", "close_4.0_C_N", "close_4.0_A_C",
        "close_4.0_A_OA", "close_4.0_N_OA", "close_4.0_A_N", "close_4.0_HD_N", "close_4.0_HD_HD",
        "close_4.0_A_HD", "close_4.0_OA_OA", "close_4.0_C_OA", "close_4.0_N_N", "close_4.0_C_SA",
        "close_4.0_HD_SA", "close_4.0_OA_SA", "close_4.0_N_SA", "close_4.0_A_A", "close_4.0_HD_S",
        "close_4.0_S_ZN", "close_4.0_N_ZN", "close_4.0_HD_ZN", "close_4.0_A_SA", "close_4.0_OA_ZN",
        "close_4.0_C_ZN", "close_4.0_C_NA", "close_4.0_NA_OA", "close_4.0_HD_NA", "close_4.0_N_NA",
        "close_4.0_A_NA", "close_4.0_BR_C", "close_4.0_HD_P", "close_4.0_F_N", "close_4.0_F_HD",
        "close_4.0_C_CL", "close_4.0_CL_HD",
        "LA_N_exists", "LA_HD_exists", # Renamed for clarity (binary 0/1)
        "ElSum_C_C", "ElSum_HD_OA", "ElSum_C_HD", "ElSum_C_N", "ElSum_A_C", "ElSum_A_OA",
        "ElSum_N_OA", "ElSum_A_N", "ElSum_HD_HD", "ElSum_A_HD", "ElSum_OA_OA", "ElSum_C_OA",
        "ElSum_N_N", "ElSum_C_SA", "ElSum_HD_SA", "ElSum_OA_SA", "ElSum_N_SA", "ElSum_A_A",
        "ElSum_N_S", "ElSum_HD_S", "ElSum_OA_S", "ElSum_A_SA", "ElSum_C_NA", "ElSum_NA_OA",
        "ElSum_HD_NA", "ElSum_N_NA", "ElSum_A_NA", "ElSum_BR_C", "ElSum_HD_P", "ElSum_OA_P",
        "ElSum_N_P", "ElSum_C_F", "ElSum_F_N", "ElSum_A_F", "ElSum_CL_OA", "ElSum_C_CL",
        "ElSum_CL_N", "ElSum_A_CL",
        "BPF_ALPHA_SIDECHAIN", "BPF_ALPHA_BACKBONE", "BPF_BETA_SIDECHAIN", "BPF_BETA_BACKBONE",
        "BPF_OTHER_SIDECHAIN", "BPF_OTHER_BACKBONE",
        "HC_ALPHA_SIDECHAIN", "HC_ALPHA_BACKBONE", "HC_BETA_SIDECHAIN", "HC_BETA_BACKBONE",
        "HC_OTHER_SIDECHAIN", "HC_OTHER_BACKBONE",
        "HB_ALPHA_SIDECHAIN_LIGAND", "HB_BETA_SIDECHAIN_LIGAND", "HB_BETA_BACKBONE_LIGAND",
        "HB_OTHER_SIDECHAIN_LIGAND", "HB_OTHER_BACKBONE_LIGAND",
        "HB_ALPHA_SIDECHAIN_RECEPTOR", "HB_ALPHA_BACKBONE_RECEPTOR", "HB_BETA_SIDECHAIN_RECEPTOR",
        "HB_BETA_BACKBONE_RECEPTOR", "HB_OTHER_SIDECHAIN_RECEPTOR", "HB_OTHER_BACKBONE_RECEPTOR",
        "SB_ALPHA", "SB_BETA", "SB_OTHER",
        "piStack_ALPHA", "piStack_BETA", "piStack_OTHER",
        "tStack_ALPHA", "tStack_BETA", "tStack_OTHER",
        "catPi_BETA_LIGAND", "catPi_OTHER_LIGAND",
        "nRot"
    ]
    # Create a default DataFrame structure
    default_binana_df = pd.DataFrame({col: [default_value] for col in binana_feature_cols})

    try:
        # Assuming binana.Binana needs string block and PDB object
        # Reconstruct minimal block if needed, or pass lines if Binana handles it
        ligand_block_str = "\n".join(ligand_pdbqt_block_lines)
        main_binana_out = binana.Binana(ligand_block_str, protein_object).out
        binana_features_calc = {} # Store calculated features here

        # Helper function to safely get and convert feature
        def get_binana_feature(category, key, default=default_value):
            val = main_binana_out.get(category, {}).get(key)
            try:
                return np.float32(val) if val is not None else default
            except (ValueError, TypeError):
                return default

        # Extract closest contacts
        for contact_def in ["2.5 (HD, OA)", "2.5 (HD, HD)", "2.5 (HD, N)", "2.5 (C, HD)", "2.5 (OA, ZN)", "2.5 (HD, ZN)", "2.5 (A, HD)"]:
            binana_key = contact_def.split('(')[-1].split(')')[0].replace(', ', '_')
            df_key = f"closest_{contact_def.split(' ')[0]}_{binana_key}"
            binana_features_calc[df_key] = get_binana_feature('closest', binana_key)

        # Extract close contacts
        for contact_def in ["4.0 (C, C)", "4.0 (HD, OA)", "4.0 (C, HD)", "4.0 (C, N)", "4.0 (A, C)", "4.0 (A, OA)", "4.0 (N, OA)", "4.0 (A, N)", "4.0 (HD, N)", "4.0 (HD, HD)", "4.0 (A, HD)", "4.0 (OA, OA)", "4.0 (C, OA)", "4.0 (N, N)", "4.0 (C, SA)", "4.0 (HD, SA)", "4.0 (OA, SA)", "4.0 (N, SA)", "4.0 (A, A)", "4.0 (HD, S)", "4.0 (S, ZN)", "4.0 (N, ZN)", "4.0 (HD, ZN)", "4.0 (A, SA)", "4.0 (OA, ZN)", "4.0 (C, ZN)", "4.0 (C, NA)", "4.0 (NA, OA)", "4.0 (HD, NA)", "4.0 (N, NA)", "4.0 (A, NA)", "4.0 (BR, C)", "4.0 (HD, P)", "4.0 (F, N)", "4.0 (F, HD)", "4.0 (C, CL)", "4.0 (CL, HD)"]:
            binana_key = contact_def.split('(')[-1].split(')')[0].replace(', ', '_')
            df_key = f"close_{contact_def.split(' ')[0]}_{binana_key}"
            binana_features_calc[df_key] = get_binana_feature('close', binana_key)

        # Extract ligand atoms (as 0/1)
        for atom_def in ["LA N", "LA HD"]:
             binana_key = atom_def.split()[-1]
             df_key = f"{atom_def.replace(' ','_')}_exists"
             binana_features_calc[df_key] = np.float32(1.0) if main_binana_out.get('ligand_atoms', {}).get(binana_key) is not None else np.float32(0.0)


        # Extract electrostatics sums
        for elsum_def in ["ElSum (C, C)", "ElSum (HD, OA)", "ElSum (C, HD)", "ElSum (C, N)", "ElSum (A, C)", "ElSum (A, OA)", "ElSum (N, OA)", "ElSum (A, N)", "ElSum (HD, HD)", "ElSum (A, HD)", "ElSum (OA, OA)", "ElSum (C, OA)", "ElSum (N, N)", "ElSum (C, SA)", "ElSum (HD, SA)", "ElSum (OA, SA)", "ElSum (N, SA)", "ElSum (A, A)", "ElSum (N, S)", "ElSum (HD, S)", "ElSum (OA, S)", "ElSum (A, SA)", "ElSum (C, NA)", "ElSum (NA, OA)", "ElSum (HD, NA)", "ElSum (N, NA)", "ElSum (A, NA)", "ElSum (BR, C)", "ElSum (HD, P)", "ElSum (OA, P)", "ElSum (N, P)", "ElSum (C, F)", "ElSum (F, N)", "ElSum (A, F)", "ElSum (CL, OA)", "ElSum (C, CL)", "ElSum (CL, N)", "ElSum (A, CL)"]:
            binana_key = elsum_def.split('(')[-1].split(')')[0].replace(', ', '_')
            df_key = elsum_def.split(' ')[0] + "_" + binana_key # e.g., ElSum_C_C
            binana_features_calc[df_key] = get_binana_feature('elsums', binana_key)

        # Extract other features (BPF, HC, HB, SB, Stacking, CationPi, nRot)
        feature_map = {
            "BPF_ALPHA_SIDECHAIN": ('bpfs', "SIDECHAIN_ALPHA"), "BPF_ALPHA_BACKBONE": ('bpfs', "BACKBONE_ALPHA"),
            "BPF_BETA_SIDECHAIN": ('bpfs', "SIDECHAIN_BETA"), "BPF_BETA_BACKBONE": ('bpfs', "BACKBONE_BETA"),
            "BPF_OTHER_SIDECHAIN": ('bpfs', "SIDECHAIN_OTHER"), "BPF_OTHER_BACKBONE": ('bpfs', "BACKBONE_OTHER"),
            "HC_ALPHA_SIDECHAIN": ('hydrophobics', "SIDECHAIN_ALPHA"), "HC_ALPHA_BACKBONE": ('hydrophobics', "BACKBONE_ALPHA"),
            "HC_BETA_SIDECHAIN": ('hydrophobics', "SIDECHAIN_BETA"), "HC_BETA_BACKBONE": ('hydrophobics', "BACKBONE_BETA"),
            "HC_OTHER_SIDECHAIN": ('hydrophobics', "SIDECHAIN_OTHER"), "HC_OTHER_BACKBONE": ('hydrophobics', "BACKBONE_OTHER"),
            "HB_ALPHA_SIDECHAIN_LIGAND": ('hbonds', "HDONOR_LIGAND_SIDECHAIN_ALPHA"), "HB_BETA_SIDECHAIN_LIGAND": ('hbonds', "HDONOR_LIGAND_SIDECHAIN_BETA"),
            "HB_BETA_BACKBONE_LIGAND": ('hbonds', "HDONOR_LIGAND_BACKBONE_BETA"), "HB_OTHER_SIDECHAIN_LIGAND": ('hbonds', "HDONOR_LIGAND_SIDECHAIN_OTHER"),
            "HB_OTHER_BACKBONE_LIGAND": ('hbonds', "HDONOR_LIGAND_BACKBONE_OTHER"), "HB_ALPHA_SIDECHAIN_RECEPTOR": ('hbonds', "HDONOR_RECEPTOR_SIDECHAIN_ALPHA"),
            "HB_ALPHA_BACKBONE_RECEPTOR": ('hbonds', "HDONOR_RECEPTOR_BACKBONE_ALPHA"), "HB_BETA_SIDECHAIN_RECEPTOR": ('hbonds', "HDONOR_RECEPTOR_SIDECHAIN_BETA"),
            "HB_BETA_BACKBONE_RECEPTOR": ('hbonds', "HDONOR_RECEPTOR_BACKBONE_BETA"), "HB_OTHER_SIDECHAIN_RECEPTOR": ('hbonds', "HDONOR_RECEPTOR_SIDECHAIN_OTHER"),
            "HB_OTHER_BACKBONE_RECEPTOR": ('hbonds', "HDONOR_RECEPTOR_BACKBONE_OTHER"),
            "SB_ALPHA": ('salt_bridges', "SALT-BRIDGE_ALPHA"), "SB_BETA": ('salt_bridges', "SALT-BRIDGE_BETA"), "SB_OTHER": ('salt_bridges', "SALT-BRIDGE_OTHER"),
            "piStack_ALPHA": ('stacking', "STACKING ALPHA"), "piStack_BETA": ('stacking', "STACKING BETA"), "piStack_OTHER": ('stacking', "STACKING OTHER"),
            "tStack_ALPHA": ('t_stacking', "T-SHAPED_ALPHA"), "tStack_BETA": ('t_stacking', "T-SHAPED_BETA"), "tStack_OTHER": ('t_stacking', "T-SHAPED_OTHER"),
            "catPi_BETA_LIGAND": ('pi_cation', "PI-CATION_LIGAND-CHARGED_BETA"), "catPi_OTHER_LIGAND": ('pi_cation', "PI-CATION_LIGAND-CHARGED_OTHER"),
            "nRot": ('nrot', None) # Special case for nrot (directly accessible)
        }

        for df_key, (category, binana_key) in feature_map.items():
             if category == 'nrot':
                 val = main_binana_out.get('nrot')
                 binana_features_calc[df_key] = np.float32(val) if val is not None else default_value
             else:
                 binana_features_calc[df_key] = get_binana_feature(category, binana_key)

        # Create DataFrame from calculated features, ensuring all columns are present
        # Use reindex to guarantee all expected columns exist, filling missing ones with default
        calculated_df = pd.DataFrame([binana_features_calc])
        final_binana_df = calculated_df.reindex(columns=binana_feature_cols, fill_value=default_value)

        # Ensure dtype is float32
        return final_binana_df.astype(np.float32)

    except Exception as e:
        print(f"Error calculating BINANA features: {e}. Returning defaults.")
        # Return the default DataFrame on error
        return default_binana_df.astype(np.float32)


# Pruning function remains similar, but assumes input `df` has numeric features
# and SC1_features.json lists the numeric columns we need.
def prune_df_headers(df, reference_headers_list):
    """
    Filter DataFrame to include only the required feature columns.
    Ensures selected columns are present, filling with NaN if missing.
    Converts selected columns to float32.

    Args:
        df: DataFrame with calculated features
        reference_headers_list: List of required feature column names

    Returns:
        DataFrame with only the required numeric columns (float32)
    """
    try:
        # Ensure all reference headers exist, fill missing with NaN
        pruned_df = df.reindex(columns=reference_headers_list, fill_value=np.nan)

        # Convert all columns to float32 for consistency
        for col in pruned_df.columns:
            # Use errors='coerce' to turn uncastable values into NaN
            pruned_df[col] = pd.to_numeric(pruned_df[col], errors='coerce')

        return pruned_df.astype(np.float32)

    except Exception as e:
        print(f"Error pruning DataFrame headers: {e}")
        # Return an empty DataFrame with expected columns filled with NaN
        return pd.DataFrame(columns=reference_headers_list, dtype=np.float32).fillna(np.nan)


# --- Molecule Processing Function (Revised) ---
def process_molecule(molecule, ligand_path, protein_object, pdbid, protein_path, reference_feature_list):
    """
    Process a single ligand molecule file and extract NUMERIC features from all poses.
    Adds 'Id' (filename) and 'Pose' number columns.

    Args:
        molecule: Filename of the ligand file
        ligand_path: Path to the directory containing ligand files
        protein_object: Pre-loaded Protein PDB object (from binana)
        pdbid: PDB ID being processed (for context, not used directly here)
        protein_path: Path to the protein file (used by ECIF)
        reference_feature_list: List of expected numeric feature column names (from SC1_features.json)

    Returns:
        Pandas DataFrame containing numeric features + 'Id' and 'Pose' for all valid poses.
        Returns an empty DataFrame if the molecule fails critically or has no valid poses.
    """
    all_pose_results = []
    expected_final_columns = None # Store final structure

    try:
        ligand_file = os.path.join(ligand_path, molecule)
        with open(ligand_file, 'r') as f:
            lig_text = f.read()

        # Split ligand file into poses (handle single-pose files)
        if 'MODEL ' in lig_text:
            lig_poses_raw = lig_text.split('MODEL ')[1:]
        else:
            lig_poses_raw = [lig_text] if lig_text.strip() else []

        if not lig_poses_raw:
             print(f"Warning: No content found in {molecule}")
             return pd.DataFrame() # Return empty if file is empty

        # Process each pose
        for i, pose_raw in enumerate(lig_poses_raw):
            pose_index = i + 1 # 1-based pose index
            pose_df = None # DataFrame for the current pose

            try:
                # Minimal PDBQT block reconstruction/cleaning
                pose_lines = pose_raw.strip().split('\n')
                # Keep only essential lines for most parsers; adjust if more needed
                # This cleaning might need refinement based on PDBQT variations
                clean_atom_hetatm_lines = [line for line in pose_lines if line.startswith(('ATOM', 'HETATM'))]
                if not clean_atom_hetatm_lines:
                    # print(f"Debug: No ATOM/HETATM lines in pose {pose_index} of {molecule}")
                    continue # Skip empty/invalid poses

                # Reconstruct minimal text block if needed by functions
                pose_text_block = "\n".join(clean_atom_hetatm_lines)

                # --- Calculate all feature sets ---
                # These functions now return DataFrames with numeric types or defaults
                ecif_df = calculate_ecifs(pose_text_block, protein_path)
                kier_val, rdkit_df = kier_flexibility(pose_text_block)
                binana_df = run_binana(clean_atom_hetatm_lines, protein_object) # Pass lines if needed

                # --- Combine Numeric Features ---
                # Concatenate the numeric feature DataFrames
                numeric_df_pose = pd.concat([ecif_df, binana_df, rdkit_df], axis=1)
                numeric_df_pose['Kier Flexibility'] = kier_val # Add Kier value

                # --- Prune to Reference Numeric Features ---
                # Pass the list of required numeric columns from the JSON
                pruned_numeric_df = prune_df_headers(numeric_df_pose, reference_feature_list)

                # --- Add Identifiers ---
                # Add 'Id' (object/string) and 'Pose' (int) columns
                pruned_numeric_df['Id'] = molecule # Filename as string
                pruned_numeric_df['Pose'] = np.int32(pose_index) # Pose number as integer

                pose_df = pruned_numeric_df

                # Store column structure from the first successfully processed pose
                if expected_final_columns is None:
                    expected_final_columns = pose_df.columns.tolist()

                all_pose_results.append(pose_df)

            except Exception as e:
                print(f"Error processing pose {pose_index} in {molecule}: {e}")
                # traceback.print_exc() # Uncomment for detailed debug
                continue # Skip this pose on error

        # --- Combine results from all poses within this molecule ---
        if all_pose_results:
            try:
                # Concatenate all valid pose DataFrames for this molecule
                molecule_results_df = pd.concat(all_pose_results, ignore_index=True)

                # Ensure final DataFrame has consistent columns & types
                if expected_final_columns:
                     # Reindex to ensure all columns are present, fill missing numeric with NaN
                     molecule_results_df = molecule_results_df.reindex(columns=expected_final_columns)
                     # Re-apply types for safety, esp. after concat might change them
                     for col in molecule_results_df.columns:
                          if col == 'Id':
                               molecule_results_df[col] = molecule_results_df[col].astype(object)
                          elif col == 'Pose':
                               molecule_results_df[col] = pd.to_numeric(molecule_results_df[col], errors='coerce').fillna(0).astype(np.int32)
                          else: # Assume others should be float32
                               molecule_results_df[col] = pd.to_numeric(molecule_results_df[col], errors='coerce').astype(np.float32)
                return molecule_results_df

            except Exception as e:
                print(f"Error concatenating results for {molecule}: {e}")
                # Return empty DataFrame with expected columns if possible
                if expected_final_columns:
                    return pd.DataFrame(columns=expected_final_columns)
                return pd.DataFrame() # Fallback generic empty
        else:
            # No valid poses processed for this molecule
            # Return empty DF potentially with columns if known, else generic empty
             if expected_final_columns:
                 return pd.DataFrame(columns=expected_final_columns)
             return pd.DataFrame()

    except Exception as e:
        print(f"Critical error processing molecule file {molecule}: {e}")
        traceback.print_exc() # Detailed error for critical failures
        # Return generic empty DataFrame
        return pd.DataFrame()


# --- PDB ID Processing Function (Using Dask) ---
def process_pdbid(pdbid, protein_base_path, molecule_path, des_path, reference_feature_list):
    """
    Process a single PDB ID using Dask for parallelism and save to Parquet.

    Args:
        pdbid: The PDB ID to process
        protein_base_path: Path to directory containing protein PDBQT files
        molecule_path: Path to directory containing molecule PDBQT files
        des_path: Directory to save output files (will be a directory per PDBID)
        reference_feature_list: List of required numeric feature column names
    """
    print(f"Starting processing for {pdbid}")
    # Define output path (Parquet writes to a directory)
    output_dir = os.path.join(des_path, f'{pdbid}_features.parquet')

    # Find the protein file
    protein_path_list = glob.glob(os.path.join(protein_base_path, f'{pdbid}*.pdbqt'))
    if not protein_path_list:
        print(f'Protein file not found for {pdbid} in {protein_base_path}')
        return
    protein_path = protein_path_list[0] # Use the first match

    # Check if output *directory* already exists
    # Consider adding an --overwrite flag if needed
    if os.path.exists(output_dir):
        print(f'Output directory {output_dir} already exists - skipping {pdbid}')
        return

    # Check if molecule directory exists and find molecule files
    molecule_dir = os.path.join(molecule_path, pdbid)
    if not os.path.isdir(molecule_dir): # Check if it's a directory
        print(f'Molecules directory not found for {pdbid} at {molecule_dir}')
        return
    try:
        # Ensure we only list files, not subdirectories
        molecules = [f for f in os.listdir(molecule_dir) if os.path.isfile(os.path.join(molecule_dir, f))]
    except OSError as e:
        print(f"Error listing molecules in {molecule_dir}: {e}")
        return

    if not molecules:
        print(f'No molecule files found in {molecule_dir} for {pdbid}')
        return

    # Read protein content and prepare protein object (do this once)
    try:
        print(f"Loading protein {protein_path} for {pdbid}")
        # Assuming binana PDB class can load directly from path
        protein_object = PDB()
        protein_object.load_PDB(protein_path)
        protein_object.assign_secondary_structure()
        print(f"Protein {pdbid} loaded successfully.")

    except Exception as e:
        print(f'Error loading or processing protein {protein_path} for {pdbid}: {e}')
        traceback.print_exc()
        return

    # --- Dask Execution ---
    # Wrap the molecule processing function with dask.delayed
    # Use partial to fix the arguments that are the same for all molecules
    delayed_process_molecule = dask.delayed(partial(
        process_molecule,
        ligand_path=molecule_dir,
        # Pass pdbid for context if needed, though not used directly in process_molecule now
        pdbid=pdbid,
        protein_object=protein_object, # Pass the loaded object
        protein_path=protein_path,
        reference_feature_list=reference_feature_list # Pass the required features
    ))

    # Create a list of delayed tasks, one for each molecule
    print(f"Setting up Dask delayed tasks for {len(molecules)} molecules in {pdbid}...")
    delayed_results = [delayed_process_molecule(molecule) for molecule in molecules]

    if not delayed_results:
        print(f"No tasks to process for {pdbid}")
        return

    # --- Get Metadata for Dask DataFrame ---
    # Run one task immediately to get the structure (columns and dtypes)
    print(f"Computing metadata for Dask DataFrame for {pdbid}...")
    meta_df = None
    processed_meta = False
    # Try a few molecules in case the first one fails or is empty
    for i, task in enumerate(delayed_results[:min(5, len(delayed_results))]): # Try up to 5 samples
         try:
             sample_result = task.compute()
             if isinstance(sample_result, pd.DataFrame) and not sample_result.empty:
                 meta_df = sample_result.head(0) # Get empty DF with correct structure
                 print(f"Metadata computed successfully for {pdbid} using molecule {i+1}.")
                 processed_meta = True
                 break # Stop after first success
             # else: print(f"Debug: Sample {i+1} for {pdbid} was empty or not a DataFrame.")
         except Exception as e:
             print(f"Warning: Error computing metadata sample {i+1} for {pdbid}: {e}")

    if not processed_meta or meta_df is None:
        print(f"Error: Failed to obtain valid metadata for {pdbid} after checking samples. Cannot create Dask DataFrame.")
        # Optional: Define meta manually if structure is absolutely fixed
        # known_numeric_cols = reference_feature_list + ['Kier Flexibility']
        # meta_structure = {col: np.float32 for col in known_numeric_cols}
        # meta_structure['Id'] = object
        # meta_structure['Pose'] = np.int32
        # try:
        #      meta_df = pd.DataFrame(columns=meta_structure.keys()).astype(meta_structure)
        #      print("Warning: Using manually defined metadata structure.")
        # except:
        #      print("Error: Manual metadata definition also failed.")
        #      return
        return # Exit if meta cannot be determined


    # --- Create and Compute Dask DataFrame ---
    try:
        # Create a Dask DataFrame from the delayed tasks
        print(f"Creating Dask DataFrame for {pdbid}...")
        # Ensure meta includes 'Id' (object) and 'Pose' (int) along with numeric features
        dask_df = dd.from_delayed(delayed_results, meta=meta_df)

        print(f"Writing features for {pdbid} to Parquet directory: {output_dir}")
        # Write directly to Parquet directory using Dask
        os.makedirs(des_path, exist_ok=True) # Ensure base output dir exists
        # Use compute=True for to_parquet to block until completion
        dask_df.to_parquet(
            output_dir,
            engine='pyarrow', # or 'fastparquet'
            write_index=False,
            overwrite=True, # Replace directory if it exists
            compute=True # Ensure computation happens and blocks
        )

        # Can't easily get exact row count without another compute, but confirm completion
        print(f"Successfully saved features for {pdbid} to {output_dir}")

    except Exception as e:
        print(f"Error during Dask computation or saving for {pdbid}: {e}")
        traceback.print_exc()


# --- Main function adjusted for Dask Client ---
def main(args):
    """
    Main execution function using Dask for parallel processing.
    """
    des_path = args.output_dir
    protein_base_path = args.protein_dir
    molecule_path = args.ligand_dir
    num_cores = args.num_cores

    os.makedirs(des_path, exist_ok=True)

    # --- Load Reference Feature List ---
    # Assuming SC1_features.json is in the same directory as the script
    reference_feature_list = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, 'SC1_features.json')
        with open(json_path, 'r') as f:
            reference_headers = json.load(f)
            # Using '492_models_58' key as in prune_df_headers
            reference_feature_list = reference_headers.get('492_models_58')
            if not reference_feature_list:
                 print(f"Error: Key '492_models_58' not found or empty in {json_path}")
                 sys.exit(1)
            print(f"Loaded {len(reference_feature_list)} reference feature names from JSON.")
    except FileNotFoundError:
        print(f"Error: SC1_features.json not found in script directory ({script_dir})")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or parsing SC1_features.json: {e}")
        sys.exit(1)


    # --- Setup Dask Client ---
    if num_cores is None:
        processes = max(1, os.cpu_count() - 1) # Leave one core free
    else:
        processes = min(num_cores, os.cpu_count())

    print(f"Setting up Dask LocalCluster with {processes} workers.")
    # Using threads_per_worker=1 often good for C-extension heavy tasks (RDKit, etc.)
    # Adjust memory_limit based on typical task memory usage; 'auto' is often reasonable.
    cluster = LocalCluster(n_workers=processes, threads_per_worker=1, memory_limit='auto')
    client = Client(cluster)
    print(f"Dask dashboard link: {client.dashboard_link}")


    # --- Get list of PDB IDs ---
    if args.pdbids:
        pdbids = args.pdbids.split(',')
        print(f"Processing {len(pdbids)} specified PDB IDs")
    else:
        # Extract PDB IDs from filenames in protein directory
        pdbids = set()
        try:
            for filename in os.listdir(protein_base_path):
                # Check for .pdbqt extension and try to extract PDBID prefix
                if filename.endswith(".pdbqt"):
                    # Assuming format PDBID_something.pdbqt or just PDBID.pdbqt
                    pdbid = filename.split('_')[0].split('.')[0]
                    if len(pdbid) == 4: # Basic check for PDB ID format
                         pdbids.add(pdbid)
                    # Add more sophisticated PDB ID extraction if needed
            pdbids = sorted(list(pdbids))
            if not pdbids:
                 print(f"Warning: No files matching PDBID*.pdbqt found in {protein_base_path}")
            else:
                 print(f"Found {len(pdbids)} potential PDB ID prefixes in {protein_base_path}")
        except FileNotFoundError:
             print(f"Error: Protein directory not found: {protein_base_path}")
             client.close()
             cluster.close()
             sys.exit(1)
        except Exception as e:
             print(f"Error listing protein directory {protein_base_path}: {e}")
             client.close()
             cluster.close()
             sys.exit(1)

    if not pdbids:
        print("No PDB IDs to process. Exiting.")
        client.close()
        cluster.close()
        sys.exit(0)


    # --- Process each PDB ID ---
    print("Starting PDB processing loop...")
    # Wrap the loop with tqdm for overall progress
    for pdbid in tqdm(pdbids, desc="Overall PDB Processing"):
        try:
            # Pass the loaded reference feature list to process_pdbid
            process_pdbid(pdbid, protein_base_path, molecule_path, des_path, reference_feature_list)
        except Exception as e:
            # Catch errors at the PDBID level to allow processing to continue
            print(f"FATAL ERROR during processing of PDBID {pdbid}: {e}")
            traceback.print_exc() # Print full traceback for debugging

    # --- Shutdown Dask Client ---
    print("\nProcessing loop finished. Shutting down Dask client.")
    try:
        client.close()
        cluster.close()
        print("Dask client and cluster closed.")
    except Exception as e:
        print(f"Warning: Error shutting down Dask client/cluster: {e}")


# --- Argument Parser and Main Execution Block ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SCORCH2 Feature Extraction Tool (Dask/Parquet Accelerated)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--output-dir', dest='output_dir', type=str, required=True,
                        help="Directory to save output Parquet directories (one per PDBID).")
    parser.add_argument('--protein-dir', dest='protein_dir', type=str, required=True,
                        help="Directory containing protein PDBQT files.")
    parser.add_argument('--ligand-dir', dest='ligand_dir', type=str, required=True,
                        help="Base directory containing ligand PDBQT files (organized by PDBID subdirectories).")

    # Optional arguments
    parser.add_argument('--pdbids', type=str, default=None,
                        help="Comma-separated list of specific PDB IDs to process (optional).")
    parser.add_argument('--num-cores', dest='num_cores', type=int, default=None,
                        help="Number of CPU cores for Dask workers (default: all available cores minus 1).")
    # Removed verbose argument as output is primarily through print statements now

    args = parser.parse_args()

    print("================================================================")
    print(" SCORCH2 Feature Extraction Tool (Dask/Parquet Accelerated) ")
    print("================================================================")
    print(f"Protein directory: {args.protein_dir}")
    print(f"Ligand directory : {args.ligand_dir}")
    print(f"Output directory : {args.output_dir}")
    if args.pdbids: print(f"PDB IDs        : Processing specified list")
    if args.num_cores: print(f"Num Cores      : {args.num_cores}")

    try:
        main(args)
        print("\nFeature extraction completed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in main: {e}")
        traceback.print_exc()
        sys.exit(1)