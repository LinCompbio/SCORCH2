#!/usr/bin/env python3
"""
SCORCH2 Rescoring Code

This script performs the complete SCORCH2 workflow including feature extraction,
normalization, and rescoring in a single streamlined process.

Usage:
    python scorch2_rescoring.py --protein-dir protein/ --ligand-dir molecule/ \
                               --sc2_ps_model sc2_ps.xgb --sc2_pb_model sc2_pb.xgb \
                               --output results.csv

Authors: Lin Chen
License: MIT
"""

import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import joblib
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional


def run_feature_extraction(protein_dir: str, ligand_dir: str, output_dir: str, num_cores: int = None) -> bool:
    """
    Run feature extraction using the SCORCH2 feature extraction utility.
    
    Args:
        protein_dir: Directory containing protein PDBQT files
        ligand_dir: Directory containing ligand PDBQT files
        output_dir: Directory to save extracted features
        num_cores: Number of CPU cores to use (default: os.cpu_count()-1)
        
    Returns:
        True if successful, False otherwise
    """
    if num_cores is None:
        num_cores = max(1, os.cpu_count() - 1)
    
    print("🔄 Running feature extraction...")
    print(f"  - Using {num_cores} CPU cores")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run feature extraction with progress monitoring
    cmd = [
        sys.executable, "utils/scorch2_feature_extraction.py",
        "--protein-dir", protein_dir,
        "--ligand-dir", ligand_dir, 
        "--output-dir", output_dir,
        "--num-cores", str(num_cores)
    ]
    
    try:
        # Use tqdm to show progress
        with tqdm(desc="Feature extraction", unit="step") as pbar:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Monitor the process
            while process.poll() is None:
                pbar.update(1)
                import time
                time.sleep(1)
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print("✓ Feature extraction completed successfully")
                return True
            else:
                print(f"❌ Feature extraction failed: {stderr}")
                return False
                
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False


def normalize_features(feature_file: str, ps_scaler_path: str, pb_scaler_path: str, 
                      output_dir: str) -> Tuple[str, str]:
    """
    Normalize features using the pre-trained scalers.
    
    Args:
        feature_file: Path to the raw feature CSV file
        ps_scaler_path: Path to SC2-PS scaler
        pb_scaler_path: Path to SC2-PB scaler
        output_dir: Directory to save normalized features
        
    Returns:
        Tuple of (ps_normalized_file, pb_normalized_file)
    """
    print("🔄 Normalizing features...")
    
    # Load raw features
    df = pd.read_csv(feature_file)
    df.fillna(0, inplace=True,axis=1)
    print(f"✓ Loaded features: {df.shape[0]} compounds, {df.shape[1]} features")
    
    # Check if Id column exists, if not, it's likely the last column
    if 'Id' not in df.columns:
        # Assume the last column is the Id column
        df.columns = list(df.columns[:-1]) + ['Id']
        print("✓ Detected Id column as the last column")
    
    # Create output directories
    ps_output_dir = os.path.join(output_dir, "sc2_ps")
    pb_output_dir = os.path.join(output_dir, "sc2_pb")
    os.makedirs(ps_output_dir, exist_ok=True)
    os.makedirs(pb_output_dir, exist_ok=True)
    
    # Extract base filename
    base_filename = os.path.basename(feature_file).replace('_features.csv', '_normalized.csv')
    
    # Load scalers
    try:
        ps_scaler = joblib.load(ps_scaler_path)
        pb_scaler = joblib.load(pb_scaler_path)
        print("✓ Scalers loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load scalers: {e}")
    
    # Separate IDs from features first
    ids = df['Id'].copy()
    X_all = df.drop(['Id'], axis=1)
    
    # Remove features that are not expected by the scalers
    features_to_remove = []
    if 'NumHeterocycles' in X_all.columns:
        features_to_remove.append('NumHeterocycles')
    
    if features_to_remove:
        X_all = X_all.drop(features_to_remove, axis=1)
        print(f"✓ Removed unexpected features: {features_to_remove}")
    
    # Prepare PS features (remove NumRotatableBonds if present)
    X_ps = X_all.copy()
    if 'NumRotatableBonds' in X_ps.columns:
        X_ps = X_ps.drop(['NumRotatableBonds'], axis=1)
    
    # PB features use all features (after removing unexpected ones)
    X_pb = X_all.copy()
    
    # Normalize features
    try:
        X_ps_normalized = ps_scaler.transform(X_ps)
        X_pb_normalized = pb_scaler.transform(X_pb)
        
        # Create normalized DataFrames
        df_ps_norm = pd.DataFrame(X_ps_normalized, columns=X_ps.columns)
        df_ps_norm.insert(0, 'Id', ids)
        
        df_pb_norm = pd.DataFrame(X_pb_normalized, columns=X_pb.columns)
        df_pb_norm.insert(0, 'Id', ids)
        
        # Save normalized features
        ps_output_file = os.path.join(ps_output_dir, base_filename)
        pb_output_file = os.path.join(pb_output_dir, base_filename)
        
        df_ps_norm.to_csv(ps_output_file, index=False)
        df_pb_norm.to_csv(pb_output_file, index=False)
        
        print(f"✓ PS normalized features saved: {ps_output_file}")
        print(f"✓ PB normalized features saved: {pb_output_file}")
        
        return ps_output_file, pb_output_file
        
    except Exception as e:
        raise RuntimeError(f"Failed to normalize features: {e}")


def load_models(ps_model_path: str, pb_model_path: str, use_gpu: bool = False) -> Tuple[xgb.Booster, xgb.Booster]:
    """
    Load the SC2-PS and SC2-PB XGBoost models.
    
    Args:
        ps_model_path: Path to SC2-PS model file
        pb_model_path: Path to SC2-PB model file  
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Tuple of (sc2_ps_model, sc2_pb_model)
    """
    try:
        # Load models
        sc2_ps = xgb.Booster()
        sc2_ps.load_model(ps_model_path)
        sc2_pb = xgb.Booster()
        sc2_pb.load_model(pb_model_path)
        
        # Set computation parameters
        params = {'tree_method': 'hist', 'device': 'cuda' if use_gpu else 'cpu'}
        sc2_ps.set_param(params)
        sc2_pb.set_param(params)
        
        print(f"✓ Successfully loaded models:")
        print(f"  - SC2-PS: {os.path.basename(ps_model_path)}")
        print(f"  - SC2-PB: {os.path.basename(pb_model_path)}")
        print(f"  - Computation device: {'GPU' if use_gpu else 'CPU'}")
        
        return sc2_ps, sc2_pb
        
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")


def load_normalized_features(ps_feature_path: str, pb_feature_path: str) -> Tuple[xgb.DMatrix, xgb.DMatrix, pd.Series]:
    """
    Load normalized feature data for both models.
    
    Args:
        ps_feature_path: Path to SC2-PS normalized feature CSV file
        pb_feature_path: Path to SC2-PB normalized feature CSV file
        
    Returns:
        Tuple of (ps_features, pb_features, ids)
    """
    try:
        # Load PS features
        df_ps = pd.read_csv(ps_feature_path)
        df_ps.fillna(0, inplace=True)
        
        # Load PB features  
        df_pb = pd.read_csv(pb_feature_path)
        df_pb.fillna(0, inplace=True)
        
        # Extract IDs (should be identical in both files)
        ids = df_ps['Id'].copy()
        
        # Prepare feature matrices
        X_ps = df_ps.drop(['Id'], axis=1, errors='ignore')
        X_pb = df_pb.drop(['Id'], axis=1, errors='ignore')
        
        # Convert to XGBoost DMatrix
        ps_features = xgb.DMatrix(X_ps, feature_names=X_ps.columns.tolist())
        pb_features = xgb.DMatrix(X_pb, feature_names=X_pb.columns.tolist())
        
        print(f"✓ Loaded normalized features:")
        print(f"  - PS features: {X_ps.shape[0]} compounds, {X_ps.shape[1]} features")
        print(f"  - PB features: {X_pb.shape[0]} compounds, {X_pb.shape[1]} features")
        
        return ps_features, pb_features, ids
        
    except Exception as e:
        raise RuntimeError(f"Failed to load normalized feature data: {e}")


def get_base_compound_name(compound_id: str) -> str:
    """
    Extract base compound name by removing pose information.
    
    Args:
        compound_id: Full compound identifier
        
    Returns:
        Base compound name without pose suffix
    """
    # Split by '_pose' to get the base compound name
    if '_pose' in compound_id:
        return compound_id.split('_pose')[0]
    
    # Fallback: remove common pose suffixes like _out_pose1, etc.
    base_name = re.sub(r'_(out_)?pose\d+$', '', compound_id)
    base_name = re.sub(r'_\d+$', '', base_name)  # Remove trailing numbers
    return base_name


def aggregate_poses(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple poses by selecting the one with highest confidence.
    
    Args:
        results_df: DataFrame with compound results
        
    Returns:
        DataFrame with aggregated results and pose information
    """
    results_df['base_compound'] = results_df['compound_id'].apply(get_base_compound_name)
    
    # Find the pose with maximum confidence for each compound
    idx_max = results_df.groupby('base_compound')['sc2_score'].idxmax()
    aggregated_df = results_df.loc[idx_max].copy()
    
    # Add information about selected pose
    aggregated_df['selected_pose'] = aggregated_df['compound_id'].apply(
        lambda x: x.replace(get_base_compound_name(x), '').lstrip('_') or 'original'
    )
    
    # Count total poses per compound
    pose_counts = results_df.groupby('base_compound').size()
    aggregated_df['total_poses'] = aggregated_df['base_compound'].map(pose_counts)
    
    return aggregated_df


def perform_rescoring(ps_features: xgb.DMatrix, pb_features: xgb.DMatrix, ids: pd.Series,
                     sc2_ps: xgb.Booster, sc2_pb: xgb.Booster, 
                     ps_weight: float = 0.7, pb_weight: float = 0.3) -> pd.DataFrame:
    """
    Perform rescoring using the consensus SCORCH2 model.
    
    Args:
        ps_features: SC2-PS feature matrix
        pb_features: SC2-PB feature matrix
        ids: Compound identifiers
        sc2_ps: SC2-PS model
        sc2_pb: SC2-PB model
        ps_weight: Weight for PS model predictions
        pb_weight: Weight for PB model predictions
        
    Returns:
        DataFrame with rescoring results
    """
    print("🔄 Performing rescoring...")
    
    # Make predictions
    preds_ps = sc2_ps.predict(ps_features)
    preds_pb = sc2_pb.predict(pb_features)
    
    # Calculate consensus score
    consensus_scores = preds_ps * ps_weight + preds_pb * pb_weight
    
    # Create results dataframe (without weights in the main data)
    results_df = pd.DataFrame({
        'compound_id': ids,
        'sc2_ps_score': preds_ps,
        'sc2_pb_score': preds_pb, 
        'sc2_score': consensus_scores
    })
    
    # Sort by consensus score (descending)
    results_df = results_df.sort_values('sc2_score', ascending=False).reset_index(drop=True)
    results_df['rank'] = range(1, len(results_df) + 1)
    
    print(f"✓ Rescoring completed for {len(results_df)} compounds")
    
    return results_df


def save_results(results_df: pd.DataFrame, output_path: str, should_aggregate_poses: bool = False,
                ps_model_path: str = "", pb_model_path: str = "", 
                ps_weight: float = 0.7, pb_weight: float = 0.3) -> None:
    """
    Save rescoring results to CSV file with detailed metadata.
    
    Args:
        results_df: DataFrame with results
        output_path: Path to save results
        should_aggregate_poses: Whether poses were aggregated
        ps_model_path: Path to PS model (for metadata)
        pb_model_path: Path to PB model (for metadata)
        ps_weight: PS model weight
        pb_weight: PB model weight
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    
    # Prepare final output
    if should_aggregate_poses:
        final_df = aggregate_poses(results_df)
        print(f"✓ Aggregated {len(results_df)} poses into {len(final_df)} unique compounds")
    else:
        final_df = results_df.copy()
        final_df['selected_pose'] = 'N/A'
        final_df['total_poses'] = 1
    
    # Remove PS/PB weights from the output CSV
    output_columns = ['compound_id', 'sc2_ps_score', 'sc2_pb_score', 'sc2_score', 'rank']
    if should_aggregate_poses:
        output_columns.extend(['selected_pose', 'total_poses'])
    
    final_output_df = final_df[output_columns].copy()
    
    # Add metadata header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'w') as f:
        # Write metadata as comments
        f.write(f"# SCORCH2 Rescoring Results\n")
        f.write(f"# Generated: {timestamp}\n")
        f.write(f"# SC2-PS Model: {os.path.basename(ps_model_path)}\n")
        f.write(f"# SC2-PB Model: {os.path.basename(pb_model_path)}\n")
        f.write(f"# Consensus Weights: PS={ps_weight:.2f}, PB={pb_weight:.2f}\n")
        f.write(f"# Poses Aggregated: {'Yes' if should_aggregate_poses else 'No'}\n")
        f.write(f"# Total Compounds: {len(final_output_df)}\n")
        f.write("#\n")
        
        # Write CSV data
        final_output_df.to_csv(f, index=False)
    
    print(f"✓ Results saved to: {output_path}")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"  - Mean SC2 Score: {final_df['sc2_score'].mean():.4f}")
    print(f"  - Std SC2 Score: {final_df['sc2_score'].std():.4f}")
    print(f"  - Score Range: {final_df['sc2_score'].min():.4f} - {final_df['sc2_score'].max():.4f}")
    
    if should_aggregate_poses:
        multi_pose_compounds = final_df[final_df['total_poses'] > 1]
        if len(multi_pose_compounds) > 0:
            print(f"  - Compounds with multiple poses: {len(multi_pose_compounds)}")
            print(f"  - Average poses per multi-pose compound: {multi_pose_compounds['total_poses'].mean():.1f}")
    
    # Show top results (ensure they are sorted by score descending)
    print(f"\n🏆 Top 5 Compounds:")
    top_compounds = final_df.sort_values('sc2_score', ascending=False).head(5)
    for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
        print(f"  {i}. {row['compound_id']} - Score: {row['sc2_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="SCORCH2 Rescoring Code",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Complete workflow with demo data (raw results, no aggregation)
  python scorch2_rescoring.py --protein-dir example_data/protein --ligand-dir example_data/molecule \\
                              --sc2_ps_model /path/to/sc2_ps.xgb --sc2_pb_model /path/to/sc2_pb.xgb \\
                              --ps_scaler /path/to/sc2_ps_scaler --pb_scaler /path/to/sc2_pb_scaler \\
                              --output results.csv --gpu
  
  # Complete workflow with pose aggregation
  python scorch2_rescoring.py --protein-dir example_data/protein --ligand-dir example_data/molecule \\
                              --sc2_ps_model /path/to/sc2_ps.xgb --sc2_pb_model /path/to/sc2_pb.xgb \\
                              --ps_scaler /path/to/sc2_ps_scaler --pb_scaler /path/to/sc2_pb_scaler \\
                              --output results.csv --aggregate --gpu
  
  # Skip feature extraction if features already exist
  python scorch2_rescoring.py --features existing_features.csv \\
                              --sc2_ps_model /path/to/sc2_ps.xgb --sc2_pb_model /path/to/sc2_pb.xgb \\
                              --ps_scaler /path/to/sc2_ps_scaler --pb_scaler /path/to/sc2_pb_scaler \\
                              --output results.csv
        """
    )
    
    # Model paths (required)
    parser.add_argument('--sc2_ps_model', type=str, required=True,
                       help="Path to the SC2-PS model file")
    parser.add_argument('--sc2_pb_model', type=str, required=True, 
                       help="Path to the SC2-PB model file")
    parser.add_argument('--ps_scaler', type=str, required=True,
                       help="Path to the SC2-PS scaler file")
    parser.add_argument('--pb_scaler', type=str, required=True,
                       help="Path to the SC2-PB scaler file")
    parser.add_argument('--output', type=str, required=True,
                       help="Output CSV file path to save rescoring results")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--protein-dir', type=str,
                            help="Directory containing protein PDBQT files (requires --ligand-dir)")
    input_group.add_argument('--features', type=str,
                            help="Path to existing feature CSV file (skip feature extraction)")
    
    # Additional required for protein-dir option
    parser.add_argument('--ligand-dir', type=str,
                       help="Directory containing ligand PDBQT files (required with --protein-dir)")
    
    # Processing options
    parser.add_argument('--num-cores', type=int, default=None,
                       help="Number of CPU cores for feature extraction")
    parser.add_argument('--aggregate', action='store_true',
                       help="Aggregate multiple poses by selecting the highest scoring pose (default: False, keep all poses)")
    parser.add_argument('--ps_weight', type=float, default=0.7,
                       help="Weight for SC2-PS predictions")
    parser.add_argument('--pb_weight', type=float, default=0.3,
                       help="Weight for SC2-PB predictions")
    parser.add_argument('--gpu', action='store_true',
                       help="Use GPU for prediction if available")
    
    # Output options
    parser.add_argument('--temp-dir', type=str, default="temp_scorch2",
                       help="Temporary directory for intermediate files")
    parser.add_argument('--keep-temp', action='store_true',
                       help="Keep temporary files after completion")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.protein_dir and not args.ligand_dir:
        parser.error("--ligand-dir is required when using --protein-dir")
    
    if abs(args.ps_weight + args.pb_weight - 1.0) > 1e-6:
        parser.error("PS and PB weights must sum to 1.0")
    
    print("=" * 60)
    print("SCORCH2 Rescoring Code")
    print("=" * 60)
    
    try:
        # Load models
        sc2_ps, sc2_pb = load_models(args.sc2_ps_model, args.sc2_pb_model, args.gpu)
        
        # Determine feature file path
        if args.features:
            # Use existing features
            feature_file = args.features
            print(f"✓ Using existing feature file: {feature_file}")
        else:
            # Extract features from protein/ligand directories
            print(f"📁 Processing structures from:")
            print(f"  - Proteins: {args.protein_dir}")
            print(f"  - Ligands: {args.ligand_dir}")
            
            # Create temporary directory
            temp_features_dir = os.path.join(args.temp_dir, "features")
            
            # Use provided num_cores or default to os.cpu_count()-1
            num_cores = args.num_cores if args.num_cores is not None else max(1, os.cpu_count() - 1)
            
            # Run feature extraction
            if not run_feature_extraction(args.protein_dir, args.ligand_dir, 
                                        temp_features_dir, num_cores):
                raise RuntimeError("Feature extraction failed")
            
            # Find the generated feature file
            feature_files = [f for f in os.listdir(temp_features_dir) if f.endswith('_features.csv')]
            if not feature_files:
                raise RuntimeError("No feature files found after extraction")
            
            feature_file = os.path.join(temp_features_dir, feature_files[0])
            print(f"✓ Feature extraction completed: {feature_file}")
        
        # Normalize features
        temp_normalized_dir = os.path.join(args.temp_dir, "normalized_features")
        ps_normalized_file, pb_normalized_file = normalize_features(
            feature_file, args.ps_scaler, args.pb_scaler, temp_normalized_dir
        )
        
        # Load normalized features
        ps_features, pb_features, ids = load_normalized_features(ps_normalized_file, pb_normalized_file)
        
        # Perform rescoring
        results_df = perform_rescoring(ps_features, pb_features, ids, sc2_ps, sc2_pb, 
                                     args.ps_weight, args.pb_weight)
        
        # Save results
        save_results(results_df, args.output, args.aggregate, 
                    args.sc2_ps_model, args.sc2_pb_model, args.ps_weight, args.pb_weight)
        
        # Clean up temporary files if requested
        if not args.keep_temp and os.path.exists(args.temp_dir):
            import shutil
            shutil.rmtree(args.temp_dir)
            print(f"✓ Temporary files cleaned up")
        
        print(f"\n🎉 SCORCH2 rescoring completed successfully!")
        print(f"📄 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
