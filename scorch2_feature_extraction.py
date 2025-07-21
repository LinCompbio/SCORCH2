import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_features(adata):
    """Extract features from single-cell RNA sequencing data.

    Args:
        adata: AnnData object containing single-cell RNA sequencing data

    Returns:
        DataFrame containing extracted features
    """
    # Calculate basic statistics
    n_genes = np.sum(adata.X > 0, axis=1)
    total_counts = np.sum(adata.X, axis=1)
    
    # Create feature matrix
    features = pd.DataFrame({
        'n_genes': n_genes,
        'total_counts': total_counts,
        'log_counts': np.log1p(total_counts)
    })
    
    # Add cell-type specific features if available
    if 'cell_type' in adata.obs:
        cell_type_means = pd.get_dummies(adata.obs['cell_type'])
        features = pd.concat([features, cell_type_means], axis=1)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features = pd.DataFrame(scaled_features, columns=features.columns)
    
    return scaled_features