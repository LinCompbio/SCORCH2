import numpy as np
from scipy import stats

def calculate_gene_scores(adata, method='zscore'):
    """Calculate gene scores using different methods.

    Args:
        adata: AnnData object containing single-cell RNA sequencing data
        method: Scoring method ('zscore', 'rank', or 'custom')

    Returns:
        Array of gene scores
    """
    if method == 'zscore':
        scores = stats.zscore(adata.X, axis=0)
    elif method == 'rank':
        scores = stats.rankdata(adata.X, axis=0)
    elif method == 'custom':
        # Custom scoring method
        total_expr = np.sum(adata.X, axis=0)
        cell_count = np.sum(adata.X > 0, axis=0)
        scores = total_expr * np.log2(cell_count + 1)
    else:
        raise ValueError(f"Unknown scoring method: {method}")
    
    return scores

def rescore_cells(adata, gene_scores):
    """Rescore cells based on gene scores.

    Args:
        adata: AnnData object containing single-cell RNA sequencing data
        gene_scores: Array of gene scores

    Returns:
        Array of cell scores
    """
    # Weight expression by gene scores
    weighted_expr = adata.X * gene_scores
    
    # Calculate cell scores
    cell_scores = np.sum(weighted_expr, axis=1)
    
    return cell_scores