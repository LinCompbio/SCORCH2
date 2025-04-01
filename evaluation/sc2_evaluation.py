import os
import argparse
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit.ML.Scoring import Scoring
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
import re
import xgboost as xgb
from tqdm import tqdm
import json
import pickle
import math
from scipy.stats import spearmanr, pearsonr, kendalltau

# Target databases
true_decoy_gap = [
"Q9NWZ3", "P42336", "P27986", "O60674", "P43405", "P10721", "O14965", "P15056",
"P23458", "P11362", "P29597", "O00329", "P28482", "Q96GD4", "P00533", "Q06187",
"O14757", "P36888", "P00519", "Q05397", "Q16539", "P08581", "P11802", "P06239",
"O75116", "Q9Y233", "Q14432", "Q08499", "P00918", "Q16790", "Q00987", "P78536",
"P09874", "P35372", "P20309", "P41143", "P35462", "P07550", "P14416", "P08172",
"P34969", "P08908", "P28335", "P30542", "P29274", "P34972", "P21453", "O43614",
"P28223", "P08253", "P03952", "P14780", "P00742", "P45452", "P27487", "P08246",
"P07711", "P00734", "P03372", "Q92731", "P51449", "P04150", "Q96RI1", "P31645",
"P01375", "O15379", "Q9UBN7", "Q07817", "Q07820", "P10415", "O60341"]

dekois1 = ['ACE', 'ACHE', 'ADRB2', 'CATL', 'DHFR', 'ERBB2', 'HDAC2', 'HIV1PR', 'HSP90', 'JAK3', 'JNK2',
 'MDM2', 'P38-alpha', 'PI3Kg', 'PNP', 'PPARG','Thrombin', 'TS']

dekois2_unseen = [
'11BETAHSD1', 'ACE2', "ALR2", "COX1", "CYP2A6", "ER-BETA", "INHA", "MMP2", 
"PPARA", "PPARG", "TK"]

dude_unseen = [
"ALDR", "CP2C9", "CP3A4", "DHI1", "HXK4", "KITH", "NOS1", "PGH1", 
"PPARA", "PPARG", "PYRD", "SAHH"]


def vs_load_and_prepare_data(path, args, drop_columns=None):
    """
    Loads dataset and prepares features and labels for VS evaluation.
    
    Args:
        path: Path to the CSV file
        args: Command line arguments
        drop_columns: Columns to drop from features
        
    Returns:
        Features, labels, and IDs
    """
    df = pd.read_csv(path)
    df.fillna(0, inplace=True)

    if args.keyword == 'active':
        # Assign labels based on 'Id' containing 'active'
        df['label'] = df['Id'].apply(lambda x: 1 if 'active' in x else 0)
    elif args.keyword == 'inactive':
        # Assign labels based on 'Id' containing 'inactive'
        df['label'] = df['Id'].apply(lambda x: 0 if 'inactive' in x else 1)
    else:
        raise ValueError("Invalid keyword. Please choose 'active' or 'inactive'.")

    y_val = df['label']
    X_val = df.drop(drop_columns, axis=1)

    return X_val, y_val, df['Id']


def ranking_load_and_prepare_data(path, drop_columns=None):
    """
    Loads dataset and prepares features for ranking evaluation.

    Args:
        path: Path to the CSV file
        drop_columns: Columns to drop from features
        
    Returns:
        XGBoost DMatrix of features and IDs
    """
    df = pd.read_csv(path)
    df.fillna(0, inplace=True)
    
    id_values = df['Id']
    X_val = df.drop(drop_columns, axis=1)
    features = df.drop(drop_columns, axis=1)
    features = xgb.DMatrix(features, feature_names=X_val.columns.tolist())
    
    return features, id_values


def get_base_name(filename):
    """
    Extracts the base name from a filename by removing the last underscore part.
    
    Args:
        filename: Input filename
        
    Returns:
        Base name without last underscore part
    """
    return re.sub(r'_[^_]*$', '', filename)


def run_vs_evaluation(args):
    """
    Run virtual screening evaluation.

    Args:
        args: Command line arguments
    """
    # Load trained models
    sc2_ps = xgb.Booster()
    sc2_ps.load_model(args.sc2_ps)
    sc2_pb = xgb.Booster()
    sc2_pb.load_model(args.sc2_pb)

    params = {'tree_method': 'hist', 'device': 'cuda' if args.gpu else 'cpu'}
    sc2_ps.set_param(params)
    sc2_pb.set_param(params)

    # Get list of files for selected targets
    if args.targets:
        valid_targets = ['dekois1', 'dekois2_unseen', 'dude_unseen', 'true_decoy_gap']
        assert args.targets in valid_targets, f"Choose the targets from {valid_targets}"

        target_dict = {
            'dekois1': [i.lower() for i in dekois1],
            'dekois2_unseen': [i.lower() for i in dekois2_unseen],
            'dude_unseen': [i.lower() for i in dude_unseen],
            'true_decoy_gap': [i for i in true_decoy_gap]
        }
        targets = target_dict[args.targets]

        files = [f for f in os.listdir(args.sc2_ps_feature_repo) if
                 '_normalized' in f and f.split('_normalized')[0] in targets]

    else:
        files = os.listdir(args.sc2_ps_feature_repo)

    # Initialize metric storage
    total_metrics = defaultdict(list)
    groups = []

    # Process each target file
    for file in tqdm(files, desc="Processing targets"):
        try:
            path_ps = os.path.join(args.sc2_ps_feature_repo, file)
            path_pb = os.path.join(args.sc2_pb_feature_repo, file)
            
            group = path_ps.split('/')[-1].split('_normalized')[0]
            groups.append(group)

            # Load data
            X1, y1, id1 = vs_load_and_prepare_data(path_ps, args, drop_columns=['Id', 'label'])
            # Identical y,id
            X2, _, _ = vs_load_and_prepare_data(path_pb, args, drop_columns=['Id', 'label'])

            # Convert to XGBoost DMatrix
            X1 = xgb.DMatrix(X1, feature_names=X1.columns.tolist())
            X2 = xgb.DMatrix(X2, feature_names=X2.columns.tolist())

            # Predict
            preds1 = sc2_ps.predict(X1)
            preds2 = sc2_pb.predict(X2)
            # Weighted consensus
            preds = preds1 * args.ps_consensus_weight + preds2 * args.pb_consensus_weight
            
            # Create results DataFrame
            if args.aggregate:
                res = pd.DataFrame({'Id': id1, 'Confidence': preds.squeeze(), 'Label': y1})
                res['base_name'] = res['Id'].apply(get_base_name)
                result_sorted = res.groupby('base_name').agg({'Confidence': 'max', 'Label': 'first'}).reset_index()
                result_sorted = result_sorted.sort_values('Confidence', ascending=False)
            else:
                res = pd.DataFrame({'Id': id1, 'Confidence': preds.squeeze(), 'Label': y1})
                result_sorted = res.sort_values('Confidence', ascending=False)
            
            # Prepare input for enrichment calculations
            feed_df = np.column_stack((result_sorted['Label'].values, result_sorted['Confidence'].values))
            
            # Compute Enrichment Factors and BEDROC
            ef_values = Scoring.CalcEnrichment(feed_df, 0, [0.005, 0.01, 0.02, 0.05])
            ef_dict = dict(zip([0.005, 0.01, 0.02, 0.05], ef_values))
            bedroc = Scoring.CalcBEDROC(feed_df, 0, alpha=80.5)
            
            # Compute classification metrics
            auc_roc = roc_auc_score(y1, preds)
            auc_pr = average_precision_score(y1, preds)
            
            # Store metrics
            total_metrics['EF_0.5%'].append(ef_dict[0.005])
            total_metrics['EF_1%'].append(ef_dict[0.01])
            total_metrics['EF_2%'].append(ef_dict[0.02])
            total_metrics['EF_5%'].append(ef_dict[0.05])
            total_metrics['BEDROC'].append(bedroc)
            total_metrics['AUC-ROC'].append(auc_roc)
            total_metrics['AUCPR'].append(auc_pr)
            
            # Print Results in Single Line (CML-Friendly)
            print(f"{group:} | EF0.5%: {ef_dict[0.005]:.2f} | EF1%: {ef_dict[0.01]:.2f} | "
                  f"EF5%: {ef_dict[0.05]:.2f} | BEDROC: {bedroc:.4f} | "
                  f"AUROC: {auc_roc:.4f} | AUCPR: {auc_pr:.4f}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Compute Final Summary Metrics
    avg_metrics = {k: np.mean(v) for k, v in total_metrics.items()}
    med_metrics = {k: np.median(v) for k, v in total_metrics.items()}
    
    print("\n### Summary Results ###")
    print(f"Avg | EF0.5%: {avg_metrics['EF_0.5%']:.2f} | EF1%: {avg_metrics['EF_1%']:.2f} | "
          f"EF5%: {avg_metrics['EF_5%']:.2f} | BEDROC: {avg_metrics['BEDROC']:.4f} | "
          f"AUROC: {avg_metrics['AUC-ROC']:.4f} | AUCPR: {avg_metrics['AUCPR']:.4f}")
    
    # Median metrics
    print(f"Med | EF0.5%: {med_metrics['EF_0.5%']:.2f} | EF1%: {med_metrics['EF_1%']:.2f} | "
          f'EF5%: {med_metrics["EF_5%"]:.2f} | BEDROC: {med_metrics["BEDROC"]:.4f} | '
          f'AUROC: {med_metrics["AUC-ROC"]:.4f} | AUCPR: {med_metrics["AUCPR"]:.4f}')
    
    # Save as CSV
    results_df = pd.DataFrame({'Target': groups, **total_metrics})
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results_df.to_csv(args.output, index=False)


def run_ranking_evaluation(args):
    """
    Run ranking evaluation.
    Args:
        args: Command line arguments
    """
    # Load trained models
    sc2_ps = xgb.Booster()
    sc2_ps.load_model(args.sc2_ps)
    sc2_pb = xgb.Booster()
    sc2_pb.load_model(args.sc2_pb)
    
    # Set model parameters
    params = {'tree_method': 'hist', 'device':'cuda'} if args.gpu else {'tree_method': 'hist'}
    sc2_ps.set_param(params)
    sc2_pb.set_param(params)
    
    # Get files from feature repository
    feature_repo_ps = args.sc2_ps_feature_repo
    feature_repo_pb = args.sc2_pb_feature_repo
    files = os.listdir(feature_repo_ps)
    
    # Directory with experimental data
    exp_result_repo = args.exp_repo
    
    # Initialize lists for correlation values
    all_pearsonr = []
    all_spearmanr = []
    all_kendalltau = []
    groups = []
    
    # Process each file
    for i in tqdm(range(len(files)), desc="Processing ranking files"):
        try:
            # Load feature files

            path_ps = os.path.join(feature_repo_ps, files[i])
            path_pb = os.path.join(feature_repo_pb, files[i])
            
            # Extract group name
            group = path_ps.split('/')[-1].split('_normalized')[0]
            groups.append(group)
            
            # Load and prepare data
            features_ps, id_ps = ranking_load_and_prepare_data(path_ps, drop_columns=['Id'])
            features_pb, id_pb = ranking_load_and_prepare_data(path_pb, drop_columns=['Id'])
            
            # Make predictions
            preds_ps = sc2_ps.predict(features_ps)
            preds_pb = sc2_pb.predict(features_pb)
            
            # Apply weights
            preds_ps = preds_ps * args.ps_consensus_weight
            preds_pb = preds_pb * args.pb_consensus_weight
            
            # Combine predictions
            preds = preds_ps + preds_pb
            
            # Process IDs
            id_values = [i.split('.pdbqt')[0].strip() for i in id_ps]
            
            # Special handling for SHP2 target
            if group == 'shp2':
                id_values = [i.replace('_', '') if 'Example' in i else i for i in id_values]
                id_values[4] = 'SHP099-1_Example7'
            
            # Create results dataframe
            res = pd.DataFrame({'Id': id_values, 'Confidence': preds.squeeze()})
            
            # Read experimental data
            exp_data = pd.read_csv(os.path.join(exp_result_repo, f'{group}_results_5ns.csv'))
            
            # Process experimental data IDs
            exp_data['Id'] = exp_data['Ligand'].astype(str).str.strip().str.replace("/", "_")
            
            # Convert numeric-like strings to integers
            exp_data['Id'] = exp_data['Id'].apply(
                lambda x: str(int(float(x))) if x.replace('.', '', 1).isdigit() else x)
            
            # Merge predictions with experimental data
            merged_data = exp_data.merge(res, on='Id', how='left')
            
            # Drop rows where predictions are missing
            merged_data = merged_data.dropna(subset=['Confidence']).reset_index(drop=True)
            
            print(f'Group: {group}, Data points: {len(merged_data)}')
            
            # Extract experimental values
            try:
                #Keep the trend identical
                exp_number = merged_data['Exp. ΔG'].values * -1
            except KeyError:
                exp_number = merged_data['Exp. Binding (ΔG)'].values * -1
            
            preds_ordered = merged_data['Confidence']

            # Calculate correlations
            pearson_corr = pearsonr(preds_ordered, exp_number)[0]
            spearman_corr = spearmanr(preds_ordered, exp_number)[0]
            kendall_corr = kendalltau(preds_ordered, exp_number)[0]

            all_pearsonr.append(pearson_corr)
            all_spearmanr.append(spearman_corr)
            all_kendalltau.append(kendall_corr)

            print(f"Pearson correlation: {pearson_corr:.4f}")
            print(f"Spearman correlation: {spearman_corr:.4f}")
            print(f"Kendall correlation: {kendall_corr:.4f}")

        except Exception as e:
            print(f'Error processing {files[i]}: {e}')
            continue

    # Calculate and print averages
    print("\n### Summary Results ###")
    print(f"Average Pearson correlation: {np.mean(all_pearsonr):.4f}")
    print(f"Average Spearman correlation: {np.mean(all_spearmanr):.4f}")
    print(f"Average Kendall correlation: {np.mean(all_kendalltau):.4f}")

    # Save correlation results
    results = {
        'Target': groups,
        'Pearson': all_pearsonr,
        'Spearman': all_spearmanr,
        'Kendall': all_kendalltau
    }

    results_df = pd.DataFrame(results)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results_df.to_csv(args.output, index=False)


def main(args):
    """
    Main function to run the selected evaluation mode.
    
    Args:
        args: Command line arguments
    """
    if args.mode == 'vs':
        run_vs_evaluation(args)
    elif args.mode == 'ranking':
        run_ranking_evaluation(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'vs' or 'ranking'.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SCORCH2 Virtual Screening and Ranking Evaluation")

    # Create subparsers
    subparsers = parser.add_subparsers(dest='mode', required=True, help="Mode of operation")

    # ========================
    # Virtual Screening Parser
    # ========================
    vs_parser = subparsers.add_parser('vs', help='Run virtual screening mode')

    # Model paths
    vs_parser.add_argument('--sc2_ps', type=str, required=True, help="Path to the SC2-PS model file")
    vs_parser.add_argument('--sc2_pb', type=str, required=True, help="Path to the SC2-PB model file")

    # Feature repositories
    vs_parser.add_argument('--sc2_ps_feature_repo', type=str, required=True,
                           help="Path to the SC2-PS feature directory")
    vs_parser.add_argument('--sc2_pb_feature_repo', type=str, required=True,
                           help="Path to the SC2-PB feature directory")

    # VS-specific options
    vs_parser.add_argument('--aggregate', action='store_true',
                           help="Aggregate results by taking the maximum confidence")
    vs_parser.add_argument('--keyword', type=str, required=True,
                           help="Keyword to assign labels: 'active' if dataset uses active/decoy, 'inactive' if it uses active/inactive")
    vs_parser.add_argument('--targets',type=str,
                           help="Only get the result from preferred targets")


    # Common options
    vs_parser.add_argument('--ps_consensus_weight', type=float, default=0.7,
                           help="Weight for SC2-PS predictions (default: 0.7)")
    vs_parser.add_argument('--pb_consensus_weight', type=float, default=0.3,
                           help="Weight for SC2-PB predictions (default: 0.3)")
    vs_parser.add_argument('--gpu', action='store_true', help="Use GPU for prediction if available")
    vs_parser.add_argument('--output', type=str, help="Output CSV file path to save predictions")

    # =====================
    # Ranking Mode Parser
    # =====================
    ranking_parser = subparsers.add_parser('ranking', help='Run binding affinity ranking mode')

    # Model paths
    ranking_parser.add_argument('--sc2_ps', type=str, required=True, help="Path to the SC2-PS model file")
    ranking_parser.add_argument('--sc2_pb', type=str, required=True, help="Path to the SC2-PB model file")

    # Feature repositories
    ranking_parser.add_argument('--sc2_ps_feature_repo', type=str, required=True,
                                help="Path to the SC2-PS feature directory")
    ranking_parser.add_argument('--sc2_pb_feature_repo', type=str, required=True,
                                help="Path to the SC2-PB feature directory")

    # Ranking-specific: Experimental data
    ranking_parser.add_argument('--exp_repo', type=str, required=True, help="Path to the experimental data directory")

    # Common options
    ranking_parser.add_argument('--ps_consensus_weight', type=float, default=0.7,
                                help="Weight for SC2-PS predictions (default: 0.7)")
    ranking_parser.add_argument('--pb_consensus_weight', type=float, default=0.3,
                                help="Weight for SC2-PB predictions (default: 0.3)")
    ranking_parser.add_argument('--gpu', action='store_true', help="Use GPU for prediction if available")
    ranking_parser.add_argument('--output', type=str, help="Output CSV file path to save predictions")


    args = parser.parse_args()
    main(args)





