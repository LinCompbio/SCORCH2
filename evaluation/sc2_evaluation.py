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

true_decoy_gap = [
"Q9NWZ3", "P42336", "P27986", "O60674", "P43405", "P10721", "O14965", "P15056",
"P23458", "P11362", "P29597", "O00329", "P28482", "Q96GD4", "P00533", "Q06187",
"O14757", "P36888", "P00519", "Q05397", "Q16539", "P08581", "P11802", "P06239",
"O75116", "Q9Y233", "Q14432", "Q08499", "P00918", "Q16790", "Q00987", "P78536",
"P09874", "P35372", "P20309", "P41143", "P35462", "P07550", "P14416", "P08172",
"P34969", "P08908", "P28335", "P30542", "P29274", "P34972", "P21453", "O43614",
"P28223", "P08253", "P03952", "P14780", "P00742", "P45452", "P27487", "P08246",
"P07711", "P00734", "P03372", "Q92731", "P51449", "P04150", "Q96RI1", "P31645",
"P01375", "O15379", "Q9UBN7", "Q07817", "Q0782git push origin main0", "P10415", "O60341"]


dekois1 = ['ACE', 'ACHE', 'ADRB2', 'CATL', 'DHFR', 'ERBB2', 'HDAC2', 'HIV1PR', 'HSP90', 'JAK3', 'JNK2',
 'MDM2', 'P38-alpha', 'PI3Kg', 'PNP', 'PPARG','Thrombin', 'TS']
# dekois1 = [i.lower() for i in dekois1]

dekois2_unseen = [
'11BETAHSD1',
'ACE2',
"ALR2",
"COX1",
"CYP2A6",
"ER-BETA",
"INHA",
"MMP2",
"PPARA",
"PPARG",
"TK"]
# dekois2_unseen = [i.lower() for i in dekois2_unseen]

dude_unseen = [
"ALDR",
"CP2C9",
"CP3A4",
"DHI1",
"HXK4",
"KITH",
"NOS1",
"PGH1",
"PPARA",
"PPARG",
"PYRD",
"SAHH"]


def load_and_prepare_data(path, drop_columns=None):
    """Loads dataset and prepares features and labels for model input."""
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


def get_base_name(filename):
    """Extracts the base name from a filename by removing the last underscore part."""
    return re.sub(r'_[^_]*$', '', filename)


def main(args):
    # Load trained models
    sc2_ps = joblib.load(args.sc2_ps)
    sc2_pb = joblib.load(args.sc2_pb)
    params = {'predictor': 'gpu_predictor', 'tree_method': 'gpu_hist'}
    sc2_ps.set_param(params)
    sc2_pb.set_param(params)

    # Get list of files for selected targets
    files = [f for f in os.listdir(args.sc2_ps_feature_repo)]

    # Initialize metric storage
    total_metrics = defaultdict(list)
    groups = []

    # Process each target file
    for file in tqdm(files, desc="Processing targets"):
        try:
            path1 = os.path.join(args.sc2_ps_feature_repo, file)
            path2 = os.path.join(args.sc2_pb_feature_repo, file)

            group = file.split('_')[0].upper()
            groups.append(group)

            # Load data
            X1, y1, id1 = load_and_prepare_data(path1, drop_columns=['Id', 'label'])
            #Identical y,id
            X2, _, _ = load_and_prepare_data(path2, drop_columns=['Id', 'label'])

            # Convert to XGBoost DMatrix
            X1 = xgb.DMatrix(X1, feature_names=X1.columns.tolist())
            X2 = xgb.DMatrix(X2, feature_names=X2.columns.tolist())

            #Predict
            preds1 = sc2_ps.predict(X1)
            preds2 = sc2_pb.predict(X2)
            #Weighted consensus
            preds = preds1 * 0.7 + preds2 * 0.3

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

            # **Print Results in Single Line (CML-Friendly)**
            print(f"{group:} | EF0.5%: {ef_dict[0.005]:.2f} | EF1%: {ef_dict[0.01]:.2f} | "
                  f"EF5%: {ef_dict[0.05]:.2f} | BEDROC: {bedroc:.4f} | "
                  f"AUROC: {auc_roc:.4f} | AUCPR: {auc_pr:.4f}")

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    # **Compute Final Summary Metrics**
    avg_metrics = {k: np.mean(v) for k, v in total_metrics.items()}
    med_metrics = {k: np.median(v) for k, v in total_metrics.items()}

    print("\n### Summary Results ###")
    print(f"Avg | EF0.5%: {avg_metrics['EF_0.5%']:.2f} | EF1%: {avg_metrics['EF_1%']:.2f} | "
          f"EF5%: {avg_metrics['EF_5%']:.2f} | BEDROC: {avg_metrics['BEDROC']:.4f} | "
          f"AUROC: {avg_metrics['AUC-ROC']:.4f} | AUCPR: {avg_metrics['AUCPR']:.4f}")

    #median_metrics
    print(f"Med | EF0.5%: {med_metrics['EF_0.5%']:.2f} | EF1%: {med_metrics['EF_1%']:.2f} | "
          f'EF5%: {med_metrics["EF_5%"]:.2f} | BEDROC: {med_metrics["BEDROC"]:.4f} | '
          f'AUROC: {med_metrics["AUC-ROC"]:.4f} | AUCPR: {med_metrics["AUCPR"]:.4f}')

    # Save as CSV
    results_df = pd.DataFrame({'Target': groups, **total_metrics})
    if not os.path.exists('output'):
        os.makedirs('output')
        results_df.to_csv(args.output, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run scoring and evaluation on SBVS results.")

    parser.add_argument('--sc2_ps', type=str, required=False, help="Path to the sc2_ps pkl file.",
                        default='/home/s2523227/sc2_final/weight/SC2_PS_deduplicated_bedroc_0.953_aucpr_0.707.pkl')
    parser.add_argument('--sc2_pb', type=str, required=False, help="Path to the sc2_pb pkl file.",
                        default='/home/s2523227/sc2_final/weight/rmsd_3_aucpr_0.7025.pkl')

    parser.add_argument('--sc2_ps_feature_repo', type=str, required=False, help="Path to the first feature directory.", default=
                        '/home/s2523227/sc2_final/evaluation_feature/vsds/sc2_ps_flare_vsds')
    parser.add_argument('--sc2_pb_feature_repo', type=str, required=False, help="Path to the second feature directory.",default=
                        '/home/s2523227/sc2_final/evaluation_feature/vsds/sc2_pb_flare_vsds')

    parser.add_argument('--targets', type=str, nargs='+', required=False, help="List of target names (space-separated).")
    parser.add_argument('--aggregate', type=str, required=False, help="Aggregate results by taking the maximum confidence",default = True)
    parser.add_argument('--keyword', type=str, required=False, help="Keyword to assign labels to the dataset. "
                                                                  "If the dataset use "
                                                                  "active / inactive to differentiate compounds, then it should be 'inactive'. "
                                                                  "If the dataset use "
                                                                  " decoy / active to differentiate compounds, then it should be 'active'.",

                        default='active')
    # parser.add_argument('--output', type=str, default='output/SC2_dekois_results.csv', help="Output CSV file name.")

    args = parser.parse_args()
    main(args)

