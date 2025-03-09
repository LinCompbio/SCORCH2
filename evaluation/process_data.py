import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
import json


def create_normalizer(args):
    """Creates and saves a scaler based on the data in feature_dir."""
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    dfs = []
    files = os.listdir(args.feature_dir)
    for file in tqdm(files, desc="Loading files"):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(args.feature_dir, file))
            if args.drop_columns:
                df = df.drop(columns=args.drop_columns, errors='ignore')

            dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)

    if args.scaler_type == 'standard':
        scaler = StandardScaler()
    elif args.scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif args.scaler_type == 'maxabs':
        scaler = MaxAbsScaler()
    else:
        raise ValueError("Invalid scaler_type. Choose from 'standard', 'minmax', or 'maxabs'.")

    scaler.fit(combined_df)
    joblib.dump(scaler, args.save_path)
    print(f"Scaler saved to {args.save_path}")

def inference_sc2_pb(args):
    """Applies sc2_pb scaler to feature files and saves the normalized data."""
    os.makedirs(args.save_path, exist_ok=True)
    files = os.listdir(args.feature_dir)
    sc2_pb_scaler = joblib.load(args.sc2_pb_scaler_path)

    def process_file(file):
        if file.endswith('.csv'):
            name = file.split('_protein')[0]
            df = pd.read_csv(os.path.join(args.feature_dir, file))
            df_id = df['Id']

            # Process sc2_pb
            df_pb = df.copy()
            df_pb = df_pb.drop(columns=['Id'], errors='ignore')

            val_pb_trans = sc2_pb_scaler.transform(df_pb)
            val_pb = pd.DataFrame(val_pb_trans, columns=df_pb.columns)
            val_pb['Id'] = df_id

            val_pb.fillna(0, inplace=True)
            val_pb.to_csv(f'{args.save_path}/{name}_normalized_pb.csv', index=False)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_file, files), total=len(files), desc="Processing files (sc2_pb)"))


def inference_sc2_ps(args):
    """Applies sc2_ps scaler to feature files and saves the normalized data."""
    os.makedirs(args.save_path, exist_ok=True)
    files = os.listdir(args.feature_dir)
    sc2_ps_scaler = joblib.load(args.sc2_ps_scaler_path)

    def process_file(file):
        if file.endswith('.csv'):
            name = file.split('_protein')[0]
            df = pd.read_csv(os.path.join(args.feature_dir, file))
            df_id = df['Id']

            # Process sc2_ps
            df_ps = df.copy()
            df_ps = df_ps.drop(columns=['Id', 'NumRotatableBonds'], errors='ignore')

            val_ps_trans = sc2_ps_scaler.transform(df_ps)
            val_ps = pd.DataFrame(val_ps_trans, columns=df_ps.columns)
            val_ps['Id'] = df_id

            val_ps.fillna(0, inplace=True)
            val_ps.to_csv(f'{args.save_path}/{name}_normalized_ps.csv', index=False)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_file, files), total=len(files), desc="Processing files (sc2_ps)"))

def main():
    parser = argparse.ArgumentParser(description="Feature Normalization Script")
    subparsers = parser.add_subparsers(dest='mode', help='Select mode: create_normalizer or scale features.')

    # Subparser for create_normalizer mode
    create_parser = subparsers.add_parser('create_normalizer', help='Create a new scaler.')
    create_parser.add_argument('--feature_dir', required=True, help='Path to feature directory.')
    create_parser.add_argument('--save_path', required=True, help='Path to save the scaler.')
    create_parser.add_argument('--scaler_type', default='maxabs', choices=['standard', 'minmax', 'maxabs'], help='Scaler type.')
    create_parser.add_argument('--drop_columns', nargs='*', help='Columns to drop before scaling.')
    create_parser.add_argument('--reference_headers_key', default='492_models_58', help='Key for the features in SC1_features.json.')
    create_parser.set_defaults(func=create_normalizer)

    # Subparser for inference sc2_pb mode
    inference_pb_parser = subparsers.add_parser('inference_pb', help='Apply sc2_pb scaler.')
    inference_pb_parser.add_argument('--feature_dir', required=True, help='Path to feature directory.')
    inference_pb_parser.add_argument('--save_path', required=True, help='Path to save normalized sc2_pb data.')
    inference_pb_parser.add_argument('--sc2_pb_scaler_path', required=True, help='Path to sc2_pb scaler.')
    inference_pb_parser.set_defaults(func=inference_sc2_pb)

    # Subparser for inference sc2_ps mode
    inference_ps_parser = subparsers.add_parser('inference_ps', help='Apply sc2_ps scaler.')
    inference_ps_parser.add_argument('--feature_dir', required=True, help='Path to feature directory.')
    inference_ps_parser.add_argument('--save_path', required=True, help='Path to save normalized sc2_ps data.')
    inference_ps_parser.add_argument('--sc2_ps_scaler_path', required=True, help='Path to sc2_ps scaler.')
    inference_ps_parser.set_defaults(func=inference_sc2_ps)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()