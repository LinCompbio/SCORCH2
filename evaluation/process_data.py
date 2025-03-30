import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse


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


# Define process_file at the top level so it can be pickled by multiprocessing
def process_file(file, args, pb_dir, ps_dir, sc2_pb_scaler, sc2_ps_scaler):
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
        val_pb.to_csv(f'{pb_dir}/{name}_normalized.csv', index=False)

        # Process sc2_ps
        df_ps = df.copy()
        df_ps = df_ps.drop(columns=['Id', 'NumRotatableBonds'], errors='ignore')

        val_ps_trans = sc2_ps_scaler.transform(df_ps)
        val_ps = pd.DataFrame(val_ps_trans, columns=df_ps.columns)
        val_ps['Id'] = df_id

        val_ps.fillna(0, inplace=True)
        val_ps.to_csv(f'{ps_dir}/{name}_normalized.csv', index=False)


def scale_sc2(args):
    """Applies scalers to feature files and saves the normalized data in subdirectories."""
    # Create main output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Create subdirectories for pb and ps
    pb_dir = os.path.join(args.output_path, "sc2_pb")
    ps_dir = os.path.join(args.output_path, "sc2_ps")
    os.makedirs(pb_dir, exist_ok=True)
    os.makedirs(ps_dir, exist_ok=True)

    files = os.listdir(args.feature_dir)
    sc2_pb_scaler = joblib.load(args.pb_scaler_path)
    sc2_ps_scaler = joblib.load(args.ps_scaler_path)

    # Use ProcessPoolExecutor to parallelize file processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_file, files, [args] * len(files), [pb_dir] * len(files), [ps_dir] * len(files),
                               [sc2_pb_scaler] * len(files), [sc2_ps_scaler] * len(files)), total=len(files),
                  desc="Processing files"))

def main():
    parser = argparse.ArgumentParser(description="Feature Normalization Script")
    subparsers = parser.add_subparsers(dest='mode', help='Select mode: create_normalizer or scale features.')

    # Subparser for create_normalizer mode
    create_parser = subparsers.add_parser('create_normalizer', help='Create a new scaler.')
    create_parser.add_argument('--feature_dir', required=True, help='Path to feature directory.')
    create_parser.add_argument('--save_path', required=True, help='Path to save the scaler.')
    create_parser.add_argument('--scaler_type', default='maxabs', choices=['standard', 'minmax', 'maxabs'], help='Scaler type.')
    create_parser.add_argument('--drop_columns', nargs='*', help='Columns to drop before scaling.')
    create_parser.set_defaults(func=create_normalizer)

    # Subparser for inference mode
    inference_parser = subparsers.add_parser('scaling', help='Apply scaler and create normalized data.')
    inference_parser.add_argument('--feature_dir', required=True, help='Path to feature directory.')
    inference_parser.add_argument('--output_path', required=True, help='Path to save normalized data (pb and ps subdirectories will be created).')
    inference_parser.add_argument('--pb_scaler_path', required=True, help='Path to SC2-PB scaler.')
    inference_parser.add_argument('--ps_scaler_path', required=True, help='Path to SC2-PS scaler.')
    inference_parser.set_defaults(func=scale_sc2)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()