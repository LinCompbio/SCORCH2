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
        val_pb.to_csv(f'{pb_dir}/{name}_normalized_pb.csv', index=False)

        # Process sc2_ps
        df_ps = df.copy()
        df_ps = df_ps.drop(columns=['Id', 'NumRotatableBonds'], errors='ignore')

        val_ps_trans = sc2_ps_scaler.transform(df_ps)
        val_ps = pd.DataFrame(val_ps_trans, columns=df_ps.columns)
        val_ps['Id'] = df_id

        val_ps.fillna(0, inplace=True)
        val_ps.to_csv(f'{ps_dir}/{name}_normalized_ps.csv', index=False)


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
    inference_parser.add_argument('--feature_dir', required=True, help='Path to feature directory.',default='/home/s2523227/sc2_final/utils/feature')
    inference_parser.add_argument('--output_path', required=True, help='Path to save normalized data (pb and ps subdirectories will be created).',
                                  default='/home/s2523227/sc2_final/utils/feature/output')
    inference_parser.add_argument('--pb_scaler_path', required=True, help='Path to SC2-PB scaler.',
                                  default='/home/s2523227/sc2_final/')
    inference_parser.add_argument('--ps_scaler_path', required=True, help='Path to SC2-PS scaler.')
    inference_parser.set_defaults(func=scale_sc2)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

















#
# def process_molecule(molecule, ligand_path, protein_object, pdbid, protein_path):
#     """
#     Process a single ligand molecule file and extract features from all poses.
#
#     Args:
#         molecule: Filename of the ligand file
#         ligand_path: Path to the directory containing ligand files
#         pdbid: PDB ID being processed
#         protein_object: Protein content from binana PDB object
#         protein_path: Path to the protein file
#
#     Returns:
#         DataFrame containing features for all poses in the ligand file
#     """
#     try:
#         # Read the ligand file
#         ligand_file = os.path.join(ligand_path, molecule)
#         with open(ligand_file, 'r') as f:
#             lig_text = f.read()
#
#         # Split into individual poses (models)
#         lig_poses = lig_text.split('MODEL')
#         results = []
#
#         # Process each pose
#         for i, pose in enumerate(lig_poses):
#             try:
#                 # Clean up the pose content
#                 lines = pose.split('\n')
#                 clean_lines = [line for line in lines if not line.strip().lstrip().isnumeric() and 'ENDMDL' not in line]
#
#                 # Skip if not enough content
#                 if len(clean_lines) < 3:
#                     continue
#
#                 # Join cleaned lines back into a string
#                 pose_text = '\n'.join(clean_lines)
#
#                 # Calculate Kier flexibility and RDKit descriptors
#                 k, rdkit_descriptors = kier_flexibility(pose_text)
#                 entropy_df = pd.DataFrame([rdkit_descriptors])
#
#                 # Calculate BINANA features
#                 binana_features = run_binana(clean_lines, protein_object)
#                 binana_df = pd.DataFrame([binana_features])
#
#                 # Calculate ECIF features
#                 ecif_df = calculate_ecifs(pose_text, protein_path)
#
#                 # Combine all features
#                 df = pd.concat([ecif_df, binana_df], axis=1)
#                 df['Kier Flexibility'] = k
#
#                 try:
#                     # Prune to required columns and add identifier
#                     pruned_df = prune_df_headers(df)
#                     combined_df = pd.concat([entropy_df, pruned_df], axis=1)
#                     combined_df['Id'] = molecule
#                     results.append(combined_df)
#                 except Exception as e:
#                     print(f"Error in pruning/combining dataframes for {molecule}: {e}")
#                     # Create a basic fallback dataframe to avoid losing computation
#                     basic_df = pd.concat([entropy_df, df], axis=1)
#                     basic_df['Id'] = molecule
#                     results.append(basic_df)
#
#             except Exception as e:
#                 print(f"Error processing pose {i} in {molecule}: {e}")
#                 continue
#
#         # Combine results from all poses
#         if results:
#             try:
#                 return pd.concat(results, ignore_index=True)
#             except Exception as e:
#                 print(f"Error concatenating results for {molecule}: {e}")
#                 # If concat fails, return the first result (better than nothing)
#                 if len(results) > 0:
#                     return results[0]
#                 return pd.DataFrame()
#         else:
#             return pd.DataFrame()
#
#     except Exception as e:
#         print(f"Error processing molecule {molecule}: {e}")
#         return pd.DataFrame()
#
#
# def process_pdbid(pdbid, protein_base_path, molecule_path, des_path, num_cores=None):
#     """
#     Process a single PDB ID by extracting features from complex.
#
#     Args:
#         pdbid: The PDB ID to process
#         protein_base_path: Path to directory containing protein PDBQT files
#         molecule_path: Path to directory containing molecule PDBQT files
#         des_path: Directory to save output files
#         num_cores: Number of CPU cores to use (defaults to all available cores minus 1)
#     """
#     # Find the protein file
#     protein_path = glob.glob(f'{protein_base_path}/{pdbid}*.pdbqt')
#     if not protein_path:
#         print(f'Protein file not found for {pdbid}')
#         return
#     protein_path = protein_path[0]
#
#     # Check if output file already exists
#     output_file = os.path.join(des_path, f'{pdbid}_protein_features.csv')
#     if os.path.exists(output_file):
#         print(f'PDBID {pdbid} Feature File exists - skipping')
#         return
#
#     # Check if molecule directory exists
#     molecule_dir = os.path.join(molecule_path, pdbid)
#     if not os.path.exists(molecule_dir):
#         print(f'Molecules not found for {pdbid}')
#         return
#     molecules = os.listdir(molecule_dir)
#
#     # Read protein content and start processing
#     try:
#         with open(protein_path, 'r') as f:
#             protein_content = list(f.readlines())
#             protein_object = PDB()
#             protein_object.load_PDB(protein_path, protein_content)
#             protein_object.assign_secondary_structure()
#
#         # Determine number of processes to use
#         if num_cores is None:
#             processes = max(1, os.cpu_count() - 1)
#         else:
#             processes = min(num_cores, os.cpu_count())
#
#         # Process molecules in parallel
#         with Pool(processes=processes) as pool:
#             process_func = partial(
#                 process_molecule,
#                 ligand_path=molecule_dir,
#                 pdbid=pdbid,
#                 protein_object=protein_object,
#                 protein_path=protein_path
#             )
#             futures = [pool.apply_async(process_func, (molecule,)) for molecule in molecules]
#
#             results = []
#             for i, future in enumerate(tqdm(futures, desc=f"Processing {pdbid} molecules", leave=False)):
#                 try:
#                     result = future.get(timeout=5)  # 5-minute timeout per molecule
#                     if not result.empty:
#                         results.append(result)
#                 except TimeoutError:
#                     print(f"Processing molecule {i} for {pdbid} timed out")
#                 except Exception as e:
#                     print(f"Error processing {pdbid} molecule {i}: {e}")
#
#             # Combine results and save to file
#             if results:
#                 try:
#                     # Use dask for efficient concatenation of potentially large dataframes
#                     dask_results = dd.from_pandas(pd.concat(results, ignore_index=True), npartitions=8)
#                     total = dask_results.compute()
#
#                     if not total.empty:
#                         os.makedirs(des_path, exist_ok=True)
#                         total.to_csv(output_file, index=False)
#                         print(f"Saved features for {pdbid} ({len(total)} poses)")
#                 except Exception as e:
#                     print(f"Error saving results for {pdbid}: {e}")
#             else:
#                 print(f"No valid results for {pdbid}")
#
#     except Exception as e:
#         print(f'Error processing {pdbid}: {e}')
#
#
# def main(args):
#     """
#     Main execution function that processes features for protein-ligand pairs.
#
#     Args:
#         args: Command line arguments
#     """
#     des_path = args.output_dir
#     protein_base_path = args.protein_dir
#     molecule_path = args.ligand_dir
#     num_cores = args.num_cores
#
#     # Create output directory if it doesn't exist
#     os.makedirs(des_path, exist_ok=True)
#
#     # Get list of PDB IDs
#     if args.pdbids:
#         pdbids = args.pdbids.split(',')
#         print(f"Processing {len(pdbids)} specified PDB IDs")
#     else:
#         # Extract PDB IDs from filenames in protein directory
#         pdbids = [i.split("_")[0] for i in os.listdir(protein_base_path)]
#         print(f"Found {len(pdbids)} PDB IDs in {protein_base_path}")
#
#     # Process each PDB ID with progress bar
#     for pdbid in tqdm(pdbids, desc="Processing PDB structures"):
#         process_pdbid(pdbid, protein_base_path, molecule_path, des_path, num_cores)
#





