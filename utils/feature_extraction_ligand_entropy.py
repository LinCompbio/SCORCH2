import os
import pandas as pd
import binana_accelerate
import random
from tqdm import tqdm
import kier
import ecif_accelerate
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem, RDLogger
import json
import dask.dataframe as dd
import multiprocessing
from functools import partial
from binana_accelerate import PDB
from multiprocessing import Pool, TimeoutError
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit.Chem.rdFreeSASA import CalcSASA,classifyAtoms
import functools
import signal
import time
from functools import wraps
import csv
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import glob
# Mute all RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
pybel.ob.obErrorLog.StopLogging()

class TimeoutException(Exception):
    pass


def time_limit(seconds):
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutException(f"Function exceeded time limit of {seconds} seconds")

        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            except TimeoutException:
                return pd.DataFrame()  # 超时返回空的 DataFrame
            finally:
                signal.alarm(0)

            return result

        return wrapper

    return decorator


def calculate_ecifs(ligand_pdbqt_block, receptor_content):
    ECIF_data = ecif_accelerate.GetECIF(receptor_content, ligand_pdbqt_block, distance_cutoff=6.0)
    ECIFHeaders = [header.replace(';', '') for header in ecif_accelerate.PossibleECIF]
    ECIF_data = dict(zip(ECIFHeaders, ECIF_data))
    return pd.DataFrame(ECIF_data, index=[0])

def get_vdw_radii(mol):
    periodic_table = Chem.GetPeriodicTable()
    return [periodic_table.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]

def has_hydrogen_atoms(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:  # Hydrogen has atomic number 1
            return True
    return False

def kier_flexibility(ligand_pdbqt_block):

    invariant_2d_descriptors = [
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

    descriptor_names = [desc[0] for desc in Descriptors._descList]

    mol = kier.SmilePrep(ligand_pdbqt_block)
    mol.GetRingInfo()
    mol_without_H = Chem.RemoveHs(mol)

    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = calculator.CalcDescriptors(mol_without_H)

    features = {}

    for name, value in zip(descriptor_names, descriptors):
        if name in invariant_2d_descriptors:
            features[name] = value
        else:
            continue

    return kier.CalculateFlexibility(mol),features



def run_binana(ligand_pdbqt_block, receptor_content):
    binana_features = {}
    main_binana_out = binana_accelerate.Binana(ligand_pdbqt_block, receptor_content).out

    # define the features we want
    keep_closest_contacts = ["2.5 (HD, OA)",
                             "2.5 (HD, HD)",
                             "2.5 (HD, N)",
                             "2.5 (C, HD)",
                             "2.5 (OA, ZN)",
                             "2.5 (HD, ZN)",
                             "2.5 (A, HD)"]

    keep_close_contacts = ["4.0 (C, C)",
                           "4.0 (HD, OA)",
                           "4.0 (C, HD)",
                           "4.0 (C, N)",
                           "4.0 (A, C)",
                           "4.0 (A, OA)",
                           "4.0 (N, OA)",
                           "4.0 (A, N)",
                           "4.0 (HD, N)",
                           "4.0 (HD, HD)",
                           "4.0 (A, HD)",
                           "4.0 (OA, OA)",
                           "4.0 (C, OA)",
                           "4.0 (N, N)",
                           "4.0 (C, SA)",
                           "4.0 (HD, SA)",
                           "4.0 (OA, SA)",
                           "4.0 (N, SA)",
                           "4.0 (A, A)",
                           "4.0 (HD, S)",
                           "4.0 (S, ZN)",
                           "4.0 (N, ZN)",
                           "4.0 (HD, ZN)",
                           "4.0 (A, SA)",
                           "4.0 (OA, ZN)",
                           "4.0 (C, ZN)",
                           "4.0 (C, NA)",
                           "4.0 (NA, OA)",
                           "4.0 (HD, NA)",
                           "4.0 (N, NA)",
                           "4.0 (A, NA)",
                           "4.0 (BR, C)",
                           "4.0 (HD, P)",
                           "4.0 (F, N)",
                           "4.0 (F, HD)",
                           "4.0 (C, CL)",
                           "4.0 (CL, HD)"]

    keep_ligand_atoms = ["LA N",
                         "LA HD"]

    keep_elsums = ["ElSum (C, C)",
                   "ElSum (HD, OA)",
                   "ElSum (C, HD)",
                   "ElSum (C, N)",
                   "ElSum (A, C)",
                   "ElSum (A, OA)",
                   "ElSum (N, OA)",
                   "ElSum (A, N)",
                   "ElSum (HD, HD)",
                   "ElSum (A, HD)",
                   "ElSum (OA, OA)",
                   "ElSum (C, OA)",
                   "ElSum (N, N)",
                   "ElSum (C, SA)",
                   "ElSum (HD, SA)",
                   "ElSum (OA, SA)",
                   "ElSum (N, SA)",
                   "ElSum (A, A)",
                   "ElSum (N, S)",
                   "ElSum (HD, S)",
                   "ElSum (OA, S)",
                   "ElSum (A, SA)",
                   "ElSum (C, NA)",
                   "ElSum (NA, OA)",
                   "ElSum (HD, NA)",
                   "ElSum (N, NA)",
                   "ElSum (A, NA)",
                   "ElSum (BR, C)",
                   "ElSum (HD, P)",
                   "ElSum (OA, P)",
                   "ElSum (N, P)",
                   "ElSum (C, F)",
                   "ElSum (F, N)",
                   "ElSum (A, F)",
                   "ElSum (CL, OA)",
                   "ElSum (C, CL)",
                   "ElSum (CL, N)",
                   "ElSum (A, CL)"]

    # add closest contacts to binana_features dict
    for contact in keep_closest_contacts:
        binana_name = contact.split('(')[-1].split(')')[0].replace(', ', '_')
        binana_features[contact] = main_binana_out['closest'].get(binana_name)

    # add close contacts to binana_features dict
    for contact in keep_close_contacts:
        binana_name = contact.split('(')[-1].split(')')[0].replace(', ', '_')
        binana_features[contact] = main_binana_out['close'].get(binana_name)

    # add ligand atoms to binana_features dict as binary tallies
    for atom in keep_ligand_atoms:
        binana_name = atom.split()[-1]
        if main_binana_out['ligand_atoms'].get(binana_name) is None:
            binana_features[atom] = 0
        else:
            binana_features[atom] = 1

    # add electrostatics to binana_features dict
    for elsum in keep_elsums:
        binana_name = elsum.split('(')[-1].split(')')[0].replace(', ', '_')
        binana_features[elsum] = main_binana_out['elsums'].get(binana_name)

    # add active site flexibility features to binana_features
    binana_features["BPF ALPHA SIDECHAIN"] = main_binana_out['bpfs'].get("SIDECHAIN_ALPHA")
    binana_features["BPF ALPHA BACKBONE"] = main_binana_out['bpfs'].get("BACKBONE_ALPHA")
    binana_features["BPF BETA SIDECHAIN"] = main_binana_out['bpfs'].get("SIDECHAIN_BETA")
    binana_features["BPF BETA BACKBONE"] = main_binana_out['bpfs'].get("BACKBONE_BETA")
    binana_features["BPF OTHER SIDECHAIN"] = main_binana_out['bpfs'].get("SIDECHAIN_OTHER")
    binana_features["BPF OTHER BACKBONE"] = main_binana_out['bpfs'].get("BACKBONE_OTHER")

    # add hydrophobic features to binana_features
    binana_features["HC ALPHA SIDECHAIN"] = main_binana_out['hydrophobics'].get("SIDECHAIN_ALPHA")
    binana_features["HC ALPHA BACKBONE"] = main_binana_out['hydrophobics'].get("BACKBONE_ALPHA")
    binana_features["HC BETA SIDECHAIN"] = main_binana_out['hydrophobics'].get("SIDECHAIN_BETA")
    binana_features["HC BETA BACKBONE"] = main_binana_out['hydrophobics'].get("BACKBONE_BETA")
    binana_features["HC OTHER SIDECHAIN"] = main_binana_out['hydrophobics'].get("SIDECHAIN_OTHER")
    binana_features["HC OTHER BACKBONE"] = main_binana_out['hydrophobics'].get("BACKBONE_OTHER")

    # add hydrogen bond features to binana_features
    binana_features["HB ALPHA SIDECHAIN LIGAND"] = main_binana_out['hbonds'].get("HDONOR_LIGAND_SIDECHAIN_ALPHA")
    binana_features["HB BETA SIDECHAIN LIGAND"] = main_binana_out['hbonds'].get("HDONOR_LIGAND_SIDECHAIN_BETA")
    binana_features["HB BETA BACKBONE LIGAND"] = main_binana_out['hbonds'].get("HDONOR_LIGAND_BACKBONE_BETA")
    binana_features["HB OTHER SIDECHAIN LIGAND"] = main_binana_out['hbonds'].get("HDONOR_LIGAND_SIDECHAIN_OTHER")
    binana_features["HB OTHER BACKBONE LIGAND"] = main_binana_out['hbonds'].get("HDONOR_LIGAND_BACKBONE_OTHER")
    binana_features["HB ALPHA SIDECHAIN RECEPTOR"] = main_binana_out['hbonds'].get("HDONOR_RECEPTOR_SIDECHAIN_ALPHA")
    binana_features["HB ALPHA BACKBONE RECEPTOR"] = main_binana_out['hbonds'].get("HDONOR_RECEPTOR_BACKBONE_ALPHA")
    binana_features["HB BETA SIDECHAIN RECEPTOR"] = main_binana_out['hbonds'].get("HDONOR_RECEPTOR_SIDECHAIN_BETA")
    binana_features["HB BETA BACKBONE RECEPTOR"] = main_binana_out['hbonds'].get("HDONOR_RECEPTOR_BACKBONE_BETA")
    binana_features["HB OTHER SIDECHAIN RECEPTOR"] = main_binana_out['hbonds'].get("HDONOR_RECEPTOR_SIDECHAIN_OTHER")
    binana_features["HB OTHER BACKBONE RECEPTOR"] = main_binana_out['hbonds'].get("HDONOR_RECEPTOR_BACKBONE_OTHER")

    # add salt bridge features to binana_features
    binana_features["SB ALPHA"] = main_binana_out['salt_bridges'].get("SALT-BRIDGE_ALPHA")
    binana_features["SB BETA"] = main_binana_out['salt_bridges'].get("SALT-BRIDGE_BETA")
    binana_features["SB OTHER"] = main_binana_out['salt_bridges'].get("SALT-BRIDGE_OTHER")

    # add aromatic stacking features to binana_features
    binana_features["piStack ALPHA"] = main_binana_out['stacking'].get("STACKING ALPHA")
    binana_features["piStack BETA"] = main_binana_out['stacking'].get("STACKING BETA")
    binana_features["piStack OTHER"] = main_binana_out['stacking'].get("STACKING OTHER")
    binana_features["tStack ALPHA"] = main_binana_out['t_stacking'].get("T-SHAPED_ALPHA")
    binana_features["tStack BETA"] = main_binana_out['t_stacking'].get("T-SHAPED_BETA")
    binana_features["tStack OTHER"] = main_binana_out['t_stacking'].get("T-SHAPED_OTHER")

    # add cation pi features to binana_features
    binana_features["catPi BETA LIGAND"] = main_binana_out['pi_cation'].get("PI-CATION_LIGAND-CHARGED_BETA")
    binana_features["catPi OTHER LIGAND"] = main_binana_out['pi_cation'].get("PI-CATION_LIGAND-CHARGED_OTHER")

    # add rotatable bond count to binana features
    binana_features["nRot"] = main_binana_out['nrot']

    # return dictionary
    return binana_features


def prune_df_headers(df):
    reference_headers = json.load(open(os.path.join('features.json')))
    headers_58 = reference_headers.get('492_models_58')
    return df[headers_58]


# @time_limit(30)
def process_decoy(decoy, ligand_path, pdbid, protein_content, protein_path):

    with open(os.path.join(ligand_path, pdbid, decoy), 'r') as f:
        lig_text = f.read()

    lig_poses = lig_text.split('MODEL')
    results = []

    for pose in lig_poses:
        try:
            lines = pose.split('\n')
            clean_lines = [line for line in lines if not line.strip().lstrip().isnumeric() and 'ENDMDL' not in line]
            if len(clean_lines) < 3:
                continue
            else:
                pose = '\n'.join(clean_lines)
                # k,entropy_features = kier_flexibility(pose)
                k, rdkit_2d_descriptors = kier_flexibility(pose)
                entropy_df = pd.DataFrame([rdkit_2d_descriptors])
                binana_features = run_binana(clean_lines, protein_content)
                binana_df = pd.DataFrame([binana_features])
                ecif_df = calculate_ecifs(pose, protein_path)
                df = pd.concat([ecif_df, binana_df], axis=1)
                df['Kier Flexibility'] = k
                df = prune_df_headers(df)
                df = pd.concat([entropy_df,df], axis=1)
                df['Id'] = decoy
                results.append(df)

        except Exception as e:
            print(f"Error processing pose in {decoy}: {e}")

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def process_pdbid(pdbid, protein_base_path, decoys_path, des_path):

    # protein_path = f'{protein_base_path}/{pdbid}_receptor.pdbqt'
    protein_path = glob.glob(f'{protein_base_path}/{pdbid}*.pdbqt')
    if not protein_path:
        print(f'Protein file not found for {pdbid}')
        return
    else:
        protein_path = protein_path[0]

    rec_or_pro = 'protein'

    # if not os.path.exists(protein_path):
    #     if os.path.exists(f'{protein_base_path}/{pdbid}_protein.pdbqt'):
    #         protein_path = f'{protein_base_path}/{pdbid}_protein.pdbqt'
    #         rec_or_pro = 'protein'
    #     else:
    #         print(f'Protein file not found for {pdbid}')
    #         return

    output_file = os.path.join(des_path, f'{pdbid}_{rec_or_pro}_features.csv')
    if os.path.exists(output_file):
        print(f'PBDID {pdbid} Feature File exists')
        return

    pdbid_path = os.path.join(decoys_path, pdbid)
    if not os.path.exists(pdbid_path):
        print(f'Decoys not found for {pdbid}')
        return
    else:
        decoys = os.listdir(pdbid_path)
        # decoys = [decoy for decoy in decoys if 'ligand' in decoy]
        # selected_decoys = random.sample(decoys, min(len(decoys), 150))

    try:
        with open(protein_path, 'r') as f:
            protein_content = list(f.readlines())
            receptor_object = PDB()
            receptor_object.load_PDB(protein_path, protein_content)
            receptor_object.assign_secondary_structure()

        with Pool(processes=os.cpu_count()-1) as pool:
            process_func = partial(process_decoy, ligand_path=decoys_path, pdbid=pdbid, protein_content=receptor_object, protein_path=protein_path)
            futures = [pool.apply_async(process_func, (decoy,)) for decoy in decoys]

            results = []
            for future in futures:
                try:
                    result = future.get()  # 等待每个子进程的结果，并设置超时
                    results.append(result)
                except TimeoutError:
                    print(f"Processing decoy timed out")
                    break
                except Exception as e:
                    print(f"Error: {e}")

            if results:
                dask_results = dd.from_pandas(pd.concat(results, ignore_index=True), npartitions=os.cpu_count() - 1)
                total = dask_results.compute()

                if not total.empty:
                    output_directory = os.path.join(des_path)
                    os.makedirs(output_directory, exist_ok=True)
                    total.to_csv(output_file, index=False)


            # if results:

                # # Use StringIO as a buffer
                # buffer = StringIO()
                # csv_writer = csv.writer(buffer)
                #
                # # Write header
                # if results[0] is not None:
                #     csv_writer.writerow(results[0].columns)
                #
                # # Write data in chunks
                # chunk_size = 10000000  # Adjust this value based on your system's memory
                # for i in range(0, len(results), chunk_size):
                #     chunk = results[i:i + chunk_size]
                #     for df in chunk:
                #         if df is not None:
                #             csv_writer.writerows(df.values)
                #
                # # Write buffer contents to file
                # output_directory = os.path.join(des_path, task)
                # os.makedirs(output_directory, exist_ok=True)
                # with open(output_file, 'w', newline='') as f:
                #     f.write(buffer.getvalue())

    except Exception as E:
        print(f'Error processing {pdbid}: {E}')


if __name__ == "__main__":


    des_path = '/home/s2523227/sc2_dapanther/ranking_test/fep_benchmark_feature'
    os.makedirs(des_path, exist_ok=True)
    # task = 'rmsd_4'
    protein_base_path = '/home/s2523227/sc2_dapanther/ranking_test/fep_protein_pdbqt'
    decoys_path = '/home/s2523227/sc2_dapanther/ranking_test/fep_molecule_pdbqt'

    # pdbids = pd.read_csv('/home/s2523227/single_group/pdbid_saved.csv')['pdbid'].tolist()
    pdbids = [i.split('_')[0] for i in os.listdir(protein_base_path)]

    for pdbid in tqdm(pdbids):
        process_pdbid(pdbid, protein_base_path, decoys_path, des_path)

