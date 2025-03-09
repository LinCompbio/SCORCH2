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
import multiprocessing
from functools import partial
from binana import PDB
from multiprocessing import Pool, TimeoutError
from tqdm import tqdm
import signal
from functools import wraps
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import glob
import argparse

# Mute all RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
pybel.ob.obErrorLog.StopLogging()


def calculate_ecifs(ligand_pdbqt_block, receptor_content):
    ECIF_data = ecif.GetECIF(receptor_content, ligand_pdbqt_block, distance_cutoff=6.0)
    ECIFHeaders = [header.replace(';', '') for header in ecif.PossibleECIF]
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
    main_binana_out = binana.Binana(ligand_pdbqt_block, receptor_content).out

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
    reference_headers = json.load(open(os.path.join('SC1_features.json')))
    headers_58 = reference_headers.get('492_models_58')
    return df[headers_58]


def process_molecule(decoy, ligand_path, pdbid, protein_content, protein_path):

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


def process_pdbid(pdbid, protein_base_path, molecule_path, des_path):
    protein_path = glob.glob(f'{protein_base_path}/{pdbid}*.pdbqt')
    if not protein_path:
        print(f'Protein file not found for {pdbid}')
        return
    protein_path = protein_path[0]

    output_file = os.path.join(des_path, f'{pdbid}_features.csv')
    if os.path.exists(output_file):
        print(f'PDBID {pdbid} Feature File exists')
        return

    molecule_path = os.path.join(molecule_path, pdbid)
    if not os.path.exists(molecule_path):
        print(f'Decoys not found for {pdbid}')
        return
    decoys = os.listdir(molecule_path)

    # Read protein content and start processing
    try:
        with open(protein_path, 'r') as f:
            protein_content = list(f.readlines())
            receptor_object = PDB()
            receptor_object.load_PDB(protein_path, protein_content)
            receptor_object.assign_secondary_structure()

        with Pool(processes=os.cpu_count()-1) as pool:
            process_func = partial(process_molecule, ligand_path=molecule_path, pdbid=pdbid, protein_content=receptor_object, protein_path=protein_path)
            futures = [pool.apply_async(process_func, (decoy,)) for decoy in decoys]

            results = []
            for future in futures:
                try:
                    result = future.get()  # Wait for each sub-process result
                    results.append(result)
                except TimeoutError:
                    print(f"Processing decoy timed out")
                except Exception as e:
                    print(f"Error: {e}")

            if results:
                dask_results = dd.from_pandas(pd.concat(results, ignore_index=True), npartitions=8)
                total = dask_results.compute()

                if not total.empty:
                    output_directory = os.path.join(des_path)
                    os.makedirs(output_directory, exist_ok=True)
                    total.to_csv(output_file, index=False)

    except Exception as E:
        print(f'Error processing {pdbid}: {E}')


def main(args):
    des_path = args.des_path
    protein_base_path = args.protein_base_path
    molecule_path = args.molecule_path
    pdbids = [i.split("_")[0] for i in os.listdir(protein_base_path)]


    os.makedirs(des_path, exist_ok=True)

    for pdbid in tqdm(pdbids):
        process_pdbid(pdbid, protein_base_path, molecule_path, des_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process PDBID features from decoy protein-ligand pairs")
    parser.add_argument('--des_path', type=str, required=True, help="Directory to save the results.")
    parser.add_argument('--protein_base_path', type=str, required=True, help="Directory containing protein pdbqt files.")
    parser.add_argument('--molecule_path', type=str, required=True, help="Directory containing molecule pdbqt files.")

    args = parser.parse_args()
    pdbids = args.pdbids.split(',')  # Convert the comma-separated string into a list of pdbids

    main(args)