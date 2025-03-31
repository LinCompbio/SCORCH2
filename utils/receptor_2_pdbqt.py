import subprocess
from pathlib import Path
from tqdm import tqdm
import os
import multiprocessing
import argparse


def prepare_receptor_pdbqt(pdb_file, pdbqt_file):
    """
    Convert a .pdb file to .pdbqt format using ADFRsuite' `prepare_receptor`.

    Parameters:
    - pdb_file (Path): Input PDB file path.
    - pdbqt_file (Path): Output PDBQT file path.

    Notes:
    - Skips conversion if output already exists.
    - Uses absolute paths and includes specific flags for cleaning the receptor.
    """
    # Ensure output directory exists
    pdbqt_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to absolute paths for safety
    abs_pdb_file = Path(pdb_file).resolve()
    abs_pdbqt_file = Path(pdbqt_file).resolve()

    # Skip conversion if output already exists
    if abs_pdbqt_file.exists():
        print(f"[SKIP] {abs_pdbqt_file} already exists.")
        return

    # Build AutoDockTools prepare_receptor command
    command = (
        f'prepare_receptor -r {abs_pdb_file} -o {abs_pdbqt_file} '
        f'-U nphs_lps_waters_nonstdres -A bonds_hydrogens -e'
    )

    try:
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        # Catch and print any errors from the command
        print(f"[ERROR] Failed on {abs_pdb_file.name}:\n{e.stderr}")


def process_file(args):
    """
    Wrapper function to handle a single task (input file → output file).

    Parameters:
    - args (tuple): (Path to .pdb file, Path to output directory)
    """
    file, output_repo = args
    name = Path(file).stem
    pdbqt_file = output_repo / f'{name}.pdbqt'
    prepare_receptor_pdbqt(file, pdbqt_file)


def main(source_dir, output_dir, n_proc):
    """
    Main batch processing function.

    Parameters:
    - source_dir (str): Directory containing input .pdb files.
    - output_dir (str): Directory to save output .pdbqt files.
    - n_proc (int): Number of processes to use for parallel conversion.
    """
    source_repo = Path(source_dir).resolve()
    output_repo = Path(output_dir).resolve()
    output_repo.mkdir(parents=True, exist_ok=True)

    # Find all .pdb files in the source directory
    pdb_files = sorted(source_repo.glob('*.pdb'))

    # Prepare task list for multiprocessing
    tasks = [(file, output_repo) for file in pdb_files]

    # Start multiprocessing pool and monitor progress with tqdm
    with multiprocessing.Pool(processes=n_proc) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc="Converting"):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert .pdb to .pdbqt using prepare_receptor (AutoDockTools)")
    parser.add_argument('--source_dir', type=str, required=True, help='Path to directory with .pdb files')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to directory for .pdbqt files')
    parser.add_argument('--n_proc', type=int, default=os.cpu_count()-1, help='Number of parallel processes')

    args = parser.parse_args()
    main(args.source_dir, args.output_dir, args.n_proc)
