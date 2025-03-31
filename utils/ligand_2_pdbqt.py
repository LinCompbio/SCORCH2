import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse


def convert_ligand_to_pdbqt(input_file, output_file):
    """
    Convert ligand file to .pdbqt using AutoDockTools' `prepare_ligand`.

    Parameters:
    - input_file (str): Full path to the input ligand file (e.g., .pdb, .mol2)
    - output_file (str): Full path to the output .pdbqt file
    """
    try:
        input_dir = Path(input_file).parent
        input_filename = Path(input_file).name

        # Construct the command
        command = (
            f'prepare_ligand -l {input_filename} -o {output_file} '
            f'-A bonds_hydrogens -U nphs_lps'
        )

        subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=input_dir
        )
        return f'[OK] {input_file} → {output_file}'

    except subprocess.CalledProcessError as e:
        return f'[ERROR] {input_file}: {e.stderr.strip()}'


def collect_tasks(input_base_dir, output_base_dir, input_format):
    """
    Generator to yield (input_file, output_file) pairs for processing.

    Parameters:
    - input_base_dir (Path): Directory with ligand files
    - output_base_dir (Path): Target directory for .pdbqt files
    - input_format (str): File extension to search (e.g., 'pdb', 'mol2')
    """
    input_base_dir = Path(input_base_dir).resolve()
    output_base_dir = Path(output_base_dir).resolve()

    ligand_files = list(input_base_dir.rglob(f'*.{input_format.lower()}'))
    for ligand_file in ligand_files:
        relative_path = ligand_file.relative_to(input_base_dir)
        output_file = output_base_dir / relative_path.with_suffix('.pdbqt')
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if not output_file.exists():
            yield str(ligand_file), str(output_file)


def main(input_dir, output_dir, input_format, n_proc):
    """
    Main function to run the ligand conversion pipeline.

    Parameters:
    - input_dir (str): Folder containing ligand files
    - output_dir (str): Folder to save converted .pdbqt files
    - input_format (str): Input file extension (without dot)
    - n_proc (int): Number of parallel processes
    """
    tasks = list(collect_tasks(input_dir, output_dir, input_format))

    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        futures = [executor.submit(convert_ligand_to_pdbqt, inp, out) for inp, out in tasks]

        with tqdm(total=len(futures), desc=f'Converting *.{input_format} → .pdbqt') as pbar:
            for future in as_completed(futures):
                print(future.result())
                pbar.update(1)

    print("All conversions completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch convert ligand files to .pdbqt using AutoDockTools")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing ligand files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save .pdbqt files')
    parser.add_argument('--input_format', type=str, required=True, help='Ligand input file format (e.g., pdb, mol2)')
    parser.add_argument('--n_proc', type=int, default=os.cpu_count()-1, help='Number of parallel processes')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.input_format, args.n_proc)
