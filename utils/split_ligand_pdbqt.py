import os
import argparse
import numpy as np
from tqdm import tqdm

def extract_top_poses(input_dir: str, output_dir: str, max_poses: int = 100):
    for root, dirs, _ in tqdm(os.walk(input_dir), desc="Processing directories"):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            pdbqt_files = [f for f in os.listdir(dir_path) if f.endswith(".pdbqt")]
            for pdbqt_file in tqdm(pdbqt_files, desc=f"Processing files in {dir}", leave=False):
                pdbqt_path = os.path.join(dir_path, pdbqt_file)
                os.makedirs(os.path.join(output_dir, dir), exist_ok=True)

                with open(pdbqt_path, 'r') as file:
                    lines = file.readlines()

                poses, current_pose = [], []
                for line in lines:
                    if line.startswith('MODEL'):
                        current_pose = [line]
                    elif line.startswith('ENDMDL'):
                        current_pose.append(line)
                        poses.append(current_pose)
                        current_pose = []
                    else:
                        current_pose.append(line)

                pose_indices = np.linspace(0, len(poses) - 1, min(max_poses, len(poses)), dtype=int)
                for index in pose_indices:
                    pose_lines = poses[index]
                    output_file = f"{pdbqt_file[:-6]}_pose{index + 1}.pdbqt"
                    output_path = os.path.join(output_dir, dir, output_file)
                    with open(output_path, 'w') as f_out:
                        f_out.writelines(pose_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract top poses from PDBQT files.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input directory containing PDBQT files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory to save extracted poses.")
    parser.add_argument('--max_poses', type=int, default=100, help="Maximum number of poses to extract per file.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    extract_top_poses(args.input_dir, args.output_dir, args.max_poses)