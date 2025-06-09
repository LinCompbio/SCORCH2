# SCORCH2 Data Structure Requirements

This document describes the expected directory structure and file formats for SCORCH2 input data.

## Directory Structure

```
project_root/
├── protein/
│   └── {pdb_id}_protein.pdbqt
├── molecule/
│   └── {pdb_id}/
│       ├── {pdb_id}_ligand.pdbqt
│       ├── {pdb_id}_ligand_out_pose1.pdbqt
│       ├── {pdb_id}_ligand_out_pose2.pdbqt
│       └── ... (additional poses)
└── features/ (generated after feature extraction)
    ├── sc2_ps/
    │   └── {pdb_id}_normalized.csv
    └── sc2_pb/
        └── {pdb_id}_normalized.csv
```

## File Naming Conventions

### Protein Files
- **Location**: `protein/`
- **Format**: `{pdb_id}_protein.pdbqt`
- **Example**: `4a9r_protein.pdbqt`

### Ligand Files
- **Location**: `molecule/{pdb_id}/`
- **Formats**:
  - Original ligand: `{pdb_id}_ligand.pdbqt`
  - Docked poses: `{pdb_id}_ligand_out_pose{N}.pdbqt`
- **Examples**:
  - `4a9r/4a9r_ligand.pdbqt`
  - `4a9r/4a9r_ligand_out_pose1.pdbqt`
  - `4a9r/4a9r_ligand_out_pose2.pdbqt`

### Feature Files (Generated)
- **Location**: `features/sc2_ps/` and `features/sc2_pb/`
- **Format**: `{pdb_id}_normalized.csv`
- **Example**: `4a9r_normalized.csv`

## File Content Requirements

### PDBQT Files
- Must be valid PDBQT format (AutoDock Tools compatible)
- Protein files should contain receptor structure with proper atom types
- Ligand files should contain small molecule structures with rotatable bonds defined

### Feature CSV Files
- Must contain an 'Id' column with compound identifiers
- Feature columns should match the training data feature space
- Missing values will be filled with zeros during processing

## Example Data Structure

Based on the provided example data (`example_data/`):

```
example_data/
├── protein/
│   └── 4a9r_protein.pdbqt          # Receptor structure (5,388 lines)
└── molecule/
    └── 4a9r/                       # Target directory
        ├── 4a9r_ligand.pdbqt       # Original ligand (49 lines)
        ├── 4a9r_ligand_out_pose1.pdbqt   # Pose 1 (56 lines)
        ├── 4a9r_ligand_out_pose2.pdbqt   # Pose 2 (56 lines)
        ├── 4a9r_ligand_out_pose3.pdbqt   # Pose 3 (56 lines)
        ├── 4a9r_ligand_out_pose5.pdbqt   # Pose 5 (56 lines)
        ├── 4a9r_ligand_out_pose18.pdbqt  # Pose 18 (56 lines)
        └── 4a9r_ligand_out_pose22.pdbqt  # Pose 22 (56 lines)
```

## Multi-Pose Handling

When multiple poses are available for a single compound:

1. **Feature Extraction**: All poses will be processed individually
2. **Rescoring**: Each pose receives a separate score
3. **Aggregation** (optional): The highest-scoring pose can be selected as representative
4. **Output**: Results include information about which pose was selected

## Data Preparation Workflow

1. **Structure Preparation**:
   - Convert protein structures to PDBQT format using `receptor_2_pdbqt.py`
   - Convert ligand structures to PDBQT format using `ligand_2_pdbqt.py`

2. **Feature Extraction**:
   ```bash
   python utils/scorch2_feature_extraction.py \
       --protein-dir protein/ \
       --ligand-dir molecule/ \
       --output-dir features/
   ```

3. **Feature Normalization**:
   ```bash
   python evaluation/process_data.py scaling \
       --feature_dir features/ \
       --output_path normalized_features/ \
       --pb_scaler_path models/sc2_pb_scaler.pkl \
       --ps_scaler_path models/sc2_ps_scaler.pkl
   ```

4. **Rescoring**:
   ```bash
   python scorch2_rescoring.py \
       --sc2_ps models/sc2_ps.xgb \
       --sc2_pb models/sc2_pb.xgb \
       --features normalized_features/ \
       --output results.csv \
       --aggregate
   ```

## Notes

- PDB IDs should be consistent across protein and molecule directories
- The system automatically handles missing poses (files that don't exist are skipped)
- Feature extraction can be parallelized using the `--num-cores` parameter
- GPU acceleration is available for rescoring using the `--gpu` flag 