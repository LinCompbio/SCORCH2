# SC2 Evaluation 

## Data Processing

Before running the evaluation, use the `process_data.py` script to normalize your feature data:

```bash
# Create normalizers from your training data
python process_data.py create_normalizer \
    --feature_dir /path/to/training/features \
    --save_path /path/to/save/pb_scaler.pkl \
    --scaler_type maxabs \
    --drop_columns Id

python process_data.py create_normalizer \
    --feature_dir /path/to/training/features \
    --save_path /path/to/save/ps_scaler.pkl \
    --scaler_type maxabs \
    --drop_columns Id NumRotatableBonds

# Apply normalizers to your feature data
python process_data.py inference \
    --feature_dir /path/to/feature/data \
    --output_path /path/to/output/normalized \
    --pb_scaler_path /path/to/pb_scaler.pkl \
    --ps_scaler_path /path/to/ps_scaler.pkl
```

This will create two subdirectories (`pb` and `ps`) under your output path containing the normalized data for SC2-PB and SC2-PS models.

## Model Evaluation

The SC2 evaluation tool supports two main modes of evaluation:

### Virtual Screening Mode

This mode evaluates how well the models can distinguish between active compounds and decoys/inactive compounds.

```bash
python sc2_evaluation.py --mode vs \
    --sc2_ps /path/to/sc2_ps.pkl \
    --sc2_pb /path/to/sc2_pb.pkl \
    --sc2_ps_feature_repo /path/to/normalized_features/ps \
    --sc2_pb_feature_repo /path/to/normalized_features/pb \
    --keyword active \
    --aggregate \
    --output results/vs_results.csv
```

Key metrics reported:
- Enrichment Factors (EF) at 0.5%, 1%, 2%, and 5%
- BEDROC (Boltzmann-Enhanced Discrimination of ROC)
- Area Under ROC Curve (AUC-ROC)
- Area Under Precision-Recall Curve (AUC-PR)

### Ranking Mode

This mode evaluates how well the models can rank compounds based on their binding affinities, compared to experimental data.

```bash
python sc2_evaluation.py --mode ranking \
    --sc2_ps /path/to/sc2_ps.pkl \
    --sc2_pb /path/to/sc2_pb.pkl \
    --sc2_ps_feature_repo /path/to/normalized_features/ps \
    --sc2_pb_feature_repo /path/to/normalized_features/pb \
    --exp_repo /path/to/experimental_data \
    --output results/ranking_results.csv
```

Key metrics reported:
- Pearson correlation coefficient
- Spearman's rank correlation coefficient
- Kendall's tau correlation coefficient

## Command Line Arguments

### Data Processing Arguments:

#### Create Normalizer Mode:
- `--feature_dir`: Path to feature directory containing CSV files
- `--save_path`: Path to save the scaler pkl file
- `--scaler_type`: Type of scaler to use (`standard`, `minmax`, or `maxabs`)
- `--drop_columns`: Column names to exclude from normalization

#### Inference Mode:
- `--feature_dir`: Path to directory containing feature CSV files to normalize
- `--output_path`: Path to save normalized data (pb and ps subdirectories will be created)
- `--pb_scaler_path`: Path to the SC2-PB scaler
- `--ps_scaler_path`: Path to the SC2-PS scaler

### Evaluation Arguments:

#### Common Arguments:
- `--mode`: Evaluation mode (`vs` for virtual screening or `ranking` for affinity ranking)
- `--sc2_ps`: Path to the SC2-PS model file
- `--sc2_pb`: Path to the SC2-PB model file
- `--sc2_ps_feature_repo`: Path to the SC2-PS feature directory
- `--sc2_pb_feature_repo`: Path to the SC2-PB feature directory
- `--ps_consensus_weight`: Weight for SC2-PS predictions (default: 0.7)
- `--pb_consensus_weight`: Weight for SC2-PB predictions (default: 0.3)
- `--gpu`: Use GPU for prediction if available
- `--output`: Output CSV file name to save results

#### Virtual Screening Mode Arguments:
- `--aggregate`: Aggregate results by taking the maximum confidence
- `--keyword`: Keyword to assign labels (`active` if dataset uses active/decoy, `inactive` if it uses active/inactive)

#### Ranking Mode Arguments:
- `--exp_repo`: Path to the experimental data directory

## Examples

### Data Processing:

```bash![img.png](img.png)

# Create your own normalizer 
python process_data.py create_normalizer \
    --feature_dir /path/to/features \
    --save_path /path/to/save/scaler \
    --scaler_type maxabs \
    --drop_columns XXX

# Apply the normalizers to feature data
python process_data.py inference \
    --feature_dir /path/to/features \
    --output_path /path/to/normalized_features \
    --pb_scaler_path /path/to/pb_scaler.pkl \
    --ps_scaler_path /path/to/ps_scaler.pkl
```

### Virtual Screening Evaluation:

```bash
python sc2_evaluation.py --mode vs \
    --sc2_ps /path/to/sc2_ps.pkl \
    --sc2_pb /path/to/sc2_pb.pkl \
    --sc2_ps_feature_repo /path/to/normalized_features \
    --sc2_pb_feature_repo /path/to/normalized_features \
    --keyword active \
    --aggregate \
    --gpu \
    --output XXX
```

### Ranking Evaluation:

```bash
python sc2_evaluation.py --mode ranking \
    --sc2_ps /path/to/sc2_ps.pkl \
    --sc2_pb /path/to/sc2_pb.pkl \
    --sc2_ps_feature_repo /path/to/normalized_features \
    --sc2_pb_feature_repo /path/to/normalized_features \
    --exp_repo /path/to/experimental_data \
    --gpu \
    --output output/ranking_results.csv
```
