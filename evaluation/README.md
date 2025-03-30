# SC2 Evaluation 

## Data Processing

Before running the evaluation, use the `process_data.py` script to normalize your feature data. This script supports two modes: `create_normalizer` for generating scalers and `scaling` for applying them to your data.

### Creating Your Own Normalizer

Use this step to create normalizers from your training data:

```bash
python process_data.py create_normalizer \
    --feature_dir /path/to/training/features \
    --save_path /path/to/save/scaler \
    --scaler_type maxabs \
    --drop_columns XXX

### Applying Normalizers

If you would like to use SCORCH2 to rescore your data, take the code below to create corresponding features.

```bash
python process_data.py scaling \
    --feature_dir /path/to/feature/data \
    --output_path /path/to/output/normalized \
    --pb_scaler_path /path/to/sc2_pb_scaler \
    --ps_scaler_path /path/to/sc2_ps_scaler
```

This will create two subdirectories (`sc2_pb` and `sc2_ps`) under your output path containing the normalized data for SC2-PB and SC2-PS models.

## Model Evaluation

The SC2 evaluation tool supports two main modes of evaluation:

### Virtual Screening Mode

This mode evaluates how well the models can distinguish between active compounds and decoys/inactive compounds.

```bash
python sc2_evaluation.py vs \
    --sc2_ps /path/to/sc2_ps.xgb \
    --sc2_pb /path/to/sc2_pb.xgb \
    --sc2_ps_feature_repo /path/to/normalized_features/sc2_ps \
    --sc2_pb_feature_repo /path/to/normalized_features/sc2_pb \
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
python sc2_evaluation.py ranking \
    --sc2_ps /path/to/sc2_ps.xgb \
    --sc2_pb /path/to/sc2_pb.xgb \
    --sc2_ps_feature_repo /path/to/normalized_features/sc2_ps \
    --sc2_pb_feature_repo /path/to/normalized_features/sc2_pb \
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
- `--save_path`: Path to save the scaler xgb file
- `--scaler_type`: Type of scaler to use (`standard`, `minmax`, or `maxabs`)
- `--drop_columns`: Column names to exclude from normalization

#### Scaling Mode:
- `--feature_dir`: Path to directory containing feature CSV files to normalize
- `--output_path`: Path to save normalized data (sc2_pb and sc2_ps subdirectories will be created)
- `--pb_scaler_path`: Path to the SC2-PB scaler
- `--ps_scaler_path`: Path to the SC2-PS scaler

### Evaluation Arguments:

#### Common Arguments:
- `--sc2_ps`: Path to the SC2-PS model file
- `--sc2_pb`: Path to the SC2-PB model file
- `--sc2_ps_feature_repo`: Path to the SC2-PS feature directory
- `--sc2_pb_feature_repo`: Path to the SC2-PB feature directory
- `--ps_consensus_weight`: Weight for SC2-PS predictions (default: 0.7)
- `--pb_consensus_weight`: Weight for SC2-PB predictions (default: 0.3)
- `--gpu`: Use GPU for prediction if available
- `--output`: Output CSV file path to save results

#### Virtual Screening Mode Arguments:
- `--aggregate`: Aggregate results by taking the maximum confidence (DUD-E,DEKOIS 2.0 (vina,ledock,surflex,gold) do not support result aggregation
- since only one pose for each compound is available)
- `--keyword`: Keyword to assign labels (`active` if dataset uses active/decoy, `inactive` if it uses active/inactive)

#### Ranking Mode Arguments:
- `--exp_repo`: Path to the experimental data directory

## Examples

### Data Processing:

```bash
# Create your own normalizer 
python process_data.py create_normalizer \
    --feature_dir /path/to/features \
    --save_path /path/to/save/scaler.xgb \
    --scaler_type maxabs \
    --drop_columns Id NumRotatableBonds

# Apply the normalizers to feature data
python process_data.py scaling \
    --feature_dir /path/to/features \
    --output_path /path/to/normalized_features \
    --pb_scaler_path /path/to/pb_scaler.xgb \
    --ps_scaler_path /path/to/ps_scaler.xgb
```

### Virtual Screening Evaluation:

```bash
python sc2_evaluation.py vs \
    --sc2_ps /path/to/sc2_ps.xgb \
    --sc2_pb /path/to/sc2_pb.xgb \
    --sc2_ps_feature_repo /path/to/normalized_features/sc2_ps \
    --sc2_pb_feature_repo /path/to/normalized_features/sc2_pb \
    --keyword active \
    --aggregate \
    --gpu \
    --output results/vs_results.csv
```

### Ranking Evaluation:

```bash
python sc2_evaluation.py ranking \
    --sc2_ps /path/to/sc2_ps.xgb \
    --sc2_pb /path/to/sc2_pb.xgb \
    --sc2_ps_feature_repo /path/to/normalized_features/sc2_ps \
    --sc2_pb_feature_repo /path/to/normalized_features/sc2_pb \
    --exp_repo /path/to/experimental_data \
    --gpu \
    --output results/ranking_results.csv
```