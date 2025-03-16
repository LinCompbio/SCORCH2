# SCORCH2

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) [![XGBoost](https://img.shields.io/badge/XGBoost-enabled-orange)](https://github.com/dmlc/xgboost)

## Introduction

SCORCH2 (SC2) is a consensus docking pose rescoring ML model for interaction-based virtual screening. SC2 is a continual work build upon SC1 (https://github.com/SMVDGroup/SCORCH/), where two XGBoost models are curated and trained separately to derive optimal knowledge pattern and then merge their result together for top-tier actives enrichment. 

<div style="text-align: center;">
  <img src="main.jpg" style="width: 90%;" />
</div>

## Dataset
PDBScreen data is derived from [Equiscore](https://github.com/CAODH/EquiScore?tab=readme-ov-file) and can be accessed from: [Zenodo](https://zenodo.org/records/8049380).

SCORCH1 data is from https://github.com/SMVDGroup/SCORCH/

PDBBind data is from https://www.pdbbind-plus.org.cn

## Data prep

SC2 accepts the same data curation setting in SC1, including `.pdbqt` format input for receptor and ligand, SC2 officially takes [ADFRsuite](https://ccsb.scripps.edu/adfr/downloads/) for format conversion.

Just so you know, SC2 is a rescoring method; the input is hypothesized to be docked poses, and SC2 does not generate the docking poses itself.

🧬 Ligand Batch Conversion to .pdbqt Format
<pre>
python receptor_2_pdbqt.py \
  --input_dir /path/to/input_ligands \
  --output_dir /path/to/output_pdbqt \
  --input_format mol2 \
  --n_proc XX
</pre>

🧬 Receptor Batch Conversion to .pdbqt Format
<pre>
python ligand_2_pdbqt.py \
  --input_dir /path/to/input_ligands \
  --output_dir /path/to/output_pdbqt \
  --input_format pdb \
  --n_proc XX
</pre>

PS: Adding charges by Openbabel may cause result deviation.

## Feature extraction

SC2 feature is majorly combined with three parts, BINANA, ECIF, and Rdkit, to extract features for rescoring, run the script below:
<pre>
python feature_extraction_ligand_entropy.py --des_path /path/to/results --protein_base_path /path/to/proteins --molecule_path /path/to/molecules
</pre>

PS: The speed for running will depend on the size of the receptor, crop redundant parts will significantly increase the efficiency.

## Normalization

SC2 combines two XGBoost models with separate normalizer, with the feature generated, you will need to scale them into distinct feature spaces.

## Retrain the model 

Though we recommend taking official weights since SC2 has already been explored for optimal generalizability and top enrichment in current scope, but you could run the code below to retrain the model on your own data.

```bash
python train_xgboost.py --train_features <path_to_train_features> \
                        --train_labels <path_to_train_labels> \
                        --test_features <path_to_test_features> \
                        --test_labels <path_to_test_labels> \
                        --output_dir <path_to_output_directory> \
                        [--n_trials <number_of_trials>]
```


## Running

### Reproduce the result on common benchmarks

Download the prepared features (DUD-E, DEKOIS, VSDS-VD, MERCK FEP benchmark), and model weight for two XGBoost models from zenodo: XXX. Unzip the files to the root repository. Then run one of the scripts below:

#### DEKOIS 2.0
<pre>
python sc2_evaluation.py --mode vs \
    --sc2_ps /path/to/models/sc2_ps.pkl \
    --sc2_pb /path/to/models/sc2_pb.pkl \
    --sc2_ps_feature_repo /path/to/features/sc2_ps \
    --sc2_pb_feature_repo /path/to/features/sc2_pb \
    --keyword active \
    --aggregate \
    --gpu \
    --output output/vs_results.csv
</pre>

Features from different docking methods are integrated inside and simply change the child path to get SC2 result on other DEKOIS2 docking poses, for example:
<pre>
--sc2_ps_feature_repo evaluation_feature/dekois/sc2_ps/sc2_ps_flare --sc2_pb_feature_repo evaluation_feature/dekois/sc2_pb/sc2_pb_flare
</pre>

#### DUD-E 
<pre>
python3 sc2_evaluation.py --sc2_ps sc2_ps.pkl --sc2_pb sc2_pb.pkl  --sc2_ps_feature_repo evaluation_feature/dude/sc2_ps_equiscore_dude --sc2_pb_feature_repo evaluation_feature/dude/sc2_pb_equiscore_dude 
--keyword active
</pre>
without --aggregate since only one pose for each molecule is available

#### VSDS-vd
<pre>
python3 sc2_evaluation.py --sc2_ps sc2_ps.pkl --sc2_pb sc2_pb.pkl  --sc2_ps_feature_repo evaluation_feature/vsds/sc2_ps_flare_vsds --sc2_pb_feature_repo evaluation_feature/vsds/sc2_pb_flare_vsds 
--aggregate --keyword inactive
</pre>


#### MERCK FEP benchmark
<pre> 
python sc2_evaluation.py --mode ranking \
    --sc2_ps /path/to/models/sc2_ps.pkl \
    --sc2_pb /path/to/models/sc2_pb.pkl \
    --sc2_ps_feature_repo /path/to/features/sc2_ps_normalized \
    --sc2_pb_feature_repo /path/to/features/sc2_pb_normalized \
    --exp_repo /path/to/experimental_data \
    --gpu \
    --save_predictions \
    --output output/ranking_results.csv
</pre>

### Running on private data

You may change the code on `sc2_evaluation.py` and create customized results, and in the future SC2 will be available on the Lab server as a common service to the community.

### Issue report

This project is under development and please feel free to open an issue or get in touch with: lin.chen@ed.ac.uk ✉️.
 

## 📖 Citation

If you think our work helps, please consider citing the following papers 👇 :

```bibtex
@article{mcgibbon2023scorch,
  title={SCORCH: Improving structure-based virtual screening with machine learning classifiers, data augmentation, and uncertainty estimation},
  author={McGibbon, Miles and Money-Kyrle, Sam and Blay, Vincent and Houston, Douglas R},
  journal={Journal of Advanced Research},
  volume={46},
  pages={135--147},
  year={2023},
  publisher={Elsevier}
}

@article{lin2025scorch2,
  title={SCORCH2: advancing generalized interaction-based virtual screening from model consensus with different knowledge pattern},
  author={McGibbon, Miles and Money-Kyrle, Sam and Blay, Vincent and Houston, Douglas R},
  journal={Journal of Advanced Research},
  volume={46},
  pages={135--147},
  year={2023},
  publisher={Elsevier}
}




