# SCORCH2

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) [![XGBoost](https://img.shields.io/badge/XGBoost-enabled-orange)](https://github.com/dmlc/xgboost)

# Introduction

SCORCH2 (SC2) is a consensus docking pose rescoring ML model for interaction-based virtual screening. SC2 is a continual work build upon SC1 (https://github.com/SMVDGroup/SCORCH/), where two XGBoost models are curated and trained separately to derive optimal knowledge pattern and then merge their result together for top-tier actives enrichment. 



# Dataset
PDBScreen data is derived from [Equiscore](https://github.com/CAODH/EquiScore?tab=readme-ov-file) and can be accessed from: [Zenodo](https://zenodo.org/records/8049380).

SCORCH1 data is from https://github.com/SMVDGroup/SCORCH/


## 📖 Citation

If you think our work helps, please consider citing the following papers:

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




