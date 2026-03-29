# TvTPPIMI

## Introduction
This repository contains the PyTorch implementation of TvTPPIMI, a boundary-aware deep learning framework for predicting protein–protein interaction (PPI) modulators using atom-level molecular representations, residue-level protein representations, and cross-modal interaction modeling.

## Framework
![TvTPPIMI Framework](figure/figure.tif)


## Acknowledgements
This implementation is inspired and partially based on earlier works [1].

## Environment

The code was developed and tested under the following environment:

- OS: Linux
- Python: 3.8.20
- CUDA: 11.3
- PyTorch: 1.12.1

We recommend using `conda` to manage the Python environment.

### Step 1: Create conda environment
```bash
conda create -n TvT python=3.8.20
conda activate TvT
```

### Step 2: Install PyTorch
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

### Step 3: Install basic dependencies
```bash
pip install timm==0.6.13 \
  yacs==0.1.8 \
  pandas==2.0.3 \
  scikit-learn==1.3.0 \
  prettytable==0.7.2 \
  ogb==1.3.5
```

### Step 4: Install PyTorch Geometric
For PyTorch 1.12.1 with CUDA 11.3, install the following wheels:
```bash
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.15%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl

pip install torch_geometric
```

### Step 5: Install RDKit
```bash
pip install rdkit
```


## Data
The `data` folder contains dataset folds:
- `MC`: Modulator cold-start
- `TC`: PPI-target cold-start
- `PC`: Paired Cold-Start


## Feature preparation
Protein ESM2 features:
```bash
python tools/extract_esm2_csv.py \
  --csv data/features/protein_sequences.csv \
  --out-dir data/features/protein_esm2
```

Compound GraphMVP features:
```bash
python tools/extract_graphmvp.py \
  --input-csv <CSV_WITH_ALL_SMILES> \
  --output-dir data/features/compound_graphmvp/pt \
  --mapping data/features/compound_graphmvp/index.csv \
  --weight weights/GraphMVP_C.model \
  --device cpu
```

## Reproduce results
Train with MC/TC/PC configs:
```bash
python main.py --model configs/model/MC.yaml --data configs/data/MC.yaml
python main.py --model configs/model/TC.yaml --data configs/data/TC.yaml
python main.py --model configs/model/PC.yaml --data configs/data/PC.yaml
```

Outputs are saved under `results/<SAVE>` as defined by `SOLVER.SAVE` in the data config.


## References
[1] Nie, Z.; Zhang, H.; Jiang, H.; Liu, Y.; Huang, X.; Xu, F.; Fu, J.; Ren, Z.; Tian, Y.; Zhang, W.-B.; Chen, J. OmniESI: A Unified Framework for Enzyme-Substrate Interaction Prediction with Progressive Conditional Deep Learning. arXiv June 22, 2025. https://doi.org/10.48550/arXiv.2506.17963.
