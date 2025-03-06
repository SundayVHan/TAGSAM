# TAGSAM
## Preparation
1. Install pytorch manually
```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```
2. Install requirements
```shell
pip install -r requirements.txt
```

## Pre-process
You first need to train a **teacher/expert model** on the original TAG. This process is generally referred to as the **buffer**.
```shell
python buffer.py --dataset_name computer
```

## Condensation
Then you can **condense/distill** TAG into a **smaller** one.
```shell
python distill.py --dataset_name computer --syn_size 200
```