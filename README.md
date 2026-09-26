# GHDiT
## Install
```bash
python -m pip install -r requirement.txt
```
## Quick Test
Run this example, or replace the text with your own Chinese sentence (5–8 characters recommended).
```bash
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python generate.py \
  --text '他做了顶好的靴子' \
  --output-dir ./outputs/test_01 \
  --seed 20260926
```
- `--text`: input Chinese sentence. Unsupported characters are rejected.
- `--output-dir`: output directory; must not already exist.
- `--seed`: random seed for generation.
- `CUDA_VISIBLE_DEVICES`: GPU index.

Trajectory length is predicted automatically. The output directory contains PNG images, NPZ trajectories, and JSON metadata.

# GHDiT pretrain model
[download GHDiT-weight, password:cvmt](https://pan.baidu.com/s/12xoSQyv_JsPVaNux7E_djQ)

# IAHCT-UCAS2025

[download GHDiT-delta2, password:cvmt](https://pan.baidu.com/s/14hWxryx36ssWRNsDeghfKg)


