# How Much Does Learning the Graph Help? Frozen-Parameter Controls for Spatio-Temporal Traffic Forecasting

Minkyoung Kim, Gun Il Kim, Hyunjung Byun, and Beakcheol Jang, Graduate School of Information, Yonsei University

Code and configuration files for the manuscript above, which is under review. A preliminary version is available as a preprint: [arXiv:2605.07577](https://arxiv.org/abs/2605.07577). Tag `v1.0` marks the code as submitted.

This code builds on the codebase of "Over-squashing in Spatiotemporal Graph Neural Networks" (https://arxiv.org/abs/2506.15507).

## Environment

Python 3.11.14, PyTorch 2.7.0 built for CUDA 12.8. Install the pinned packages with

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

`requirements.txt` lists the packages this code imports:

```
cycler==0.12.1
einops==0.8.2
lightning==2.6.1
matplotlib==3.10.8
networkx==3.6.1
numba==0.63.1
numpy==2.2.6
omegaconf==2.3.0
pandas==2.3.3
POT==0.9.6.post1
pytorch-lightning==2.6.1
PyYAML==6.0.3
scipy==1.13.1
torch==2.7.0+cu128
torch-geometric==2.7.0
torch_sparse==0.6.18+pt27cu128
torch_spatiotemporal==0.9.5
```

## Datasets

All datasets are loaded through the Torch Spatiotemporal loaders (`torch_spatiotemporal` 0.9.5), which download them on first use into the library's storage directory:

| dataset | `dataset=` | loader |
|---|---|---|
| PeMS04 | `pems04` | `tsl.datasets.PeMS04()` |
| PeMS07 | `pems07` | `tsl.datasets.PeMS07()` |
| PeMS08 | `pems08` | `tsl.datasets.PeMS08()` |
| METR-LA | `la` | `tsl.datasets.MetrLA(impute_zeros=True)` |
| PeMS-BAY | `bay` | `tsl.datasets.PemsBay()` |
| AirQuality | `airquality` | `tsl.datasets.AirQuality()` |

## Seeds

Every mean ± standard deviation entry uses the five seeds 42, 123, 456, 789, 1024. The learning-rate sweep of the fresh-batch control uses the first three seeds of this set: 42, 123, 456.

## Reproducing the tables

Each script runs every configuration of one table for all five seeds through `experiments/run_realworld_rewired.py`, from the package root. Set `PYTHON` to the interpreter of the environment above (default `python`).

| table | command |
|---|---|
| Survey | `bash scripts/table_survey.sh` |
| Decomposition | `bash scripts/table_decomposition.sh` |
| Factorial | `bash scripts/table_factorial.sh` |
| Compute controls | `bash scripts/table_compute_controls.sh` |
| T-sweep | `bash scripts/table_tsweep.sh` |
| Graph-identity ladder | `bash scripts/table_graph_identity_ladder.sh` |
| Graph WaveNet control | `bash scripts/table_gwnet_control.sh` |
| AGCRN control | `bash scripts/table_agcrn_control.sh` |
| Backbone scope | `bash scripts/table_backbone_scope.sh` |
| Same-backbone distillation | `bash scripts/table_distill_same.sh` |
| Cross-backbone transfer | `bash scripts/table_distill_cross.sh` |
| End-to-end baseline | `bash scripts/table_e2e_baseline.sh` |
| Per-horizon | `bash scripts/table_per_horizon.sh` |

The factorial, same-backbone distillation and cross-backbone transfer scripts load learned graphs from `learned_graphs/*.npy`. These files are produced from bilevel checkpoints with `scripts/extract_learned_graph.py --ckpt <checkpoint> --out learned_graphs/<name>.npy`.

The AirQuality bandwidth table needs no training: `python scripts/aq_anatomy.py --outdir <dir>` writes it to `<dir>/bandwidth_ablation.csv`.

## Approximate wall-clock time per table on one GPU

Sum of the logged durations of all runs in the table.

| table | hours |
|---|---|
| Survey | 442 |
| Decomposition | 114 |
| Factorial | 481 |
| Compute controls | 106 |
| T-sweep | 483 |
| Graph-identity ladder | 16 |
| Graph WaveNet control | 116 |
| AGCRN control | 72 |
| Backbone scope | 269 |
| Same-backbone distillation | 529 |
| Cross-backbone transfer | 37 |
| End-to-end baseline | 287 |
| Per-horizon | 1108 |

## Citation

If you use this code, please cite the preprint:

```bibtex
@article{kim2026bilevel,
  title   = {Bilevel Graph Structure Learning, Revisited: Inner-Channel Origins of the Reported Gain},
  author  = {Kim, Minkyoung and Jang, Beakcheol},
  journal = {arXiv preprint arXiv:2605.07577},
  year    = {2026}
}
```

## License

A license will be added upon publication of the manuscript. Components adapted from the upstream codebase remain under their original licenses.
