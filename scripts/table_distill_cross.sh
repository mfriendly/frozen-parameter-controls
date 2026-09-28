#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
# DiffConv$\to$MPGRU | Vanilla (target)
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# DiffConv$\to$MPGRU | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 ++adj_load=learned_graphs/pems04_bilevel_seed42.npy seed=$SEED; done
# MPGRU$\to$DiffConv | Vanilla (target)
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# MPGRU$\to$DiffConv | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 ++adj_load=learned_graphs/pems04_mpgru_bilevel_seed42.npy seed=$SEED; done
# DiffConv$\to$DCRNN (coupled) | Vanilla (target)
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=dcrnn rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# DiffConv$\to$DCRNN (coupled) | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=dcrnn rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 ++adj_load=learned_graphs/pems04_bilevel_seed42.npy seed=$SEED; done
