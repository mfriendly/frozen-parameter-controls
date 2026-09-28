#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
# DiffConv | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# DiffConv | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# ChebConv | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_cheb rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# ChebConv | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_cheb rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# MPGRU | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# MPGRU | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mp_gru rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# DCRNN | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=dcrnn rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# DCRNN | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=dcrnn rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
