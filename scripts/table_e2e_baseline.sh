#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
# Vanilla | PeMS04
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# Vanilla | PeMS07
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# Vanilla | PeMS08
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# E2E reweight | PeMS04
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=e2e_reweight epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# E2E reweight | PeMS07
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=e2e_reweight epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# E2E reweight | PeMS08
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=e2e_reweight epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# Bilevel | PeMS04
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# Bilevel | PeMS07
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# Bilevel | PeMS08
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
