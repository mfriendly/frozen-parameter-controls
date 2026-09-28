#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
# PeMS04 | DiffConv | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS04 | DiffConv | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 ++adj_load=learned_graphs/pems04_bilevel_seed42.npy seed=$SEED; done
# PeMS04 | DiffConv | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS04 | ChebConv | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_cheb rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS04 | ChebConv | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_cheb rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 ++adj_load=learned_graphs/pems04_chebconv_bilevel_seed42.npy seed=$SEED; done
# PeMS04 | ChebConv | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_cheb rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS04 | MPGRU | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# PeMS04 | MPGRU | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 ++adj_load=learned_graphs/pems04_mpgru_bilevel_seed42.npy seed=$SEED; done
# PeMS04 | MPGRU | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mp_gru rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS07 | DiffConv | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS07 | DiffConv | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 ++adj_load=learned_graphs/pems07_bilevel_seed42.npy seed=$SEED; done
# PeMS07 | DiffConv | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS07 | ChebConv | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_cheb rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS07 | ChebConv | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_cheb rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 ++adj_load=learned_graphs/pems07_chebconv_bilevel_seed42.npy seed=$SEED; done
# PeMS07 | ChebConv | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_cheb rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS07 | MPGRU | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# PeMS07 | MPGRU | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 ++adj_load=learned_graphs/pems07_mpgru_bilevel_seed42.npy seed=$SEED; done
# PeMS07 | MPGRU | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mp_gru rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS08 | DiffConv | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS08 | DiffConv | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 ++adj_load=learned_graphs/pems08_bilevel_seed42.npy seed=$SEED; done
# PeMS08 | DiffConv | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS08 | ChebConv | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_cheb rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS08 | ChebConv | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_cheb rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 ++adj_load=learned_graphs/pems08_chebconv_bilevel_seed42.npy seed=$SEED; done
# PeMS08 | ChebConv | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_cheb rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS08 | MPGRU | Vanilla
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# PeMS08 | MPGRU | Distilled
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 ++adj_load=learned_graphs/pems08_mpgru_bilevel_seed42.npy seed=$SEED; done
# PeMS08 | MPGRU | Bilevel
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mp_gru rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
