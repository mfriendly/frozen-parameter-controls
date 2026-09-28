#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
# PeMS04, ChebConv | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_cheb rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS04, ChebConv | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_cheb rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS04, DiffConv | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=none epochs=100 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS04, DiffConv | Frozen-$\phi$ | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 rewiring.freeze_phi=true seed=$SEED; done
# PeMS04, DiffConv | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS04, MPGRU | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# PeMS04, MPGRU | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mp_gru rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS04, DCRNN | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=dcrnn rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# PeMS04, DCRNN | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=dcrnn rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS07, ChebConv | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_cheb rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS07, ChebConv | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_cheb rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS07, DiffConv | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS07, DiffConv | Frozen-$\phi$ | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 rewiring.freeze_phi=true seed=$SEED; done
# PeMS07, DiffConv | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS07, MPGRU | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# PeMS07, MPGRU | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mp_gru rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS07, DCRNN | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=dcrnn rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# PeMS07, DCRNN | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=dcrnn rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS08, ChebConv | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_cheb rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS08, ChebConv | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_cheb rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS08, DiffConv | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=none epochs=100 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# PeMS08, DiffConv | Frozen-$\phi$ | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 rewiring.freeze_phi=true seed=$SEED; done
# PeMS08, DiffConv | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS08, MPGRU | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mp_gru rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# PeMS08, MPGRU | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mp_gru rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# PeMS08, DCRNN | Vanilla | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=dcrnn rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# PeMS08, DCRNN | Bilevel | $h{=}3$
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=dcrnn rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
