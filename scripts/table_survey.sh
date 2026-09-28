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
# Vanilla | METR-LA
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=la model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# Vanilla | PeMS-BAY
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=bay model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# Vanilla | AirQuality
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=airquality model=mptcn_diff rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=6 model.min_tcn_layers=6 seed=$SEED; done
# FoSR | PeMS04
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=fosr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# FoSR | PeMS07
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=fosr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# FoSR | PeMS08
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=fosr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# FoSR | METR-LA
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=la model=mptcn_diff rewiring=fosr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# FoSR | PeMS-BAY
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=bay model=mptcn_diff rewiring=fosr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# FoSR | AirQuality
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=airquality model=mptcn_diff rewiring=fosr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=6 model.min_tcn_layers=6 seed=$SEED; done
# BORF | PeMS04
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=borf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# BORF | PeMS07
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=borf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# BORF | PeMS08
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=borf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# BORF | METR-LA
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=la model=mptcn_diff rewiring=borf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# BORF | PeMS-BAY
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=bay model=mptcn_diff rewiring=borf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# BORF | AirQuality
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=airquality model=mptcn_diff rewiring=borf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=6 model.min_tcn_layers=6 seed=$SEED; done
# SDRF | PeMS04
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=sdrf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.num_iterations=50 seed=$SEED; done
# SDRF | PeMS07
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=sdrf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.num_iterations=50 seed=$SEED; done
# SDRF | PeMS08
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=sdrf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.num_iterations=50 seed=$SEED; done
# SDRF | METR-LA
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=la model=mptcn_diff rewiring=sdrf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# SDRF | PeMS-BAY
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=bay model=mptcn_diff rewiring=sdrf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# SDRF | AirQuality
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=airquality model=mptcn_diff rewiring=sdrf epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=6 model.min_tcn_layers=6 seed=$SEED; done
# GTR | PeMS04
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=gtr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# GTR | PeMS07
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=gtr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# GTR | PeMS08
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=gtr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# GTR | METR-LA
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=la model=mptcn_diff rewiring=gtr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# GTR | PeMS-BAY
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=bay model=mptcn_diff rewiring=gtr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 seed=$SEED; done
# GTR | AirQuality
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=airquality model=mptcn_diff rewiring=gtr epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=6 model.min_tcn_layers=6 seed=$SEED; done
# Bilevel | PeMS04
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# Bilevel | PeMS07
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# Bilevel | PeMS08
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# Bilevel | METR-LA
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=la model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# Bilevel | PeMS-BAY
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=bay model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=3 model.min_tcn_layers=3 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
# Bilevel | AirQuality
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=airquality model=mptcn_diff rewiring=bilevel epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.mode=tas model.hparams.hidden_size=64 model.hparams.n_layers=2 model.hparams.tcn_layers=6 model.min_tcn_layers=6 rewiring.policy_type=relative_reweight rewiring.outer_lr=0.001 rewiring.inner_steps=10 seed=$SEED; done
