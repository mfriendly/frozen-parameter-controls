#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
# PeMS04 | No-adaptive
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=gwnet rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 model.hparams.learned_adjacency=false seed=$SEED; done
# PeMS04 | Frozen emb.
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=gwnet rewiring=none freeze_node_emb=true epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 seed=$SEED; done
# PeMS04 | Trainable
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=gwnet rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 seed=$SEED; done
# PeMS07 | No-adaptive
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=gwnet rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 model.hparams.learned_adjacency=false seed=$SEED; done
# PeMS07 | Frozen emb.
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=gwnet rewiring=none freeze_node_emb=true epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 seed=$SEED; done
# PeMS07 | Trainable
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=gwnet rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 seed=$SEED; done
# PeMS08 | No-adaptive
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=gwnet rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 model.hparams.learned_adjacency=false seed=$SEED; done
# PeMS08 | Frozen emb.
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=gwnet rewiring=none freeze_node_emb=true epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 seed=$SEED; done
# PeMS08 | Trainable
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=gwnet rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 seed=$SEED; done
