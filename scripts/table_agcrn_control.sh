#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
# Graph WaveNet, this harness | PeMS04
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=gwnet rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 seed=$SEED; done
# Graph WaveNet, this harness | PeMS07
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=gwnet rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 seed=$SEED; done
# Graph WaveNet, this harness | PeMS08
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=gwnet rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 model.hparams.hidden_size=64 seed=$SEED; done
# AGCRN, this harness | PeMS04
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=agcrn rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# AGCRN, this harness | PeMS07
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=agcrn rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# AGCRN, this harness | PeMS08
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=agcrn rewiring=none epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# AGCRN, frozen embeddings | PeMS04
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems04 model=agcrn rewiring=none freeze_node_emb=true epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# AGCRN, frozen embeddings | PeMS07
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems07 model=agcrn rewiring=none freeze_node_emb=true epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
# AGCRN, frozen embeddings | PeMS08
for SEED in 42 123 456 789 1024; do "$PY" experiments/run_realworld_rewired.py dataset=pems08 model=agcrn rewiring=none freeze_node_emb=true epochs=100 patience=30 lr_scheduler.hparams.T_max=100 seed=$SEED; done
