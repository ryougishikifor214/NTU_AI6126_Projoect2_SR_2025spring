#!/bin/bash
conda activate /home/featurize/work/AI6126project2/env

PROJECT_ROOT=$(dirname "$(realpath "$0")")
MODELS_CONFIGS_DIR=$(realpath "$PROJECT_ROOT/assets/configs")
CUR_MODEL_CFG_SUFFIX="swinir/swinir_20250420.py"

CUR_MODEL_CFG=$(realpath "$MODELS_CONFIGS_DIR/$CUR_MODEL_CFG_SUFFIX")
MMAGIC_TRAIN_TOOL=$(realpath "$PROJECT_ROOT/mmagic/tools/dist_train.sh")

bash $MMAGIC_TRAIN_TOOL $CUR_MODEL_CFG 4