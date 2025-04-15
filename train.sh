#!/bin/bash
conda activate /home/featurize/work/6126p2

PROJECT_ROOT=$(dirname "$(realpath "$0")")
MODELS_CONFIGS_DIR=$(realpath "$PROJECT_ROOT/assets/configs")
CUR_MODEL_CFG_SUFFIX="edsr/edsr_20250415_cfg.py"

CUR_MODEL_CFG=$(realpath "$MODELS_CONFIGS_DIR/$CUR_MODEL_CFG_SUFFIX")
MMAGIC_TRAIN_TOOL=$(realpath "$PROJECT_ROOT/mmagic/tools/train.py")

python $MMAGIC_TRAIN_TOOL $CUR_MODEL_CFG