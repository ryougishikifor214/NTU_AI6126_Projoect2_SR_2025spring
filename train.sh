#!/bin/bash
conda activate /home/featurize/work/AI6126project2/env

PROJECT_ROOT=$(dirname "$(realpath "$0")")
MODELS_CONFIGS_DIR=$(realpath "$PROJECT_ROOT/assets/configs")
CUR_MODEL_CFG_SUFFIX="edsr/edsr_20250424.py"
# CUR_MODEL_CFG_SUFFIX="srgan_resnet/srgan_20250424.py"

CUR_MODEL_CFG=$(realpath "$MODELS_CONFIGS_DIR/$CUR_MODEL_CFG_SUFFIX")
MMAGIC_TRAIN_TOOL=$(realpath "$PROJECT_ROOT/mmagic/tools/train.py")

CUDA_VISIBLE_DEVICES=0 python $MMAGIC_TRAIN_TOOL $CUR_MODEL_CFG
