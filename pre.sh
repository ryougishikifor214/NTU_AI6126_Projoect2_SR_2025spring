#!/bin/bash

DATA_DST_DIR="/home/featurize/data/ffhq"

PROJECT_ROOT=$(dirname "$(realpath "$0")")
DATA_SRC_DIR=$(realpath "$PROJECT_ROOT/data")

if [ ! -d "$DATA_DST_DIR" ]; then
    echo "directory created: $DATA_DST_DIR"
    mkdir -p "$DATA_DST_DIR"
    cp -r "$DATA_SRC_DIR"/* "$DATA_DST_DIR"
else
    echo "directory existed already: $DATA_DST_DIR"
fi