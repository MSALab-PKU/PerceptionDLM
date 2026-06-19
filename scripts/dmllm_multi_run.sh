#!/usr/bin/env bash

set -x

FILE=$1
DATA_PATH=$2
TRAIN_CONFIG=$3
GPUS=$4

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$TRAIN_CONFIG" ]; then
    LOG_FILENAME="$(basename "${TRAIN_CONFIG%.*}")"
    LOG_FILE="logs/${LOG_FILENAME}_${TIMESTAMP}.log"
else
    LOG_FILE="logs/log_${TIMESTAMP}.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((28500 + $RANDOM % 2000))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


echo "Using torchrun mode."
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun --nnodes=${NNODES} \
--node_rank=${NODE_RANK} \
--nproc_per_node=${GPUS} \
--master_addr=${MASTER_ADDR} \
--master_port=${PORT} \
smallvlm/${FILE}.py --data_path ${DATA_PATH} --config ${TRAIN_CONFIG} --launcher pytorch "${@:5}"
