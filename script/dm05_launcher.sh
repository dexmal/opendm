#!/usr/bin/env bash
# Local / generic launcher for DM05 SFT (train & inference).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP="$(cd "$SCRIPT_DIR/.." && pwd)/playground/dm05_libero.py"
NPROC_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
EXP_ARGS=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp)
            EXP="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --)
            shift
            EXP_ARGS+=("$@")
            break
            ;;
        *)
            EXP_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Running: $EXP"
echo "Args: ${EXP_ARGS[*]}"

if [[ " ${EXP_ARGS[*]} " == *" --task inference "* ]]; then
    python "$EXP" "${EXP_ARGS[@]}"
    exit $?
fi

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=24
export NCCL_SOCKET_TIMEOUT=7200
export OMP_NUM_THREADS=4

echo "torchrun: nproc_per_node=$NPROC_PER_NODE nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT"

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$EXP" \
    "${EXP_ARGS[@]}"
