#!/bin/bash
# launch_4gpu_grpo.sh
# 4-GPU GRPO训练启动脚本
# 在启动脚本中添加
export WANDB_MODE=offline
export WANDB_OFFLINE=true
export WANDB_PROJECT="reasoner_grpo_4gpu"
export WANDB_RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
set -e

echo "=== 4-GPU GRPO Training Launch Script ==="
echo "Starting training at $(date)"

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Detected ${GPU_COUNT} GPUs"

if [ $GPU_COUNT -lt 4 ]; then
    echo "Warning: Less than 4 GPUs detected. Training may not use all intended resources."
fi

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_NTHREADS=4

# 训练参数设置
SCRIPT_NAME="multi_gpu_grpo_reasoner.py"
CONFIG_FILE="accelerate_4gpu_config.yaml"

# 可调整的训练参数
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
NUM_GENERATIONS=${NUM_GENERATIONS:-16}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-256}
LEARNING_RATE=${LEARNING_RATE:-5e-7}
NUM_EPOCHS=${NUM_EPOCHS:-2}

echo "Training Configuration:"
echo "  Per Device Batch Size: ${PER_DEVICE_BATCH_SIZE}"
echo "  Gradient Accumulation Steps: ${GRAD_ACCUM_STEPS}"
echo "  Effective Batch Size: $((PER_DEVICE_BATCH_SIZE * 4 * GRAD_ACCUM_STEPS))"
echo "  Num Generations: ${NUM_GENERATIONS}"
echo "  Max Completion Length: ${MAX_COMPLETION_LENGTH}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Num Epochs: ${NUM_EPOCHS}"

# 检查必需文件
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Training script $SCRIPT_NAME not found!"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    echo "Creating default config file..."
    cat > $CONFIG_FILE << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    echo "Default config file created."
fi

# 启动训练
echo "Launching 4-GPU GRPO training..."
echo "Command: accelerate launch --config_file $CONFIG_FILE $SCRIPT_NAME"

accelerate launch \
    --config_file $CONFIG_FILE \
    $SCRIPT_NAME \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_generations $NUM_GENERATIONS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps 50 \
    --logging_steps 5

echo "Training completed at $(date)"
echo "Check output directory: ./checkpoints/Music_data/Reasoner_4GPU_GRPO"