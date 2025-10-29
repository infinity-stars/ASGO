#!/bin/bash

# Interactive input for configuration
echo "=== Training Configuration ==="
echo "Please enter the following parameters:"

# Get CUDA_VISIBLE_DEVICES
echo -n "Enter CUDA_VISIBLE_DEVICES (e.g., 0,1,2,3): "
read cuda_devices

# Get optimizer
echo -n "Enter optimizer (e.g., asgo, dasgo, adam, muon, Shampoo): "
read optimizer_type

echo -n 'Enter wandb project name: '
read wandb_project

echo -n 'Enter dataset path: '
read dataset_path


# Validate inputs
if [ -z "$cuda_devices" ]; then
    echo "Error: CUDA_VISIBLE_DEVICES cannot be empty"
    exit 1
fi

if [ -z "$optimizer_type" ]; then
    echo "Error: Optimizer cannot be empty"
    exit 1
fi

if [ -z "$wandb_project" ]; then
    echo "Error: Wandb project name cannot be empty"
    exit 1
fi

if [ -z "$dataset_path" ]; then
    echo "Error: Dataset path cannot be empty"
    exit 1
fi

# Count number of GPUs
gpu_count=$(echo $cuda_devices | tr ',' '\n' | wc -l)

# Set environment variables
export CUDA_VISIBLE_DEVICES=$cuda_devices
export OMP_NUM_THREADS=6

# Print environment information
echo ""
echo "=== Training Environment Information ==="
echo "Start time: $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $gpu_count"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Optimizer: $optimizer_type"

# Print training job description
echo ""
echo "=== Training Job Description ==="
echo "This job trains a GPT-2 model on OpenWebText dataset"
echo "Using optimizer: $optimizer_type"
echo "Using $gpu_count GPU(s): $cuda_devices"

# Run training command
echo ""
echo "=== Starting Training ==="
python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=$gpu_count \
    main.py \
    dataset=openwebtext \
    model=gpt2 \
    train.train_steps=2400 \
    train.wandb_entity='AdaShampoo' \
    train.wandb_project='Debug' \
    train.wandb_project=$wandb_project \
    train.dataset_path=$dataset_path \
    train.batch_size=32 \
    optimizer=$optimizer_type \
    # train.DDP=False \
    # optimizer.learning_rate=0.01 \
    # optimizer.update_freq=1 \
    # optimizer.StateWarmup_steps=0 \

echo ""
echo "=== Training Completed ==="
echo "End time: $(date)" 