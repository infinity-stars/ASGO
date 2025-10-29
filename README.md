# ASGO: Adaptive Structured Gradient Optimization

Official implementation of the paper "ASGO: Adaptive Structured Gradient Optimization" by Kang An, Yuxing Liu, Rui Pan, Yi Ren, Shiqian Ma, Donald Goldfarb, and Tong Zhang.

Paper: [https://arxiv.org/abs/2503.20762](https://arxiv.org/abs/2503.20762)

## Overview

ASGO is an optimizer that combines adaptive preconditioning with structured gradient optimization for training deep neural networks. This repository provides implementations of ASGO and related optimizers for language model training.

## Repository Structure

```
ASGO/
├── optimizers/          # Core optimizer implementations
│   ├── asgo.py         # Adaptive Structured Gradient Optimization
│   ├── dasgo.py        # Diagonal Adaptive Structured Gradient Optimization
│   ├── muon.py         # Muon optimizer
│   └── shampoo.py      # Shampoo optimizer
├── LLMmodels/          # Language model training scripts
│   ├── config/         # Hydra configuration files
│   ├── utils/          # Training utilities
│   └── main.py         # Main training script
└── requirements.txt    # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

To run the training script:

```bash
cd LLMmodels
bash script.sh
```

The script will interactively prompt for:
- CUDA_VISIBLE_DEVICES (e.g., 0,1,2,3)
- Optimizer choice (e.g., asgo, muon, shampoo)
- Weights & Biases project name
- Dataset Path to download and load the Data

### Manual Training

For more control, you can run the training script directly:

```bash
python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    LLMmodels/main.py \
    dataset=openwebtext \
    model=gpt2 \
    optimizer=asgo \
    train.batch_size=32 \
    train.train_steps=2400
```

## Available Optimizers

- `asgo`: Adaptive Structured Gradient Optimization (main method)
- `dasgo`: Diagonal ASGO variant
- `muon`: Muon optimizer
- `shampoo`: Shampoo optimizer
- `adamw`: adamw optimizer

## Configuration

Training configurations are managed using Hydra. Configuration files are located in `LLMmodels/config/`:

- `model/`: Model architectures (gpt2, nanogpt)
- `optimizer/`: Optimizer hyperparameters
- `dataset/`: Dataset configurations
- `train_config.yaml`: Main training configuration

Override any parameter from the command line:

```bash
python LLMmodels/main.py optimizer.learning_rate=0.01 train.batch_size=64
```

## Citation

```bibtex
@inproceedings{an2025asgo,
  title={ASGO: Adaptive Structured Gradient Optimization},
  author={An, Kang and Liu, Yuxing and Pan, Rui and Ren, Yi and Ma, Shiqian and Goldfarb, Donald and Zhang, Tong},
  booktitle={NeurIPS},
  year={2025},
  url={https://arxiv.org/abs/2503.20762}
}
```

## License

MIT License
