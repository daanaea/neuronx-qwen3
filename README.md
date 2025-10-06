# Qwen3-14B Training on AWS Trn2 with Neuron SDK

Team: Ananya, Sara

This repository provides a complete setup for training Qwen3-14B models on AWS Trainium (Trn2) instances using the AWS Neuron SDK.

## Overview

![Qwen3 Architecture](qwen3-architecture.png)

- **Model**: Qwen3-14B (14.8B parameters)
- **Hardware**: AWS Trn2 instances
- **Framework**: PyTorch with AWS Neuron SDK
- **Precision**: BF16
- **Context Length**: 32,768 tokens (extendable to 131,072)

## Quick Start

1. **Setup Environment**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Training**
   ```bash
   cp configs/qwen3_14b_config.yaml configs/my_config.yaml
   # Edit configs/my_config.yaml with your settings
   ```

4. **Start Training**
   ```bash
   python src/train.py --config configs/my_config.yaml
   ```

## Repository Structure

```
├── configs/                 # Training configurations
│   ├── qwen3_14b_config.yaml
│   └── neuron_config.yaml
├── scripts/                 # Setup and utility scripts
│   ├── setup.sh
│   ├── launch_training.sh
│   └── monitor.sh
├── src/                     # Source code
│   ├── train.py            # Main training script
│   ├── model.py            # Model definition
│   ├── data_loader.py      # Data loading utilities
│   └── utils.py            # Helper functions
├── data/                    # Training data
├── docs/                    # Documentation
└── requirements.txt         # Python dependencies
```

## Model Configuration

Based on the Qwen3-14B architecture:
- **Total Parameters**: 14.8B (13.2B non-embedding)
- **Layers**: 40
- **Attention Heads**: 40 (Query), 8 (Key/Value)
- **Hidden Size**: 5120
- **Vocab Size**: 152064

## AWS Trn2 Instance Types

Recommended instance types:
- `trn2.xlarge` (1 Trainium chip, 32GB memory) - For experimentation
- `trn2.8xlarge` (1 Trainium chip, 128GB memory) - For small-scale training
- `trn2.48xlarge` (6 Trainium chips, 768GB memory) - For distributed training

## Optimization
- Uses XLA compiler optimizations
- Implements gradient accumulation for larger effective batch sizes
- Utilizes mixed precision training with BF16
- Supports tensor parallelism and data parallelism
- **Gradient Checkpointing**: Memory optimization
- **Dynamic Loss Scaling**: For numerical stability
- **Monitoring**: Real-time metrics and logging

## Prerequisites

- AWS EC2 Trn2 instance (recommended: trn2.48xlarge for distributed training)
- Python 3.8+
- AWS Neuron SDK 2.x

## Installation
```bash
# Install AWS Neuron SDK
pip install torch-neuronx neuronx-cc

# Install project dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
