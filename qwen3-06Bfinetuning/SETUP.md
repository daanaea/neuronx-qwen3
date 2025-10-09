# Qwen3 Fine-tuning on AWS Trainium (trn2.48xlarge)

## Instance Requirements
- **Instance Type**: `trn2.48xlarge`
- **Trainium Chips**: 16 Trainium chips (32 NeuronCores each = 512 total cores)
- **Region**: us-east-1 or us-west-2 (Trainium availability)

## Recommended AMI
Use the **AWS Deep Learning AMI Neuron PyTorch** (Ubuntu 22.04 or Amazon Linux 2):
- Search for "Deep Learning AMI Neuron" in EC2 AMI catalog
- Example AMI name: `Deep Learning AMI Neuron PyTorch 2.1 (Ubuntu 22.04)`
- The AMI comes pre-installed with:
  - AWS Neuron SDK
  - PyTorch with Neuron support
  - Required Neuron drivers and runtime

## Setup Steps

### 1. Launch Instance
```bash
# Use AWS CLI or Console to launch trn2.48xlarge with the Neuron AMI
# Ensure you have sufficient EBS storage (at least 500GB recommended)
```

**Note:** No special IAM permissions needed! The auto-shutdown feature uses OS-level shutdown (no AWS API calls).

### 2. SSH into Instance
```bash
ssh -i your-key.pem ubuntu@<instance-ip>
```

### 3. Activate Neuron Environment
```bash
# The AMI comes with pre-configured conda environments
source /opt/aws_neuron_venvs_pt/pytorch_venv/bin/activate
# OR if using conda:
# conda activate aws_neuron_venv_pytorch
```

### 4. Clone Your Repository
```bash
git clone <your-repo-url>
cd qwen3finetuning
```

### 5. Install Additional Dependencies
```bash
pip install -r requirements.txt
```

### 6. Verify Neuron Setup
```bash
# Check Neuron devices
neuron-ls

# Should show 16 Neuron devices
# Each device has 2 NeuronCores
```

### 7. Run Training
```bash
# Test run (auto-shuts down after 1 minute)
bash finetune_qwen3.sh

# Full training (disable auto-shutdown)
AUTO_STOP_DISABLED=1 bash finetune_qwen3.sh
```

**Auto-Shutdown Feature:**
- By default, the script will automatically **shutdown the instance** after **60 seconds (1 minute)**
- This is a safety feature to prevent accidental costs during testing
- Uses OS-level `shutdown` command - no IAM permissions required
- To disable for production runs, set `AUTO_STOP_DISABLED=1`
- You can adjust the timer in `finetune_qwen3.py` line 129
- Instance will show countdown in last 10 seconds

## Configuration Details

The current configuration in `finetune_qwen3.sh`:
- **Processes**: 32 (PROCESSES_PER_NODE=32)
  - This matches the 16 Trainium chips × 2 NeuronCores per chip
- **Tensor Parallelism**: 8 (TP_DEGREE=8)
- **Data Parallelism**: 4 (32 processes / 8 TP degree)
- **Batch Size**: 1 per device
- **Gradient Accumulation**: 8 steps
- **Effective Batch Size**: 32 (4 DP × 1 BS × 8 GA)

## Important Environment Variables (already set in script)
- `NEURON_CC_FLAGS`: Compiler flags for Neuron
- `NEURON_FUSE_SOFTMAX=1`: Optimize softmax operations
- `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3`: Async execution
- `MALLOC_ARENA_MAX=64`: Prevent host OOM issues

## Monitoring

### During Training
```bash
# Monitor Neuron utilization
neuron-top

# Monitor system resources
htop
```

### Logs
Training logs will be saved to the output directory specified in the script.

## Troubleshooting

### If you see "No Neuron devices found"
1. Ensure you're using a trn2 instance (not inf2)
2. Verify Neuron driver is loaded: `neuron-ls`
3. Restart the instance if needed

### For compilation errors
- First run might take longer due to graph compilation
- Compiled graphs are cached for subsequent runs
- Set `NEURON_EXTRACT_GRAPHS_ONLY=1` to do a quick compilation test

### Out of Memory
- Reduce `per_device_train_batch_size`
- Reduce `max_seq_length` in the Python script
- Increase `gradient_accumulation_steps`

## Cost Optimization
- Use Spot Instances if workload allows interruption
- trn2.48xlarge On-Demand: ~$21.50/hour (verify current pricing)
- Stop instance when not in use
- Consider checkpointing frequently for resumable training
