# Quick Start Guide: Training with Auto-Stop

This is a condensed guide for experienced users. See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

---

## 1. Create IAM Role

**IAM Policy JSON:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["ec2:StopInstances", "ec2:DescribeInstances"],
      "Resource": "*"
    }
  ]
}
```

**Steps:**
1. IAM → Policies → Create → Name: `EC2SelfStopPolicy`
2. IAM → Roles → Create → Service: EC2 → Attach policy → Name: `EC2SelfStopRole`

---

## 2. Launch EC2 Instance

- **AMI**: Deep Learning AMI Neuron PyTorch (Ubuntu 20.04)
- **Instance**: `trn2.xlarge` (1 Trainium chip, 32GB)
- **IAM Role**: Attach `EC2SelfStopRole`
- **Security**: Allow SSH (port 22)
- **Storage**: 100GB gp3

---

## 3. Connect & Setup

```bash
# SSH to instance
ssh -i your-key.pem ubuntu@<INSTANCE_IP>

# Clone repo
git clone <YOUR_REPO>
cd <YOUR_REPO>

# Install dependencies
pip install -r requirements.txt

# Run setup
sudo ./scripts/setup.sh
source ~/.neuron_env

# Prepare config
cp configs/qwen3_0.6b_config.yaml configs/my_config.yaml
```

---

## 4. Start Training with Auto-Stop

### 1-minute test run:
```bash
python src/train.py --config configs/my_config.yaml --auto-stop 60
```

### Common durations:
```bash
# 5 minutes
python src/train.py --config configs/my_config.yaml --auto-stop 300

# 30 minutes
python src/train.py --config configs/my_config.yaml --auto-stop 1800

# 1 hour
python src/train.py --config configs/my_config.yaml --auto-stop 3600

# 2 hours
python src/train.py --config configs/my_config.yaml --auto-stop 7200
```

### Background execution:
```bash
nohup python src/train.py --config configs/my_config.yaml --auto-stop 3600 > training.log 2>&1 &
```

---

## 5. Monitor Training

```bash
# Watch logs
tail -f logs/training.log

# Monitor script
./scripts/monitor.sh

# Neuron devices
neuron-ls

# Device utilization
neuron-top
```

---

## 6. Download Results

```bash
# From local machine
scp -i your-key.pem -r ubuntu@<IP>:~/qwen3-training/output/ ./local_output/
```

---

## Command Line Arguments

```bash
python src/train.py \
  --config <path>              # Required: Path to config file
  --auto-stop <seconds>        # Optional: Auto-stop delay (e.g., 60 for 1 min)
  --auto-stop-region <region>  # Optional: AWS region (default: us-east-1)
  --output_dir <path>          # Optional: Override output directory
  --resume <checkpoint>        # Optional: Resume from checkpoint
```

---

## Auto-Stop Time Calculator

| Duration | Seconds | Command |
|----------|---------|---------|
| 1 minute | 60 | `--auto-stop 60` |
| 5 minutes | 300 | `--auto-stop 300` |
| 10 minutes | 600 | `--auto-stop 600` |
| 30 minutes | 1800 | `--auto-stop 1800` |
| 1 hour | 3600 | `--auto-stop 3600` |
| 2 hours | 7200 | `--auto-stop 7200` |
| 6 hours | 21600 | `--auto-stop 21600` |
| 12 hours | 43200 | `--auto-stop 43200` |
| 24 hours | 86400 | `--auto-stop 86400` |

---

## Troubleshooting

### Auto-stop not working?
```bash
# Check IAM role
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/

# Should return: EC2SelfStopRole
```

### Out of memory?
```yaml
# Reduce batch size in configs/my_config.yaml
training:
  batch_size: 8  # or lower
  max_seq_length: 1024  # or lower
```

### Slow training?
```bash
# Check Neuron compilation (first run is slower)
neuron-top  # Should show high utilization
```

---

## Cost Estimate

| Instance | Cost/Hour | 1 min | 1 hour | 6 hours |
|----------|-----------|-------|--------|---------|
| trn2.xlarge | $1.34 | $0.02 | $1.34 | $8.04 |
| trn2.8xlarge | $10.73 | $0.18 | $10.73 | $64.38 |

**Always use `--auto-stop` to prevent unnecessary charges!**

---

## Complete Example

```bash
# 1. SSH to instance
ssh -i trn2-key.pem ubuntu@54.123.45.67

# 2. Setup
cd ~/qwen3-training
pip install -r requirements.txt
source ~/.neuron_env

# 3. Configure
cp configs/qwen3_0.6b_config.yaml configs/my_config.yaml

# 4. Train with 1-hour auto-stop
nohup python src/train.py \
  --config configs/my_config.yaml \
  --auto-stop 3600 \
  > training.log 2>&1 &

# 5. Monitor
tail -f training.log

# Instance will automatically stop after 1 hour!
```
