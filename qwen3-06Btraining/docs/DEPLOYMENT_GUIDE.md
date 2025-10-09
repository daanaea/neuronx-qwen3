# Complete Deployment Guide: Qwen3-0.6B on AWS Trn2

This guide walks you through the complete process of launching an AWS EC2 Trn2 instance, setting up the environment, and training the Qwen3-0.6B model with auto-stop functionality.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Create IAM Role for Auto-Stop](#step-1-create-iam-role-for-auto-stop)
3. [Step 2: Launch EC2 Trn2 Instance](#step-2-launch-ec2-trn2-instance)
4. [Step 3: Connect to Your Instance](#step-3-connect-to-your-instance)
5. [Step 4: Install Dependencies](#step-4-install-dependencies)
6. [Step 5: Clone and Setup the Repository](#step-5-clone-and-setup-the-repository)
7. [Step 6: Configure Training](#step-6-configure-training)
8. [Step 7: Run Training with Auto-Stop](#step-7-run-training-with-auto-stop)
9. [Step 8: Monitor Training](#step-8-monitor-training)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have:
- An AWS account with appropriate permissions
- AWS CLI installed (optional, but recommended)
- Basic familiarity with SSH and command line

---

## Step 1: Create IAM Role for Auto-Stop

The auto-stop functionality requires an IAM role that allows the EC2 instance to stop itself.

### 1.1 Create IAM Policy

1. Open the **AWS Console** and navigate to **IAM** → **Policies**
2. Click **Create policy**
3. Switch to the **JSON** tab
4. Paste the following policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ec2:StopInstances",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "ec2:DescribeInstances",
      "Resource": "*"
    }
  ]
}
```

5. Click **Next: Tags** (optional: add tags)
6. Click **Next: Review**
7. Name the policy: `EC2SelfStopPolicy`
8. Add description: `Allows EC2 instance to stop itself for auto-shutdown after training`
9. Click **Create policy**

### 1.2 Create IAM Role

1. In IAM Console, go to **Roles** → **Create role**
2. Select **Trusted entity type**: **AWS service**
3. Select **Use case**: **EC2**
4. Click **Next**
5. In the **Permissions policies** section:
   - Search for `EC2SelfStopPolicy` (the policy you just created)
   - Check the box next to it
6. Click **Next**
7. **Role name**: `EC2SelfStopRole`
8. **Description**: `Role for EC2 instances to stop themselves after training completion`
9. Review and click **Create role**

✅ **IAM Role Created!** You'll attach this to your EC2 instance in the next step.

---

## Step 2: Launch EC2 Trn2 Instance

### 2.1 Navigate to EC2 Console

1. Open **AWS Console** → **EC2**
2. Click **Launch Instance**

### 2.2 Configure Instance

**Name and Tags:**
- **Name**: `Qwen3-Training-Trn2` (or your preferred name)

**Application and OS Images (AMI):**
- Click **Browse more AMIs**
- Search for: `Deep Learning AMI Neuron PyTorch`
- Select: **Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04)**
  - Or the latest available Neuron AMI
- Click **Select**

**Instance Type:**
- Click **All instance types**
- Search for: `trn2`
- Select: `trn2.xlarge` (Recommended for Qwen3-0.6B)
  - **Specs**: 1 Trainium chip, 8 vCPUs, 32GB memory
  - **Alternative**: `trn2.8xlarge` for larger batch sizes

**Key Pair (login):**
- Select an existing key pair, OR
- Click **Create new key pair**:
  - **Name**: `trn2-training-key`
  - **Type**: RSA
  - **Format**: `.pem` (for Mac/Linux) or `.ppk` (for Windows/PuTTY)
  - Click **Create key pair**
  - **Save the downloaded key file securely!**

**Network Settings:**
- Keep default VPC settings
- **Auto-assign public IP**: Enable
- **Firewall (security groups)**:
  - Create security group or use existing
  - Ensure **SSH (port 22)** is allowed from your IP
  - Optional: Add **Custom TCP 6006** for TensorBoard access

**Configure Storage:**
- **Root volume**: 100 GB gp3 (minimum recommended)
- Add additional EBS volume if needed for datasets

**Advanced Details:**
- **IAM instance profile**: Select `EC2SelfStopRole` (the role you created in Step 1)
- **User data** (optional): Leave empty for now

### 2.3 Launch Instance

1. Review your configuration in the **Summary** panel
2. Click **Launch instance**
3. Wait for instance to reach **Running** state (2-3 minutes)

✅ **Instance Launched!** Note the **Public IPv4 address** from the instance details.

---

## Step 3: Connect to Your Instance

### 3.1 Set Key Permissions (Mac/Linux)

```bash
chmod 400 ~/Downloads/trn2-training-key.pem
```

### 3.2 SSH into Instance

```bash
ssh -i ~/Downloads/trn2-training-key.pem ubuntu@<YOUR_INSTANCE_PUBLIC_IP>
```

Replace `<YOUR_INSTANCE_PUBLIC_IP>` with the actual IP address from the EC2 console.

### 3.3 Verify Neuron Devices

Once connected, verify Trainium chips are detected:

```bash
neuron-ls
```

Expected output:
```
+--------+--------+--------+---------+
| DEVICE | TYPE   | STATUS | MEMORY  |
+--------+--------+--------+---------+
| 0      | NeuronCore | OK | 16GB    |
+--------+--------+--------+---------+
```

---

## Step 4: Install Dependencies

### 4.1 Update System Packages

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 4.2 Verify AWS Neuron SDK

The Deep Learning AMI comes with Neuron SDK pre-installed. Verify:

```bash
python3 -c "import torch_neuronx; print('Neuron PyTorch installed successfully')"
```

### 4.3 Install Git (if not already installed)

```bash
sudo apt-get install -y git
```

---

## Step 5: Clone and Setup the Repository

### 5.1 Clone Your Repository

```bash
cd ~
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
```

Or if you're uploading files directly:

```bash
mkdir ~/qwen3-training
cd ~/qwen3-training
# Upload your files via scp or sftp
```

### 5.2 Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch Neuron packages
- Transformers
- boto3 (for auto-stop)
- All other dependencies

### 5.3 Run Setup Script

```bash
chmod +x scripts/setup.sh
sudo ./scripts/setup.sh
```

This script will:
- Configure environment variables
- Create necessary directories
- Set up monitoring tools
- Create sample data

### 5.4 Source Environment

```bash
source ~/.neuron_env
```

Add to your `.bashrc` for future sessions:
```bash
echo "source ~/.neuron_env" >> ~/.bashrc
```

---

## Step 6: Configure Training

### 6.1 Copy and Edit Configuration

```bash
cp configs/qwen3_0.6b_config.yaml configs/my_config.yaml
```

### 6.2 Edit Configuration (Optional)

```bash
nano configs/my_config.yaml
```

Key settings to adjust:
- `batch_size`: Adjust based on available memory
- `learning_rate`: Tune for your dataset
- `num_epochs`: Set training duration
- `data.train_path`: Path to your training data

### 6.3 Prepare Your Training Data

Place your training data in the `data/` directory:

```bash
# Example: Upload training data
scp -i ~/Downloads/trn2-training-key.pem your_train_data.jsonl ubuntu@<IP>:~/qwen3-training/data/

# Or use the sample data created by setup.sh
mv data/sample_train.jsonl data/train.jsonl
mv data/sample_validation.jsonl data/validation.jsonl
```

Expected format (JSONL):
```json
{"text": "Your training text here..."}
{"text": "Another training example..."}
```

---

## Step 7: Run Training with Auto-Stop

### 7.1 Start Training with 1-Minute Auto-Stop

For testing (stops after 1 minute):

```bash
python src/train.py \
  --config configs/my_config.yaml \
  --auto-stop 60 \
  --auto-stop-region us-east-1
```

### 7.2 Common Auto-Stop Durations

**5 minutes:**
```bash
python src/train.py --config configs/my_config.yaml --auto-stop 300
```

**30 minutes:**
```bash
python src/train.py --config configs/my_config.yaml --auto-stop 1800
```

**1 hour:**
```bash
python src/train.py --config configs/my_config.yaml --auto-stop 3600
```

**2 hours:**
```bash
python src/train.py --config configs/my_config.yaml --auto-stop 7200
```

### 7.3 Run Without Auto-Stop

For full training without automatic shutdown:

```bash
python src/train.py --config configs/my_config.yaml
```

### 7.4 Run in Background with nohup

To keep training running even if SSH disconnects:

```bash
nohup python src/train.py \
  --config configs/my_config.yaml \
  --auto-stop 3600 \
  > training.log 2>&1 &
```

Check the process:
```bash
tail -f training.log
```

---

## Step 8: Monitor Training

### 8.1 Using the Monitor Script

```bash
./scripts/monitor.sh
```

Shows:
- Neuron device status
- Memory usage
- Training logs

### 8.2 Watch Training Logs

```bash
tail -f logs/training.log
```

### 8.3 TensorBoard (Optional)

If you enabled TensorBoard in config:

```bash
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
```

Access via: `http://<YOUR_INSTANCE_IP>:6006`

### 8.4 Monitor Neuron Device Utilization

```bash
neuron-top
```

### 8.5 Check Auto-Stop Status

The auto-stop timer runs in the background. You'll see log messages like:

```
[Auto-Stop] Timer started. Instance will stop in 60 seconds (1.0 minutes)...
```

---

## Step 9: After Training Completes

### 9.1 Check Outputs

```bash
ls -lh output/
```

Your checkpoints and models will be saved here.

### 9.2 Download Trained Model

From your local machine:

```bash
scp -i ~/Downloads/trn2-training-key.pem -r \
  ubuntu@<IP>:~/qwen3-training/output/ \
  ./local_output/
```

### 9.3 Instance Auto-Stop

The instance will automatically stop after the specified time. Check in AWS Console:
- **EC2** → **Instances** → Your instance should be in **Stopped** state

### 9.4 Restart Instance for More Training

1. In EC2 Console, select your instance
2. **Instance state** → **Start instance**
3. SSH back in and resume training:

```bash
python src/train.py \
  --config configs/my_config.yaml \
  --resume output/checkpoint-epoch-X
```

---

## Troubleshooting

### Issue: "Permission denied" when SSH connecting

**Solution:**
```bash
chmod 400 ~/Downloads/trn2-training-key.pem
```

### Issue: Auto-stop doesn't work

**Verify IAM role is attached:**
```bash
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
```

Should return: `EC2SelfStopRole`

**Check logs:**
```bash
grep "Auto-Stop" logs/training.log
```

### Issue: "No Neuron devices found"

**Check instance type:**
```bash
curl http://169.254.169.254/latest/meta-data/instance-type
```

Should return: `trn2.xlarge` or similar

**Verify Neuron driver:**
```bash
neuron-ls
```

### Issue: Out of memory during training

**Solution 1: Reduce batch size**
```yaml
# In configs/my_config.yaml
training:
  batch_size: 8  # Reduce from 16
```

**Solution 2: Reduce sequence length**
```yaml
training:
  max_seq_length: 1024  # Reduce from 2048
```

**Solution 3: Use larger instance**
- Upgrade to `trn2.8xlarge` for more memory

### Issue: Training is very slow

**Check Neuron compilation:**
- First run will be slower due to XLA compilation
- Subsequent runs use cached compiled graphs

**Verify Neuron utilization:**
```bash
neuron-top
```

Should show high utilization on NeuronCores.

### Issue: Auto-stop fails with API error

**Fallback to local shutdown:**
The script automatically falls back to `sudo shutdown -h now` if AWS API fails.

**Check IAM permissions:**
Go to **IAM** → **Roles** → **EC2SelfStopRole** → Verify `ec2:StopInstances` permission is attached.

---

## Cost Management

### Pricing Estimate (as of 2024)

- **trn2.xlarge**: ~$1.34/hour
- **trn2.8xlarge**: ~$10.73/hour

**Example costs with auto-stop:**
- 1 minute test: ~$0.02
- 1 hour training: ~$1.34
- Full day: ~$32.16

### Best Practices

1. **Always use auto-stop** for cost control
2. **Stop instances** when not in use
3. **Use Spot Instances** for 60-70% savings (advanced)
4. **Monitor costs** in AWS Cost Explorer

---

## Quick Reference Commands

```bash
# Launch training with 1-minute auto-stop
python src/train.py --config configs/my_config.yaml --auto-stop 60

# Monitor training
tail -f logs/training.log

# Check Neuron devices
neuron-ls

# Monitor device utilization
neuron-top

# Check instance will auto-stop
grep "Auto-Stop" logs/training.log

# Download trained model
scp -r ubuntu@<IP>:~/qwen3-training/output/ ./local_output/

# Stop instance manually
sudo shutdown -h now
```

---

## Support and Resources

- **AWS Neuron Documentation**: https://awsdocs-neuron.readthedocs-hosted.com/
- **Qwen3 Model**: https://huggingface.co/Qwen/Qwen3-0.6B
- **AWS EC2 Documentation**: https://docs.aws.amazon.com/ec2/

---

## Summary

You've successfully:
✅ Created an IAM role for auto-stop
✅ Launched a Trn2 instance
✅ Set up the training environment
✅ Configured and started training
✅ Enabled automatic instance shutdown

Your instance will automatically stop after the specified time, preventing unnecessary charges!
