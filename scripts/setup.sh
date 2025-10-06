#!/bin/bash

# Qwen3-14B Training Setup Script for AWS Trn2 Instances
# This script sets up the environment for training Qwen3-14B on AWS Trainium

set -e  # Exit on any error

echo "=========================================="
echo "Qwen3-14B AWS Trn2 Training Setup"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if running on supported instance
check_instance() {
    log "Checking instance type..."
    
    if command -v aws &> /dev/null; then
        INSTANCE_TYPE=$(aws ec2 describe-instances \
            --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id) \
            --query 'Reservations[0].Instances[0].InstanceType' \
            --output text 2>/dev/null || echo "unknown")
    else
        INSTANCE_TYPE="unknown"
    fi
    
    if [[ $INSTANCE_TYPE == trn2* ]]; then
        log "Detected AWS Trn2 instance: $INSTANCE_TYPE"
    else
        warn "Instance type: $INSTANCE_TYPE (Trn2 recommended for optimal performance)"
    fi
}

# Check and install Python
setup_python() {
    log "Setting up Python environment..."
    
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log "Python version: $PYTHON_VERSION"
    
    if ! command -v pip3 &> /dev/null; then
        log "Installing pip..."
        curl https://bootstrap.pypa.io/get-pip.py | python3
    fi
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
}

# Install AWS Neuron SDK
install_neuron_sdk() {
    log "Installing AWS Neuron SDK..."
    
    # Add Neuron repository
    . /etc/os-release
    if [[ $ID == "ubuntu" ]]; then
        # Ubuntu setup
        curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-NEURON.PUB | apt-key add -
        echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" | tee /etc/apt/sources.list.d/neuron.list
        apt-get update -y
        
        # Install Neuron runtime and tools
        apt-get install aws-neuronx-dkms aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools -y
        
    elif [[ $ID == "amzn" ]]; then
        # Amazon Linux setup
        tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
metadata_expire=0
EOF
        rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-NEURON.PUB
        yum install aws-neuronx-dkms aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools -y
    else
        warn "Unsupported OS for automated Neuron SDK installation: $ID"
        log "Please install Neuron SDK manually: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu20.html"
    fi
    
    # Install Python Neuron packages
    log "Installing Python Neuron packages..."
    python3 -m pip install torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com
}

# Install PyTorch and other dependencies
install_dependencies() {
    log "Installing PyTorch and dependencies..."
    
    # Install PyTorch (CPU version for compilation, Neuron for execution)
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install training dependencies
    python3 -m pip install \
        transformers \
        datasets \
        accelerate \
        tensorboard \
        wandb \
        pyyaml \
        psutil \
        numpy \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        tqdm \
        jupyter \
        ipywidgets
}

# Setup environment variables
setup_environment() {
    log "Setting up environment variables..."
    
    # Create environment setup file
    cat > ~/.neuron_env << 'EOF'
# AWS Neuron Environment Variables
export PATH=/opt/aws/neuron/bin:$PATH
export NEURON_CC_FLAGS="--model-type=transformer --auto-cast=all"
export NEURON_FRAMEWORK_DEBUG=0
export XLA_USE_BF16=1
export MALLOC_ARENA_MAX=64
export OMP_NUM_THREADS=1

# Python path
export PYTHONPATH=$PYTHONPATH:/opt/aws/neuron/lib/python3.8/site-packages

# Neuron monitoring
export NEURON_MONITOR_CW_REGION=us-west-2
EOF
    
    # Add to bashrc if not already present
    if ! grep -q "source ~/.neuron_env" ~/.bashrc; then
        echo "source ~/.neuron_env" >> ~/.bashrc
    fi
    
    # Source for current session
    source ~/.neuron_env
    
    log "Environment variables configured"
}

# Create directories
create_directories() {
    log "Creating project directories..."
    
    mkdir -p data
    mkdir -p logs
    mkdir -p output
    mkdir -p neuron_cache
    mkdir -p xla_cache
    mkdir -p checkpoints
    
    # Set permissions
    chmod -R 755 data logs output neuron_cache xla_cache checkpoints
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring tools..."
    
    # Install neuron-monitor if available
    if command -v neuron-monitor &> /dev/null; then
        log "Neuron Monitor available"
    else
        warn "Neuron Monitor not available - install AWS Neuron Tools for hardware monitoring"
    fi
    
    # Create monitoring script
    cat > scripts/monitor.sh << 'EOF'
#!/bin/bash
# Monitoring script for Neuron training

echo "=== Neuron Device Status ==="
if command -v neuron-ls &> /dev/null; then
    neuron-ls
else
    echo "neuron-ls not available"
fi

echo -e "\n=== System Resources ==="
echo "Memory Usage:"
free -h

echo -e "\nDisk Usage:"
df -h

echo -e "\nCPU Usage:"
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"% CPU usage"}'

echo -e "\n=== Training Logs (last 10 lines) ==="
if [ -f "training.log" ]; then
    tail -10 training.log
else
    echo "No training.log found"
fi
EOF
    
    chmod +x scripts/monitor.sh
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    
    # Check Python packages
    log "Checking Python packages..."
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || error "PyTorch import failed"
    python3 -c "import torch_neuronx; print('Neuron PyTorch: OK')" || error "torch_neuronx import failed"
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || error "Transformers import failed"
    
    # Check Neuron runtime
    if command -v neuron-ls &> /dev/null; then
        log "Neuron devices:"
        neuron-ls
    else
        warn "neuron-ls command not available"
    fi
    
    # Check environment
    log "Environment check:"
    echo "NEURON_CC_FLAGS: ${NEURON_CC_FLAGS:-'Not set'}"
    echo "XLA_USE_BF16: ${XLA_USE_BF16:-'Not set'}"
    
    log "Installation verification completed successfully!"
}

# Create sample data
create_sample_data() {
    log "Creating sample training data..."
    
    python3 -c "
import json
import os

# Create sample training data
sample_data = []
for i in range(1000):
    sample_data.append({
        'text': f'This is sample training text number {i}. It demonstrates the format for training data.'
    })

os.makedirs('data', exist_ok=True)
with open('data/sample_train.jsonl', 'w') as f:
    for item in sample_data:
        f.write(json.dumps(item) + '\n')

# Create sample validation data
val_data = []
for i in range(100):
    val_data.append({
        'text': f'This is sample validation text number {i}. It is used for model evaluation.'
    })

with open('data/sample_validation.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')

print('Sample data created successfully!')
"
}

# Main setup function
main() {
    log "Starting Qwen3-14B training environment setup..."
    
    # Check if running as root for system-level installations
    if [[ $EUID -eq 0 ]]; then
        log "Running as root - proceeding with system installations"
    else
        warn "Not running as root - some system-level installations may fail"
        log "For full setup, consider running: sudo $0"
    fi
    
    check_instance
    setup_python
    
    # Only install system packages if running as root
    if [[ $EUID -eq 0 ]]; then
        install_neuron_sdk
    else
        warn "Skipping Neuron SDK system installation (requires root)"
        log "Install manually or run with sudo"
    fi
    
    install_dependencies
    setup_environment
    create_directories
    setup_monitoring
    create_sample_data
    verify_installation
    
    log "Setup completed successfully!"
    log ""
    log "Next steps:"
    log "1. Source the environment: source ~/.neuron_env"
    log "2. Configure training: cp configs/qwen3_14b_config.yaml configs/my_config.yaml"
    log "3. Edit your configuration file as needed"
    log "4. Start training: python src/train.py --config configs/my_config.yaml"
    log ""
    log "Monitor training: ./scripts/monitor.sh"
}

# Run main function
main "$@"