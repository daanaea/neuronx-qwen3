#!/bin/bash

# Qwen3-14B Training Launch Script for AWS Trn2
# This script launches training with proper environment setup and monitoring

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE="configs/qwen3_14b_config.yaml"
OUTPUT_DIR="./output"
RESUME_FROM=""
DISTRIBUTED=false
NUM_DEVICES=1
LOG_LEVEL="INFO"

# Logging functions
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

heading() {
    echo -e "${BLUE}$1${NC}"
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Launch Qwen3-14B training on AWS Trn2 instances

OPTIONS:
    -c, --config FILE       Configuration file (default: configs/qwen3_14b_config.yaml)
    -o, --output DIR        Output directory (default: ./output)
    -r, --resume PATH       Resume from checkpoint
    -d, --distributed       Enable distributed training
    -n, --devices NUM       Number of devices (default: 1)
    -l, --log-level LEVEL   Log level (default: INFO)
    -h, --help              Show this help message

EXAMPLES:
    # Basic training
    $0

    # Training with custom config
    $0 --config my_config.yaml --output ./my_output

    # Distributed training on 6 devices
    $0 --distributed --devices 6

    # Resume from checkpoint
    $0 --resume ./output/checkpoint-epoch-1

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -r|--resume)
                RESUME_FROM="$2"
                shift 2
                ;;
            -d|--distributed)
                DISTRIBUTED=true
                shift
                ;;
            -n|--devices)
                NUM_DEVICES="$2"
                shift 2
                ;;
            -l|--log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
}

# Validate inputs
validate_inputs() {
    log "Validating inputs..."
    
    # Check config file
    if [[ ! -f "$CONFIG_FILE" ]]; then
        error "Configuration file not found: $CONFIG_FILE"
    fi
    
    # Check if resume checkpoint exists
    if [[ -n "$RESUME_FROM" && ! -d "$RESUME_FROM" ]]; then
        error "Checkpoint directory not found: $RESUME_FROM"
    fi
    
    # Validate number of devices
    if [[ ! "$NUM_DEVICES" =~ ^[0-9]+$ ]] || [[ "$NUM_DEVICES" -lt 1 ]]; then
        error "Invalid number of devices: $NUM_DEVICES"
    fi
    
    log "Input validation passed"
}

# Setup environment
setup_environment() {
    log "Setting up training environment..."
    
    # Source Neuron environment if available
    if [[ -f ~/.neuron_env ]]; then
        source ~/.neuron_env
        log "Neuron environment loaded"
    else
        warn "Neuron environment file not found - run setup.sh first"
    fi
    
    # Set Neuron-specific environment variables
    export NEURON_CC_FLAGS="--model-type=transformer --auto-cast=all"
    export NEURON_FRAMEWORK_DEBUG=0
    export XLA_USE_BF16=1
    export MALLOC_ARENA_MAX=64
    export OMP_NUM_THREADS=1
    
    # Set distributed training environment if enabled
    if [[ "$DISTRIBUTED" == true ]]; then
        export WORLD_SIZE=$NUM_DEVICES
        export MASTER_ADDR="localhost"
        export MASTER_PORT="29500"
        
        log "Distributed training enabled with $NUM_DEVICES devices"
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs
    mkdir -p neuron_cache
    
    log "Environment setup completed"
}

# Check system status
check_system() {
    log "Checking system status..."
    
    # Check available memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    log "Available memory: ${MEMORY_GB}GB"
    
    if [[ $MEMORY_GB -lt 32 ]]; then
        warn "Low memory detected: ${MEMORY_GB}GB (recommended: 128GB+)"
    fi
    
    # Check disk space
    DISK_AVAIL=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    log "Available disk space: ${DISK_AVAIL}GB"
    
    if [[ $DISK_AVAIL -lt 100 ]]; then
        warn "Low disk space: ${DISK_AVAIL}GB (recommended: 500GB+)"
    fi
    
    # Check Neuron devices
    if command -v neuron-ls &> /dev/null; then
        log "Neuron devices:"
        neuron-ls
        
        DEVICE_COUNT=$(neuron-ls | grep -c "NeuronDevice" || echo "0")
        if [[ $DEVICE_COUNT -eq 0 ]]; then
            error "No Neuron devices found"
        elif [[ $DEVICE_COUNT -lt $NUM_DEVICES ]]; then
            error "Insufficient Neuron devices: found $DEVICE_COUNT, need $NUM_DEVICES"
        fi
    else
        warn "neuron-ls not available - cannot verify Neuron devices"
    fi
    
    log "System check completed"
}

# Pre-training setup
pre_training_setup() {
    log "Performing pre-training setup..."
    
    # Clear compilation cache if requested
    if [[ -d "neuron_cache" ]]; then
        CACHE_SIZE=$(du -sh neuron_cache | cut -f1)
        log "Neuron cache size: $CACHE_SIZE"
        
        # Optionally clear cache for fresh compilation
        # rm -rf neuron_cache/*
    fi
    
    # Validate data files
    python3 -c "
import yaml
import os

# Load config
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Check data files
train_path = config['data']['train_path']
if not os.path.exists(train_path):
    print(f'ERROR: Training data not found: {train_path}')
    exit(1)

val_path = config['data'].get('validation_path')
if val_path and not os.path.exists(val_path):
    print(f'WARNING: Validation data not found: {val_path}')

print('Data validation passed')
" || error "Data validation failed"
    
    log "Pre-training setup completed"
}

# Launch training
launch_training() {
    log "Launching Qwen3-14B training..."
    
    # Build training command
    TRAIN_CMD="python3 src/train.py --config $CONFIG_FILE"
    
    if [[ -n "$OUTPUT_DIR" ]]; then
        TRAIN_CMD="$TRAIN_CMD --output_dir $OUTPUT_DIR"
    fi
    
    if [[ -n "$RESUME_FROM" ]]; then
        TRAIN_CMD="$TRAIN_CMD --resume $RESUME_FROM"
    fi
    
    # Set up logging
    LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"
    
    heading "=========================================="
    heading "Starting Qwen3-14B Training"
    heading "=========================================="
    heading "Config: $CONFIG_FILE"
    heading "Output: $OUTPUT_DIR"
    heading "Devices: $NUM_DEVICES"
    heading "Distributed: $DISTRIBUTED"
    heading "Log file: $LOG_FILE"
    heading "=========================================="
    
    # Launch training with proper logging
    if [[ "$DISTRIBUTED" == true ]]; then
        # Distributed training launch
        log "Launching distributed training..."
        
        # Use torchrun for distributed training
        LAUNCH_CMD="python3 -m torch.distributed.run \
            --nproc_per_node=$NUM_DEVICES \
            --master_addr=localhost \
            --master_port=29500 \
            src/train.py --config $CONFIG_FILE"
            
        if [[ -n "$OUTPUT_DIR" ]]; then
            LAUNCH_CMD="$LAUNCH_CMD --output_dir $OUTPUT_DIR"
        fi
        
        if [[ -n "$RESUME_FROM" ]]; then
            LAUNCH_CMD="$LAUNCH_CMD --resume $RESUME_FROM"
        fi
        
        # Execute with logging
        eval $LAUNCH_CMD 2>&1 | tee "$LOG_FILE"
    else
        # Single device training
        log "Launching single device training..."
        eval $TRAIN_CMD 2>&1 | tee "$LOG_FILE"
    fi
    
    TRAIN_EXIT_CODE=$?
    
    if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
        log "Training completed successfully!"
    else
        error "Training failed with exit code: $TRAIN_EXIT_CODE"
    fi
}

# Post-training cleanup and reporting
post_training() {
    log "Post-training cleanup and reporting..."
    
    # Show final model size
    if [[ -d "$OUTPUT_DIR" ]]; then
        OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
        log "Output directory size: $OUTPUT_SIZE"
        
        # List checkpoints
        if ls "$OUTPUT_DIR"/checkpoint-* &> /dev/null; then
            log "Available checkpoints:"
            ls -la "$OUTPUT_DIR"/checkpoint-* | awk '{print "  " $9 " (" $5 " bytes)"}'
        fi
        
        # Show best model if available
        if [[ -f "$OUTPUT_DIR/best_model/pytorch_model.bin" ]]; then
            BEST_MODEL_SIZE=$(du -sh "$OUTPUT_DIR/best_model" | cut -f1)
            log "Best model size: $BEST_MODEL_SIZE"
        fi
    fi
    
    # Show cache sizes
    if [[ -d "neuron_cache" ]]; then
        CACHE_SIZE=$(du -sh neuron_cache | cut -f1)
        log "Neuron compilation cache: $CACHE_SIZE"
    fi
    
    # Generate training report
    log "Training session completed at $(date)"
}

# Signal handlers for graceful shutdown
cleanup() {
    log "Received interrupt signal - cleaning up..."
    
    # Kill any running training processes
    pkill -f "python.*train.py" || true
    
    log "Cleanup completed"
    exit 130
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main function
main() {
    heading "Qwen3-14B Training Launcher"
    
    parse_args "$@"
    validate_inputs
    setup_environment
    check_system
    pre_training_setup
    launch_training
    post_training
    
    log "Training session completed successfully!"
}

# Run main function
main "$@"