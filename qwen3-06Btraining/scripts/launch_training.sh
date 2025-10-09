#!/bin/bash
# Launch Training Script with Auto-Stop
# Qwen3-0.6B Training on AWS Trn2

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONFIG="configs/qwen3_0.6b_config.yaml"
AUTO_STOP=""
REGION="us-east-1"
BACKGROUND=false
OUTPUT_DIR=""
RESUME_FROM=""

# Display help
show_help() {
    cat << EOF
${BLUE}Qwen3-0.6B Training Launcher with Auto-Stop${NC}

USAGE:
    ./scripts/launch_training.sh [OPTIONS]

OPTIONS:
    -c, --config FILE          Path to config file (default: configs/qwen3_0.6b_config.yaml)
    -t, --time SECONDS         Auto-stop after N seconds (e.g., 60, 3600)
    -m, --minutes MINUTES      Auto-stop after N minutes (converted to seconds)
    -r, --region REGION        AWS region (default: us-east-1)
    -b, --background           Run in background with nohup
    -o, --output DIR           Output directory override
    --resume PATH              Resume from checkpoint
    -h, --help                 Show this help message

EXAMPLES:
    ${GREEN}# Train for 1 minute then auto-stop (testing)${NC}
    ./scripts/launch_training.sh -m 1

    ${GREEN}# Train for 30 minutes in background${NC}
    ./scripts/launch_training.sh -m 30 -b

    ${GREEN}# Train for 1 hour with custom config${NC}
    ./scripts/launch_training.sh -c configs/my_config.yaml -m 60

    ${GREEN}# Train without auto-stop${NC}
    ./scripts/launch_training.sh

    ${GREEN}# Resume training from checkpoint${NC}
    ./scripts/launch_training.sh --resume output/checkpoint-epoch-1

COMMON TIME CONVERSIONS:
    ${YELLOW}1 minute${NC}   = 60 seconds    | Use: ${GREEN}-m 1${NC}  or ${GREEN}-t 60${NC}
    ${YELLOW}5 minutes${NC}  = 300 seconds   | Use: ${GREEN}-m 5${NC}  or ${GREEN}-t 300${NC}
    ${YELLOW}30 minutes${NC} = 1800 seconds  | Use: ${GREEN}-m 30${NC} or ${GREEN}-t 1800${NC}
    ${YELLOW}1 hour${NC}     = 3600 seconds  | Use: ${GREEN}-m 60${NC} or ${GREEN}-t 3600${NC}
    ${YELLOW}2 hours${NC}    = 7200 seconds  | Use: ${GREEN}-m 120${NC} or ${GREEN}-t 7200${NC}

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -t|--time)
            AUTO_STOP="$2"
            shift 2
            ;;
        -m|--minutes)
            AUTO_STOP=$((60 * $2))
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -b|--background)
            BACKGROUND=true
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Verify config file exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
    exit 1
fi

# Check if resuming from checkpoint
if [ -n "$RESUME_FROM" ] && [ ! -d "$RESUME_FROM" ]; then
    echo -e "${RED}Error: Checkpoint directory not found: $RESUME_FROM${NC}"
    exit 1
fi

# Build command
CMD="python src/train.py --config $CONFIG"

if [ -n "$AUTO_STOP" ]; then
    CMD="$CMD --auto-stop $AUTO_STOP --auto-stop-region $REGION"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

if [ -n "$RESUME_FROM" ]; then
    CMD="$CMD --resume $RESUME_FROM"
fi

# Display configuration
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   Qwen3-0.6B Training Launch${NC}"
echo -e "${BLUE}==========================================${NC}"
echo -e "Config file:    ${GREEN}$CONFIG${NC}"

if [ -n "$AUTO_STOP" ]; then
    MINUTES=$((AUTO_STOP / 60))
    HOURS=$((MINUTES / 60))

    if [ $HOURS -gt 0 ]; then
        REMAINING_MINS=$((MINUTES % 60))
        echo -e "Auto-stop:      ${YELLOW}$AUTO_STOP seconds (${HOURS}h ${REMAINING_MINS}m)${NC}"
    else
        echo -e "Auto-stop:      ${YELLOW}$AUTO_STOP seconds ($MINUTES minutes)${NC}"
    fi

    echo -e "AWS Region:     ${GREEN}$REGION${NC}"
else
    echo -e "Auto-stop:      ${YELLOW}Disabled${NC}"
fi

if [ -n "$OUTPUT_DIR" ]; then
    echo -e "Output dir:     ${GREEN}$OUTPUT_DIR${NC}"
else
    echo -e "Output dir:     ${GREEN}./output${NC} (default)"
fi

if [ -n "$RESUME_FROM" ]; then
    echo -e "Resume from:    ${GREEN}$RESUME_FROM${NC}"
fi

echo -e "Background:     ${GREEN}$BACKGROUND${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Warnings and confirmations
if [ -n "$AUTO_STOP" ]; then
    MINUTES=$((AUTO_STOP / 60))
    echo -e "${YELLOW}⚠️  AUTO-STOP ENABLED${NC}"
    echo -e "${YELLOW}    Instance will automatically stop in $MINUTES minutes!${NC}"
    echo ""
fi

# Check if running on EC2
if ! curl -s --connect-timeout 1 http://169.254.169.254/latest/meta-data/instance-id &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: Not running on EC2${NC}"
    echo -e "${YELLOW}    Auto-stop may not work properly${NC}"
    echo ""
fi

# Confirmation prompt
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Launch training
echo -e "${GREEN}Starting training...${NC}"
echo -e "${BLUE}Command:${NC} $CMD"
echo ""

if [ "$BACKGROUND" = true ]; then
    echo -e "${GREEN}Running in background with nohup...${NC}"

    # Create logs directory if it doesn't exist
    mkdir -p logs

    # Generate timestamp for log file
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="logs/training_${TIMESTAMP}.log"

    nohup $CMD > "$LOG_FILE" 2>&1 &
    PID=$!

    echo -e "${GREEN}✓ Training started!${NC}"
    echo -e "  PID:      ${YELLOW}$PID${NC}"
    echo -e "  Log file: ${YELLOW}$LOG_FILE${NC}"
    echo ""
    echo -e "${BLUE}Monitor with:${NC}"
    echo -e "  tail -f $LOG_FILE"
    echo -e "  ./scripts/monitor.sh"
    echo ""
    echo -e "${BLUE}Stop training:${NC}"
    echo -e "  kill $PID"
    echo ""

    if [ -n "$AUTO_STOP" ]; then
        MINUTES=$((AUTO_STOP / 60))
        echo -e "${YELLOW}⏰ Instance will auto-stop in $MINUTES minutes${NC}"
    fi
else
    # Run in foreground
    echo -e "${GREEN}Running in foreground...${NC}"
    echo -e "${BLUE}Press Ctrl+C to stop training${NC}"
    echo ""

    $CMD

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Training completed successfully!${NC}"
    else
        echo ""
        echo -e "${RED}✗ Training failed with exit code: $EXIT_CODE${NC}"
        exit $EXIT_CODE
    fi
fi

echo -e "${GREEN}Done!${NC}"
