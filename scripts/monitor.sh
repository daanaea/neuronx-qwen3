#!/bin/bash

# Qwen3-14B Training Monitoring Script for AWS Trn2
# Real-time monitoring of training progress and system resources

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
REFRESH_INTERVAL=10
LOG_FILE="training.log"
SHOW_LIVE=false
EXPORT_METRICS=false
METRICS_FILE="metrics.json"

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Monitor Qwen3-14B training on AWS Trn2 instances

OPTIONS:
    -i, --interval SECONDS  Refresh interval (default: 10)
    -l, --log-file FILE     Training log file (default: training.log)
    -w, --live             Show live log tail
    -e, --export           Export metrics to JSON
    -m, --metrics FILE      Metrics output file (default: metrics.json)
    -h, --help             Show this help message

EXAMPLES:
    # Basic monitoring
    $0

    # Monitor with 5-second refresh
    $0 --interval 5

    # Show live logs
    $0 --live

    # Export metrics
    $0 --export --metrics training_metrics.json

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--interval)
                REFRESH_INTERVAL="$2"
                shift 2
                ;;
            -l|--log-file)
                LOG_FILE="$2"
                shift 2
                ;;
            -w|--live)
                SHOW_LIVE=true
                shift
                ;;
            -e|--export)
                EXPORT_METRICS=true
                shift
                ;;
            -m|--metrics)
                METRICS_FILE="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# Header display
show_header() {
    clear
    echo -e "${BLUE}=========================================="
    echo -e "Qwen3-14B Training Monitor"
    echo -e "$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "==========================================${NC}"
}

# System information
show_system_info() {
    echo -e "\n${CYAN}=== SYSTEM INFORMATION ===${NC}"
    
    # Instance type
    if command -v aws &> /dev/null; then
        INSTANCE_TYPE=$(aws ec2 describe-instances \
            --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null) \
            --query 'Reservations[0].Instances[0].InstanceType' \
            --output text 2>/dev/null || echo "unknown")
        echo -e "Instance Type: ${GREEN}$INSTANCE_TYPE${NC}"
    fi
    
    # Uptime
    UPTIME=$(uptime -p 2>/dev/null || uptime | awk '{print $3,$4}')
    echo -e "Uptime: ${GREEN}$UPTIME${NC}"
    
    # Load average
    LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | xargs)
    echo -e "Load Average: ${GREEN}$LOAD_AVG${NC}"
}

# Neuron device status
show_neuron_status() {
    echo -e "\n${CYAN}=== NEURON DEVICES ===${NC}"
    
    if command -v neuron-ls &> /dev/null; then
        neuron-ls 2>/dev/null || echo -e "${RED}No Neuron devices found${NC}"
    else
        echo -e "${YELLOW}neuron-ls not available${NC}"
    fi
    
    # Neuron device utilization
    if command -v neuron-monitor &> /dev/null; then
        echo -e "\n${CYAN}Device Utilization:${NC}"
        neuron-monitor | head -20 2>/dev/null || echo -e "${YELLOW}Neuron monitor data not available${NC}"
    fi
}

# Memory usage
show_memory_usage() {
    echo -e "\n${CYAN}=== MEMORY USAGE ===${NC}"
    
    # System memory
    free -h | while read line; do
        if [[ $line == Mem:* ]]; then
            USED=$(echo $line | awk '{print $3}')
            TOTAL=$(echo $line | awk '{print $2}')
            PERCENT=$(echo $line | awk '{printf "%.1f", ($3/$2)*100}')
            
            if (( $(echo "$PERCENT > 80" | bc -l) )); then
                COLOR=$RED
            elif (( $(echo "$PERCENT > 60" | bc -l) )); then
                COLOR=$YELLOW
            else
                COLOR=$GREEN
            fi
            
            echo -e "System Memory: ${COLOR}$USED / $TOTAL (${PERCENT}%)${NC}"
        fi
    done
    
    # GPU/Neuron memory if available
    if command -v neuron-monitor &> /dev/null; then
        echo -e "Neuron Memory: ${GREEN}$(neuron-monitor | grep -i memory | head -1 || echo 'N/A')${NC}"
    fi
}

# Disk usage
show_disk_usage() {
    echo -e "\n${CYAN}=== DISK USAGE ===${NC}"
    
    # Current directory
    CURRENT_USAGE=$(df -h . | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}')
    echo -e "Current Directory: ${GREEN}$CURRENT_USAGE${NC}"
    
    # Important directories
    for dir in "output" "logs" "neuron_cache" "data"; do
        if [[ -d "$dir" ]]; then
            SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo -e "$dir: ${GREEN}$SIZE${NC}"
        fi
    done
}

# Training progress
show_training_progress() {
    echo -e "\n${CYAN}=== TRAINING PROGRESS ===${NC}"
    
    if [[ -f "$LOG_FILE" ]]; then
        # Get latest training metrics
        LATEST_LOSS=$(grep -o "Loss: [0-9.]*" "$LOG_FILE" | tail -1 | cut -d' ' -f2)
        LATEST_LR=$(grep -o "LR: [0-9.e-]*" "$LOG_FILE" | tail -1 | cut -d' ' -f2)
        LATEST_STEP=$(grep -o "Step [0-9]*" "$LOG_FILE" | tail -1 | cut -d' ' -f2)
        LATEST_EPOCH=$(grep -o "Epoch [0-9]*" "$LOG_FILE" | tail -1 | cut -d' ' -f2)
        
        # Current status
        if pgrep -f "python.*train.py" > /dev/null; then
            STATUS="${GREEN}RUNNING${NC}"
        else
            STATUS="${RED}STOPPED${NC}"
        fi
        
        echo -e "Status: $STATUS"
        
        if [[ -n "$LATEST_EPOCH" ]]; then
            echo -e "Current Epoch: ${GREEN}$LATEST_EPOCH${NC}"
        fi
        
        if [[ -n "$LATEST_STEP" ]]; then
            echo -e "Current Step: ${GREEN}$LATEST_STEP${NC}"
        fi
        
        if [[ -n "$LATEST_LOSS" ]]; then
            echo -e "Latest Loss: ${GREEN}$LATEST_LOSS${NC}"
        fi
        
        if [[ -n "$LATEST_LR" ]]; then
            echo -e "Learning Rate: ${GREEN}$LATEST_LR${NC}"
        fi
        
        # Training speed
        RECENT_LOGS=$(tail -100 "$LOG_FILE" | grep "Step Time")
        if [[ -n "$RECENT_LOGS" ]]; then
            AVG_STEP_TIME=$(echo "$RECENT_LOGS" | grep -o "[0-9.]*s" | sed 's/s//' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
            if [[ "$AVG_STEP_TIME" != "N/A" ]]; then
                echo -e "Avg Step Time: ${GREEN}${AVG_STEP_TIME}s${NC}"
            fi
        fi
        
        # Error checking
        ERROR_COUNT=$(grep -c "ERROR\|Exception\|Error" "$LOG_FILE" 2>/dev/null || echo "0")
        if [[ $ERROR_COUNT -gt 0 ]]; then
            echo -e "Errors Found: ${RED}$ERROR_COUNT${NC}"
        fi
        
    else
        echo -e "${YELLOW}Training log not found: $LOG_FILE${NC}"
    fi
}

# Recent logs
show_recent_logs() {
    echo -e "\n${CYAN}=== RECENT LOGS ===${NC}"
    
    if [[ -f "$LOG_FILE" ]]; then
        echo -e "${PURPLE}Last 5 log entries:${NC}"
        tail -5 "$LOG_FILE" | while read line; do
            # Color code based on log level
            if [[ $line == *"ERROR"* ]]; then
                echo -e "${RED}$line${NC}"
            elif [[ $line == *"WARNING"* || $line == *"WARN"* ]]; then
                echo -e "${YELLOW}$line${NC}"
            elif [[ $line == *"INFO"* ]]; then
                echo -e "${GREEN}$line${NC}"
            else
                echo "$line"
            fi
        done
    else
        echo -e "${YELLOW}No log file found${NC}"
    fi
}

# Process information
show_process_info() {
    echo -e "\n${CYAN}=== TRAINING PROCESSES ===${NC}"
    
    # Training processes
    TRAIN_PIDS=$(pgrep -f "python.*train.py" 2>/dev/null || echo "")
    
    if [[ -n "$TRAIN_PIDS" ]]; then
        echo -e "${GREEN}Active training processes:${NC}"
        for pid in $TRAIN_PIDS; do
            PROCESS_INFO=$(ps -p "$pid" -o pid,ppid,%cpu,%mem,etime,cmd --no-headers 2>/dev/null || echo "Process $pid not found")
            echo "  PID $pid: $PROCESS_INFO"
        done
    else
        echo -e "${YELLOW}No training processes running${NC}"
    fi
    
    # Python processes using significant resources
    echo -e "\n${PURPLE}High-resource Python processes:${NC}"
    ps aux | grep python | grep -v grep | awk '$3 > 5 || $4 > 5 {printf "  PID %s: CPU %.1f%% MEM %.1f%% %s\n", $2, $3, $4, $11}' | head -5
}

# Export metrics to JSON
export_metrics() {
    if [[ "$EXPORT_METRICS" == true ]]; then
        echo -e "\n${CYAN}=== EXPORTING METRICS ===${NC}"
        
        python3 - << EOF
import json
import re
import os
from datetime import datetime

metrics = {
    "timestamp": datetime.now().isoformat(),
    "training": {},
    "system": {},
    "neuron": {}
}

# Parse training log if available
if os.path.exists("$LOG_FILE"):
    with open("$LOG_FILE", "r") as f:
        log_content = f.read()
    
    # Extract latest metrics
    loss_matches = re.findall(r"Loss: ([0-9.]+)", log_content)
    if loss_matches:
        metrics["training"]["latest_loss"] = float(loss_matches[-1])
    
    lr_matches = re.findall(r"LR: ([0-9.e-]+)", log_content)
    if lr_matches:
        metrics["training"]["latest_lr"] = float(lr_matches[-1])
    
    step_matches = re.findall(r"Step ([0-9]+)", log_content)
    if step_matches:
        metrics["training"]["latest_step"] = int(step_matches[-1])
    
    epoch_matches = re.findall(r"Epoch ([0-9]+)", log_content)
    if epoch_matches:
        metrics["training"]["latest_epoch"] = int(epoch_matches[-1])

# System metrics
import psutil
memory = psutil.virtual_memory()
metrics["system"]["memory_percent"] = memory.percent
metrics["system"]["memory_used_gb"] = memory.used / (1024**3)
metrics["system"]["memory_total_gb"] = memory.total / (1024**3)

disk = psutil.disk_usage('.')
metrics["system"]["disk_percent"] = disk.percent
metrics["system"]["disk_used_gb"] = disk.used / (1024**3)
metrics["system"]["disk_total_gb"] = disk.total / (1024**3)

# Save metrics
with open("$METRICS_FILE", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Metrics exported to $METRICS_FILE")
EOF
    fi
}

# Live log tail
show_live_logs() {
    if [[ "$SHOW_LIVE" == true ]]; then
        echo -e "\n${CYAN}=== LIVE LOGS ===${NC}"
        echo -e "${PURPLE}Press Ctrl+C to return to monitoring${NC}\n"
        
        if [[ -f "$LOG_FILE" ]]; then
            tail -f "$LOG_FILE" | while read line; do
                # Color code based on log level
                if [[ $line == *"ERROR"* ]]; then
                    echo -e "${RED}$line${NC}"
                elif [[ $line == *"WARNING"* || $line == *"WARN"* ]]; then
                    echo -e "${YELLOW}$line${NC}"
                elif [[ $line == *"INFO"* ]]; then
                    echo -e "${GREEN}$line${NC}"
                else
                    echo "$line"
                fi
            done
        else
            echo -e "${RED}Log file not found: $LOG_FILE${NC}"
        fi
    fi
}

# Main monitoring loop
monitor_loop() {
    while true; do
        show_header
        show_system_info
        show_neuron_status
        show_memory_usage
        show_disk_usage
        show_training_progress
        show_recent_logs
        show_process_info
        export_metrics
        
        if [[ "$SHOW_LIVE" != true ]]; then
            echo -e "\n${PURPLE}Refreshing in ${REFRESH_INTERVAL}s... (Press Ctrl+C to exit)${NC}"
            sleep "$REFRESH_INTERVAL"
        else
            break
        fi
    done
}

# Signal handler for clean exit
cleanup() {
    echo -e "\n${GREEN}Monitoring stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main function
main() {
    parse_args "$@"
    
    # Validate inputs
    if [[ ! "$REFRESH_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$REFRESH_INTERVAL" -lt 1 ]]; then
        echo "Invalid refresh interval: $REFRESH_INTERVAL"
        exit 1
    fi
    
    if [[ "$SHOW_LIVE" == true ]]; then
        show_live_logs
    else
        monitor_loop
    fi
}

# Run main function
main "$@"