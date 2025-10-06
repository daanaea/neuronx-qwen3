#!/usr/bin/env python3

import logging
import os
import time
import json
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.distributed as dist
import torch_neuronx


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def setup_distributed_training() -> Tuple[int, int]:
    """Setup distributed training if available"""
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        
        # Initialize the process group
        dist.init_process_group(
            backend='xla',  # Use XLA backend for Neuron
            world_size=world_size,
            rank=rank
        )
        
        return world_size, rank
    else:
        return 1, 0


class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    checkpoint_path: str,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    **kwargs
):
    """Save model checkpoint"""
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save model state
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
        
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
        
    torch.save(checkpoint, os.path.join(checkpoint_path, 'pytorch_model.bin'))
    
    # Save tokenizer and config if available
    if hasattr(model, 'config'):
        with open(os.path.join(checkpoint_path, 'config.json'), 'w') as f:
            json.dump(model.config, f, indent=2)
            
    logging.info(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    strict: bool = True
) -> Dict:
    """Load model checkpoint"""
    checkpoint_file = os.path.join(checkpoint_path, 'pytorch_model.bin')
    
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_file}")
        
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    # Load scaler state
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    logging.info(f"Checkpoint loaded from {checkpoint_path}")
    logging.info(f"Resuming from epoch {checkpoint.get('epoch', 0)}, step {checkpoint.get('step', 0)}")
    
    return checkpoint


def calculate_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Calculate training metrics"""
    with torch.no_grad():
        # Accuracy (for next token prediction)
        mask = (labels != -100)
        correct = (predictions.argmax(dim=-1) == labels) & mask
        accuracy = correct.float().sum() / mask.float().sum()
        
        # Perplexity
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_predictions = predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        flat_predictions = shift_predictions.view(-1, shift_predictions.size(-1))
        flat_labels = shift_labels.view(-1)
        
        losses = loss_fct(flat_predictions, flat_labels)
        mask = (flat_labels != -100)
        
        avg_loss = losses[mask].mean()
        perplexity = torch.exp(avg_loss)
        
        return {
            'accuracy': accuracy.item(),
            'perplexity': perplexity.item(),
            'loss': avg_loss.item()
        }


def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Get model size information"""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in bytes (assuming float32)
    model_size_mb = param_count * 4 / (1024 * 1024)
    
    return {
        'total_params': param_count,
        'trainable_params': trainable_param_count,
        'non_trainable_params': param_count - trainable_param_count,
        'size_mb': model_size_mb
    }


def print_model_info(model: torch.nn.Module):
    """Print model information"""
    info = get_model_size(model)
    
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Non-trainable parameters: {info['non_trainable_params']:,}")
    print(f"Model size: {info['size_mb']:.2f} MB")
    print("="*50 + "\n")


def get_neuron_device_info():
    """Get information about available Neuron devices"""
    try:
        import torch_neuronx
        
        device_count = torch_neuronx.device_count()
        devices = []
        
        for i in range(device_count):
            device_info = {
                'device_id': i,
                'device_name': f'neuron:{i}',
                'memory_info': torch_neuronx.memory_info(i) if hasattr(torch_neuronx, 'memory_info') else None
            }
            devices.append(device_info)
            
        return devices
    except Exception as e:
        logging.warning(f"Could not get Neuron device info: {e}")
        return []


def monitor_training_resources():
    """Monitor training resources and log them"""
    try:
        # Monitor Neuron devices
        devices = get_neuron_device_info()
        if devices:
            logging.info(f"Available Neuron devices: {len(devices)}")
            for device in devices:
                logging.info(f"Device {device['device_id']}: {device['device_name']}")
                
        # Monitor memory usage
        import psutil
        memory = psutil.virtual_memory()
        logging.info(f"System memory usage: {memory.percent}% ({memory.used // 1024**3}GB / {memory.total // 1024**3}GB)")
        
    except ImportError:
        logging.warning("psutil not available for resource monitoring")
    except Exception as e:
        logging.warning(f"Error monitoring resources: {e}")


def create_learning_rate_schedule(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    schedule_type: str = "linear"
):
    """Create learning rate schedule"""
    from transformers import get_scheduler
    
    scheduler = get_scheduler(
        schedule_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return scheduler


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class TrainingTimer:
    """Timer for tracking training time"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        self.end_time = time.time()
        
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
        
    def elapsed_str(self) -> str:
        return format_time(self.elapsed())


def validate_config(config: Dict) -> bool:
    """Validate training configuration"""
    required_fields = [
        'model',
        'training',
        'data',
        'output_dir'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
            
    # Validate model config
    model_required = ['vocab_size', 'hidden_size', 'num_layers', 'num_attention_heads']
    for field in model_required:
        if field not in config['model']:
            raise ValueError(f"Missing required model config field: {field}")
            
    # Validate training config
    training_required = ['batch_size', 'learning_rate', 'num_epochs']
    for field in training_required:
        if field not in config['training']:
            raise ValueError(f"Missing required training config field: {field}")
            
    # Validate data config
    if 'train_path' not in config['data']:
        raise ValueError("Missing required data config field: train_path")
        
    return True


def setup_environment():
    """Setup environment variables for Neuron training"""
    # Set environment variables for optimal Neuron performance
    os.environ.setdefault('NEURON_CC_FLAGS', '--model-type=transformer')
    os.environ.setdefault('NEURON_FRAMEWORK_DEBUG', '1')
    os.environ.setdefault('XLA_USE_BF16', '1')
    
    # Set up distributed training environment if not already set
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        
    logging.info("Environment setup completed for Neuron training")


if __name__ == "__main__":
    # Test utilities
    setup_logging()
    setup_environment()
    monitor_training_resources()
    
    # Test timer
    timer = TrainingTimer()
    timer.start()
    time.sleep(1)
    timer.stop()
    print(f"Timer test: {timer.elapsed_str()}")