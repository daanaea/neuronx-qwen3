#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch_neuronx
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model import Qwen3Model
from data_loader import create_dataloader
from utils import (
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    calculate_metrics,
    AverageMeter,
    setup_distributed_training
)

logger = logging.getLogger(__name__)


class Qwen3Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cpu')
        self.world_size = 1
        self.rank = 0
        
        if config.get('distributed', False):
            self.world_size, self.rank = setup_distributed_training()
            
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_logging()
        
    def setup_model(self):
        model_config = self.config['model']
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['name'],
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = Qwen3Model(
            vocab_size=model_config['vocab_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_attention_heads=model_config['num_attention_heads'],
            num_key_value_heads=model_config['num_key_value_heads'],
            max_position_embeddings=model_config['max_position_embeddings'],
            use_gradient_checkpointing=model_config.get('gradient_checkpointing', True)
        )
        
        if self.config.get('resume_from_checkpoint'):
            self.model = load_checkpoint(self.model, self.config['resume_from_checkpoint'])
            
        # Compile model for Neuron
        logger.info("Compiling model for Neuron...")
        if self.config['training'].get('compile_model', True):
            self.model = torch_neuronx.trace(
                self.model,
                example_inputs=torch.randint(0, self.config['model']['vocab_size'], 
                                           (1, self.config['training']['max_seq_length'])),
                compiler_workdir=self.config.get('compiler_workdir', './neuron_cache')
            )
            
    def setup_data(self):
        train_config = self.config['training']
        
        self.train_dataloader = create_dataloader(
            data_path=self.config['data']['train_path'],
            tokenizer=self.tokenizer,
            max_length=train_config['max_seq_length'],
            batch_size=train_config['batch_size'],
            num_workers=train_config.get('num_workers', 4),
            shuffle=True
        )
        
        if self.config['data'].get('validation_path'):
            self.val_dataloader = create_dataloader(
                data_path=self.config['data']['validation_path'],
                tokenizer=self.tokenizer,
                max_length=train_config['max_seq_length'],
                batch_size=train_config['batch_size'],
                num_workers=train_config.get('num_workers', 4),
                shuffle=False
            )
        else:
            self.val_dataloader = None
            
    def setup_optimizer(self):
        train_config = self.config['training']
        
        # Create optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": train_config.get('weight_decay', 0.01),
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=train_config['learning_rate'],
            betas=(train_config.get('beta1', 0.9), train_config.get('beta2', 0.999)),
            eps=train_config.get('epsilon', 1e-8)
        )
        
        # Setup learning rate scheduler
        num_training_steps = len(self.train_dataloader) * train_config['num_epochs']
        num_warmup_steps = int(num_training_steps * train_config.get('warmup_ratio', 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Setup automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=train_config.get('fp16', False))
        
    def setup_logging(self):
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=self.config.get('log_dir', './logs'))
        else:
            self.writer = None
            
    def train_epoch(self, epoch: int):
        self.model.train()
        losses = AverageMeter()
        train_config = self.config['training']
        
        for step, batch in enumerate(self.train_dataloader):
            start_time = time.time()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16, 
                               enabled=train_config.get('bf16', True)):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                if train_config.get('gradient_accumulation_steps', 1) > 1:
                    loss = loss / train_config['gradient_accumulation_steps']
            
            # Backward pass
            if train_config.get('fp16', False):
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Gradient accumulation
            if (step + 1) % train_config.get('gradient_accumulation_steps', 1) == 0:
                # Gradient clipping
                if train_config.get('max_grad_norm'):
                    if train_config.get('fp16', False):
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        train_config['max_grad_norm']
                    )
                
                # Optimizer step
                if train_config.get('fp16', False):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.scheduler.step()
                
            losses.update(loss.item())
            step_time = time.time() - start_time
            
            # Logging
            if step % train_config.get('logging_steps', 100) == 0 and self.rank == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch}, Step {step}/{len(self.train_dataloader)}, "
                    f"Loss: {losses.avg:.4f}, LR: {current_lr:.2e}, "
                    f"Step Time: {step_time:.2f}s"
                )
                
                if self.writer:
                    global_step = epoch * len(self.train_dataloader) + step
                    self.writer.add_scalar('train/loss', losses.avg, global_step)
                    self.writer.add_scalar('train/learning_rate', current_lr, global_step)
                    self.writer.add_scalar('train/step_time', step_time, global_step)
                    
            # Save checkpoint
            if (step % train_config.get('save_steps', 1000) == 0 and 
                step > 0 and self.rank == 0):
                checkpoint_path = os.path.join(
                    self.config['output_dir'], 
                    f"checkpoint-epoch-{epoch}-step-{step}"
                )
                save_checkpoint(self.model, self.optimizer, epoch, step, checkpoint_path)
                
        return losses.avg
        
    def validate(self, epoch: int):
        if self.val_dataloader is None:
            return None
            
        self.model.eval()
        losses = AverageMeter()
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                with torch.autocast(device_type='cpu', dtype=torch.bfloat16, 
                                   enabled=self.config['training'].get('bf16', True)):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                losses.update(loss.item())
                
        if self.rank == 0:
            logger.info(f"Validation Loss: {losses.avg:.4f}")
            if self.writer:
                self.writer.add_scalar('val/loss', losses.avg, epoch)
                
        return losses.avg
        
    def train(self):
        logger.info("Starting training...")
        train_config = self.config['training']
        best_val_loss = float('inf')
        
        for epoch in range(train_config['num_epochs']):
            logger.info(f"Starting epoch {epoch + 1}/{train_config['num_epochs']}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate(epoch)
            
            # Save best model
            if val_loss is not None and val_loss < best_val_loss and self.rank == 0:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.config['output_dir'], 'best_model')
                save_checkpoint(self.model, self.optimizer, epoch, 0, best_model_path)
                logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
                
            # Save epoch checkpoint
            if self.rank == 0:
                epoch_checkpoint_path = os.path.join(
                    self.config['output_dir'], 
                    f"checkpoint-epoch-{epoch}"
                )
                save_checkpoint(self.model, self.optimizer, epoch, 0, epoch_checkpoint_path)
                
        logger.info("Training completed!")
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-14B on AWS Trn2")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, help='Output directory override')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override config with command line arguments
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.resume:
        config['resume_from_checkpoint'] = args.resume
        
    # Setup output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Setup logging
    setup_logging(config.get('log_level', 'INFO'))
    
    # Initialize trainer
    trainer = Qwen3Trainer(config)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()