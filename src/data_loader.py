#!/usr/bin/env python3

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        text_column: str = "text",
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} examples from {self.data_path}")
        
    def _load_data(self) -> List[Dict]:
        data = []
        
        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if self.text_column in item:
                            data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")
                        
        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                if isinstance(raw_data, list):
                    for item in raw_data:
                        if isinstance(item, dict) and self.text_column in item:
                            data.append(item)
                elif isinstance(raw_data, dict) and self.text_column in raw_data:
                    data.append(raw_data)
                    
        elif self.data_path.suffix == '.txt':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Split by double newlines to get separate examples
                examples = content.split('\n\n')
                for example in examples:
                    if example.strip():
                        data.append({self.text_column: example.strip()})
                        
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
        return data
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item[self.text_column]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # For language modeling, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class ConversationDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        conversation_format: str = "chatml",  # or "alpaca"
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversation_format = conversation_format
        
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} conversations from {self.data_path}")
        
    def _load_data(self) -> List[Dict]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            if self.data_path.suffix == '.jsonl':
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        return data
        
    def _format_conversation_chatml(self, conversation: List[Dict]) -> str:
        formatted = ""
        for message in conversation:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted
        
    def _format_conversation_alpaca(self, conversation: List[Dict]) -> str:
        if len(conversation) >= 2:
            instruction = conversation[0].get("content", "")
            response = conversation[1].get("content", "")
            if len(conversation) > 2:
                context = conversation[1].get("content", "")
                return f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{response}"
            else:
                return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        return ""
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Handle different conversation formats
        if "conversations" in item:
            conversation = item["conversations"]
        elif "messages" in item:
            conversation = item["messages"]
        else:
            # Assume the item itself is a conversation
            conversation = item
            
        # Format the conversation
        if self.conversation_format == "chatml":
            text = self._format_conversation_chatml(conversation)
        elif self.conversation_format == "alpaca":
            text = self._format_conversation_alpaca(conversation)
        else:
            raise ValueError(f"Unsupported conversation format: {self.conversation_format}")
            
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class CustomCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Get the maximum length in the batch
        max_len = min(
            max(len(item['input_ids']) for item in batch),
            self.max_length
        )
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            labels = item['labels']
            
            # Truncate if necessary
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                labels = labels[:max_len]
                
            # Pad to max_len
            pad_length = max_len - len(input_ids)
            if pad_length > 0:
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id
                    
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_length, dtype=attention_mask.dtype)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_length,), -100, dtype=labels.dtype)  # -100 is ignored in loss
                ])
                
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            
        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'labels': torch.stack(batch_labels)
        }


def create_dataloader(
    data_path: Union[str, Path],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    dataset_type: str = "text",  # "text" or "conversation"
    conversation_format: str = "chatml",
    text_column: str = "text",
    **kwargs
) -> DataLoader:
    
    # Create dataset
    if dataset_type == "conversation":
        dataset = ConversationDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            conversation_format=conversation_format
        )
    else:
        dataset = TextDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            text_column=text_column
        )
        
    # Create collator
    collator = CustomCollator(tokenizer, max_length)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        **kwargs
    )
    
    return dataloader


def prepare_sample_data(output_path: str, num_samples: int = 1000):
    """Create sample training data for testing"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data.",
    ]
    
    data = []
    for i in range(num_samples):
        text = sample_texts[i % len(sample_texts)]
        data.append({"text": f"Sample {i}: {text}"})
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            
    logger.info(f"Created sample data with {num_samples} examples at {output_path}")


if __name__ == "__main__":
    # Create sample data for testing
    prepare_sample_data("data/sample_train.jsonl", 1000)
    prepare_sample_data("data/sample_val.jsonl", 100)