from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from src.nli.transformer.model import get_tokenizer
import pandas as pd
import os

class TransformerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Map labels to indices
        self.label_map = {
            'entailment': 0,
            'neutral': 1,
            'contradiction': 2,
            0: 0,  # Handle numeric labels
            1: 1,
            2: 2
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Combine premise and hypothesis
        text = f"{item['premise']} [SEP] {item['hypothesis']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get label - handle both string and numeric labels
        label = self.label_map[item['label']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_data_loaders(batch_size=32, max_length=128, model_name="bert-base-uncased"):
    # Load local data
    data_dir = os.path.join("data", "training_data", "NLI")
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    dev_data = pd.read_csv(os.path.join(data_dir, "dev.csv"))
    
    # Get tokenizer
    tokenizer = get_tokenizer(model_name)
    
    # Create datasets
    train_dataset = TransformerDataset(train_data, tokenizer, max_length)
    validation_dataset = TransformerDataset(dev_data, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, validation_loader 