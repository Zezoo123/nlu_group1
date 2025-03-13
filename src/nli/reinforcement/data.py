import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

class NLIDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        vocab: Dict[str, int],
        max_length: int = 50,
        is_training: bool = True
    ):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        self.is_training = is_training
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data.iloc[idx]
        
        # Tokenize and convert to IDs
        premise_words = word_tokenize(str(item['premise']).lower())
        hypothesis_words = word_tokenize(str(item['hypothesis']).lower())
        
        premise_ids = [self.vocab['<BOS>']] + [self.vocab.get(word, self.vocab['<UNK>']) for word in premise_words] + [self.vocab['<EOS>']]
        hypothesis_ids = [self.vocab['<BOS>']] + [self.vocab.get(word, self.vocab['<UNK>']) for word in hypothesis_words] + [self.vocab['<EOS>']]
        
        # Pad sequences
        premise_ids = self._pad_sequence(premise_ids)
        hypothesis_ids = self._pad_sequence(hypothesis_ids)
        
        # Convert to tensors
        premise_ids = torch.tensor(premise_ids, dtype=torch.long)
        hypothesis_ids = torch.tensor(hypothesis_ids, dtype=torch.long)
        
        # Get label
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return {
            'premise_ids': premise_ids,
            'hypothesis_ids': hypothesis_ids,
            'labels': label
        }
    
    def _pad_sequence(self, sequence: List[int]) -> List[int]:
        """Pad sequence to max_length."""
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        return sequence + [self.vocab['<PAD>']] * (self.max_length - len(sequence))

def get_data_loaders(
    batch_size: int = 64,
    max_length: int = 50,
    min_freq: int = 2,
    max_vocab_size: int = 50000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        batch_size: Batch size for data loaders
        max_length: Maximum sequence length
        min_freq: Minimum word frequency for vocabulary
        max_vocab_size: Maximum vocabulary size
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, vocab)
    """
    # Load data
    train_data = pd.read_csv('data/training_data/NLI/train.csv')
    test_data = pd.read_csv('data/training_data/NLI/test.csv')
    
    # Create vocabulary from training data
    vocab = create_vocabulary(
        train_data,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_length
    )
    
    # Split training data into train and validation
    train_size = int(len(train_data) * train_ratio)
    val_size = int(len(train_data) * val_ratio)
    
    train_dataset = NLIDataset(
        train_data[:train_size],
        vocab,
        max_length=max_length,
        is_training=True
    )
    
    val_dataset = NLIDataset(
        train_data[train_size:train_size + val_size],
        vocab,
        max_length=max_length,
        is_training=False
    )
    
    test_dataset = NLIDataset(
        test_data,
        vocab,
        max_length=max_length,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader, vocab 