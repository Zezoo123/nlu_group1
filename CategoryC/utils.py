import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import List, Dict, Tuple
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from data_augmentor import NLIAugmentor

class NLIDataset(Dataset):
    def __init__(self, premises: List[str], hypotheses: List[str], labels: List[int],
                 tokenizer, max_length: int = 128, augment: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augmentor = NLIAugmentor() if augment else None

        # Store original data
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels

        # Create augmented dataset if needed
        if augment:
            self.augmented_data = self._create_augmented_dataset()
        else:
            self.augmented_data = list(zip(premises, hypotheses, labels))

    def _create_augmented_dataset(self) -> List[Tuple[str, str, int]]:
        augmented_data = []
        for premise, hypothesis, label in zip(self.premises, self.hypotheses, self.labels):
            augmented_pairs = self.augmentor.augment(premise, hypothesis, label)
            augmented_data.extend(augmented_pairs)
        return augmented_data

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        premise, hypothesis, label = self.augmented_data[idx]

        # Combine premise and hypothesis with [SEP] token
        text = f"{premise} [SEP] {hypothesis}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, train_loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'f1': f1
    }

def load_and_predict(model, tokenizer, input_file='test.csv', output_file='predictions.csv'):
    """Load data from CSV, make predictions, and save results."""
    # Load data
    df = pd.read_csv(input_file)
    
    # Create dummy labels (0) for prediction
    dummy_labels = [0] * len(df)
    
    # Create dataset
    dataset = NLIDataset(
        premises=df['premise'],
        hypotheses=df['hypothesis'],
        labels=dummy_labels,
        tokenizer=tokenizer,
        max_length=128,
        augment=False
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    
    # Make predictions
    model.eval()
    predictions = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    # Save predictions
    pd.DataFrame({'prediction': predictions}).to_csv(output_file, index=False)
    
    return predictions 