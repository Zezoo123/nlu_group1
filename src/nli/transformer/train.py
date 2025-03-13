import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
from collections import Counter

# Set environment variable to disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.nli.transformer.model import TransformerModel
from src.nli.transformer.data import get_data_loaders

def calculate_class_weights(train_loader):
    """Calculate class weights based on dataset distribution."""
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].numpy())
    
    # Count occurrences of each class
    class_counts = Counter(all_labels)
    total_samples = len(all_labels)
    
    # Calculate weights (inverse of frequency)
    class_weights = {
        class_idx: total_samples / (len(class_counts) * count)
        for class_idx, count in class_counts.items()
    }
    
    # Convert to tensor and normalize
    weights = torch.tensor([class_weights[i] for i in range(len(class_counts))])
    weights = weights / weights.sum()
    
    return weights

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(
    model_name="bert-base-uncased",
    hidden_dim=256,
    num_layers=2,
    batch_size=32,
    max_length=128,
    learning_rate=0.001,
    num_epochs=10,
    patience=5,
    min_delta=0.0005,
    device=None
):
    # Check device availability
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    print(f"Using device: {device}")
    if device.type == "mps":
        print("Using Apple Silicon GPU (MPS)")
    elif device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", "transformer", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, validation_loader = get_data_loaders(
        batch_size=batch_size,
        max_length=max_length,
        model_name=model_name
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_loader)
    print("\nClass weights:", class_weights)
    
    # Initialize model
    model = TransformerModel(
        model_name=model_name,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    # Initialize optimizer with weight decay
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Initialize weighted loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    print(f"\nTraining for up to {num_epochs} epochs with early stopping (patience={patience}, min_delta={min_delta})")
    
    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        # Calculate training metrics
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        
        # Validation
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch in validation_loader:
                # Move data to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                total_val_correct += (predictions == labels).sum().item()
                total_val_samples += labels.size(0)
        
        # Calculate validation metrics
        val_loss = total_val_loss / len(validation_loader)
        val_acc = total_val_correct / total_val_samples
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    model.eval()
    
    # Generate classification report
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in validation_loader:
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            predictions = outputs.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Save classification report
    report = classification_report(all_labels, all_predictions)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    # Set device for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_model(device=device) 