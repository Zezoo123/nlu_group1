import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from datetime import datetime

# Set environment variable to disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.nli.reinforcement.model import PolicyNetwork
from src.nli.reinforcement.data import get_data_loaders
from src.nli.reinforcement.utils import (
    compute_returns,
    compute_gae,
    evaluate_model,
    save_checkpoint,
    load_checkpoint,
    compute_entropy,
    compute_kl_divergence
)

def train_model(
    model_name: str = "bert-base-uncased",
    hidden_dim: int = 256,
    num_layers: int = 2,
    batch_size: int = 32,
    max_length: int = 128,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    gamma: float = 0.99,  # Discount factor
    lambda_: float = 0.95,  # GAE parameter
    epsilon: float = 0.2,  # PPO clip parameter
    c1: float = 1.0,  # Value loss coefficient
    c2: float = 0.01,  # Entropy coefficient
    device: torch.device = None
):
    """
    Train the reinforcement learning model using PPO.
    
    Args:
        model_name: Name of the BERT model to use
        hidden_dim: Hidden dimension of LSTM
        num_layers: Number of LSTM layers
        batch_size: Batch size for training
        max_length: Maximum sequence length
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        gamma: Discount factor for reinforcement learning
        lambda_: GAE parameter
        epsilon: PPO clip parameter
        c1: Value loss coefficient
        c2: Entropy coefficient
        device: Device to train on (CPU/GPU)
    """
    # Set device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    print(f"Using device: {device}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", "reinforcement", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size,
        max_length=max_length,
        model_name=model_name
    )
    
    # Initialize model
    model = PolicyNetwork(
        model_name=model_name,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        total_reward = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get policy logits and state values
            policy_logits, state_values = model(input_ids, attention_mask)
            
            # Sample actions from policy
            action_probs = F.softmax(policy_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)
            
            # Calculate rewards (1 for correct prediction, -1 for incorrect)
            rewards = (actions == labels).float() * 2 - 1
            
            # Calculate returns and advantages
            returns = compute_returns(rewards.tolist(), gamma)
            advantages = compute_gae(rewards.tolist(), state_values.squeeze().tolist(), gamma, lambda_)
            
            # Calculate policy loss (PPO)
            old_log_probs = log_probs.detach()
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(state_values.squeeze(), returns)
            
            # Calculate entropy bonus
            entropy = compute_entropy(policy_logits).mean()
            
            # Total loss
            loss = policy_loss + c1 * value_loss - c2 * entropy
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_reward += rewards.sum().item()
            total_correct += (actions == labels).sum().item()
            total_samples += labels.size(0)
        
        # Calculate metrics
        avg_reward = total_reward / total_samples
        accuracy = total_correct / total_samples
        
        # Validation
        val_acc, val_report = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Reward: {avg_reward:.4f}, Train Acc: {accuracy:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, os.path.join(output_dir, "best_model.pt"))
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
    
    # Load best model for testing
    load_checkpoint(model, optimizer, os.path.join(output_dir, "best_model.pt"))
    test_acc, test_report = evaluate_model(model, test_loader, device)
    
    print("\nTest Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        test_report['true_labels'],
        test_report['predictions'],
        target_names=['entailment', 'contradiction']
    ))

if __name__ == "__main__":
    train_model() 