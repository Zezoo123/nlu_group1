import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def compute_returns(rewards: List[float], gamma: float = 0.99, lambda_: float = 0.95) -> torch.Tensor:
    """
    Compute returns using GAE (Generalized Advantage Estimation).
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        lambda_: GAE parameter
        
    Returns:
        Tensor of computed returns
    """
    returns = []
    running_return = 0
    
    for r in reversed(rewards):
        running_return = r + gamma * running_return
        returns.insert(0, running_return)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns

def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        gamma: Discount factor
        lambda_: GAE parameter
        
    Returns:
        Tensor of computed advantages
    """
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards) - 1)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages

def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, Dict]:
    """
    Evaluate the model on the given data loader.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        
    Returns:
        Tuple of (accuracy, classification report)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            policy_logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(policy_logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    return accuracy, report

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_accuracy: float,
    path: str
) -> None:
    """
    Save a checkpoint of the model.
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        val_accuracy: Current validation accuracy
        path: Path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy
    }, path)

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str
) -> Tuple[int, float]:
    """
    Load a checkpoint of the model.
    
    Args:
        model: The model to load the state into
        optimizer: The optimizer to load the state into
        path: Path to the checkpoint
        
    Returns:
        Tuple of (epoch, validation accuracy)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_accuracy']

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of the policy distribution.
    
    Args:
        logits: Policy logits
        
    Returns:
        Tensor of entropy values
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

def compute_kl_divergence(old_logits: torch.Tensor, new_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between old and new policy distributions.
    
    Args:
        old_logits: Old policy logits
        new_logits: New policy logits
        
    Returns:
        Tensor of KL divergence values
    """
    old_probs = F.softmax(old_logits, dim=-1)
    new_log_probs = F.log_softmax(new_logits, dim=-1)
    kl_div = (old_probs * (torch.log(old_probs + 1e-10) - new_log_probs)).sum(dim=-1)
    return kl_div 