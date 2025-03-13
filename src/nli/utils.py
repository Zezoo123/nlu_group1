import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_training_history(history, save_path):
    """Save training history to a file."""
    with open(save_path, 'w') as f:
        for epoch, metrics in history.items():
            f.write(f"Epoch {epoch}:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")

def load_model(model, model_path, device):
    """Load model from checkpoint."""
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def predict_single_example(model, tokenizer, premise, hypothesis, device):
    """Predict the relationship between a premise and hypothesis."""
    model.eval()
    
    # Prepare input
    text = f"{premise} [SEP] {hypothesis}"
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = outputs.argmax(dim=1).item()
    
    # Map prediction to label
    label_map = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }
    
    return label_map[prediction] 