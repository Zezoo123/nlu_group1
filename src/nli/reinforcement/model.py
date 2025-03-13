import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class PolicyNetwork(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dim=256, num_layers=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Additional layers for policy
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=0.1
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # 2 actions: entailment and contradiction
        )
        
        # Value head for state value estimation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_outputs.last_hidden_state
        
        # Apply LSTM
        lstm_output, _ = self.lstm(sequence_output)
        
        # Apply attention
        attn_output, _ = self.attention(
            lstm_output.transpose(0, 1),
            lstm_output.transpose(0, 1),
            lstm_output.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)
        
        # Global average pooling
        pooled_output = torch.mean(attn_output, dim=1)
        
        # Get policy logits and state value
        policy_logits = self.policy_head(pooled_output)
        state_value = self.value_head(pooled_output)
        
        return policy_logits, state_value
    
    def tokenize(self, premise, hypothesis):
        """Tokenize premise and hypothesis for inference."""
        # Combine premise and hypothesis with special tokens
        text = f"{premise} [SEP] {hypothesis}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoding 