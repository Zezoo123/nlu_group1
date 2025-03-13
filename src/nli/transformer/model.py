import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

class TransformerModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dim=256, num_layers=2, num_labels=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Add additional layers
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, num_labels)  # * 2 for bidirectional
        
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get sequence output
        sequence_output = bert_outputs.last_hidden_state
        
        # Process through LSTM
        lstm_out, _ = self.lstm(sequence_output)
        
        # Get final state
        final_state = lstm_out[:, -1, :]
        
        # Apply dropout
        final_state = self.dropout(final_state)
        
        # Classification
        logits = self.fc(final_state)
        return logits 