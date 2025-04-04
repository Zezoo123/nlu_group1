import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerModel(nn.Module):
    def __init__(self, model_name: str = "roberta-large", num_labels: int = 2):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze all layers except the last 4
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last 4 layers
        for layer in self.model.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_output)

        return logits 