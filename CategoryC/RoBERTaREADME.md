# BERT approach to NLI task

This project implements a Natural Language Inference (NLI) model using RoBERTa-large with custom data augmentation and training optimizations. NLI is the task of determining the logical relationship between two sentences (premise and hypothesis).

## Features

- RoBERTa-large model with custom classification head
- Custom data augmentation using WordNet synonyms
- Advanced training techniques including:
  - Mixed precision training
  - OneCycleLR scheduler
  - Early stopping with patience of 3
  - Layer-wise unfreezing (last 4 layers only)
  - Gradient checkpointing
- Comprehensive evaluation metrics

## Project Structure

```
CategoryC/
├── README.md              # Project documentation
├── nli_demo.py           # Main demo script for loading model and making predictions
├── model.py              # Model architecture implementation
├── utils.py             # Utility functions for data handling and evaluation
├── data_augmentor.py    # Data augmentation implementation
├── RoBERTa-notebook.ipynb # Jupyter notebook containing model implementation and training
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Process

The model is trained using the following configuration:
- Dataset: train.csv and dev.csv provided
- Model: RoBERTa-large with custom classification head
- Training Parameters:
  - Batch size: 16 (training), 32 (evaluation)
  - Number of epochs: 5
  - Learning rate: 2e-5
  - Early stopping patience: 3
  - Mixed precision training enabled
  - Data augmentation enabled
  - Warmup ratio: 0.1
  - Weight decay: 0.1
  - Gradient checkpointing enabled
- Training Strategy:
  - Only last 4 layers of RoBERTa are trainable
  - OneCycleLR scheduler for learning rate optimization
  - Cross-entropy loss with AdamW optimizer
  - Gradient clipping and weight decay for regularization

## Usage

The demo script is designed to load a pre-trained model and make predictions on test data:

```bash
python nli_demo.py
```

The script will:
1. Download the pre-trained model from Google Drive
2. Load test data from `../test.csv`
3. Make predictions and save them to `predictions.csv`
4. Display sample predictions in the console

For more detailed information about the model architecture, training process, and evaluation metrics, please refer to the `RoBERTa-model-card.md` file.

## Model Architecture

The model uses a RoBERTa-large backbone with a custom classification head:
- Freezes all layers except the last 4
- Uses a 3-layer MLP with LayerNorm and GELU activation
- Implements dropout for regularization
- Includes attention pooling and residual connections

## Data Augmentation

The project includes a sophisticated data augmentation pipeline:
- Synonym replacement using WordNet
- Controlled augmentation to maintain semantic meaning
- Batch-wise augmentation during training

## Performance

The model achieves:
- Validation Accuracy: 87%
- Validation F1 Score: 88%
- Training Time: ~15 minutes per epoch (on GPU)
- Total Training Time: ~1 hour
- Model Size: 1.42GB

## Requirements

- Python 3.8+
- PyTorch 1.11.0+cu113
- Transformers 4.18.0
- NLTK
- Pandas
- Scikit-learn
- CUDA toolkit
- Hardware:
  - RAM: at least 16 GB
  - Storage: at least 2GB
  - NVIDIA GPU with CUDA support

## License

This project is licensed under the MIT License - see the LICENSE file for details.
