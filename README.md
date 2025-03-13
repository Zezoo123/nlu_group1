# Natural Language Inference (NLI) Project

This project implements two different approaches for Natural Language Inference (NLI) task:

1. **Transformer-based Model**: A supervised learning approach using BERT for fine-tuning
2. **Reinforcement Learning Model**: A policy gradient approach using LSTM with attention

## Project Structure

```
.
├── data/
│   └── training_data/
│       └── NLI/
│           ├── train.csv
│           └── test.csv
├── outputs/
│   ├── transformer/
│   └── reinforcement/
├── src/
│   └── nli/
│       ├── transformer/
│       │   ├── model.py
│       │   ├── data.py
│       │   └── train.py
│       └── reinforcement/
│           ├── model.py
│           ├── data.py
│           └── train.py
├── run_transformer.py
└── run_reinforcement.py
```

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
```

## Usage

### Training Models

1. For the transformer model:
```bash
python3 run_transformer.py
```

2. For the reinforcement learning model:
```bash
python3 run_reinforcement.py
```

### Model Details

#### Transformer Model
- Uses BERT for fine-tuning
- Supervised learning approach
- Direct classification of sentence relationships

#### Reinforcement Learning Model
- Uses LSTM with attention mechanism
- Policy gradient approach
- Learns through trial and error
- Components:
  - Policy network (actor): Makes decisions about sentence relationships
  - Value network (critic): Estimates state values
  - Attention mechanism: Focuses on relevant parts of sentences

## Output

Both models will:
1. Train on the training data
2. Validate on a validation set
3. Test on the test set
4. Save the best model
5. Print classification reports

The outputs will be saved in the `outputs/` directory with timestamps. 