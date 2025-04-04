# Natural Language Inference (NLI) Task - Multiple Approaches

This directory contains two different implementations for the Natural Language Inference (NLI) task. Each implementation takes a unique approach to solving the problem of determining the logical relationship between premise and hypothesis pairs.

## Directory Structure

```
Deliverable2/
├── CategoryA/           # SVM-BERT Hybrid Approach
├── CategoryC/           # RoBERTa-based Approach
└── test.csv            # Test dataset for evaluation
```

## Approach Comparison

### CategoryA: SVM-BERT Hybrid Approach
- **Architecture**: Combines SBERT embeddings with TF-IDF features and SVM classifier
- **Key Features**:
  - Uses all-MiniLM-L6-v2 for sentence embeddings
  - Incorporates TF-IDF features for enhanced text representation
  - Leverages SVM for robust classification
  - Lightweight and efficient inference
- **Implementation**: 
  - Training notebook: `svm-SBERT.ipynb`
  - Demo script: `svm_demo.py`
- **Output**: `Group_55_nli.csv`

### CategoryC: RoBERTa-based Approach
- **Architecture**: Fine-tuned RoBERTa-large with custom classification head
- **Key Features**:
  - Custom data augmentation using WordNet synonyms
  - Mixed precision training
  - Layer-wise unfreezing (last 4 layers)
  - Gradient checkpointing
  - Advanced training techniques (OneCycleLR, Early stopping)
- **Implementation**:
  - Training notebook: `RoBERTa-notebook.ipynb`
  - Demo script: `nli_demo.py`
- **Performance**:
  - Validation Accuracy: 87%
  - Validation F1 Score: 88%

## Usage

### Running CategoryA (SVM-BERT)
1. Install dependencies:
```bash
cd CategoryA
pip install -r requirements.txt
```
2. Run the demo script:
```bash
python svm_demo.py
```

### Running CategoryC (RoBERTa)
1. Install dependencies:
```bash
cd CategoryC
pip install -r requirements.txt
```
2. Run the demo script:
```bash
python nli_demo.py
```

## Model Citations

### CategoryA Models
1. **Sentence-BERT (SBERT)**
   - Model: all-MiniLM-L6-v2
   - Authors: Nils Reimers and Iryna Gurevych
   - Paper: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

2. **TF-IDF Implementation**
   - Paper: Question classification using support vector machines
   - Authors: Dell Zhang and Wee Sun Lee
   - Conference: ACM SIGIR 2003
   - Pages: 26-32

### CategoryC Models
- **RoBERTa**
  - Base Model: RoBERTa-large
  - Custom implementation with advanced training techniques
  - Enhanced with WordNet-based data augmentation

## Hardware Requirements

### CategoryA (SVM-BERT)
- Standard CPU machine
- Minimum 8GB RAM
- No specific GPU requirements

### CategoryC (RoBERTa)
- NVIDIA GPU with CUDA support
- RAM: at least 16 GB
- Storage: at least 2GB
- CUDA toolkit

## Test Data

Both implementations use the same test dataset (`test.csv`) located in the root directory. The file should contain:
- 'premise' column: The premise text
- 'hypothesis' column: The hypothesis text to be evaluated against the premise

## Output Format

Both implementations generate prediction files in CSV format containing the model's predictions for the NLI relationships between premise-hypothesis pairs. 
