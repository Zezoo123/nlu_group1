# SVM-BERT Natural Language Inference (NLI) Classification

This directory contains the implementation of a hybrid SVM-BERT model for Natural Language Inference (NLI) classification. The model combines the power of BERT embeddings with SVM classification to determine the relationship between premise and hypothesis pairs.

## Directory Structure

```
CategoryA/
├── svm-SBERT.ipynb        # Training notebook for the SVM-BERT model
├── svm_demo.py            # Demo script for making predictions on test data
├── Model/                 # Contains the trained SVM model and vectorizer
├── all-MiniLM-L6-v2-local/ # Local BERT model for sentence embeddings
├── training_data/        # Training data directory
├── Test/                 # Test data directory
└── predictions.csv       # Model predictions output
```

## Implementation Details

1. **Model Architecture**
   - The implementation uses a hybrid approach combining:
     - Sentence-BERT (SBERT) embeddings using the all-MiniLM-L6-v2 model
     - TF-IDF features for additional text representation
     - Support Vector Machine (SVM) classifier

2. **Training Process** (`svm-SBERT.ipynb`)
   - Data preprocessing including text cleaning and lemmatization
   - Feature extraction using SBERT embeddings and TF-IDF
   - SVM model training and hyperparameter tuning
   - Model evaluation and performance metrics
   - Saving the trained model and vectorizer

3. **Prediction Pipeline** (`svm_demo.py`)
   - Loads the trained SVM model and TF-IDF vectorizer
   - Preprocesses new test data using the same pipeline
   - Generates predictions for test instances
   - Outputs predictions to a CSV file

## Usage

1. **Training the Model**
   - Open and run `svm-SBERT.ipynb` to train the model
   - The notebook will save the trained model in the `Model/` directory

2. **Making Predictions**
   - Place your test data in the correct location (same directory as `test.csv`)
   - Run the demo script:
   ```bash
   python svm_demo.py
   ```
   - The script will generate `Group_55_nli.csv` with predictions

## Data Format

- Input data should be in CSV format with 'premise' and 'hypothesis' columns
- The model predicts the relationship between premise-hypothesis pairs
- Output predictions are saved in CSV format

## Dependencies

- pandas
- numpy
- scikit-learn
- sentence-transformers
- nltk
- joblib
- Local BERT model 

## Models Used

### Sentence-BERT (SBERT)
- **Model**: all-MiniLM-L6-v2
- **Source**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Paper**: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- **Authors**: Nils Reimers and Iryna Gurevych
- **Year**: 2019

The model was downloaded and saved locally using:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('all-MiniLM-L6-v2-local')  # Saves to a local folder
```

This citation includes:
1. The specific model used
2. Link to the model on Hugging Face
3. Reference to the original SBERT paper
4. The code used to save it locally

### TF-IDF
- **Technique**: Term Frequency-Inverse Document Frequency
- **Paper**: Question classification using support vector machines
- **Authors**: Dell Zhang and Wee Sun Lee
- **Conference**: Proceedings of the 26th annual international ACM SIGIR conference on Research and development in information retrieval
- **Year**: 2003
- **Pages**: 26-32

The TF-IDF implementation follows the approach outlined in Zhang and Lee's paper, where they demonstrated its effectiveness when combined with SVMs for text classification tasks. This technique helps capture the importance of words in the context of the entire document collection, providing complementary features to the BERT embeddings.

## Model Performance

The model has been trained and optimized for NLI classification tasks, combining the semantic understanding capabilities of BERT with the robust classification abilities of SVM.

## Output

The model generates predictions in the following format:
- `Group_55_nli.csv`: Contains predictions for the test dataset
