# Natural Language Inference (NLI) Project

This project implements Natural Language Inference (NLI) using a sophisticated hybrid approach that combines TF-IDF and SBERT features with Support Vector Machine (SVM) for robust sentence pair classification.

## Project Overview

This project is part of the COMP34812 coursework, focusing on Natural Language Inference tasks. The implementation leverages a hybrid approach that combines traditional text features (TF-IDF) with modern transformer-based embeddings (SBERT) to achieve robust sentence pair classification through SVM.

## Project Structure

```
.
├── data/                  # Main dataset directory
├── CategoryA/            # Category A implementation
│   ├── data/            # Category A specific data
│   ├── all-MiniLM-L6-v2-local/  # Local SBERT model files
│   ├── svm-SBERT.ipynb  # Hybrid SVM implementation with TF-IDF and SBERT
│   ├── localSBERT.py    # SBERT implementation
│   └── requirements.txt # Project dependencies
├── CategoryC/            # Category C implementation
└── README.md             # Project documentation
```

## Implementation Details

### Hybrid SVM with Combined Features
The implementation uses a sophisticated approach that combines multiple feature extraction methods with SVM:

1. **Feature Extraction**:
   - TF-IDF features: Captures term frequency and importance in the text
   - SBERT embeddings: Provides semantic understanding through transformer-based embeddings
   - Feature combination: Concatenates both feature sets for comprehensive text representation

2. **Classification**:
   - Support Vector Machine (SVM) classifier
   - Optimized hyperparameters for the combined feature space
   - Cross-validation for robust model evaluation

3. **Key Features**:
   - Multi-view learning through feature combination
   - Enhanced semantic understanding through SBERT
   - Statistical significance through TF-IDF
   - Robust classification through SVM

## Setup and Installation

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

2. Install required packages:
```bash
# Navigate to CategoryA directory
cd CategoryA

# Install dependencies
pip install -r requirements.txt
```

3. Download NLTK data:
```python
# Open Python and run:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

4. Download the SBERT model:
```python
# Open Python and run:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

## Usage

### Running the Notebook

1. Navigate to the CategoryA directory:
```bash
cd CategoryA
```

2. Start Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `svm-SBERT.ipynb` in your browser

4. Make sure your data is in the correct location:
   - Place your dataset in the `CategoryA/data` directory
   - Update the data path in the notebook if needed

5. Run all cells in the notebook

Note: The first run might take some time as it downloads the SBERT model and processes the data.

## Model Performance

The project evaluates the hybrid approach using standard metrics:
- Accuracy
- Precision
- Recall
- F1-Score

## Future Improvements

1. Hyperparameter optimization for SVM
2. Experimentation with different feature combinations
3. Advanced feature fusion techniques
4. Ensemble methods with different feature weights
5. Integration of additional feature types

## Contributing

This project is developed as part of the COMP34812 coursework. For any questions or suggestions, please contact the team members.

## License

This project is part of academic coursework and should be used accordingly. 