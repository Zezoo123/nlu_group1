---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/Zezoo123/nlu_group1

---

# Model Card for p45493za-d99547jh-nli

<!-- Provide a quick summary of what the model is/does. -->

This model performs natural language inference by predicting the relationship
      between a pair of texts (premise and hypothesis) using a combination of SBERT and TF-IDF embeddings.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model combines Sentence-BERT (MiniLM) embeddings with TF-IDF features to
      represent input text pairs. These combined features are used to train a logistic regression classifier
      to identify relationships between premise-hypothesis pairs (entailment, contradiction, neutral).

- **Developed by:** Joseph Hayes & Zeyad Awadalla
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** SVM (SBERT + TF-IDF Features)
- **Finetuned from model [optional]:** sentence-transformers/all-MiniLM-L6-v2

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **Paper or documentation:** https://arxiv.org/abs/1908.10084

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

Natural language inference dataset from /kaggle/input/nlidata/train.csv.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - C (logistic regression): 0.1 (tuned via GridSearchCV)
      - solver: liblinear
      - max_iter: 1000
      - SBERT model: all-MiniLM-L6-v2
      - TF-IDF: binary=True, ngram_range=(1,3), min_df=3
      - combined features: [SBERT | TF-IDF]
    

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: ~5 minutes
      - model size: ~127KB (SBERT) + vectorized features
      - grid search folds: 3
    

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Held-out dev set from /kaggle/input/nlidata/dev.csv containing text pairs and labels.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - Precision (Macro & Weighted)
      - Recall (Macro & Weighted)
      - F1-score (Macro & Weighted)
      - Matthews Correlation Coefficient (MCC)
    

### Results


      - Accuracy: 0.6317
      - Macro-F1: 0.6306
      - Weighted F1: 0.6313
      - MCC: 0.2618
    

## Technical Specifications

### Hardware


      - RAM: 8–16 GB
      - GPU: Not required
      - Storage: ~500MB for model + data
    

### Software


      - sentence-transformers
      - scikit-learn
      - nltk
      - pandas, numpy, tqdm
    

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Model relies on pretrained SBERT embeddings and TF-IDF statistics,
      which may reflect biases present in the original corpora.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The combination of semantic embeddings and statistical features
      provided improved performance over the baseline. Hyperparameters were selected using GridSearchCV.
