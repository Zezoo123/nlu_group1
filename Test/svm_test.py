import pandas as pd
import numpy as np
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# Load saved model and TF-IDF vectorizer
saved_data = joblib.load('Models/svm_model.joblib')
model = saved_data['model']
tfidf_vectorizer = saved_data['tfidf_vectorizer']


test_df = pd.read_csv('Data/Input/test.csv')

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
sbert_model = SentenceTransformer('../CategoryA/all-MiniLM-L6-v2-local')  # Use same path as training

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    text = re.sub(r'([.,!?])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if len(token) > 1]
    return ' '.join(processed_tokens)

# Apply preprocessing exactly like before
test_df['premise'] = test_df['premise'].apply(preprocess_text)
test_df['hypothesis'] = test_df['hypothesis'].apply(preprocess_text)
test_df['text'] = '[P] ' + test_df['premise'] + ' [H] ' + test_df['hypothesis']

# Vectorize using exact same pipeline
X_sbert = sbert_model.encode(test_df['text'], show_progress_bar=True)
X_tfidf = tfidf_vectorizer.transform(test_df['text'])
X_combined = np.concatenate([X_sbert, X_tfidf.toarray()], axis=1)

# Predict
predictions = model.predict(X_combined)

# Save
# pd.DataFrame({'prediction': predictions}).to_csv('Data/Output/predictions.csv', index=False, header=True)
# print("predictions.csv saved")

pd.DataFrame({'prediction': predictions}).to_csv('Data/Output/Group_55_nli.csv', index=False, header=True)
print("Group_55_nli.csv saved")