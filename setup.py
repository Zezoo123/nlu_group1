import nltk
import os

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading required NLTK data...")
    
    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.expanduser("~/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Download required NLTK data
    required_data = [
        'punkt',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'wordnet',
        'stopwords',
        'omw-1.4'  # Open Multilingual Wordnet
    ]
    
    for data in required_data:
        try:
            nltk.download(data, quiet=True)
            print(f"✓ Downloaded {data}")
        except Exception as e:
            print(f"✗ Error downloading {data}: {str(e)}")
    
    print("\nNLTK data setup complete!")

if __name__ == "__main__":
    download_nltk_data() 