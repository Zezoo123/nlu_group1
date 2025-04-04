import torch
from transformers import AutoTokenizer
from model import TransformerModel
from utils import NLIDataset, load_and_predict
import pandas as pd
import os
import gdown
import tempfile

def download_model_from_drive(file_id: str, local_path: str) -> bool:
    """Download a file from Google Drive using its file ID."""
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, local_path, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def run_demo():
    """Run the NLI demo using a pre-trained model from Google Drive."""
    print("Starting NLI Demo with Pre-trained Model...\n")
    
    # Google Drive file ID for the model
    drive_file_id = "18WOfGjX3_0qed9FCD14Fi7_VypjW7JNE"  # Replace with your Google Drive file ID
    
    # Create a temporary file to store the downloaded model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        model_path = tmp_file.name
    
    # Download the model
    print("Downloading pre-trained model from Google Drive...")
    if not download_model_from_drive(drive_file_id, model_path):
        print("Failed to download the model. Please check the file ID and your internet connection.")
        return
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    # Load model and tokenizer
    print("Loading pre-trained model...")
    model = TransformerModel(model_name="roberta-large")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    
    # Check for test file in Test folder
    test_file = "../test.csv"
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found!")
        return
    
    # Make predictions
    print(f"\nMaking predictions on {test_file}...")
    predictions = load_and_predict(
        model=model,
        tokenizer=tokenizer,
        input_file=test_file,
        output_file='predictions.csv'
    )
    
    # Save predictions to CSV
    pd.DataFrame({'prediction': predictions}).to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")
    
    # Clean up the temporary file
    try:
        os.unlink(model_path)
    except:
        pass

if __name__ == "__main__":
    run_demo() 