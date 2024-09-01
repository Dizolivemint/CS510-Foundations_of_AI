# scripts/feature_extraction.py

import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import joblib

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_bert_embeddings(text_list, tokenizer, model):
    embeddings = []
    for text in tqdm(text_list, desc="Generating BERT embeddings"):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move tensors to GPU if available
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the CLS token's embedding as the representation for the entire sentence
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # Move back to CPU for storage
        embeddings.append(cls_embedding)
    
    return np.array(embeddings)

if __name__ == "__main__":
    # Load augmented data
    df = pd.read_csv('data/processed/augmented_data.csv', sep='|')
    
    # Initialize BERT tokenizer and model, move model to GPU
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    # Generate embeddings
    jokes = df['Joke'].tolist()
    embeddings = get_bert_embeddings(jokes, tokenizer, bert_model)
    
    # Save embeddings and labels
    np.save('data/processed/joke_embeddings.npy', embeddings)
    df['Humor Type'].to_csv('data/processed/labels.csv', index=False)
    
    # Save tokenizer for future use
    joblib.dump(tokenizer, 'data/models/tokenizer.pkl')
    
    print("Feature extraction completed and saved embeddings and labels.")
