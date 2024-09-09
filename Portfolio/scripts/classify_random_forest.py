import numpy as np
import torch
import joblib
from transformers import BertModel
import argparse

# Function to get BERT embeddings
def get_bert_embedding(text, tokenizer, model, device):
    # Tokenize the input text and move to the appropriate device (CPU/GPU)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(device)
    
    # Generate BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the [CLS] token embedding (typically used for classification tasks)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    
    return cls_embedding

# Function to classify the joke
def classify_joke(joke_text):
    # Load the tokenizer and classifier
    tokenizer = joblib.load('data/models/tokenizer.pkl')
    classifier = joblib.load('data/models/bert_joke_classifier.pkl')
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the BERT model and move it to the appropriate device
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    # Generate the BERT embedding for the joke
    embedding = get_bert_embedding(joke_text, tokenizer, bert_model, device)
    
    # Predict the humor type using the classifier
    prediction = classifier.predict([embedding])[0]
    probabilities = classifier.predict_proba([embedding])[0]
    
    # Create a probability distribution for each humor type
    classes = classifier.classes_
    prob_distribution = dict(zip(classes, probabilities * 100))
    sorted_probs = dict(sorted(prob_distribution.items(), key=lambda item: item[1], reverse=True))
    
    return prediction, sorted_probs

# Main script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify the type of humor in a joke.")
    parser.add_argument('joke', type=str, help='The joke text to classify.')
    args = parser.parse_args()
    
    # Classify the joke
    predicted_type, probabilities = classify_joke(args.joke)
    
    # Output the results
    print(f"\nJoke: {args.joke}")
    print(f"\nPredicted Humor Type: {predicted_type}\n")
    print("Probability Distribution:")
    for humor_type, prob in probabilities.items():
        print(f"{humor_type}: {prob:.2f}%")
