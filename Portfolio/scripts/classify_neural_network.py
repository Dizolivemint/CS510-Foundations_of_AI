import numpy as np
import torch
import joblib
from transformers import BertModel
import argparse

# Neural network class (must match the architecture used during training)
class JokeClassifierNN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(JokeClassifierNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

# Function to classify the joke using the neural network
def classify_joke(joke_text):
    # Load the tokenizer and neural network classifier
    tokenizer = joblib.load('data/models/tokenizer.pkl')
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the BERT model and move it to the appropriate device
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    # Generate the BERT embedding for the joke
    embedding = get_bert_embedding(joke_text, tokenizer, bert_model, device)
    embedding_tensor = torch.tensor([embedding], dtype=torch.float32).to(device)
    
    # Load the trained neural network
    model = JokeClassifierNN(input_size=embedding.shape[0], num_classes=44).to(device)  # Adjust num_classes if necessary
    model.load_state_dict(torch.load('data/models/joke_classifier_nn.pth'))
    model.eval()
    
    # Predict the humor type using the neural network
    with torch.no_grad():
        outputs = model(embedding_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    
    # Load the classes (make sure they match those from training)
    classes = joblib.load('data/models/humor_classes.pkl')  # Assuming you saved classes during training
    predicted_idx = np.argmax(probabilities)
    predicted_type = classes[predicted_idx]
    
    # Create a probability distribution for each humor type
    prob_distribution = dict(zip(classes, probabilities * 100))
    sorted_probs = dict(sorted(prob_distribution.items(), key=lambda item: item[1], reverse=True))
    
    return predicted_type, sorted_probs

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
