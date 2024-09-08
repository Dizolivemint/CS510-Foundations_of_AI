import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertModel
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Data Augmentation Function: Synonym Replacement
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_sentence = words.copy()
    random_words = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_words)
    num_replaced = 0
    for word in random_words:
        synonyms = get_synonyms(word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_sentence = [synonym if w == word else w for w in new_sentence]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_sentence)

# Load the dataset
df = pd.read_csv('data/processed/augmented_data.csv', sep='|')

# Augment the dataset with synonym replacement
augmented_jokes = []
for joke in df['Joke']:
    augmented_jokes.append(synonym_replacement(joke, n=2))  # Replace 2 words with synonyms

# Append augmented jokes to the original dataset
df_augmented = df.copy()
df_augmented['Joke'] = augmented_jokes
df = pd.concat([df, df_augmented], ignore_index=True)

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Function to get BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Move back to CPU for numpy
    return embeddings

# Generate BERT embeddings for each joke
df['embeddings'] = df['Joke'].apply(lambda x: get_bert_embeddings(x))

# Convert embeddings into a NumPy array for model training
X = np.vstack(df['embeddings'].values)
y = df['Humor Type']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model and tokenizer
joblib.dump(clf, 'data/models/bert_joke_classifier.pkl')
joblib.dump(tokenizer, 'data/models/bert_tokenizer.pkl')
print("Model and tokenizer saved.")
