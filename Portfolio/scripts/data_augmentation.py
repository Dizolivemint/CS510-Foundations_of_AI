# scripts/data_augmentation.py

import pandas as pd
import random
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(sentence, n=2):
    words = sentence.split()
    new_sentence = words.copy()
    random_words = list(set([word for word in words if get_synonyms(word)]))
    random.shuffle(random_words)
    num_replaced = 0
    for word in random_words:
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_sentence = [synonym if w == word else w for w in new_sentence]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_sentence)

def augment_data(df, augment_factor=1):
    augmented_data = []
    for _, row in df.iterrows():
        for _ in range(augment_factor):
            augmented_joke = synonym_replacement(row['Joke'])
            augmented_data.append({
                'Humor Type': row['Humor Type'],
                'Joke': augmented_joke
            })
    return pd.DataFrame(augmented_data)

if __name__ == "__main__":
    # Load original data
    df = pd.read_csv('data/raw/Humor_classifications.csv', sep='|')
    
    # Augment data
    df_augmented = augment_data(df, augment_factor=2)  # Creates 2 augmented versions per joke
    
    # Combine original and augmented data
    df_combined = pd.concat([df, df_augmented], ignore_index=True)
    
    # Save processed data
    df_combined.to_csv('data/processed/augmented_data.csv', index=False, sep='|')
    
    print("Data augmentation completed and saved to 'data/processed/augmented_data.csv'")
