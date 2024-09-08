# joke_synonym_generator.py
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
import pandas as pd
import random

# Download necessary NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

# Function to get synonyms for contextually replaceable words
def get_contextual_synonyms(word, pos):
    synsets = wordnet.synsets(word)
    synonyms = []
    for syn in synsets:
        if syn.pos() == pos:  # Match part of speech
            for lemma in syn.lemmas():
                if lemma.name() != word:  # Avoid identity replacement
                    synonyms.append(lemma.name())
    return synonyms

# Function to replace words with synonyms based on context
def contextual_synonym_replacement(sentence):
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    new_sentence = []

    for word, tag in pos_tags:
        if tag.startswith('JJ') or tag.startswith('RB') or tag.startswith('VB'):  # Adjectives, adverbs, and verbs
            synonyms = get_contextual_synonyms(word, tag[:2].lower())
            if synonyms:
                new_word = random.choice(synonyms)
                new_sentence.append(new_word)
            else:
                new_sentence.append(word)
        else:
            new_sentence.append(word)
    
    return ' '.join(new_sentence)

# Function to expand jokes with synonyms
def expand_with_synonyms(input_file, output_file):
    df = pd.read_csv(input_file, sep='|')
    expanded_jokes = []

    for index, row in df.iterrows():
        original_joke = row['Joke']
        new_joke = contextual_synonym_replacement(original_joke)
        expanded_jokes.append([row['Humor Type'], new_joke])

    new_df = pd.DataFrame(expanded_jokes, columns=['Humor Type', 'Joke'])
    new_df.to_csv(output_file, sep='|', index=False)

if __name__ == "__main__":
    input_path = "data/raw/humor_classifications.csv"
    output_path = "data/processed/synonym_expanded_jokes.csv"
    expand_with_synonyms(input_path, output_path)
