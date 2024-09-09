import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
from nltk.corpus import brown
from nltk import pos_tag, word_tokenize
from collections import Counter
import spacy
nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')

# Load spaCy's English model for sentence validation
nlp = spacy.load("en_core_web_sm")
  
# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    else:
        return None

# Function to validate if the replacement is appropriate
def is_valid_replacement(original_word, replacement_word, sentence):
    # Ensure both words have the same part of speech using NLTK
    original_tag = nltk.pos_tag([original_word])[0][1]
    replacement_tag = nltk.pos_tag([replacement_word])[0][1]

    if original_tag != replacement_tag:
        return False

    # Validate sentence structure using spaCy after replacement
    modified_sentence = sentence.replace(original_word, replacement_word)
    doc = nlp(modified_sentence)

    # Basic check to ensure that the sentence still parses correctly
    return len(list(doc.sents)) > 0


# Function to filter out inappropriate synonyms
def get_filtered_synonyms(word, pos):
    synonyms = []
    synsets = wordnet.synsets(word, pos=pos)
    
    for synset in synsets:
        for lemma in synset.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            # Filter out archaic, weird words, or the original word itself
            if synonym != word and len(synonym.split()) == 1 and all(c.isalpha() for c in synonym):
              synonyms.append(synonym)
    return list(set(synonyms))

# Contextual synonym replacement with limit on replacements
def contextual_synonym_replacement(sentence, humor_type, replace_ratio=0.2):
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    
    new_sentence = []
    num_words = len(words)
    num_replaced = 0

    # Only log if the humor type is "Sarcasm"
    if humor_type == "Sarcasm":
        print(f"Logging for sarcasm category. Original sentence: {sentence}")
        for word, tag in pos_tags:
            print(f"Word: {word}, POS Tag: {tag}")  # Console log the word and its POS tag

    for word, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)  # Map to WordNet POS
        if wordnet_pos and random.random() < replace_ratio:  # Replace only a fraction of words
            synonyms = get_filtered_synonyms(word, wordnet_pos)  # Ensure matching POS
            if synonyms:
                for synonym in synonyms:
                    # Validate the replacement before applying it
                    if is_valid_replacement(word, synonym, sentence):
                        new_word = synonym
                        new_sentence.append(new_word)
                        num_replaced += 1
                        break
                else:
                    # No valid replacement, keep the original word
                    new_sentence.append(word)
            else:
                new_sentence.append(word)
        else:
            new_sentence.append(word)

        # Limit the number of replacements
        if num_replaced > (replace_ratio * num_words):
            break
    
    return ' '.join(new_sentence)

# Use the Brown corpus, which is U.S.-based, for word frequency data
word_freq = Counter(brown.words())

# Function to calculate similarity between original and replaced sentences
def is_semantically_similar(original_sentence, modified_sentence, threshold=0.8):
    original_doc = nlp(original_sentence)
    modified_doc = nlp(modified_sentence)
    similarity = original_doc.similarity(modified_doc)
    
    return similarity >= threshold
  
def get_filtered_synonyms_us(word, pos):
    synonyms = []
    synsets = wordnet.synsets(word, pos=pos)

    for synset in synsets:
        for lemma in synset.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            # Only use common U.S. English words and avoid rare terms
            if synonym != word and word_freq[synonym] > 100:  # Adjust threshold as needed
                synonyms.append(synonym)
    return list(set(synonyms))

# Contextual synonym replacement with semantic similarity check
def contextual_synonym_replacement_us(sentence, humor_type, replace_ratio=0.2, similarity_threshold=0.8):
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    
    new_sentence = []
    num_words = len(words)
    num_replaced = 0

    for word, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)  # Map to WordNet POS
        if wordnet_pos and random.random() < replace_ratio:  # Replace only a fraction of words
            synonyms = get_filtered_synonyms_us(word, wordnet_pos)  # Ensure U.S.-centric synonyms
            if synonyms:
                for synonym in synonyms:
                    # Temporarily replace the word
                    temp_sentence = sentence.replace(word, synonym)
                    # Check if the modified sentence retains semantic similarity
                    if is_semantically_similar(sentence, temp_sentence, similarity_threshold):
                        new_word = synonym
                        new_sentence.append(new_word)
                        num_replaced += 1
                        break
                else:
                    # No valid replacement, keep the original word
                    new_sentence.append(word)
            else:
                new_sentence.append(word)
        else:
            new_sentence.append(word)

        # Limit the number of replacements
        if num_replaced > int(replace_ratio * num_words):
            break
    
    return ' '.join(new_sentence)

# Synonym replacement based on random words (same as before but with filtering)
def synonym_replacement(sentence, n=2):
    words = sentence.split()
    new_sentence = words.copy()
    random_words = list(set([word for word in words if get_filtered_synonyms(word, 'n')]))  # Use nouns for random replacements
    random.shuffle(random_words)
    num_replaced = 0
    for word in random_words:
        synonyms = get_filtered_synonyms(word, 'n')
        if synonyms:
            synonym = random.choice(synonyms)
            new_sentence = [synonym if w == word else w for w in new_sentence]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_sentence)

# Function to augment data using both techniques
def augment_data(df, augment_factor=1, use_contextual=False):
    augmented_data = []
    for _, row in df.iterrows():
        for _ in range(augment_factor):
            if use_contextual:
                augmented_joke = contextual_synonym_replacement_us(row['Joke'], row['Humor Type'])
            else:
                augmented_joke = synonym_replacement(row['Joke'])
            augmented_data.append({
                'Humor Type': row['Humor Type'],
                'Joke': augmented_joke
            })
    return pd.DataFrame(augmented_data)

if __name__ == "__main__":
    # Load original data
    input_file = 'data/raw/humor_classifications.csv'
    output_file = 'data/processed/augmented_data.csv'
    
    df = pd.read_csv(input_file, sep='|')
    
    # Augment data with both techniques
    df_random_augmented = augment_data(df, augment_factor=2, use_contextual=False)  # Random synonym replacement
    df_contextual_augmented = augment_data(df, augment_factor=2, use_contextual=True)  # Contextual synonym replacement
    
    # Combine original and augmented data
    df_combined = pd.concat([df, df_random_augmented, df_contextual_augmented], ignore_index=True)
    
    # Save processed data
    df_combined.to_csv(output_file, index=False, sep='|')
    
    print(f"Data augmentation completed and saved to '{output_file}'")
