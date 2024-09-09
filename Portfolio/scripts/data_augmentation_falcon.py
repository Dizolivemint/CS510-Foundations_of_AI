import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import logging
from datasets import Dataset
import re  # Regular expressions for cleaning
from dotenv import load_dotenv
import os
from huggingface_hub import login

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Now you can access your environment variables
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=huggingface_token)

# Load Model from Hugging Face
model_name = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Define a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to clean and format text
def clean_text(text):
    # Remove curly braces, quotation marks, and newlines
    text = re.sub(r'[{}"]', '', text)
    text = text.replace('\n', ' ').strip()
    return text

# List of unique humor types extracted from the file
unique_humor_types = [
    'Sarcasm', 'Satire', 'Irony', 'Puns', 'Double Entendre',
    'Observational Humor', 'Dark Humor (Black Comedy)',
    'Self-Deprecating Humor', 'Slapstick', 'Parody',
    'Satirical Commentary', 'Anecdotal Humor', 'Hyperbole', 'Innuendo',
    'Deadpan', 'Absurd Humor', 'Juvenile/Toilet Humor',
    'Sarcastic Hyperbole', 'Sardonic Humor',
    'Situational Comedy (Sitcom Humor)', 'Topical Humor', 'Blue Humor',
    'Political Humor', 'Surreal Humor', 'Mockumentary Style',
    'Impressions/Imitative Humor', 'Character Comedy',
    'Punishment Humor', 'Shock Humor', 'Non-Sequitur', 'Meta Humor',
    'Blunder Humor', 'Anti-Joke', 'Anachronism', 'Cynical Humor',
    'Paraprosdokian', 'Groaners', 'Observational Comedy',
    'Play on Idioms', 'Narrative Jokes', 'Reductio ad Absurdum'
]

# Function to generate jokes for each humor type
def generate_jokes(humor_types, num_jokes=100):
    generated_jokes = []
    for humor_type in humor_types:
        prompt = f"Give me a one to two sentence {humor_type} joke."
        
        # Generate 100 jokes for each category
        responses = generator(
            prompt,
            max_length=100,
            num_return_sequences=num_jokes,
            do_sample=True,
            top_k=40,
            top_p=0.8,
            temperature=0.7,
            truncation=True
        )
        
        # Clean the generated jokes
        jokes = [clean_text(response['generated_text']) for response in responses]
        for joke in jokes:
            generated_jokes.append({'Humor Type': humor_type, 'Joke': joke})
    
    return pd.DataFrame(generated_jokes)

# Convert the Hugging Face dataset to a pandas DataFrame manually
df_jokes = generate_jokes(unique_humor_types)

dataset_jokes = Dataset.from_pandas(df_jokes)

# Convert the Hugging Face dataset to a pandas DataFrame manually
df_augmented = pd.DataFrame(dataset_jokes)

# Save the pandas DataFrame to a CSV file
df_jokes.to_csv('data/processed/augmented_data.csv', sep='|', index=False)

print("Data augmentation completed and saved to 'data/processed/augmented_data.csv'")