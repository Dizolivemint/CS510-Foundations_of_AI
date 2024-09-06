import pandas as pd
import json

# Load your dataset
df = pd.read_csv('data/raw/Humor_classifications.csv', sep='|')

# Create input-output pairs
data = []
for _, row in df.iterrows():
    humor_type = row['Humor Type']
    joke = row['Joke']
    prompt = f"{humor_type}: "
    data.append({"prompt": prompt, "completion": joke})

# Save as JSON
with open('data/raw/humor_data.json', 'w') as f:
    json.dump(data, f, indent=4)
