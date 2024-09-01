from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the token embeddings (excluding the [CLS] and [SEP] tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Convert jokes into BERT embeddings
df['embeddings'] = df['Joke'].apply(lambda x: get_bert_embeddings(x))

# Convert the embeddings into a format suitable for model training
X = np.vstack(df['embeddings'].values)
y = df['Humor Type']
