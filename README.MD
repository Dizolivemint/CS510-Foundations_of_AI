# CS510 - Foundations of AI

## AI Humor - Porfolio Project

### **1. Ensure CUDA-Enabled PyTorch Installation**

First, make sure you have a CUDA-enabled version of PyTorch installed. This allows BERT (via the `transformers` library) and any deep learning operations to utilize the GPU.

If you haven’t installed PyTorch with CUDA, you can do so with:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

This command installs PyTorch with CUDA 11.8 support. Check the [PyTorch installation page](https://pytorch.org/get-started/locally/) for different versions if you have a different CUDA version.


**Note**: The exact torch version with CUDA should be adjusted based on your system’s CUDA version. The example assumes CUDA 11.8.

### **2. Verification and Testing**

To ensure that the GPU is being utilized:

- **Check PyTorch’s Device**: Ensure that `torch.cuda.is_available()` returns `True`.
- **Monitor GPU Usage**: Use tools like `nvidia-smi` to monitor GPU usage during feature extraction and training.

```bash
nvidia-smi
```

```bash
# Ensure GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Step 1: Data Augmentation
# NLTK performing Synonym Replacement
python scripts/data_augmentation_nltk.py

# Large Language Model
# Mistral
python scripts/data_augmentation_mistral.py
# GPT2
python scripts/data_augmentation_gpt2.py
# Falcon
python scripts/data_augmentation_falcon.py

# Step 2: Feature Extraction (with GPU)
python scripts/feature_extraction.py

# Step 3: Model Training (with GPU if using a neural network)
# Random Forest
python scripts/train_random_forest.py
# Neural Network
python scripts/train_neural_network.py

# Step 4: Classify a Joke
# Random Forest
python scripts/classify_random_forest.py "Why don't skeletons fight each other? They don't have the guts!"
# Neural Network
python scripts/classify_neural_network.py "Why don't skeletons fight each other? They don't have the guts!"

```

### **3. Download Spacy Model**

```python -m spacy download en_core_web_sm```

### **4. Sign Up or Log In to Hugging Face**

- Make sure you are logged in to Hugging Face and that you have accepted the conditions of the gated model's license. You can do that by visiting the model’s page: [Mistral 7B Model](https://huggingface.co/mistralai/Mistral-7B-v0.1).

### 5. **Set Up Authentication with Hugging Face Hub**

- You’ll need to authenticate your Hugging Face account through your environment to access gated models.

- First, get your Hugging Face API token:
     1. Go to [Hugging Face account tokens](https://huggingface.co/settings/tokens).
     2. Copy your personal access token.

To create an `.env` file for storing sensitive information (like API tokens) and ensure it is not included in your version control system, follow these steps:

### 6. **Create the `.env` File**

In your project directory, create a new file named `.env`. This file will store environment variables in key-value format.

Paste your Hugging Face Token

```HUGGINGFACE_TOKEN=your_huggingface_token_here```
