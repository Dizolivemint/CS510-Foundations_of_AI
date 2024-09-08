

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
python scripts/data_augmentation.py

# Step 2: Feature Extraction (with GPU)
python scripts/feature_extraction.py

# Step 3: Model Training (with GPU if using a neural network)
python scripts/train_model.py

# Step 4: Classify a Joke
python scripts/classify_joke.py "Why don't skeletons fight each other? They don't have the guts!"
```

### **3. Download Spacy Package**
```python -m spacy download en_core_web_sm```