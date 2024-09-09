import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class JokeClassifierNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(JokeClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Load embeddings and labels
    X = np.load('data/processed/joke_embeddings.npy')
    label_df = pd.read_csv('data/processed/labels.csv')
    y, category_names = label_df['Humor Type'].factorize()

    # Convert to tensors and create DataLoader
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize the neural network
    input_size = X_tensor.shape[1]
    num_classes = len(np.unique(y))  # Dynamically calculate number of classes
    model = JokeClassifierNN(input_size, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Map indices back to humor categories
    y_true_categories = [category_names[i] for i in y_true]
    y_pred_categories = [category_names[i] for i in y_pred]

    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_true_categories, y_pred_categories, target_names=category_names))
    
    # Save the trained model
    torch.save(model.state_dict(), 'data/models/joke_classifier_nn.pth')
    
    # Save the category names for inference
    joblib.dump(category_names, 'data/models/humor_classes.pkl')
    
    print("Neural network model training and class saving completed.")
