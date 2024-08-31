import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate Fibonacci sequences using numpy
def generate_fibonacci_sequence(length):
    fib_sequence = np.zeros(length, dtype=int)
    fib_sequence[:2] = [0, 1]
    for i in range(2, length):
        fib_sequence[i] = fib_sequence[i-1] + fib_sequence[i-2]
    return fib_sequence


# Generate arithmetic sequences using numpy
def generate_arithmetic_sequence(start, step, length):
    return start + step * np.arange(length)

# Generate geometric sequences using numpy
def generate_geometric_sequence(start, ratio, length):
    return start * ratio ** np.arange(length)

# Prepare data
sequence_length = 5
X = []
y = []

# Generate a balanced mix of different types of sequences
num_samples_per_type = 100

# Arithmetic sequences (including linear sequences)
for _ in range(num_samples_per_type):
    start = np.random.randint(1, 51)
    step = np.random.randint(1, 5)
    seq = generate_arithmetic_sequence(start, step, sequence_length + 1)
    X.append(seq[:-1])
    y.append(seq[-1])

# Add a few specific linear sequences for better training
specific_linear_sequences = [
    [1, 2, 3, 4, 5, 6],
    [10, 20, 30, 40, 50, 60],
    [5, 10, 15, 20, 25, 30],
    [2, 4, 6, 8, 10, 12],
    [11, 22, 33, 44, 55, 66],
    [7, 14, 21, 28, 35, 42],
    [3, 6, 9, 12, 15, 18],
    [8, 16, 24, 32, 40, 48],
    [13, 26, 39, 52, 65, 78],
    [9, 18, 27, 36, 45, 54]
]

for seq in specific_linear_sequences:
    X.append(seq[:-1])
    y.append(seq[-1])

# Geometric sequences
for _ in range(num_samples_per_type):
    start = np.random.randint(1, 10)
    ratio = np.random.uniform(1, 2)
    seq = generate_geometric_sequence(start, ratio, sequence_length + 1)
    if seq.max() <= 100:  # Ensure the numbers don't exceed 100
        X.append(seq[:-1])
        y.append(seq[-1])

# Fibonacci sequences
for _ in range(num_samples_per_type):
    fib_sequence = generate_fibonacci_sequence(sequence_length + 1)
    start_index = np.random.randint(0, len(fib_sequence) - sequence_length)
    seq = fib_sequence[start_index:start_index + sequence_length + 1]
    X.append(seq[:-1])
    y.append(seq[-1])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# Normalize the data between 0 and 1
X = X / 100.0
y = y / 100.0

# Define the RNN model parameters
input_size = 1  # Processing one input at a time in a sequence
hidden_size = 256  # Increased hidden size
output_size = 1
learning_rate = 0.001  # Smaller learning rate for better convergence

# Initialize weights and biases with He initialization
Wxh = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
Why = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
bh = np.zeros((1, hidden_size))
by = np.zeros((1, output_size))

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU
def relu_derivative(x):
    return (x > 0).astype(float)

# Calculate loss (Mean Squared Error)
def calculate_loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)

# Forward pass
def forward(X):
    h_prev = np.zeros((X.shape[0], hidden_size))
    h_states = []  # Store hidden states for backward pass
    for t in range(sequence_length):
        xt = X[:, t].reshape(-1, 1)  # Reshape each input
        h_prev = relu(np.dot(xt, Wxh) + np.dot(h_prev, Whh) + bh)
        h_states.append(h_prev)
    y_pred = np.dot(h_prev, Why) + by
    return y_pred, h_states

# Backward pass
def backward(X, y, y_pred, h_states):
    global Wxh, Whh, Why, bh, by
    # Calculate loss and derivative
    loss = calculate_loss(y, y_pred)
    dy = 2 * (y_pred - y) / y.size

    # Gradients for Why and by
    dWhy = np.dot(h_states[-1].T, dy)
    dby = np.sum(dy, axis=0, keepdims=True)

    # Gradients for hidden layer
    dh = np.dot(dy, Why.T)

    # Initialize gradients for Wxh, Whh, and bh
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros_like(bh)
    
    dh_next = np.zeros_like(dh)

    # Reverse pass for hidden states
    for t in reversed(range(sequence_length)):
        dh = dh_next + dh * relu_derivative(h_states[t])
        xt = X[:, t].reshape(-1, 1)
        dWxh += np.dot(xt.T, dh)
        if t > 0:
            dWhh += np.dot(h_states[t-1].T, dh)
        dbh += np.sum(dh, axis=0, keepdims=True)
        dh_next = np.dot(dh, Whh.T)

    # Update weights and biases
    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy
    bh -= learning_rate * dbh
    by -= learning_rate * dby

    return loss

# Training loop
epochs = 1000  # More epochs for better training
loss_history = []

for epoch in range(epochs):
    # Shuffle the data for better training
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Forward and backward pass
    y_pred, h_states = forward(X)
    loss = backward(X, y, y_pred, h_states)
    loss_history.append(loss)

    # Print the loss every 100 epochs for debugging
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.5f}")

# Print loss history
print("Loss History:")
for i, loss in enumerate(loss_history):
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.5f}")

# Test the neural network
def predict_next_number():
    print("Enter a sequence of 5 whole numbers separated by spaces:")
    user_input = input().strip().split()
    
    if len(user_input) != 5:
        print("Please enter exactly 5 numbers.")
        return

    try:
        user_sequence = [int(num) for num in user_input]
    except ValueError:
        print("Invalid input. Please enter whole numbers only.")
        return

    # Normalize user input
    normalized_input = np.array(user_sequence) / 100.0
    normalized_input = normalized_input.reshape((1, sequence_length))

    # Predict the next number
    prediction, _ = forward(normalized_input)

    # Denormalize prediction
    denormalized_prediction = prediction * 100
    print(f"Predicted next number: {int(round(denormalized_prediction[0][0]))}")

# Run prediction
while True:
    predict_next_number()
    print("\nDo you want to make another prediction? (y/n)")
    if input().lower() != 'y':
        break