import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Randomly initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size)
        self.bh = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs, h_prev):
        # Compute hidden state
        self.h_next = np.tanh(np.dot(self.Wxh, inputs) + np.dot(self.Whh, h_prev) + self.bh)
        return self.h_next

# Binary Cross-Entropy Loss
class BinaryCrossEntropyLoss:
    def forward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.maximum(epsilon, np.minimum(1 - epsilon, y_pred))
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def backward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.maximum(epsilon, np.minimum(1 - epsilon, y_pred))
        dy = - (y_true / y_pred - (1 - y_true) / (1 - y_pred))
        return dy

# Example usage
input_size = 3
hidden_size = 4
seq_length = 5

rnn = SimpleRNN(input_size, hidden_size)
loss_fn = BinaryCrossEntropyLoss()

# Initialize hidden state
h_prev = np.zeros((hidden_size, 1))

# Fake sequential input data and labels (binary sentiment, 0 or 1)
inputs_sequence = np.random.randn(input_size, seq_length)
labels = np.random.randint(2, size=(1, seq_length))

# Training parameters
learning_rate = 0.01
epochs = 100

# Training loop
for epoch in range(epochs):
    # Forward pass
    for t in range(seq_length):
        h_prev = rnn.forward(inputs_sequence[:, t].reshape(-1, 1), h_prev)

    # Calculate loss
    y_pred = rnn.sigmoid(h_prev)
    loss = loss_fn.forward(y_pred, labels)

    # Backward pass
    dy = loss_fn.backward(y_pred, labels)
    dh_next = np.zeros_like(h_prev)

    for t in reversed(range(seq_length)):
        # Backward pass through time
        dh = dy
        dh_next = (1 - h_prev**2) * np.dot(rnn.Whh.T, dh) + dh_next

        # Update weights and
