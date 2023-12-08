import numpy as np


class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Parameters
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        self.Wy = np.random.randn(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def binary_cross_entropy_loss(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.maximum(epsilon, np.minimum(1 - epsilon, y_pred))
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def forward(self, x):
        # Unpack input
        h_prev, c_prev, x = x

        # Concatenate previous hidden state and input
        concat = np.concatenate((h_prev, x), axis=0)

        # Forget gate
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)

        # Input gate
        i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)

        # Cell state update
        c_tilde = self.tanh(np.dot(self.Wc, concat) + self.bc)
        c = f * c_prev + i * c_tilde

        # Output gate
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

        # Hidden state
        h = o * self.tanh(c)

        # Output
        y = np.dot(self.Wy, h) + self.by

        # Cache values for backpropagation
        cache = (h_prev, c_prev, x, concat, f, i, c_tilde, c, o, h)

        return y, cache

    def backward(self, dy, caches):
        gradients = {}

        # Unpack the last cache
        h_prev, c_prev, x, concat, f, i, c_tilde, c, o, h = caches[-1]

        # Gradient of the loss with respect to y
        gradients['dWy'] = np.dot(dy, h.T)
        gradients['dby'] = np.sum(dy, axis=1, keepdims=True)

        # Initialize gradients for the next time step
        dh_next = np.zeros_like(h)
        dc_next = np.zeros_like(c)

        for t in reversed(range(len(caches))):
            h_prev, c_prev, x, concat, f, i, c_tilde, c, o, h = caches[t]

            # Gradient of the loss with respect to h
            dh = np.dot(self.Wy.T, dy) + dh_next

            # Gradient of the loss with respect to o
            do = dh * self.tanh(c) * o * (1 - o)
            gradients['dWo'] = np.dot(do, concat.T)
            gradients['dbo'] = np.sum(do, axis=1, keepdims=True)

            # Gradient of the loss with respect to c
            dc = dh * o * (1 - self.tanh(c) ** 2) + dc_next

            # Gradient of the loss with respect to c_tilde
            dc_tilde = dc * i
            gradients['dWc'] = np.dot(dc_tilde, concat.T)
            gradients['dbc'] = np.sum(dc_tilde, axis=1, keepdims=True)

            # Gradient of the loss with respect to i
            di = dc * c_tilde
            gradients['dWi'] = np.dot(di, concat.T)
            gradients['dbi'] = np.sum(di, axis=1, keepdims=True)

            # Gradient of the loss with respect to f
            df = dc * c_prev
            gradients['dWf'] = np.dot(df, concat.T)
            gradients['dbf'] = np.sum(df, axis=1, keepdims=True)

            # Gradient of the loss with respect to concat
            dconcat = (np.dot(self.Wf.T, df) +
                       np.dot(self.Wi.T, di) +
                       np.dot(self.Wc.T, dc_tilde) +
                       np.dot(self.Wo.T, do))

            # Gradients for the next time step
            dh_prev = dconcat[:self.hidden_size, :]
            dc_prev = f * dc

            # Update dh_next and dc_next for the next iteration
            dh_next = dh_prev

            # Update gradients
            for param in ['Wf', 'Wi', 'Wc', 'Wo', 'bf', 'bi', 'bc', 'bo']:
                gradients['d' + param] = gradients.get(
                    'd' + param, 0) + dconcat[self.hidden_size:, :]
                gradients['d' + param] = gradients.get(
                    'd' + param, 0) + dconcat[:self.hidden_size, :]

        return gradients


# Example training loop with loss calculation
input_size = 10
hidden_size = 5
output_size = 1

# Assuming binary classification (positive/negative sentiment)
lstm = LSTM(input_size, hidden_size, output_size)

# Hyperparameters
learning_rate = 0.01
epochs = 100
sequence_length = 3

# Generate synthetic data for training
input_sequence = [np.random.randn(input_size, 1)
                  for _ in range(sequence_length)]
target_label = np.random.randint(2)  # 0 or 1

# Initialize hidden state and cell state
h_prev = np.zeros((hidden_size, 1))
c_prev = np.zeros((hidden_size, 1))

# Training loop
for epoch in range(epochs):
    # Forward pass
    for x in input_sequence:
        y, cache = lstm.forward((h_prev, c_prev, x))
        h_prev, c_prev, _, _, _, _, _, _, _, _ = cache

    # Calculate loss
    loss = lstm.binary_cross_entropy_loss(lstm.sigmoid(y), target_label)

    # Backward pass
    # Gradient of loss with respect to y
    dy = lstm.binary_cross_entropy_loss(lstm.sigmoid(y), target_label)
    gradients = lstm.backward(dy, cache)

    # Update parameters (simple gradient descent, you might want to use more advanced optimizers)
    for param in ['Wf', 'Wi', 'Wc', 'Wo', 'bf', 'bi', 'bc', 'bo', 'Wy', 'by']:
        lstm.__dict__[param] -= learning_rate * gradients['d' + param]

    # Print loss for monitoring
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

# Final output after training
final_output = lstm.sigmoid(y)
print(f"Final Output: {final_output}")
