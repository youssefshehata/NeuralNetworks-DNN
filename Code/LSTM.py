import tensorflow as tf

class myLstm(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(myLstm, self).__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Weights and biases for input gate
        self.w_i = self.add_weight(shape=(input_dim, self.units), initializer='uniform', name='w_i')
        self.u_i = self.add_weight(shape=(self.units, self.units), initializer='uniform', name='u_i')
        self.b_i = self.add_weight(shape=(self.units,), initializer='zeros', name='b_i')

        # Weights and biases for forget gate
        self.w_f = self.add_weight(shape=(input_dim, self.units), initializer='uniform', name='w_f')
        self.u_f = self.add_weight(shape=(self.units, self.units), initializer='uniform', name='u_f')
        self.b_f = self.add_weight(shape=(self.units,), initializer='zeros', name='b_f')

        # Weights and biases for cell state
        self.w_c = self.add_weight(shape=(input_dim, self.units), initializer='uniform', name='w_c')
        self.u_c = self.add_weight(shape=(self.units, self.units), initializer='uniform', name='u_c')
        self.b_c = self.add_weight(shape=(self.units,), initializer='zeros', name='b_c')

        # Weights and biases for output gate
        self.w_o = self.add_weight(shape=(input_dim, self.units), initializer='uniform', name='w_o')
        self.u_o = self.add_weight(shape=(self.units, self.units), initializer='uniform', name='u_o')
        self.b_o = self.add_weight(shape=(self.units,), initializer='zeros', name='b_o')

        self.built = True

    def call(self, inputs, states):
        h_t, c_t = states

        # Input gate
        i_t = tf.sigmoid(tf.matmul(inputs, self.w_i) + tf.matmul(h_t, self.u_i) + self.b_i)

        # Forget gate
        f_t = tf.sigmoid(tf.matmul(inputs, self.w_f) + tf.matmul(h_t, self.u_f) + self.b_f)

        # Cell state
        c_tilde_t = tf.tanh(tf.matmul(inputs, self.w_c) + tf.matmul(h_t, self.u_c) + self.b_c)
        c_t = f_t * c_t + i_t * c_tilde_t

        # Output gate
        o_t = tf.sigmoid(tf.matmul(inputs, self.w_o) + tf.matmul(h_t, self.u_o) + self.b_o)

        # Hidden state
        h_t = o_t * tf.tanh(c_t)

        return h_t, [h_t, c_t]

    def get_config(self):
        config = {'units': self.units}
        base_config = super(myLstm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Example usage:
# lstm = myLstm(units=128)
# output, final_states = lstm(inputs, initial_states)
