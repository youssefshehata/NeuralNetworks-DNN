import tensorflow as tf
from keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, Dropout, LayerNormalization, MultiHeadAttention
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import pandas as pd

def feed_forward(x, ff_dim, dropout=0.1):
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(x.shape[-1])(x)
    x = Dropout(dropout)(x)
    return x
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention( num_heads=num_heads , key_dim = 64)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = feed_forward(x, ff_dim=ff_dim, dropout=dropout)
    return x + res
def build_model(max_seq_length, vocab_size, num_classes):
    inputs = Input(shape=(max_seq_length,), dtype=tf.int32)

    # Embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=128)(inputs)

    # Transformer Encoder Block
    transformer_block = transformer_encoder(embedding_layer, head_size=128, num_heads=4, ff_dim=128, dropout=0.1)

    # Global Average Pooling layer
    pooling_layer = GlobalAveragePooling1D()(transformer_block)

    # Output layer
    outputs = Dense(num_classes, activation='sigmoid')(pooling_layer)

    model = Model(inputs=inputs, outputs=outputs, name='transformer_model')
    return model

def trans(max_seq_length, vocab_size, num_classes, train_padded, val_padded, train_labels, val_labels , test_sentences):
    num_classes = 3  # Replace with the number of classes in your classification task

    model = build_model(max_seq_length, vocab_size, num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_padded, train_labels, epochs=3, validation_data=(val_padded, val_labels), verbose=2)
    score = model.evaluate(val_padded, val_labels, verbose=2)

    print(f"Test Accuracy:", score[1])

    predictions = model.predict(test_sentences)

    predictions = np.argmax(predictions, axis=1) - 1





    data = pd.read_csv("/home/joe/School/Neural/NeuralNetworks-DNN/Data/test _no_label.csv")
    submimssion= pd.DataFrame()
    submimssion["ID"] = data['ID']
    submimssion["rating"] = predictions
    submimssion.to_csv("TransSubmission.csv", index=False)

