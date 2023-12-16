import tensorflow as tf
from keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D
import tensorflow_addons as tfa

from tensorflow.nlp.layers import TransformerEncoderBlock


def trans(max_seq_length, vocab_size, num_classes , train_padded , val_padded , train_labels , val_labels ):
    # Input layer for token indices
    inputs = Input(shape=(max_seq_length,), dtype=tf.int32)

    # Embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=128)(inputs)

    # Transformer layer
    transformer_layer = TransformerEncoderBlock(
        num_layers=1,
        d_model=128,
        num_heads=4,
        mlp_units=[128],
        dropout=0.1,
    )(embedding_layer)

    # Global average pooling layer
    pooling_layer = GlobalAveragePooling1D()(transformer_layer)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(pooling_layer)

    # Model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='transformer_model')





    num_classes = 3   # Replace with the number of classes in your classification task

    model = model(max_seq_length, vocab_size, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_padded, train_labels, epochs=6, validation_data=(val_padded, val_labels), verbose=2)
    score = model.evaluate(val_padded, val_labels, verbose=2)


    print(f"Test Accuracy:", score[1])
    predictions = model.predict(train_padded)
    predictions = [-1 if P < 0.33 else (0 if P < 0.67 else 1) for P in predictions]

    print("Actual labels : ",train_labels[10:20])    
    print("Predicted labels : ",predictions[10:20])




