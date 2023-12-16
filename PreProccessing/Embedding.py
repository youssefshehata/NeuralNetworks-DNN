from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import PreProccessing.Tokenization as tokenizer
import tensorflow as tf


def embedding():
  num_unique_words , max_sequence_length , train_padded , train_labels , val_padded , val_labels,sentences = tokenizer.tokenize()
  data =  np.concatenate((train_padded,val_padded)) 

  # Convert padded sequences back to sentences
  # (This is necessary for TfidfVectorizer to work)
  reconstructed_sentences = sentences

  # Create TF-IDF Vectorizer
  tfidf_vectorizer = TfidfVectorizer()

  # Fit and transform on the training data
  x_train , x_test = train_test_split(reconstructed_sentences , test_size=0.2)
  data= tfidf_vectorizer.fit_transform(x_train)
  data = data.toarray()

  # data = np.nonzero(data)


  # data = tf.sparse.SparseTensor(
  #     indices=[
  #         [i, j] for i, row in enumerate(data) for j , value in enumerate(row) if value != 0.0
  #     ],
  #     values=[value for row in data for value in row if value != 0.0],
  #     dense_shape=(len(data), len(data[0])),b
  # )




  return   num_unique_words , max_sequence_length , train_padded , train_labels , val_padded , val_labels , data
