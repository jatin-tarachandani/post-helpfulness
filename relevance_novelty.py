"""Implementation of paper titled "Predicting Helpful Posts in
   Open-Ended Discussion Forums: A Neural Architecture", 2019, NAACL.
"""
import sys

from datetime import datetime

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, GRU, Lambda, concatenate
from keras.layers.embeddings import Embedding
import keras.preprocessing as preprocessing
import keras.layers as layers

import numpy as np
import pandas as pd

from sklearn import model_selection

import constants
import utilities

def get_sequences(tokenizer, data, max_len, field):
    x = tokenizer.texts_to_sequences(data[field].values.astype(str))
    seq = preprocessing.sequence.pad_sequences(x, maxlen=max_len)
    return seq


def get_compiled_model(word_index, embedding_matrix, past_posts):
    """The Relevance and Novelty based post classification model"""

    def backend_reshape(x):
        return K.reshape(x, (-1, constants.MAX_LEN))


    def backend_reshape2(x):
        x = K.reshape(x, (-1, past_posts, constants.NEURON_COUNT))
        return x

    embedding = Embedding(len(word_index)+1, constants.DATA_DIM,
                          weights=[embedding_matrix], trainable=True,
                          input_length=constants.MAX_LEN, mask_zero=True)
    gru2 = GRU(constants.NEURON_COUNT, return_sequences=False)
    __dropout__ = Dropout(constants.DROPOUT_RATE)

    gru3 = GRU(constants.NEURON_COUNT, return_sequences=False)
    __dropout__ = Dropout(constants.DROPOUT_RATE)

    text = Input(shape=(constants.MAX_LEN,))
    text_past = Input((past_posts, constants.MAX_LEN,))
    text_op = Input(shape=(constants.MAX_LEN,))

    encoder = embedding(text)
    encoder_gru = gru2(encoder)
    encoder_gru = __dropout__(encoder_gru)

    past_text_model = Sequential()
    past_text_model.add(Lambda(backend_reshape, input_shape=(past_posts, constants.MAX_LEN),
                               output_shape=(constants.MAX_LEN,)))
    past_text_model.add(embedding)
    past_text_model.add(gru2)
    past_text_model.add(__dropout__)
    past_text_model.add(Lambda(backend_reshape2, output_shape=(past_posts, constants.NEURON_COUNT)))
    past_text_model.add(gru3)
    past_text_model.add(__dropout__)

    encoder_op = embedding(text_op)
    encoder_gru_op = gru2(encoder_op)
    encoder_gru_op = __dropout__(encoder_gru_op)

    relevance_w_op = layers.multiply([encoder_gru, encoder_gru_op])

    past_context = past_text_model(text_past)

    novelty = layers.multiply([past_context, encoder_gru])

    combined = concatenate([relevance_w_op, novelty])

    pred = Dense(constants.NEURON_COUNT, activation='sigmoid')(combined)
    pred = Dense(constants.NEURON_COUNT, activation='sigmoid')(pred)
    pred = Dense(1, activation='sigmoid')(pred)

    model = Model(inputs=[text, text_past, text_op], outputs=pred)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def prepare_context(data, past_context_length, tokenizer):
    """Loads up the contextual texts and stacks them up"""
    print('\n*********************\nwith context length:', past_context_length)
    print('\n*********************\n')
    x_past = []
    for i in range(0, past_context_length):
        current_context = constants.MAX_CONTEXT_LENGTH-past_context_length+i
        print('adding current context', current_context)
        seq = get_sequences(tokenizer, data, constants.MAX_LEN, 'context_'+str(current_context))
        x_past.append(seq)
    x_past = np.column_stack([x for x in x_past])
    x_past = x_past.reshape(x_past.shape[0], past_context_length, constants.MAX_LEN)
    return x_past

def run(model_name):
    """Reads data file, encodes the texts, trains the models and tests it."""
    data = pd.read_csv(constants.POSTS_FILE, sep='\t')

    y = data.helpfulCount.values
    ids = data.postID.values.astype(str)

    tokenizer = preprocessing.text.Tokenizer(num_words=constants.VOCABULARY_SIZE)
    tokenizer.fit_on_texts(list(data.postText.values.astype(str)))
    (word_index, embedding_matrix) = utilities.load_embedding(tokenizer, \
      '../mentalHealth/glove.6B.100d.txt', constants.DATA_DIM)

    x = get_sequences(tokenizer, data, constants.MAX_LEN, 'postText')
    x_op = get_sequences(tokenizer, data, constants.MAX_LEN, 'OPost')

    #Prepare contextual data and train model for past context
    x_past = prepare_context(data, constants.PAST_POST_LENGTH, tokenizer)

    ids_train, ids_val, x_train, x_val, x_op_train, x_op_val, x_past_train, x_past_val, \
    y_train, y_val = model_selection.train_test_split(ids, x, x_op, x_past, y, test_size= \
    constants.TEST_SIZE, random_state=constants.SEED, stratify=y)

    print('Shape of text_trainX: ', x_past.shape)
    model = get_compiled_model(word_index, embedding_matrix, constants.PAST_POST_LENGTH)

    #Check accuracy after each epoch through callback
    histories = utilities.Histories([ids_val, x_val, x_past_val, x_op_val], y_val, LOG_FILE,
                                    REPORT_FILE, constants.POSTS_FILE, model_name)
    model.fit([x_train, x_past_train, x_op_train], y_train, epochs=constants.NUM_EPOCHS,
              validation_data=([x_val, x_past_val, x_op_val], y_val), shuffle=True,
              batch_size=512, verbose=1, callbacks=[histories])

if __name__ == "__main__":
    CURRENT_PROGRAM_NAME = sys.argv[0]
    LOG_FILE = 'logs/'+ datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.txt'
    REPORT_FILE = LOG_FILE + '.predictions.txt'
    utilities.initialize_log_file(LOG_FILE, CURRENT_PROGRAM_NAME)
    np.random.seed(constants.SEED)
    run(CURRENT_PROGRAM_NAME)
