import numpy as np
from tqdm import tqdm
import keras

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

class Histories(keras.callbacks.Callback):
    def __init__(self, x_values, y_val, log_file, report_file, data_file, model_name):
        self._ids_val = x_values[0]
        self._x_val = x_values[1]
        self._x_past_val = x_values[2]
        self._x_op_val = x_values[3]
        self._y_val = y_val
        self._maxf1 = -1
        self._logfile = log_file
        self._report_file = report_file
        self._source_date_file = data_file
        self._model_name = model_name

    def on_train_begin(self, logs={}):
        self._maxf1 = -1

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict([self._x_val, self._x_past_val, self._x_op_val])
        y_pred = (pred > 0.5).astype('int32')
        report = classification_report(self._y_val, y_pred, digits=4)

        #Append latest classification report to log file
        f = open(self._logfile, 'a')
        f.write(report + '\n')
        f.close()
        f1s = f1_score(self._y_val, y_pred, average=None)[1]

        #Dump the predictions for the test posts if f1 is better
        if f1s > self._maxf1 or self._maxf1<0:
            self._maxf1 = f1s
            f = open(self._report_file, 'w')
            f.write('model:'+self._model_name+'\ndata:'+self._source_date_file+'\n'+report+'\n'+
                    'id\tpredicted\tactual\n')
            for i in range(0, len(y_pred)):
                f.write(self._ids_val[i]+'\t'+str(y_pred[i])+'\t'+str(self._y_val[i])+'\n')
            f.close()
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def load_embedding(tokenizer, embedding_file, embedding_dim):
    '''
    begin: use glove embedding
    '''
    word_index = tokenizer.word_index
    embeddings_index = {}
    f = open(embedding_file, encoding='utf-8')
    for line in tqdm(f):
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))


    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    found = 0
    total = 0

    f = open('uncoveredWords.txt', 'w')
    for word, i in tqdm(word_index.items()):
        total += 1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            found += 1
            embedding_matrix[i] = embedding_vector
        else:
            f.write(word+'\n')
    f.close()
    print('glove coverage:', float(found/total))
    return (word_index, embedding_matrix)

def initialize_log_file(log_file, current_program_name):
    f = open(log_file, 'w')
    f.write('logging for:'+current_program_name+'\n')
    f.close()
