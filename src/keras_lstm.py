# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import ast
import json

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.layers import LSTM
from dvclive.keras import DvcLiveCallback
from optparse import OptionParser

global dir_list
global class_list

parser = OptionParser()
parser.add_option("-a", "--action", dest="action",
                  help="Nature of the action", metavar="ACTION")

# Importation des données et typage pour matcher avec la fonction du fichier
(options, args) = parser.parse_args()
action = options.action

dir_list = ['data/train', 'data/test']
dir_lemm_list = ['data/lemm/train', 'data/lemm/test']
dir_stem_list = ['data/stem/train', 'data/stem/test']

class_list = ['data/train/negatif', 'data/train/positif',
              'data/test/negatif', 'data/test/positif']
class_lemm_list = ['data/lemm/train/negatif', 'data/lemm/train/positif',
                   'data/lemm/test/negatif', 'data/lemm/test/positif']
class_stem_list = ['data/stem/train/negatif', 'data/stem/train/positif',
                   'data/stem/test/negatif', 'data/stem/test/positif']

dict_of_dir = {'nature': dir_list, 'lemm': dir_lemm_list, 'stem': dir_stem_list}
dict_of_class = {'nature': class_list, 'lemm': class_lemm_list, 'stem': class_stem_list}

data_list = dict_of_dir[action]
class_list = dict_of_class[action]


with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
    epochs = int(params['keras_lstm']['epochs'])
    batch_size = int(params['keras_embedding']['batch_size'])
    embedding_dim = int(params['model_constants']['embedding_dim'])
    max_features = int(params['model_constants']['max_features'])
    sequence_length = int(params['model_constants']['sequence_length'])
    maxlen = int(params['model_constants']['maxlen'])
    training_samples = int(params['model_constants']['training_samples'])
    optimizer = str(params['model_constants']['optimizer'])
    validation_samples = int(params['model_constants']['validation_samples'])
    max_words = int(params['model_constants']['max_words'])
    unit_cost_of_a_bad_buzz = int(params['model_constants']['unit_cost_of_a_bad_buzz'])
    dense = int(params['keras_lstm']['dense'])
    recall = float(params['keras']['recall'])
    specificity = float(params['keras']['specificity'])
    sensitivity = float(params['keras']['sensitivity'])

print('Keras_lstm')

def cost_metric(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))

    cost = K.log(tn / (tn + fp)) * unit_cost_of_a_bad_buzz


    if tf.math.is_nan(cost) or tf.math.is_inf(cost):
        cost = 0.

    return cost

train_dir = (dir_list[0])

labels = []
texts = []

for label_type in ['negatif', 'positif']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'negatif':
                labels.append(0)
            else:
                labels.append(1)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:]
y_val = labels[training_samples:]

model = Sequential()
model.add(Embedding(max_features, dense))
model.add(LSTM(dense))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=[tf.metrics.BinaryAccuracy(), 'AUC', tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.PrecisionAtRecall(recall),
                       cost_metric,
                       tf.keras.metrics.SensitivityAtSpecificity(
                           specificity),
                       tf.keras.metrics.SpecificityAtSensitivity(
                           sensitivity)])

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val),
                    callbacks=[DvcLiveCallback(path="./lstm_logs_" + action)])

if not os.path.isdir('models'):
    os.mkdir('models')
model.save('models/lstm_model_' + action + '.h5')

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

# plt.show()
test_dir = dir_list[1]

labels = []
texts = []

for label_type in ['negatif', 'positif']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'negatif':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

#model.load_model('models/lstm_model_' + action + '.h5')
metrics = model.evaluate(x_test, y_test)
# Enregistrement des métriques plus des paramètres (mais l’enregistrement des paramètres est redondant)
df = pd.DataFrame([[w, x, y, z, a, b, c, d, e] for w, x, y, z, a, b, c, d, e in zip([metrics[0]], [metrics[1]], [metrics[2]],
                                                                              [metrics[3]],
                                                                              [metrics[4]],
                                                                              [metrics[5]],
                                                                              [metrics[6]],
                                                                              [metrics[7]],
                                                                              [metrics[8]])],

                  columns=['loss', 'binary_accuracy', 'AUC',
                           'precision',
                           'recall',
                           'precision_at_recall',
                           'cost_metric',
                           'sensitivity_at_sensitivity',
                           'specificity_at_sensitivity']) \
    .to_json(orient='records').replace('[', '').replace(']', '')

# Nécessaire pour travailler en mode expérience, puisque le programme utilise /tmp
if not os.path.isdir('metrics'):
    os.mkdir('metrics')
if not os.path.isdir('metrics/lstm'):
    os.mkdir('metrics/lstm')
f = open('metrics/lstm/lstm_metrics_' + action + '.json', 'w')
f.writelines(df)
f.close

db = 'lstm_logs_' + action + '/'
file_list = glob.glob(db + '*.tsv')
for file in file_list:
    name = file.split('/')[-1]
    dict_json = {}
    list_json = []
    x = ast.literal_eval(pd.read_csv(file, sep='\t').to_json())
    list_keys = []
    for key, value in x.items():
        if key not in list_keys:
            list_keys.append(key)
    list_str = []
    for i in range(len(x[list_keys[0]])):
        dict_str = {}
        dict_str.update({name[:-4]: x[name[:-4]][str(i)]})
        dict_str.update({'step': i})
        list_str.append(dict_str)
        del dict_str
    with open(db[:-1] + '_dvc_plots/' + name[:-4] + '.json', 'w') as convert_file:
        convert_file.write(json.dumps(list_str))
