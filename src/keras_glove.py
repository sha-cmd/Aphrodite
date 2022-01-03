# -*- coding: utf-8 -*-
import numpy as np
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import ast
import json
import tensorflow as tf

from tensorflow.keras.layers import TextVectorization
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
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
    epochs = int(params['keras_glove']['epochs'])
    batch_size = int(params['keras_embedding']['batch_size'])
    embedding_dim = int(params['model_constants']['embedding_dim'])
    max_features = int(params['model_constants']['max_features'])
    embedding_dim = int(params['model_constants']['embedding_dim'])
    sequence_length = int(params['model_constants']['sequence_length'])
    optimizer = str(params['model_constants']['optimizer'])
    maxlen = int(params['model_constants']['maxlen'])
    training_samples = int(params['model_constants']['training_samples'])
    validation_samples = int(params['model_constants']['validation_samples'])
    unit_cost_of_a_bad_buzz = int(params['model_constants']['unit_cost_of_a_bad_buzz'])
    max_words = int(params['model_constants']['max_words'])
    dense = int(params['keras_glove']['dense'])
    recall = float(params['keras']['recall'])
    specificity = float(params['keras']['specificity'])
    sensitivity = float(params['keras']['sensitivity'])

print('keras_glove')


def cost_metric(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))

    cost = K.log(tn / (tn + fp)) * unit_cost_of_a_bad_buzz

    if tf.math.is_nan(cost) or tf.math.is_inf(cost):
        cost = 0.

    return cost


dataset_train = keras.preprocessing.text_dataset_from_directory(
    dir_list[0], batch_size=batch_size, seed=42, subset='training', validation_split=0.2)
dataset_val = keras.preprocessing.text_dataset_from_directory(
    dir_list[0], batch_size=batch_size, seed=42, subset='validation', validation_split=0.2)
dataset_test = keras.preprocessing.text_dataset_from_directory(
    dir_list[1], batch_size=batch_size)

print(dataset_train)

# Now that we have our custom standardization, we can instantiate our text
# vectorization layer. We are using this layer to normalize, split, and map
# strings to integers, so we set our 'output_mode' to 'int'.
# Note that we're using the default split function,
# and the custom standardization defined above.
# We also set an explicit maximum sequence length, since the CNNs later in our
# model won't support ragged sequences.
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length
)

# Now that the vocab layer has been created, call `adapt` on a text-only
# dataset to create the vocabulary. You don't have to batch, but for very large
# datasets this means you're not keeping spare copies of the dataset in memory.

# Let's make a text-only dataset (no labels):
text_ds = dataset_train.map(lambda x, y: x)
# Let's call `adapt`:
vectorize_layer.adapt(text_ds)
vocabulary = vectorize_layer.get_vocabulary()


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Vectorize the data.
train_ds = dataset_train.map(vectorize_text)
val_ds = dataset_val.map(vectorize_text)
test_ds = dataset_test.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

glove_dir = '/home/romain/Téléchargements'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((max_words, embedding_dim))
for i, word in enumerate(vocabulary):
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(dense, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=[tf.metrics.BinaryAccuracy(), 'AUC', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                       tf.keras.metrics.PrecisionAtRecall(recall),
                       cost_metric, tf.keras.metrics.SensitivityAtSpecificity(specificity),
                       tf.keras.metrics.SpecificityAtSensitivity(sensitivity)])

history = model.fit(train_ds,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=val_ds,
                    callbacks=[DvcLiveCallback(path="./glove_logs_" + action)])

if not os.path.isdir('models'):
    os.mkdir('models')
model.save_weights('models/glove_model' + action + '.h5')

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


model.load_weights('models/glove_model' + action + '.h5')
metrics = model.evaluate(test_ds)
# Enregistrement des métriques plus des paramètres (mais l’enregistrement des paramètres est redondant)
df = pd.DataFrame([[w, x, y, z, a, b, c, d, e] for w, x, y, z, a, b, c, d, e in zip([metrics[0]],
                                                                                    [metrics[1]],
                                                                                    [metrics[2]],
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
if not os.path.isdir('metrics/glove'):
    os.mkdir('metrics/glove')
f = open('metrics/glove/glove_metrics_' + action + '.json', 'w')
f.writelines(df)
f.close

db = 'glove_logs_' + action + '/'
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
