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
import gensim.downloader
import seaborn as sns

from gensim import utils
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, GRU
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from dvclive.keras import DvcLiveCallback
from optparse import OptionParser
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

global dir_list
global class_list

path_to_model = 'word2vec/wiki-news-300d-1M.vec'


def set_weights(set_of_word, dim_of_model, w2vmodel):
    """Nous allons créer une matrice de poids à partir du modèle fasttext word2vec
    obtenu à cette adresse
    https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"""
    vocab_len = len(set_of_word) + 1
    embedding_matrix = np.zeros((vocab_len, dim_of_model))
    word_found = 0
    word_it = 0
    with open(w2vmodel) as f:
        for line in f:
            word, *vector = line.split()  # le pointeur permet de lister les valeurs à droite
            if word in set_of_word:  # Nous aurions pu utiliser aussi bien un "try, except"
                word_found += 1
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:dim_of_model]  # La valeur correspond au modèle
            word_it += 1
    print('Words not found : %s' % (vocab_len - word_found))
    return embedding_matrix


def to_tensor(text):
    return tf.convert_to_tensor([[text]])


def to_label(prob):
    return 0 if (prob < 0.5) else 4


def get_true_postive_count(cm, model_ml, df_data, label_column_name, text_column_name, target_class):
    pred_labels = [model_ml(to_tensor(str(raw_text))) for it, raw_text in
                   df_data.loc[df[label_column_name] == target_class][text_column_name].items()]
    predictions = [to_label(x) for x in pred_labels]
    tp_count = len(list(filter(lambda x: x == target_class, predictions)))
    index = np.where(cm == tp_count)
    if index[0][0] == index[1][0]:
        return index[0][0]
    else:
        raise ValueError('Could not find the index ' + index)


def resolve_labels_sequence(classes, cm, model_ml, df_data, label_column_name, text_column_name):
    target_seq = [0] * len(classes)
    for label in classes:
        index = get_true_postive_count(cm, model_ml, df_data, label_column_name, text_column_name, label)
        target_seq[index] = label
    return target_seq


def perf_confusion_matrix(model_ml, pd_test_data, label_column_name, text_column_name):
    test_labels = pd_test_data[label_column_name]
    test_labels = np.array(test_labels)
    _classes = list(set(test_labels))
    for it, raw_text in pd_test_data[text_column_name].items():
        print(str(raw_text))
        print(to_tensor(str(raw_text)))
        print(model_ml(to_tensor(str(raw_text))))
    predictions = [model_ml(to_tensor(str(raw_text))) for it, raw_text in pd_test_data[text_column_name].items()]
    pred_labels = [to_label(x) for x in predictions]
    pred_labels = np.array(pred_labels)
    eq = test_labels == pred_labels
    print("Accuracy: " + str(eq.sum() / len(test_labels)))
    cm = confusion_matrix(test_labels, pred_labels)
    labels = resolve_labels_sequence(_classes, cm, model_ml, pd_test_data, label_column_name, text_column_name)
    print(labels)
    print(confusion_matrix(test_labels, pred_labels, labels=labels))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrices/Confusion_matrix_w2v_' + action + '.jpg')


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
    epochs = int(params['keras_w2v']['epochs'])
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
    dense = int(params['keras_w2v']['dense'])
    recall = float(params['keras']['recall'])
    specificity = float(params['keras']['specificity'])
    sensitivity = float(params['keras']['sensitivity'])

print('keras_w2v')


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        X_train = pd.read_csv('data/X_' + action + '_train.csv', index_col='index')
        for it, line in X_train['tweets'].items():
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


def cost_metric(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))

    cost = K.log(tn / (tn + fp)) * unit_cost_of_a_bad_buzz
    if tf.math.is_nan(cost) or tf.math.is_inf(cost):
        cost = 0.

    return cost


df_X = pd.read_csv('data/X_' + action + '_train.csv', index_col='index')
df_y = pd.read_csv('data/y_' + action + '_train.csv', index_col='index')
df_X['tweets'] = df_X['tweets'].apply(lambda x: str(x)).astype('str')
X_train, X_val, y_train, y_val = train_test_split(df_X['tweets'].values, df_y['note'].values, test_size=0.2,
                                                  random_state=1)
texts = df_X['tweets'].tolist()
texts_train = X_train.tolist()
texts_val = X_val.tolist()
tokenized = Tokenizer()
tokenized.fit_on_texts(texts)
sequences_train = tokenized.texts_to_sequences(texts_train)
sequences_val = tokenized.texts_to_sequences(texts_val)
word_index = tokenized.word_index
vocab_len = len(word_index)
print('Nous avons %s tokens uniques.' % vocab_len)
df_X['nb_words'] = df_X['tweets'].apply(lambda x: x.split())
df_X['nb_words'] = df_X['nb_words'].apply(lambda x: len(x))
MAX_SEQUENCE_LENGTH = df_X['nb_words'].max()
print('Nous avons %s mots tout au plus dans chaque tweets' % MAX_SEQUENCE_LENGTH)
X_train_pad = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
X_val_pad = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH)
label_encoding = {
    0: 0,
    4: 1,
}

y_train = [label_encoding[x] for x in y_train.tolist()]
y_val = [label_encoding[x] for x in y_val.tolist()]
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

embeddings_m = set_weights(word_index, embedding_dim, path_to_model)
print(embeddings_m.shape)

embedding_layer = Embedding(vocab_len + 1,
                            embedding_dim,
                            weights=[embeddings_m],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(GRU(units=128,
                            dropout=0.2,
                            recurrent_dropout=0.2)))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=[tf.metrics.BinaryAccuracy(), 'AUC', tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.PrecisionAtRecall(recall),
                       cost_metric,
                       tf.keras.metrics.SensitivityAtSpecificity(
                           specificity),
                       tf.keras.metrics.SpecificityAtSensitivity(
                           sensitivity)])

# happy learning!
history = model.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val),
                    epochs=epochs, batch_size=batch_size, callbacks=[DvcLiveCallback(path="./w2v_lstm_logs_" + action)])

# Matrice de confusion
X_test = pd.read_csv('data/X_' + action + '_test.csv', index_col='index')
y_test = pd.read_csv('data/y_' + action + '_test.csv', index_col='index')
X_test['tweets'] = X_test['tweets'].apply(lambda x: str(x)).astype('str')
texts_test = X_test['tweets'].tolist()
sequences_test = tokenized.texts_to_sequences(texts_test)
X_test_pad = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
pred_labels = [(1 if (x[0] < 0.5) else 0) for x in model.predict(X_test_pad)]
df = pd.DataFrame(pred_labels)
df['note'] = y_test
df['note'] = df['note'].map({0: 0, 4: 1})
df = df.rename(columns={0: 'tweets'})
list_index = df[df['note'] == 1]['tweets'].index.tolist()
cm = confusion_matrix(df['note'], pred_labels)
df_cm = pd.DataFrame(cm, index=[0, 1], columns=[0, 1])
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True)
plt.savefig('confusion_matrices/Confusion_matrix_w2v_lstm_' + action + '.jpg')
#
# dataset_train = keras.preprocessing.text_dataset_from_directory(
#     dir_list[0], batch_size=batch_size, seed=42, subset='training', validation_split=0.2)
# dataset_val = keras.preprocessing.text_dataset_from_directory(
#     dir_list[0], batch_size=batch_size, seed=42, subset='validation', validation_split=0.2)
# dataset_test = keras.preprocessing.text_dataset_from_directory(
#     dir_list[1], batch_size=batch_size)
#
# print(dataset_train)
#
# # Now that we have our custom standardization, we can instantiate our text
# # vectorization layer. We are using this layer to normalize, split, and map
# # strings to integers, so we set our 'output_mode' to 'int'.
# # Note that we're using the default split function,
# # and the custom standardization defined above.
# # We also set an explicit maximum sequence length, since the CNNs later in our
# # model won't support ragged sequences.
# vectorize_layer = TextVectorization(
#     max_tokens=max_features,
#     output_mode="int",
#     output_sequence_length=sequence_length
# )
#
# # Now that the vocab layer has been created, call `adapt` on a text-only
# # dataset to create the vocabulary. You don't have to batch, but for very large
# # datasets this means you're not keeping spare copies of the dataset in memory.
#
# # Let's make a text-only dataset (no labels):
# text_ds = dataset_train.map(lambda x, y: x)
# # Let's call `adapt`:
# vectorize_layer.adapt(text_ds)
#
#
# def vectorize_text(text, label):
#     text = tf.expand_dims(text, -1)
#     return vectorize_layer(text), label
#
#
# # Vectorize the data.
# train_ds = dataset_train.map(vectorize_text)
# val_ds = dataset_val.map(vectorize_text)
# test_ds = dataset_test.map(vectorize_text)
#
# # Do async prefetching / buffering of the data for best performance on GPU.
# train_ds = train_ds.cache().prefetch(buffer_size=10)
# val_ds = val_ds.cache().prefetch(buffer_size=10)
# test_ds = test_ds.cache().prefetch(buffer_size=10)
#
#
#
# mbeddings_index = {}
#
# df_X = pd.read_csv('data/X_' + action + '_train.csv', index_col='index')
# df_y = pd.read_csv('data/y_' + action + '_train.csv', index_col='index')
# X_train, X_val, y_train, y_val = train_test_split(df_X['tweets'].values, df_y['note'].values, test_size=0.2, random_state=1)
# print('X_train', type(X_train), len(X_train))
# sentences = MyCorpus()
# model_w2v = gensim.models.Word2Vec(sentences=sentences, window=5, min_count=1, workers=4)
# model_w2v.train(sentences, total_examples=model_w2v.corpus_count,  epochs=epochs)
# print(model_w2v)
# model_w2v.save("models/word2vec_" + action + ".model")
# print('ok saved')
# model_w2v = Word2Vec.load('models/word2vec_' + action + '.model')
#
# print(model_w2v.wv.most_similar(positive=['you']))
# preprocesseur = Tokenizer(split=',')
# preprocesseur.fit_on_texts(X_train)
# vocabulary = preprocesseur.word_index
# print(vocabulary)
# lst_text2seq = preprocesseur.texts_to_sequences(X_train)
# X_train_seq = sequence.pad_sequences(lst_text2seq, maxlen=maxlen,
#                                      padding="post", truncating="post")
# preprocesseur = Tokenizer(split=',')
# preprocesseur.fit_on_texts(' '.join(X_val))
# vocabulary = preprocesseur.word_index
# lst_text2seq = preprocesseur.texts_to_sequences(X_val)
# X_val_seq = sequence.pad_sequences(lst_text2seq, maxlen=maxlen,
#                                      padding="post", truncating="post")
# embeddings_m = np.zeros((len(vocabulary)+1, maxlen))
# for word, idx in vocabulary.items():
#     try:
#         embeddings_m[idx] = model_w2v[word]
#     except:
#         pass
# print(embeddings_m)
# # A integer input for voca
# model = Sequential()
# model.add(Embedding(embeddings_m.shape[0], output_dim=embeddings_m.shape[1], weights=[embeddings_m], input_length=maxlen, trainable=False))
# model.add(Flatten())
# model.add(Dense(dense, activation='relu'))  # Perceptron multicouches
# model.add(Dense(dense, activation='relu'))  # Perceptron multicouches
# model.add(Dense(1, activation='sigmoid'))
# print(model.summary())
#
# model.compile(optimizer=optimizer,
#               loss='binary_crossentropy',
#               metrics=[tf.metrics.BinaryAccuracy(), 'AUC', tf.keras.metrics.Precision(),
#                        tf.keras.metrics.Recall(),
#                        tf.keras.metrics.PrecisionAtRecall(recall),
#                        cost_metric,
#                        tf.keras.metrics.SensitivityAtSpecificity(
#                            specificity),
#                        tf.keras.metrics.SpecificityAtSensitivity(
#                            sensitivity)])
# print(X_train_seq[1], y_train[1], X_val_seq[1], y_val[1])
# history = model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val),
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     callbacks=[DvcLiveCallback(path="./w2v_logs_" + action)])

if not os.path.isdir('models'):
    os.mkdir('models')
model.save_weights('models/w2v_model_' + action + '.h5')

# acc = history.history['binary_accuracy']
# val_acc = history.history['val_binary_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# # plt.show()
#
#
# model.load_weights('models/w2v_model_' + action + '.h5')
#
# # Matrix of confusion
# df = pd.read_csv('data/X_' + action + '_test.csv')
# df_y = pd.read_csv('data/y_' + action + '_test.csv')
# df = df[['tweets']]
# df['note'] = df_y['note']
# lst_text2seq = preprocesseur.texts_to_sequences(df['tweets'])
# X_test_seq = sequence.pad_sequences(lst_text2seq, maxlen=maxlen,
#                                    padding="post", truncating="post")
# df['tweets'] = X_test_seq.tolist()
# perf_confusion_matrix(model, df, 'note', 'tweets')
#
# print(test_ds)
metrics = model.evaluate(X_test_pad)

# Enregistrement des métriques plus des paramètres (mais l’enregistrement des paramètres est redondant)
df = pd.DataFrame(
    [[w, x, y, z, a, b, c, d, e] for w, x, y, z, a, b, c, d, e in zip([metrics[0]],
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
if not os.path.isdir('metrics/w2v_lstm'):
    os.mkdir('metrics/w2v_lstm')
f = open('metrics/w2v_lstm/w2v_lstm_metrics_' + action + '.json', 'w')
f.writelines(df)
f.close
db = 'w2v_lstm_logs_' + action + '/'
file_list = glob.glob(db + '*.tsv')
for file in file_list:
    name = file.split('/')[-1]
    dict_json = {}
    list_json = []
    x = ast.literal_eval(pd.read_csv(file, sep='\t').to_json())  # prix d’un bad buzz
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
