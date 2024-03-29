# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
import os
import pandas as pd
import glob
import ast
import json
import pickle

from sklearn.metrics import confusion_matrix
from dvclive.keras import DvcLiveCallback
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from optparse import OptionParser

global dir_list
global class_list


# Fonctions pour travailler avec le modèle afin de récupérer une matrice de confusion
def to_tensor(text):
    return tf.convert_to_tensor([[text]])


def to_label(prob):
    return 0 if (prob < 0.5) else 4


def get_true_postive_count(cm, model_ml, df_data, label_column_name, text_column_name, target_class):

    pred_labels = [model_ml(to_tensor(str(raw_text))) for it, raw_text in df_data.loc[df[label_column_name] == target_class][text_column_name].items()]
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
    plt.title('Matrice de confusion' + ' embedding ' + action)
    sns.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrices/Confusion_matrix_embeddings_' + action + '.jpg')


# Analyse de la ligne de commande du script pour récupérer le type d’action :
# soit nature (juste le cleaning du texte), soit lemm (lemmatisation), soit stem (stemming)
# les trois types de données sont à la fois écrit en fichier texte, dans des répertoires,
# mais aussi dans des fichiers csv, pour pallier la nervosité de keras.
parser = OptionParser()
parser.add_option("-a", "--action", dest="action",
                  help="Nature of the action", metavar="ACTION")

# Importation des données et typage pour matcher avec la fonction du fichier
(options, args) = parser.parse_args()
action = options.action

# Répertoire selon le preprocessing et l’action donnée en argument de la ligne de commande du script python
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

# Chargement du bon lot de répertoire selon l’action du stage de la commande dvc repro
data_list = dict_of_dir[action]
class_list = dict_of_class[action]

print('keras_embedding')

# Chargement des paramètres depuis le fichier de paramètres.
with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
    epochs = int(params['keras_embedding']['epochs'])
    batch_size = int(params['keras_embedding']['batch_size'])
    max_features = int(params['model_constants']['max_features'])
    embedding_dim = int(params['model_constants']['embedding_dim'])
    sequence_length = int(params['model_constants']['sequence_length'])
    optimizer = str(params['model_constants']['optimizer'])
    unit_cost_of_a_bad_buzz = int(params['model_constants']['unit_cost_of_a_bad_buzz'])
    dense = int(params['keras_embedding']['dense'])
    recall = float(params['keras']['recall'])
    specificity = float(params['keras']['specificity'])
    sensitivity = float(params['keras']['sensitivity'])


def cost_metric(y_true, y_pred):
    """Métrique métier, coût d’un bad buzz multiplier par le logarithme du rappel"""
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    cost = K.log(tn / (tn + fp)) * unit_cost_of_a_bad_buzz
    if tf.math.is_nan(cost) or tf.math.is_inf(cost):
        cost = 0.
    return cost

# Chargement des données à partir des fichiers des répertoires
dataset_train = keras.preprocessing.text_dataset_from_directory(
    dir_list[0], batch_size=batch_size, seed=42, subset='training', validation_split=0.2)
dataset_val = keras.preprocessing.text_dataset_from_directory(
    dir_list[0], batch_size=batch_size, seed=42, subset='validation', validation_split=0.2)
dataset_test = keras.preprocessing.text_dataset_from_directory(
    dir_list[1], batch_size=batch_size)

# Vectorisation
vectorizer = TextVectorization(output_mode="int")

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

text_ds = dataset_train.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)  # Returns a tensor with a length 1 axis inserted at index -1
    return vectorize_layer(text), label


train_ds = dataset_train.map(vectorize_text)
val_ds = dataset_val.map(vectorize_text)
test_ds = dataset_test.map(vectorize_text)

# Usage pour la mise en cache des données lors du traitement
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

# Écriture du modèle
inputs = tf.keras.Input(shape=(None,), dtype="int64")

x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(dense, activation="relu")(x)
x = layers.Dropout(0.5)(x)

predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[tf.metrics.BinaryAccuracy(), 'AUC',
                                                                        tf.keras.metrics.Precision(),
                                                                        tf.keras.metrics.Recall(),
                                                                        tf.keras.metrics.PrecisionAtRecall(recall),
                                                                        tf.keras.metrics.SensitivityAtSpecificity(
                                                                            specificity),
                                                                        tf.keras.metrics.SpecificityAtSensitivity(
                                                                            sensitivity),
                                                                        cost_metric])
# Entraînement
model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[DvcLiveCallback(path="./logs_" + action,
                                                                                      summary=True)])
# Évaluation du modèle, sauvegarde, prédiction des données tests pour évaluer la capacité de généralisation de la
# fonction de prédiction

if not os.path.isdir('models'):
    os.mkdir('models')
model.save('models/embedding_model_' + action + '.tf', save_format='tf')
model = load_model('models/embedding_model_' + action + '.tf', custom_objects={'cost_metric': cost_metric})
pickle.dump({'config': vectorize_layer.get_config(),
             'vocabulary': vectorize_layer.get_vocabulary(),
             'weights': vectorize_layer.get_weights()},
            open("tv_layer.pkl", "wb"))
inputs = tf.keras.Input(shape=(1,), dtype="string")
indices = vectorize_layer(inputs)
outputs = model(indices)
# Réalisation d’un modèle de bout en bout pour tester des phrases de tweets en entrée
end_to_end_model = tf.keras.Model(inputs, outputs)
end_to_end_model.compile(
    loss="binary_crossentropy", optimizer=optimizer, metrics=[tf.metrics.BinaryAccuracy(), 'AUC',
                                                              tf.keras.metrics.Precision(),
                                                              tf.keras.metrics.Recall(),
                                                              tf.keras.metrics.PrecisionAtRecall(recall),
                                                              tf.keras.metrics.SensitivityAtSpecificity(
                                                                  specificity),
                                                              tf.keras.metrics.SpecificityAtSensitivity(
                                                                  sensitivity),
                                                              cost_metric
                                                              ]
)
# Example of tweet
raw_text_data = tf.convert_to_tensor([
    ["That was an excellent movie I loved it"],
])
# Prediction of it
predictions = end_to_end_model(raw_text_data)
print('Prediction : ', predictions)

# Matrix of confusion sur les données test
predictions = end_to_end_model.predict(dataset_test)
df = pd.read_csv('data/X_' + action + '_test.csv')
df_y = pd.read_csv('data/y_' + action + '_test.csv')
df = df[['tweets']]
df['note'] = df_y['note']
perf_confusion_matrix(end_to_end_model, df, 'note', 'tweets')

# Metrics for printing DVC Studio at https://studio.iterative.ai/user/sha-cmd/views/Aphrodite-1ue5zga6kt
metrics = end_to_end_model.evaluate(dataset_test)
# Enregistrement des métriques plus des paramètres (mais l’enregistrement des paramètres est redondant)
df = pd.DataFrame(
    [[w, x, y, z, a, b, c, d, e] for w, x, y, z, a, b, c, d, e in zip([metrics[0]], [metrics[1]], [metrics[2]],
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
             'sensitivity_at_sensitivity',
             'specificity_at_sensitivity',
             'cost_metric'
             ]) \
    .to_json(orient='records').replace('[', '').replace(']', '')
# Nécessaire pour travailler en mode expérience, puisque le programme utilise /tmp
if not os.path.isdir('metrics'):
    os.mkdir('metrics')
if not os.path.isdir('metrics/embedding'):
    os.mkdir('metrics/embedding')
f = open('metrics/embedding/embedding_metrics_' + action + '.json', 'w')
f.writelines(df)
f.close
# Recopie des fichiers de logs au format json pour la beauté de l’art !
db = 'logs_' + action + '/'
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
