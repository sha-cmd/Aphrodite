# -*- coding: utf-8 -*-
"""Ce fichier prépare et sépare les données en jeu de données pour
l’entraînement et les tests"""

import pandas as pd
import ftfy
import yaml
import os
from sklearn.model_selection import StratifiedShuffleSplit

global train_size
global test_size

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
    train_size = int(params['prepare']['train_size'])
    test_size = int(params['prepare']['test_size'])


def convert(data):
    """Convertir le texte de l’encodage pour l’anglais à l’encodage utf-8, car il est
    idéal pour le machine-learning, les algorithmes le préfèrent"""
    return data.applymap(lambda x: ftfy.fix_text(x))


def prepare():
    """Prépare les jeux de test et d’entraînements à partir de la base de données téléchargé du projet.
    En mettant en plus le texte en encodage utf-8, et en utilisant un sélection stratifié et aléatoire
    pour récupérer le nombre de données voulues en valeur entière données dans le fichier params.yaml"""
    global train_size
    global test_size
    print(train_size, test_size)
    data_file_name = 'data/training.1600000.processed.noemoticon.csv'
    data_cp437 = pd.read_csv(data_file_name, names=['note', 'number', 'date', 'query', 'name', 'tweets'],
                             encoding='cp437')
    data = data_cp437.copy()
    data.iloc[:, 3:] = convert(data_cp437.iloc[:, 3:])
    data.index.name = 'index'
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size, random_state=42)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    data_list = [X_train, X_test, y_train, y_test]
    data_name_list = ['X_train', 'X_test', 'y_train', 'y_test']
    path_ = 'data/'
    format_ = '.csv'
    # Sauvegarde en fichier des données de test et d’entraînement
    if not os.path.isdir(path_):
        os.mkdir(path_)
    for it, dataset in enumerate(data_list):
        if 'X' in data_name_list[it]:
            if not os.path.isdir(path_ + data_name_list[it] ):
                os.mkdir(path_ + data_name_list[it] )
            pd.DataFrame(dataset, columns=list(data.iloc[:, 1:].columns)) \
                .to_csv(path_ + data_name_list[it] + format_, index_label='index')
        elif 'y' in data_name_list[it]:
            if not os.path.isdir(path_ + data_name_list[it]):
                os.mkdir(path_ + data_name_list[it])
            pd.DataFrame(dataset, columns=['note']) \
                .to_csv(path_ + data_name_list[it] + format_, index_label='index')


def main():
    prepare()


if __name__ == "__main__":
    main()
