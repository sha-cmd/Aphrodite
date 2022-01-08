"""Ce fichier permet d’inspecter dans le répertoire racine du projet,
en listant tous les répertoires ce terminant par _dvc_plots.
Ensuite il réalise un dictionnaire contenant
* le nom du répertoire donc du stage
* le nom de chaque métrique pour chaque répertoire
* pour chaque métrique le nom de fichier de train et de validation, assimilé au test dans les graphiques.
Dans une seconde partie le script crée un nouveau fichier de données mixant le fichier train et le fichier test.
Enfin il écrit dans un fichier texte toutes les lignes à ajouter dans le stage multi_plots du fichier dvc.yaml"""

import glob
import os
import pandas as pd
import numpy as np

# Liste des répertoires d’intérêts
dirs = glob.glob('*_dvc_plots')
dirs_dict = {}
for dir in dirs:
    metrics_dict = {}
    files = glob.glob(dir + '/*')
    for file in files:
        train_val_dict = {}
        header = file.split('/')[1][:4]
        if not (header == 'val_'):
            metric = [x.split('.') for x in file.split('/')][1][0]
            metric_file = file.split('/')[1]
            train_val_dict.update({'train': dir + '/' + metric_file,
                                   'val': dir + '/' + 'val_' + metric_file})

            metrics_dict.update({metric: train_val_dict})
    # Création d’une entrée pour chaque répertoire d’intérêt
    dirs_dict.update({dir: metrics_dict})

if not os.path.isdir('multi_plots'):
    os.mkdir('multi_plots')

try:
    os.remove('multi_plots/yamlhadoc.txt')
except FileNotFoundError as e:
    print(e)

# Itération à l’intérieur de notre dictionnaire de dictionnaire
for key_dir, metrics in dirs_dict.items():
    if not os.path.isdir('multi_plots/' + key_dir):
        os.mkdir('multi_plots/' + key_dir)
    for key_metric, learning in metrics.items():
        # Nous réalisons des actions de transformation
        # pour conformer nos données avec le modèle .dvc/plots/multi-plots.json
        df_train = pd.read_json(learning['train'])
        df_train['stage'] = 'train'
        df_val = pd.read_json(learning['val'])
        df_val['stage'] = 'val'
        cols = df_val.columns
        df_list = [df_train, df_val]
        for df in df_list:
            new_cols = []
            for col in df.columns:
                if 'val_' in col:
                    new_cols.append(col[4:])
                elif 'step' in col:
                    new_cols.append('epoch')
                else:
                    new_cols.append(col)
            df.columns = pd.Index(new_cols)
        df = df_train.append(df_val)
        df.to_csv('multi_plots/' + key_dir + '/' + key_metric + '.csv', index_label='index')
        # Fichier pour rajouter les plots dans dvc.yaml, stage multi_plots
        with open('multi_plots/yamlhadoc.txt', 'a') as f:
            f.writelines(f"- multi_plots/{key_dir}/{key_metric}.csv:\n\
            cache: false\n\
            persist: true\n\
            title: Train/Test {' '.join(key_metric.split('_')).title()} {key_dir[:-10]}\n\
            template: multi_loss\n\
            x: epoch\n\
            y: {key_metric}\n\
            \n")

