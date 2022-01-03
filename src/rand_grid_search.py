"""Permet de réaliser de lancer le programme de machine pour plusieurs hyper paramètres
à la manière du grid search, mais en assurant les mesures des métriques pour affichage
dans le navigateur web via dvc studio, et github, moyennant pour ce dernier l’ajout d’un
fichier de workflow dans le répertoire .github/workflow"""
import subprocess
import random

num_exps = 3
random.seed(42)

for n in range(num_exps):
    params = {
        "dense_high": [256, 512, 1024],
        "batch_size": [128, 256, 512],
        "dense_low": [16, 32, 64],
        "optimizer": random.choice(['rmsprop', 'adam'])
    }
    subprocess.run(["dvc", "exp", "run", "-P", "-f", "--temp", "--queue",
                    "--set-param", f"keras_embedding.dense={params['dense_high'][n]}",
                    "--set-param", f"keras_embedding.batch_size={params['batch_size'][n]}",
                    "--set-param", f"keras_glove.dense={params['dense_low'][n]}",
                    "--set-param", f"keras_lstm.dense={params['dense_low'][n]}",
                    "--set-param", f"keras_w2v.dense={params['dense_low'][n]}",
                    "--set-param", f"model_constants.optimizer={params['optimizer'][n]}",
                    "--set-param", f"keras_bert.dense={params['dense_low'][n]}"]),
