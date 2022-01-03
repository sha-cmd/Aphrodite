# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import re
import string

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from stop_words import ENGLISH_STOP_WORDS
from textblob import Word
import nltk
nltk.download('popular')

global dir_list
global dir_lemm_list
global dir_stem_list
global class_list
global class_lemm_list
global class_stem_list

dir_list = ['data/train', 'data/test']
dir_lemm_list = ['data/lemm/train', 'data/lemm/test']
dir_stem_list = ['data/stem/train', 'data/stem/test']

class_list = ['data/train/negatif', 'data/train/positif',
              'data/test/negatif', 'data/test/positif']
class_lemm_list = ['data/lemm/train/negatif', 'data/lemm/train/positif',
                   'data/lemm/test/negatif', 'data/lemm/test/positif']
class_stem_list = ['data/stem/train/negatif', 'data/stem/train/positif',
                   'data/stem/test/negatif', 'data/stem/test/positif']


def write_tweets_in_txt(X_list, y_list, dir_list, class_list):

    for i in range(len(dir_list)):
        if not os.path.isdir(dir_list[i]):
            os.mkdir(dir_list[i])
    for i in range(len(class_list)):
        if not os.path.isdir(class_list[i]):
            os.mkdir(class_list[i])
    n = 0
    for X, y in zip(X_list, y_list):
        if n < 1:
            l = 0
            m = 0
            for it, note in enumerate(list(y['note'].values)):
                if note == 0:
                    with open(class_list[0] + '/tweet_' + str(l) + '.txt', 'w') as f:
                        f.write(X.iloc[it, -1])
                    l += 1
                if note == 4:
                    with open(class_list[1] + '/tweet_' + str(m) + '.txt', 'w') as f:
                        f.write(X.iloc[it, -1])
                    m += 1
        if n == 1:
            l = 0
            m = 0
            for it, note in enumerate(list(y['note'].values)):
                if int(note) == 0:
                    with open(class_list[2] + '/tweet_' + str(l) + '.txt', 'w') as f:
                        f.write(X.iloc[it, -1])
                    l += 1
                if int(note) == 4:
                    with open(class_list[3] + '/tweet_' + str(m) + '.txt', 'w') as f:
                        f.write(X.iloc[it, -1])
                    m += 1
        n += 1


def create_dir():
    """Cette fonction crée la structure pour accueillir les données selon le type de classe"""
    global dir_list
    global class_list
    for directory in dir_list:
        shutil.rmtree(directory) if os.path.isdir(directory) else 0
        os.mkdir(directory)
    for directory in class_list:
        shutil.rmtree(directory) if os.path.isdir(directory) else 0
        os.mkdir(directory)


def custom_standardization(x, stemming):
    text = str(x).lower()
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', text)
    # Isolate punctuation
    s = re.sub(r'([.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•"«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    tweet = re.sub(r"can'?t", ' can not', s)
    tweet = re.sub(r"n't", ' not', tweet)
    tweet = re.sub(r"'s", ' is', tweet)
    tweet = re.sub(r"i'm", ' i am ', tweet)
    tweet = re.sub(r"'ll", ' will', tweet)
    tweet = re.sub(r"'ve", ' have', tweet)
    tweet = re.sub(r"'d", ' would', tweet)
    tweet = re.sub(r'\&amp;|\&gt;|&lt;|\&', ' and ', tweet)
    url = re.compile(r'(https?[^\s]*)')
    smile = re.compile(r'[8:=;][\'`\-]?[\)d]+|[)d]+[\'`\-][8:=;]')
    sad = re.compile(r'[8:=;][\'`\-]?\(+|\)+[\'`\-][8:=;]')
    lol = re.compile(r'[8:=;][\'`\-]?p+')
    tweet = re.sub(r'\@[^\s]+', ' U ', tweet)
    tweet = url.sub(' ', tweet)
    tweet = re.sub(r'\/', ' ', tweet)
    tweet = smile.sub(' H ', tweet)
    tweet = lol.sub(' H ', tweet)
    tweet = sad.sub(' S ', tweet)
    tweet = re.sub(r'([\!\?\.]){2,}', '\g<1>', tweet)
    tweet = re.sub(r'\b(\S*?)([^\s])\2{2,}\b', '\g<1>\g<2>', tweet)
    tweet = re.sub(r'\#', ' #', tweet)
    tweet = re.sub(r'[^\w\#\s\?\<\>]+', ' ', tweet)
    tweet = re.sub('\s+', ' ', tweet)
    text = re.sub('\[.*?\]', '', tweet)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    chain = ''
    if stemming == 'lemmatize':
        chain = ' '.join([Word(word).lemmatize() for word in text.split(' ') if word not in ENGLISH_STOP_WORDS])
    elif stemming == 'stemming':
        chain = ' '.join([Word(word).stem() for word in text.split(' ') if word not in ENGLISH_STOP_WORDS])
    else:
        chain = ' '.join([word for word in text.split(' ')])
    return chain


def process():
    global dir_list
    global dir_lemm_list
    global dir_stem_list
    global class_list
    global class_lemm_list
    global class_stem_list
    X_train_path = 'data/X_train.csv'
    X_test_path = 'data/X_test.csv'
    y_train_path = 'data/y_train.csv'
    y_test_path = 'data/y_test.csv'
    X_train = pd.read_csv(X_train_path, index_col='index')
    y_train = pd.read_csv(y_train_path, index_col='index')
    X_test = pd.read_csv(X_test_path, index_col='index')
    y_test = pd.read_csv(y_test_path, index_col='index')
    create_dir()
    # Écrire les tweets en fichiers textes
    X_nature_train = X_train.copy()
    X_nature_train['tweets'] = X_nature_train['tweets'].apply(lambda x: custom_standardization(x, 'nature'))
    y_nature_train = y_train.copy()
    X_nature_test = X_test.copy()
    X_nature_test['tweets'] = X_nature_test['tweets'].apply(lambda x: custom_standardization(x, 'nature'))
    y_nature_test = y_test.copy()
    X_nature_train.to_csv('data/X_nature_train.csv', index_label='index')
    X_nature_test.to_csv('data/X_nature_test.csv', index_label='index')
    y_nature_train.to_csv('data/y_nature_train.csv', index_label='index')
    y_nature_test.to_csv('data/y_nature_test.csv', index_label='index')
    X_list = [X_nature_train, X_nature_test]
    y_list = [y_nature_train, y_nature_test]
    write_tweets_in_txt(X_list, y_list, dir_list, class_list)
    X = X_nature_train['tweets'].loc[y_nature_train['note'] == 4].apply(lambda x: custom_standardization(x, 'nature'))
    word_cloud(X, 'nature', 'positive')
    X = X_nature_train['tweets'].loc[y_nature_train['note'] == 0].apply(lambda x: custom_standardization(x, 'nature'))
    word_cloud(X, 'nature', 'negative')
    del X_list
    del y_list
    X_lemm_train = X_train.copy()
    X_lemm_train['tweets'] = X_lemm_train['tweets'].apply(lambda x: custom_standardization(x, 'lemmatize'))
    y_lemm_train = y_train.copy()
    X_lemm_test = X_test.copy()
    X_lemm_test['tweets'] = X_lemm_test['tweets'].apply(lambda x: custom_standardization(x, 'lemmatize'))
    y_lemm_test = y_test.copy()
    X_lemm_train.to_csv('data/X_lemm_train.csv', index_label='index')
    X_lemm_test.to_csv('data/X_lemm_test.csv', index_label='index')
    y_lemm_train.to_csv('data/y_lemm_train.csv', index_label='index')
    y_lemm_test.to_csv('data/y_lemm_test.csv', index_label='index')
    X_list = [X_lemm_train, X_lemm_test]
    y_list = [y_lemm_train, y_lemm_test]
    write_tweets_in_txt(X_list, y_list, dir_lemm_list, class_lemm_list)

    X = X_lemm_train['tweets'].loc[y_lemm_train['note'] == 4].apply(lambda x: custom_standardization(x, 'lemmatize'))
    word_cloud(X, 'lemm', 'positive')
    X = X_lemm_train['tweets'].loc[y_lemm_train['note'] == 0].apply(lambda x: custom_standardization(x, 'lemmatize'))
    word_cloud(X, 'lemm', 'negative')

    del X_list
    del y_list
    X_stem_train = X_train.copy()
    X_stem_train['tweets'] = X_train['tweets'].apply(lambda x: custom_standardization(x, 'stemming'))
    y_stem_train = y_train.copy()
    X_stem_test = X_test.copy()
    X_stem_test['tweets'] = X_test['tweets'].apply(lambda x: custom_standardization(x, 'stemming'))
    y_stem_test = y_test.copy()
    X_stem_train.to_csv('data/X_stem_train.csv', index_label='index')
    X_stem_test.to_csv('data/X_stem_test.csv', index_label='index')
    y_stem_train.to_csv('data/y_stem_train.csv', index_label='index')
    y_stem_test.to_csv('data/y_stem_test.csv', index_label='index')
    X_list = [X_stem_train, X_stem_test]
    y_list = [y_stem_train, y_stem_test]
    write_tweets_in_txt(X_list, y_list, dir_stem_list, class_stem_list)
    X = X_stem_train['tweets'].loc[y_stem_train['note'] == 4].apply(lambda x: custom_standardization(x, 'stemming'))
    word_cloud(X, 'stemming', 'positive')
    X = X_stem_train['tweets'].loc[y_stem_train['note'] == 0].apply(lambda x: custom_standardization(x, 'stemming'))
    word_cloud(X, 'stemming', 'negative')


def word_cloud(tweets, kind, sense):
    corpus = []
    for it, row in tweets.items():
        corpus.append(str(row))
    corpus = ' '.join([word for word in corpus])
    # Generate a word cloud image
    mask = np.array(Image.open("twitter_logo.jpg"))
    wordcloud_nature = WordCloud(background_color="white", mode="RGBA", max_words=1000, mask=mask).generate(corpus)

    # create coloring from image
    image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[18, 18])
    plt.imshow(wordcloud_nature.recolor(color_func=image_colors))
    plt.axis("off")

    # store to file
    plt.savefig("wc_" + kind + "_" + sense + ".png", format="png")


def main():
    global corpus
    process()

if __name__ == "__main__":
    main()
