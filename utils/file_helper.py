#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import os
import pickle
import jieba


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DICT_DIR = os.path.join(ROOT_DIR, 'data', 'userdict')
MODEL_DIR = os.path.join(ROOT_DIR, 'data', 'model')
PLOT_DIR = os.path.join(ROOT_DIR, 'data', 'plot')
RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
with open(os.path.join(DICT_DIR, 'stopwords.pk'), 'rb') as fp:
    stopwords = pickle.load(fp)

def generate_vocab(file_path, top_k):
        '''
        获取文件前top_k个关键词形成词库
        :param file_path: 关键词文件
        :param top_k: 关键词个数
        :return: {word: index, ...}
        '''
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab = {}
            idx = 0
            for line in f:
                vocab[line.strip()] = idx
                idx += 1
                if idx == top_k:
                    return vocab
            return vocab


def stopwords_pickle2txt():
    with open(os.path.join(DICT_DIR, 'stopwords.pk'), 'rb') as fp:
        word_list = pickle.load(fp)

    with open(os.path.join(DICT_DIR, 'stopwords.txt'), 'w', encoding='utf-8') as ft:
        for word in word_list:
            ft.write('{}\n'.format(word))


def stopwords_txt2pickle():
    word_list = []
    with open(os.path.join(DICT_DIR, 'stopwords.txt'), 'r', encoding='utf-8') as ft:
        for line in ft:
            word_list.append(line.strip())

    with open(os.path.join(DICT_DIR, 'stopwords.pk'), 'wb') as fp:
        pickle.dump(word_list, fp)


def preprocess_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as fp:
        tokens = jieba.lcut(fp.read())

    return ' '.join([t for t in tokens if t not in stopwords])
