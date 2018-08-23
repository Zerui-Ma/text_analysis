#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import os
import pickle
from random import shuffle

from sklearn.feature_extraction.text import TfidfVectorizer

from utils.file_helper import ROOT_DIR

from .TextClusteringBase import TextClusteringBase


class NewsClusteringTfidf(TextClusteringBase):
    '''
    新闻文本信息聚类，使用TF-IDF生成特征向量
    '''
    def load_data(self, filename):
        pickle_path = os.path.join(self.DATA_DIR, filename + '.pk')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f_data:
                data = pickle.load(f_data)
                self.symbols = data[0]
                self.original = data[1]
                self.tokenized = data[2]

        else:
            root_path = os.path.join(ROOT_DIR, 'data', 'raw', filename)
            file_list = []
            for root, dirs, files in os.walk(root_path):
                if len(files) > 0:
                    files = [os.path.join(root, root, f) for f in files]
                    file_list.extend(files)

            shuffle(file_list)
            for news_file in file_list:
                with open(news_file, 'r', encoding='utf-8') as f_news:
                    raw = f_news.read()
                    self.symbols.append(os.path.splitext(news_file)[0].split('\\')[-1])
                    self.original.append(raw)
                    self.tokenized.append(self.preprocess(raw))

            with open(pickle_path, 'wb') as f_data:
                pickle.dump((self.symbols, self.original, self.tokenized), f_data)

    def generate_feature_vectors(self, data):
        tfidf_vectorizer = TfidfVectorizer(max_features=20000, analyzer='word')
        return tfidf_vectorizer.fit_transform(data)
