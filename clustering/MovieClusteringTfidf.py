#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from .TextClusteringBase import TextClusteringBase


class MovieClusteringTfidf(TextClusteringBase):
    '''
    电影文本信息聚类，使用TF-IDF生成特征向量
    '''
    def load_data(self, filename):
        with open(os.path.join(self.DATA_DIR, filename), 'rb') as fp:
            data = pickle.load(fp)

        for d in data:
            if d is not None:
                self.symbols.append(d[0])
                self.original.append(d[1])
                self.tokenized.append(self.preprocess(d[1]))

    def generate_feature_vectors(self, data):
        tfidf_vectorizer = TfidfVectorizer(max_features=20000, analyzer='word')
        return tfidf_vectorizer.fit_transform(data)
