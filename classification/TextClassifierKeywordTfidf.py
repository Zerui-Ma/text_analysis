#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .TextClassifierTfidf import TextClassifierTfidf


class TextClassifierKeywordTfidf(TextClassifierTfidf):
    """
    使用关键词列表生成特征向量
    """
    def generate_feature_vectors(self, text_list, vocab=None):
        if self.level == 'word':
            self.vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', vocabulary=vocab)
        elif self.level == 'ngram':
            self.vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), vocabulary=vocab)
        elif self.level == 'char':
            self.vectorizer = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3), vocabulary=vocab)

        if self.vectorizer:
            self.vectorizer.fit(self.train_x)

        # transformed = self.vectorizer.transform(text_list).toarray()
        # feature_vectors = []
        # i = 0
        # for text in text_list:
        #     cnt = 0
        #     for word in text.split():
        #         if word not in vocab:
        #             cnt += 1
        #     feature_vectors.append(np.concatenate((transformed[i], [cnt / len(text)])))
        #     i += 1

        # return np.array(feature_vectors)

        transformed = self.vectorizer.transform(text_list)
        other = []
        for text in text_list:
            splitted = text.split()
            if len(splitted) == 0:
                other.append([0])
            else:
                cnt = 0
                for word in splitted:
                    if word not in vocab:
                        cnt += 1
                other.append([cnt / len(splitted)])
        return hstack([transformed, csr_matrix(other)])
