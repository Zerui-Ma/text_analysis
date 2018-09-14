#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

from sklearn.feature_extraction.text import TfidfVectorizer

from .TextClassifierBase import TextClassifierBase


class TextClassifierTfidf(TextClassifierBase):
    def __init__(self, level, userdict=None):
        super().__init__(userdict)
        self.level = level

    def generate_feature_vectors(self, text_list, vocab=None):
        if not self.vectorizer:
            if self.level == 'word':
                self.vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
            elif self.level == 'ngram':
                self.vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
            elif self.level == 'char':
                self.vectorizer = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)

            if self.vectorizer:
                self.vectorizer.fit(self.train_x)

        return self.vectorizer.transform(text_list)
