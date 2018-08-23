#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

from random import shuffle

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from .TextClassifierBase import TextClassifierBase


class TextClassifierDoc2Vec(TextClassifierBase):
    def __init__(self, userdict=None):
        super().__init__(userdict)
        self.model = None

    def generate_feature_vectors(self, text_list, vocab=None):
        if not self.model:
            tagged_docs = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(self.train_x)]
            self.model = Doc2Vec(tagged_docs, vector_size=100, window=2, min_count=1, workers=4)
            for _ in range(10):
                shuffle(tagged_docs)
                self.model.train(tagged_docs, total_examples=self.model.corpus_count, epochs=5)

        feature_vectors = []
        for text in text_list:
            feature_vectors.append(self.model.infer_vector(text.split()))

        return np.array(feature_vectors)
