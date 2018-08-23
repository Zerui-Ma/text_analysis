#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author: mazr

import os
import sys
import unittest

from sklearn import ensemble

from classification.TextClassifierTfidf import TextClassifierTfidf
from utils.file_helper import ROOT_DIR


DATA_DIR = os.path.join(ROOT_DIR, 'data', 'classification')

class TestClassification(unittest.TestCase):
    def test_toutiao_classification(self):
        test_titles = ['徐嘉余50米仰泳预赛第二晋级 汪顺进200混决赛',\
                       '美国海军无人战斗机设计揭开神秘面纱(附图) ',\
                       '济南十八中争创新优学校, 改进教育教学方法']
        tfidf_classifier = TextClassifierTfidf('word')
        if not os.path.exists(os.path.join(DATA_DIR, 'toutiao_labels.pk')) or\
           not os.path.exists(os.path.join(DATA_DIR, 'toutiao_texts.pk')):
            tfidf_classifier.load_lines_with_labels('toutiao_cat_data.txt', '_!_', [3, 4], 2, 'toutiao')

        tfidf_classifier.load_pickle_data('toutiao')
        model, report = tfidf_classifier.train_model(ensemble.RandomForestClassifier(), 'toutiao_RF.model')
        print(report)

        model = tfidf_classifier.load_model('toutiao_RF.model')
        results = tfidf_classifier.classify(model, test_titles)
        self.assertListEqual(list(results), ['news_sports', 'news_military', 'news_edu'])


if __name__ == '__main__':
    unittest.main()
