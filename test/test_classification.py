#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author: mazr

import os
import random
import sys
import unittest

from sklearn import ensemble, metrics

from classification.KerasClassifier import KerasClassifier
from classification.TextClassifierTfidf import TextClassifierTfidf
from utils.file_helper import RAW_DIR, ROOT_DIR, MODEL_DIR


DATA_DIR = os.path.join(ROOT_DIR, 'data', 'classification')

def collect_random_data(num_samples):
    target_data_path = os.path.join(RAW_DIR, 'THUCNews')
    texts = []
    labels = []
    for category in ['教育', '体育']:
        cate_path = os.path.join(target_data_path, category)
        for fname in random.sample(os.listdir(cate_path), num_samples):
            if fname.endswith('.txt'):
                with open(os.path.join(cate_path, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(0 if category == '教育' else 1)

    return texts, labels


def test_accuracy(model_name, vectorizer_name, selector_name):
    random_texts, random_labels = collect_random_data(100)

    KC = KerasClassifier()
    KC.load_all(model_name, vectorizer_name, selector_name)
    vals = KC.classify(random_texts)
    print(metrics.classification_report(vals, random_labels, target_names=['教育', '体育']))


class TestClassification(unittest.TestCase):
    # def test_toutiao_classification(self):
    #     test_titles = ['徐嘉余50米仰泳预赛第二晋级 汪顺进200混决赛',\
    #                    '美国海军无人战斗机设计揭开神秘面纱(附图) ',\
    #                    '济南十八中争创新优学校, 改进教育教学方法']
    #     tfidf_classifier = TextClassifierTfidf('word')
    #     if not os.path.exists(os.path.join(DATA_DIR, 'toutiao_labels.pk')) or\
    #        not os.path.exists(os.path.join(DATA_DIR, 'toutiao_texts.pk')):
    #         tfidf_classifier.load_lines_with_labels('toutiao_cat_data.txt', '_!_', [3, 4], 2, 'toutiao')

    #     tfidf_classifier.load_pickle_data('toutiao')
    #     if not os.path.exists(os.path.join(MODEL_DIR, 'toutiao_RF.model')):
    #         model, report = tfidf_classifier.train_model(ensemble.RandomForestClassifier(), 'toutiao_RF.model')
    #         print(report)

    #     else:
    #         model = tfidf_classifier.load_model('toutiao_RF.model')
    #         predictions = tfidf_classifier.classify(model, tfidf_classifier.valid_x)
    #         print(metrics.classification_report(predictions, tfidf_classifier.valid_y))

    #     results = tfidf_classifier.classify(model, test_titles)
    #     self.assertListEqual(list(results), ['news_sports', 'news_military', 'news_edu'])

    # def test_keras_classification(self):
    #     # KC = KerasClassifier()
    #     # KC.load_thuc_news_dataset(RAW_DIR)
    #     # KC.train_model(layers=1, dropout_rate=0.5, epochs=50)
    #     # KC.save_all('thuc_news_mlp_model_4.h5', 'thuc_news_mlp_vct.pk', 'thuc_news_mlp_slt.pk')
    #     test_accuracy('thuc_news_mlp_model_4.h5', 'thuc_news_mlp_vct.pk', 'thuc_news_mlp_slt.pk')
    # ================================================================================================

    def test_keras_cnews(self):
        KC = KerasClassifier()
        KC.load_cnews()
        if os.path.exists(os.path.join(MODEL_DIR, 'keras_cnews_model.h5')):
            KC.load_all('keras_cnews_model.h5', 'keras_cnews_vct.pk', 'keras_cnews_slt.pk')
        else:
            KC.train_model(layers=2, dropout_rate=0.2, epochs=1000)
            KC.save_all('keras_cnews_model.h5', 'keras_cnews_vct.pk', 'keras_cnews_slt.pk')

        test_file = os.path.join(RAW_DIR, 'cnews', 'cnews_test.txt')
        test_texts = []
        test_labels = []
        with open(test_file, 'r', encoding='utf-8') as f_test:
            for line in f_test:
                split = line.strip().split()
                test_texts.append(' '.join(split[1:]))
                label = split[0]
                if label in KC.categories:
                    test_labels.append(KC.cat_to_id[label])
                else:
                    print('Unknown label:{}'.format(label))
                    break

        predictions = KC.classify(test_texts)
        print('Keras MLP:\n')
        print(metrics.classification_report(predictions, test_labels, target_names=KC.categories))

    def test_RF_cnews(self):
        RFC = TextClassifierTfidf('word')
        RFC.load_cnews()
        if os.path.exists(os.path.join(MODEL_DIR, 'RF_cnews_model.pk')):
            RFC.load_all('RF_cnews_model.pk', 'RF_cnews_vct.pk')
        else:
            RFC.train_model(ensemble.RandomForestClassifier())
            RFC.save_all('RF_cnews_model.pk', 'RF_cnews_vct.pk')

        test_file = os.path.join(RAW_DIR, 'cnews', 'cnews_test.txt')
        test_texts = []
        test_labels = []
        with open(test_file, 'r', encoding='utf-8') as f_test:
            for line in f_test:
                split = line.strip().split()
                test_texts.append(' '.join(split[1:]))
                label = split[0]
                if label in RFC.categories:
                    test_labels.append(RFC.cat_to_id[label])
                else:
                    print('Unknown label:{}'.format(label))
                    break

        predictions = RFC.classify(test_texts)
        print('Random Forest:\n')
        print(metrics.classification_report(predictions, test_labels, target_names=RFC.categories))

    def test_keras_sample(self):
        KC = KerasClassifier()
        KC.categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        KC.load_all('keras_cnews_model.h5', 'keras_cnews_vct.pk', 'keras_cnews_slt.pk')
        print('Keras MLP result: {}'.format(KC.classify_single_file(os.path.join(RAW_DIR, 'sample_news.txt'))))

    def test_RF_sample(self):
        RFC = TextClassifierTfidf('word')
        RFC.categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        RFC.load_all('RF_cnews_model.pk', 'RF_cnews_vct.pk')
        print('Random Forest result: {}'.format(RFC.classify_single_file(os.path.join(RAW_DIR, 'sample_news.txt'))))


if __name__ == '__main__':
    unittest.main()
