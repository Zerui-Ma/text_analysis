#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import json
import os
import pickle

import jieba
import opencc  # 将opencc的__init__.py 中的from version import __version__改为from .version import __version__
import pandas as pd
from sklearn import metrics, model_selection
from sklearn.externals import joblib

from utils.file_helper import DICT_DIR, MODEL_DIR, ROOT_DIR


class TextClassifierBase():
    '''
    文本分类基类:
    数据集加载函数(文本格式、pickle文件、json格式)，分隔数据集
    训练模型函数
    '''
    def __init__(self, userdict=None):
        '''
        加载停用词、自定义词典、初始化路径
        :param userdict: 自定义词典
        '''
        self.RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
        self.DATA_DIR = os.path.join(ROOT_DIR, 'data', 'classification')
        stopwords_path = os.path.join(DICT_DIR, 'stopwords.pk')
        with open(stopwords_path, 'rb') as sw:
            self.stopwords = pickle.load(sw)
        self.dataset = pd.DataFrame()
        self.dataset['texts'] = None
        self.dataset['labels'] = None
        self.train_x, self.valid_x, self.train_y, self.valid_y = None, None, None, None
        if userdict:
            jieba.load_userdict(os.path.join(DICT_DIR, userdict))

    def load_model(self, model_name):
        # 载入模型
        return joblib.load(os.path.join(MODEL_DIR, model_name))

    def load_texts_labels(self, filename_label_dict, pickle_name):
        '''
        加载文本格式数据集分为训练集和测试集; 将数据集保存成pickle文件
        :param filename_label_dict:
        :param pickle_name: 保存的pickle文件名
        :return:
        '''
        texts, labels = [], []
        # 加载数据集
        for filename, label in filename_label_dict.items():
            #label = filename_label_dict[filename]
            file_path = os.path.join(self.DATA_DIR, filename)
            data = open(file_path, 'r', encoding='utf-8')
            cnt = 0
            for line in data:
                try:
                    tokens = [t for t in jieba.lcut(line.strip()) if t not in self.stopwords]
                    text = ' ' + ' '.join(tokens)
                    # 数据预处理，繁体转简体
                    cc = opencc.OpenCC('mix2s')
                    texts.append(cc.convert(text))
                    labels.append(label)
                except Exception as e:
                    print('{}:\n{}'.format(e, data))

                cnt += 1
                if cnt % 1000 == 0:
                    print('Processed {} records.'.format(cnt))
            print('Done processing {} records.'.format(cnt))

        self.dataset['texts'] = texts
        self.dataset['labels'] = labels

        # 保存数据集为pickle文件
        pickle_texts = '{}_texts.pk'.format(pickle_name)
        pickle_labels = '{}_labels.pk'.format(pickle_name)
        with open(os.path.join(self.DATA_DIR, pickle_texts), 'wb') as f_texts:
            pickle.dump(self.dataset['texts'], f_texts)

        with open(os.path.join(self.DATA_DIR, pickle_labels), 'wb') as f_labels:
            pickle.dump(self.dataset['labels'], f_labels)

        # 分隔数据集为测试集和训练集
        self.train_x, self.valid_x, self.train_y, self.valid_y = \
            model_selection.train_test_split(self.dataset['texts'], self.dataset['labels'])

    def load_json_data(self, filename):
        '''
        加载json格式文件
        :param filename:
        :return:
        '''
        file_path = os.path.join(self.DATA_DIR, filename)
        data = json.loads(open(file_path, 'r', encoding='utf-8').read(), strict=False)
        texts, labels = [], []
        for r in data['RECORDS']:
            text = r['video_name']
            # cc = opencc.OpenCC('mix2s')
            # texts.append(cc.convert(text))
            texts.append(text.strip())
            labels.append(r['tort_result'])
        # for record in data['RECORDS']:
        #     try:
        #         if langid.classify(record['introduction'])[0] == 'zh':
        #             text = record['name']
        #             tokens = [t for t in jieba.lcut(record['introduction']) if t not in self.stopwords]
        #             text += ' ' + ' '.join(tokens)
        #             cc = opencc.OpenCC('mix2s')
        #             texts.append(cc.convert(text))
        #             labels.append(record['theme'])
        #     except Exception as e:
        #         print('{}:\n{}'.format(e, record))
        #
        #     cnt += 1
        #     if cnt % 1000 == 0:
        #         print('Processed {} records.'.format(cnt))
        # print('Done processing {} records.'.format(cnt))

        self.dataset['texts'] = texts
        self.dataset['labels'] = labels
        # 保存pickle文件
        pickle_texts = '{}_texts.pk'.format(os.path.splitext(filename)[0])
        pickle_labels = '{}_labels.pk'.format(os.path.splitext(filename)[0])
        with open(os.path.join(self.DATA_DIR, pickle_texts), 'wb') as f_texts:
            pickle.dump(self.dataset['texts'], f_texts)

        with open(os.path.join(self.DATA_DIR, pickle_labels), 'wb') as f_labels:
            pickle.dump(self.dataset['labels'], f_labels)

        # 分隔训练集、测试集
        self.train_x, self.valid_x, self.train_y, self.valid_y = \
            model_selection.train_test_split(self.dataset['texts'], self.dataset['labels'])

    def load_pickle_data(self, filename):
        '''
        加载pickle文件,分隔测试集
        :param filename: 训练集文件名
        :return:
        '''
        texts_path = os.path.join(self.DATA_DIR, '{}_texts.pk'.format(filename))
        labels_path = os.path.join(self.DATA_DIR, '{}_labels.pk'.format(filename))
        if os.path.exists(texts_path) and os.path.exists(labels_path):
            with open(texts_path, 'rb') as f_t:
                self.dataset['texts'] = pickle.load(f_t)
            with open(labels_path, 'rb') as f_l:
                labels = pickle.load(f_l)
                self.dataset['labels'] = labels

            self.train_x, self.valid_x, self.train_y, self.valid_y = \
                model_selection.train_test_split(self.dataset['texts'], self.dataset['labels'])

        else:
            print('Missing pickle file(s).')

    def load_lines_with_labels(self, filename, delimiter, text_idx, label_idx, pickle_name, simplify=False):
        file_path = os.path.join(self.RAW_DIR, filename)
        texts, labels = [], []
        with open(file_path, 'r', encoding='utf-8') as f_all:
            cnt = 0
            for line in f_all:
                splitted = line.strip().split(delimiter)
                try:
                    raw_text = ''
                    for t_idx in text_idx:
                        raw_text += splitted[t_idx] + ' '

                    tokens = [t for t in jieba.lcut(raw_text.strip()) if t not in self.stopwords]
                    text = ' '.join(tokens)
                    if simplify:
                    # 数据预处理，繁体转简体
                        cc = opencc.OpenCC('mix2s')
                        texts.append(cc.convert(text))
                    else:
                        texts.append(text)

                    label = splitted[label_idx]
                    labels.append(label)
                except Exception as e:
                    print('{}:\n{}'.format(e, line))

                cnt += 1
                if cnt % 1000 == 0:
                    print('Processed {} records.'.format(cnt))
            print('Done processing {} records.'.format(cnt))

        self.dataset['texts'] = texts
        self.dataset['labels'] = labels

        # 保存数据集为pickle文件
        pickle_texts = '{}_texts.pk'.format(pickle_name)
        pickle_labels = '{}_labels.pk'.format(pickle_name)
        with open(os.path.join(self.DATA_DIR, pickle_texts), 'wb') as f_texts:
            pickle.dump(self.dataset['texts'], f_texts)

        with open(os.path.join(self.DATA_DIR, pickle_labels), 'wb') as f_labels:
            pickle.dump(self.dataset['labels'], f_labels)

        # 分隔数据集为测试集和训练集
        self.train_x, self.valid_x, self.train_y, self.valid_y = \
            model_selection.train_test_split(self.dataset['texts'], self.dataset['labels'])

    def export_train_valid(self, filename):
        with open(os.path.join(self.DATA_DIR, '{}_train_x.pk'.format(filename)), 'wb') as f_tx:
            pickle.dump(self.train_x, f_tx)

        with open(os.path.join(self.DATA_DIR, '{}_train_y.pk'.format(filename)), 'wb') as f_ty:
            pickle.dump(self.train_y, f_ty)

        with open(os.path.join(self.DATA_DIR, '{}_valid_x.pk'.format(filename)), 'wb') as f_vx:
            pickle.dump(self.valid_x, f_vx)

        with open(os.path.join(self.DATA_DIR, '{}_valid_y.pk'.format(filename)), 'wb') as f_vy:
            pickle.dump(self.valid_y, f_vy)

    def import_train_valid(self, filename):
        with open(os.path.join(self.DATA_DIR, '{}_train_x.pk'.format(filename)), 'rb') as f_tx:
            self.train_x = pickle.load(f_tx)

        with open(os.path.join(self.DATA_DIR, '{}_train_y.pk'.format(filename)), 'rb') as f_ty:
            self.train_y = pickle.load(f_ty)

        with open(os.path.join(self.DATA_DIR, '{}_valid_x.pk'.format(filename)), 'rb') as f_vx:
            self.valid_x = pickle.load(f_vx)

        with open(os.path.join(self.DATA_DIR, '{}_valid_y.pk'.format(filename)), 'rb') as f_vy:
            self.valid_y = pickle.load(f_vy)

    def generate_feature_vectors(self, text_list, vocab=None):
        '''
        特征向量生成函数
        :param text_list:
        :param vocab:
        :return:
        '''
        raise NotImplementedError

    def train_model(self, classifier, model_name, is_neural_net=False, vocab=None):
        '''
        根据分类器训练模型，并预测结果
        :param classifier: 分类器
        :param is_neural_net:
        :param vocab: 字典
        :return:
        '''
        print('Strat training.')
        # 获取训练数据、测试数据特征和向量
        feature_vectors_train = self.generate_feature_vectors(self.train_x, vocab=vocab)
        feature_vectors_valid = self.generate_feature_vectors(self.valid_x, vocab=vocab)
        # 训练模型并预测
        classifier.fit(feature_vectors_train, self.train_y)
        predictions = classifier.predict(feature_vectors_valid)
        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        # 保存模型文件
        joblib.dump(classifier, os.path.join(MODEL_DIR, model_name))
        print('Training complete.')
        # return classifier, metrics.accuracy_score(predictions, self.valid_y)
        return classifier, metrics.classification_report(predictions, self.valid_y)

    def classify(self, classifier, text_list, vocab=None, lable_flag=True):
        '''
        根据分类器预测
        :param classifier:
        :param text_list:
        :return:
        '''

        for i in range(len(text_list)):
            # cc = opencc.OpenCC('mix2s')
            # simplified = cc.convert(text_list[i])
            tokens = [t for t in jieba.lcut(text_list[i]) if t not in self.stopwords]
            text_list[i] = ' '.join(tokens)

        feature_vectors = self.generate_feature_vectors(text_list, vocab=vocab)
        if lable_flag:
            predictions = classifier.predict(feature_vectors)
        else:
            predictions = classifier.predict_proba(feature_vectors)
        return predictions
