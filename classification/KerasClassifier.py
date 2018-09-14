#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import os
import pickle
import random

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense, Dropout

from utils.file_helper import MODEL_DIR, preprocess_file

from .TextClassifierBase import TextClassifierBase


# 向量化参数
# n-gram模型的取值范围
NGRAM_RANGE = (1, 2)

# 向量化的特征数，这里取最重要的20000个特征
TOP_K = 20000

# 对文本进行分割的单位
# 'word'-以词为单位进行分割 'char'-以字符为单位进行分割
TOKEN_MODE = 'word'

# 最低文档频率，如果某单位的文档频率低于该值则不作为特征
MIN_DOCUMENT_FREQUENCY = 2

class KerasClassifier(TextClassifierBase):
    def __init__(self, userdict=None):
        super().__init__(userdict)
        self.num_classes = None
        self.selector = None

    def generate_feature_vectors(self, text_list, vocab=None):
        '''
        生成传入文本列表对应的特征向量
        '''
        if not self.vectorizer:
            # tf-idf向量化参数
            kwargs = {
                    'ngram_range': NGRAM_RANGE,
                    'dtype': 'int32',
                    'strip_accents': 'unicode',
                    'decode_error': 'replace',
                    'analyzer': TOKEN_MODE,
                    'min_df': MIN_DOCUMENT_FREQUENCY,
            }
            self.vectorizer = TfidfVectorizer(**kwargs)

            # 由训练文本生成向量化模型并生成训练文本的向量
            self.train_vec = self.vectorizer.fit_transform(self.train_x)

            # 选择最重要的TOP-K个特征
            self.selector = SelectKBest(f_classif, k=min(TOP_K, self.train_vec.shape[1]))
            self.selector.fit(self.train_vec, self.train_y)
            self.train_vec = self.selector.transform(self.train_vec).astype('float32')

        vectorized = self.vectorizer.transform(text_list)
        return self.selector.transform(vectorized).astype('float32')


    def _get_last_layer_units_and_activation(self, num_classes):
        '''
        根据数据类别的数目决定最后一层网络的输出单位以及激活函数
        '''
        if num_classes == 2:
            activation = 'sigmoid'
            units = 1
        else:
            activation = 'softmax'
            units = num_classes
        return units, activation


    def generate_mlp_model(self, layers, units, dropout_rate, input_shape, num_classes):
        '''
        实例化一个MLP模型

        参数:
            layers: 模型中Dense layer的数量
            units: 每层输出的数据维度
            dropout_rate: Dropout layer中弃用输入数据的百分百，防止过拟合
            input_shape: 输入数据的维数
            num_classes: 输出的类别数
        '''
        op_units, op_activation = self._get_last_layer_units_and_activation(num_classes)
        self.model = models.Sequential()
        self.model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

        for _ in range(layers-1):
            self.model.add(Dense(units=units, activation='relu'))
            self.model.add(Dropout(rate=dropout_rate))

        self.model.add(Dense(units=op_units, activation=op_activation))


    def train_model(self,
                    learning_rate=1e-3,
                    epochs=1000,
                    batch_size=128,
                    layers=2,
                    units=64,
                    dropout_rate=0.2):
        '''
        训练n-gram模型

        参数:
            learning_rate: 学习率
            epoch: 训练循环次数
            batch_size: 每批次训练的样本数
            layers: 模型中Dense layer的数量
            units: 每层输出的数据维度
            dropout_rate: Dropout layer中弃用输入数据的百分百，防止过拟合
        '''
        # 验证集文本向量化
        self.valid_vec = self.generate_feature_vectors(self.valid_x)

        # 实例化模型
        self.generate_mlp_model(layers=layers,units=units,
                                dropout_rate=dropout_rate,
                                input_shape=self.train_vec.shape[1:],
                                num_classes=self.num_classes)

        # 传入训练参数
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

        # 生成回调函数，当连续两个epoch中loss没有降低时提前停止训练
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2)]

        # 训练并验证模型
        history = self.model.fit(
                self.train_vec,
                self.train_y,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=(self.valid_vec, self.valid_y),
                verbose=2,  # Logs once per epoch.
                batch_size=batch_size)

        history = history.history
        print('Validation accuracy: {acc}, loss: {loss}'.format(
                acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

        return self.model, (history['val_acc'][-1], history['val_loss'][-1])


    def save_all(self, model_name, vectorizer_name, selector_name):
        '''
        导出训练模型，向量化模型和选择器模型
        '''
        self.model.save(os.path.join(MODEL_DIR, model_name))
        with open(os.path.join(MODEL_DIR, vectorizer_name), 'wb') as f_v:
            pickle.dump(self.vectorizer, f_v)

        with open(os.path.join(MODEL_DIR, selector_name), 'wb') as f_s:
            pickle.dump(self.selector, f_s)


    def load_all(self, model_name, vectorizer_name, selector_name):
        '''
        载入训练模型，向量化模型和选择器模型
        '''
        self.model = models.load_model(os.path.join(MODEL_DIR, model_name))
        with open(os.path.join(MODEL_DIR, vectorizer_name), 'rb') as f_v:
            self.vectorizer = pickle.load(f_v)

        with open(os.path.join(MODEL_DIR, selector_name), 'rb') as f_s:
            self.selector = pickle.load(f_s)


    def classify(self, text_list, batch_size=32, verbose=0):
        '''
        输出文本列表中每条文本的分类
        '''
        if self.model and self.vectorizer and self.selector:
            vectorized = self.generate_feature_vectors(text_list)
            return self.model.predict_classes(vectorized,
                                              batch_size=batch_size,
                                              verbose=verbose)
        else:
            print('Please train or load a model.')


    def classify_single_file(self, filepath, batch_size=32, verbose=0):
        '''
        识别单一文本数据所属类别
        '''
        preprocessed = preprocess_file(filepath)
        label_id = self.classify([preprocessed], batch_size=batch_size, verbose=verbose)
        return self.categories[label_id[0]]
