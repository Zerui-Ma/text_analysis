#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import os
import pickle

from utils.file_helper import DICT_DIR


class TextSummarizerBase():
    """
    文本自动摘要基类
    """
    def __init__(self, doc_path):
        """
        定义停用词文件路径，读取待摘要的文本文件
        :param doc_path: 文本文件路径
        """
        stopwords_path = os.path.join(DICT_DIR, 'stopwords.pk')
        with open(stopwords_path, 'rb') as sw:
            self.stopwords = pickle.load(sw)
        self.doc_path = doc_path

    def summarize(self, topK=3):
        """
        生成文本的自动摘要，必须由子类实现
        :param topK: 摘要中句子的数量
        :return: [关键句1, 关键句2, 关键句3,, ...]
        """
        raise NotImplementedError
