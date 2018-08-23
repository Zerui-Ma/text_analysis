#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import os

from utils.file_helper import DICT_DIR


class KeywordExtractorBase():
    """
    关键词提取器基类，方法必须由子类实现
    """
    def __init__(self):
        """
        定义文件存放路径
        """
        self.stopwords_path = os.path.join(DICT_DIR, 'stopwords.txt')

    def extract_keywords(self, doc_path, top_k=100):
        """
        定义关键词提取方法，必须由子类实现
        :param doc_path: 待提取关键词文本的路径
        :param top_k: 提取关键词个数
        :return: 关键词及其对应权重列表 [(关键词1, 权重1), (关键词2, 权重2), ...]
        """
        raise NotImplementedError

    def export_idfs(self, idfs_path):
        """
        导出分词的idf值至指定文件，必须由子类实现
        :param idfs_path: 导出的文件路径
        :return:
        """
        raise NotImplementedError

    def import_idfs(self, idfs_path):
        """
        由指定文件导入分词的idf值，必须由子类实现
        :param idfs_path: 导入的文件路径
        :return:
        """
        raise NotImplementedError

    def set_idf_value(self, keyword_value):
        """
        改变指定分词的idf值，必须由子类实现
        :param keyword_value: 需要改变的关键词-idf值字典 {关键词1: 权重1, 关键词2: 权重2, ...}
        :return:
        """
        raise NotImplementedError
