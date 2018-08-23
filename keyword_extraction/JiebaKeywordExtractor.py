#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import os
import shutil

import jieba
import jieba.analyse

from utils.file_helper import DICT_DIR

from .KeywordExtractorBase import KeywordExtractorBase


class JiebaKeywordExtractor(KeywordExtractorBase):
    """
    使用jieba实现的关键词提取器
    """
    def __init__(self, reset=False):
        """
        实例化jieba提取器
        :param reset: 是否重置提取关键词使用的idf文件（默认为jieba.analyse中的idf.txt）
        """
        super().__init__()
        # jieba默认idf文件路径
        default_idfs_path = os.path.join(os.path.dirname(jieba.analyse.__file__), 'idf.txt')
        # 关键词提取使用的idf文件路径
        self.idfs_path = os.path.join(DICT_DIR, 'jieba_idfs.txt')
        # idf文件不存在或重置则复制jieba默认idf文件
        if not os.path.exists(self.idfs_path) or reset:
            shutil.copyfile(default_idfs_path, self.idfs_path)
        # jieba使用自定义idf文件时只有当idf文件名变化时才会重新载入idf值，需要一个临时文件完成文件名变化
        self.tmp = os.path.join(DICT_DIR, 'tmp.txt')
        with open(os.path.join(DICT_DIR, 'tmp.txt'), 'w') as f:
            f.write('tmp 0.0')
        # 设定关键词提取的停用词
        jieba.analyse.set_stop_words(self.stopwords_path)

    def extract_keywords(self, doc_path, top_k=100):
        """
        jieba实现的关键词提取方法，定义与基类相同
        """
        # 设定提取关键词使用的idf文件路径
        jieba.analyse.set_idf_path(self.idfs_path)
        # 提取关键词
        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as doc_file:
            keywords = jieba.analyse.extract_tags(doc_file.read(), topK=top_k, withWeight=True)

        # 重置idf文件路径（否则idf文件变化后不会被重新载入）
        jieba.analyse.set_idf_path(os.path.join(DICT_DIR, 'tmp.txt'))
        return keywords

    def export_idfs(self, idfs_path):
        """
        jieba实现的idf文件导出方法，定义与基类相同
        """
        fpath = os.path.split(idfs_path)[0]
        # 不存在则创建路径
        if not os.path.exists(fpath) and fpath != '':
            os.makedirs(fpath)
        # 复制idf文件至指定路径
        shutil.copyfile(self.idfs_path, idfs_path)
        print('idfs file exported')

    def import_idfs(self, idfs_path):
        """
        jieba实现的idf文件导入方法，定义与基类相同
        """
        # 将外部idf文件复制至提取关键词使用的idf文件路径
        if os.path.exists(idfs_path):
            os.remove(self.idfs_path)
            shutil.copyfile(idfs_path, self.idfs_path)
            print('idfs file imported')
        else:
            print('idfs file dose not exist')

    def set_idf_value(self, keyword_value):
        """
        jieha实现的改变分词idf值方法，定义与基类相同
        """
        # 读取idf文件
        with open(self.idfs_path, 'r', encoding='utf-8', errors='ignore') as f_read:
            lines = f_read.readlines()

        for i in range(len(lines)):
            # 若所有idf值改变完成则退出循环
            if len(keyword_value) == 0:
                break

            else:
                # 改变分词对应idf值，将其从字典中删除
                keyword = lines[i].split()[0]
                if keyword in keyword_value:
                    lines[i] = '{} {}\n'.format(keyword, keyword_value[keyword])
                    del keyword_value[keyword]

        # 原idf文件中不存在的分词直接添加idf值
        for keyword in keyword_value:
            lines.append('{} {}\n'.format(keyword, keyword_value[keyword]))

        # 将变化写入idf文件
        with open(self.idfs_path, 'w', encoding='utf-8', errors='ignore') as f_write:
            f_write.writelines(lines)
