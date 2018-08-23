#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import os
import pickle

import jieba
from gensim import corpora, models

from utils.file_helper import DICT_DIR, MODEL_DIR

from .KeywordExtractorBase import KeywordExtractorBase


class GensimKeywordExtractor(KeywordExtractorBase):
    """
    使用gensim实现的关键词提取器
    """
    def __init__(self, corpus_path=None):
        """
        实例化gensim提取器
        :param corpus_path: 语料库路径，用来生成关键词字典及tfidf模型，若为None则载入已有的字典及模型
        """
        super().__init__()
        # 关键词字典路径
        self.dict_path = os.path.join(MODEL_DIR, 'gensim.dict')
        # tfidf模型路径
        self.tfidf_path = os.path.join(MODEL_DIR, 'gensim.tfidf')
        # 载入停用词列表
        stopwords_path = os.path.join(DICT_DIR, 'stopwords.pk')
        with open(stopwords_path, 'rb') as sw:
            self.stopwords = pickle.load(sw)

        if corpus_path:
            # 存在有效的语料库路径则分词并生产新的关键词字典及tfidf模型
            if not os.path.exists(corpus_path):
                print('missing corpus file')
            else:
                corpus = open(corpus_path, 'r', encoding='utf-8', errors='ignore')
                corpus_tokens = []
                for line in corpus:
                    tokens = [t for t in jieba.lcut(line.strip()) if t not in self.stopwords]
                    corpus_tokens.append(tokens)

                self.dictionary = corpora.Dictionary(corpus_tokens)
                self.dictionary.compactify()
                self.dictionary.save(self.dict_path)
                self.tfidf = models.TfidfModel([self.dictionary.doc2bow(tokens) for tokens in corpus_tokens])
                self.tfidf.save(self.tfidf_path)

        else:
            # 不存在有效的语料库路径则载入已有的字典及模型
            if os.path.exists(self.dict_path) and os.path.exists(self.tfidf_path):
                self.dictionary = corpora.Dictionary.load(self.dict_path)
                self.tfidf = models.TfidfModel.load(self.tfidf_path)
            else:
                print('missing .dict or .tfidf files')

    def extract_keywords(self, doc_path, top_k=100):
        """
        gensim实现的关键词提取方法，定义与基类相同
        """
        keyword_weight = []
        # 对文本进行分词及向量化后提取分词的tfidf值
        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as doc_file:
            doc_tokens = [t for t in jieba.lcut(doc_file.read().strip()) if t not in self.stopwords]

        for id, weight in self.tfidf[self.dictionary.doc2bow(doc_tokens)]:
            keyword_weight.append((self.dictionary[id], weight))

        # 根据分词的tfidf值由高到低输出关键词列表
        keyword_weight.sort(key=lambda item: item[1], reverse=True)
        return keyword_weight[:top_k]

    def export_idfs(self, idfs_path):
        """
        gensim实现的idf文件导出方法，定义与基类相同
        """
        with open(idfs_path, 'w', encoding='utf-8', errors='ignore') as f:
            # 直接将模型中的idf值写入文件
            for id in self.tfidf.idfs:
                f.write('{} {}\n'.format(self.dictionary[id], self.tfidf.idfs[id]))

        print('idfs file exported')

    def import_idfs(self, idfs_path):
        """
        gensim实现的idf文件导入方法，定义与基类相同
        """
        if not os.path.exists(idfs_path):
            print('idfs file dose not exist')
        else:
            # 创建关键词-idf字典，直接调用set_idf_value方法
            keyword_value = {}
            with open(idfs_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                for line in f_in:
                    keyword, value = line.strip().split()
                    keyword_value[keyword] = value

            self.set_idf_value(keyword_value)
            print('idfs file imported')

    def set_idf_value(self, keyword_value):
        """
        gensim实现的改变分词idf值方法，定义与基类相同
        """
        for keyword in keyword_value:
            # 若分词不存在字典中则添加
            if keyword not in self.dictionary.token2id:
                self.dictionary.add_documents([[keyword]])

            # 改变添加或模型中idf值或
            self.tfidf.idfs[self.dictionary.token2id[keyword]] = keyword_value[keyword]

        # 保存做出的更改
        self.dictionary.save(self.dict_path)
        self.tfidf.save(self.tfidf_path)
