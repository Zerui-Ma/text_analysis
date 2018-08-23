#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import jieba
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.text_helper import split_sentences

from .TextSummarizerBase import TextSummarizerBase


class PagerankSummarizer(TextSummarizerBase):
    """
    使用PageRank算法给每个分句评分并排序输出关键句
    """
    def summarize(self, topK=3):
        sentences = split_sentences(self.doc_path)
        # 对每个分句进行分词
        sentences_tokens = []
        for s in sentences:
            s_tokens = jieba.lcut(s)
            sentences_tokens.append(' '.join([s for s in s_tokens if s not in self.stopwords]))
        # 生成TF-IDF矩阵，矩阵每一行代表每个分句中每个分词的TF-IDF值
        tfidf_matrix = TfidfVectorizer(min_df=1).fit_transform(sentences_tokens)
        # 生成邻接矩阵，其中的每一个元素similarity_graph[i, j]表示第i个分句与第j个分句的相似度
        # 相似度为1说明两个分句完全相同，相似度为0说明两个分句完全不同
        similarity_graph = np.dot(tfidf_matrix, tfidf_matrix.T)
        # 使用NetworkX从邻接矩阵生成图，每个顶点代表每个分句，每条边代表分句间的相关度
        nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
        # 使用PageRank算法对每个分句的关键性进行评分
        scores = nx.pagerank(nx_graph)
        # 排序并输出关键句
        top_scores = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)[:topK]
        top_sentences = [item[1] for item in top_scores]
        # 将关键句按照其在文本中出现的顺序输出
        summary = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary.append(sentence)
            if len(summary) == topK:
                break

        return summary
