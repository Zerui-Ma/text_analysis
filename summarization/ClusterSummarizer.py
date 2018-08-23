#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import jieba
import nltk
import numpy as np

from utils.text_helper import split_sentences

from .TextSummarizerBase import TextSummarizerBase


class ClusterSummarizer(TextSummarizerBase):
    def summarize(self, topK=3):
        N = 100                    # 单词数量
        CLUSTER_THRESHOLD = 5    # 单词间的距离

        def _score_sentences(sentences, topn_words):
            scores = []
            sentence_idx = -1
            for s in [list(jieba.cut(s)) for s in sentences]:
                sentence_idx += 1
                word_idx = []
                for w in topn_words:
                    try:
                        word_idx.append(s.index(w))    # 关键词出现在该句子中的索引位置
                    except ValueError:                 # w不在句子中
                        pass
                word_idx.sort()
                if len(word_idx) == 0:
                    continue
                # 对于两个连续的单词，利用单词位置索引，通过距离阈值计算簇
                clusters = []
                cluster = [word_idx[0]]
                i = 1
                while i < len(word_idx):
                    if word_idx[i] - word_idx[i-1] < CLUSTER_THRESHOLD:
                        cluster.append(word_idx[i])
                    else:
                        clusters.append(cluster[:])
                        cluster = [word_idx[i]]
                    i+=1
                clusters.append(cluster)
                # 对每个簇打分，每个簇类的最大分数是对句子的打分
                max_cluster_score = 0
                for c in clusters:
                    significant_words_in_cluster = len(c)
                    total_words_in_cluster = c[-1] - c[0] + 1
                    score = 1.0 * significant_words_in_cluster * significant_words_in_cluster / total_words_in_cluster
                    if score > max_cluster_score:
                        max_cluster_score = score
                scores.append((sentence_idx, max_cluster_score))
            return scores

        sentences = split_sentences(self.doc_path)
        words = [w for sentence in sentences for w in jieba.cut(sentence) if w not in self.stopwords if len(w)>1 and w!='\t']
        wordfre = nltk.FreqDist(words)
        topn_words = [w[0] for w in sorted(wordfre.items(), key=lambda d: d[1], reverse=True)][:N]
        scored_sentences = _score_sentences(sentences, topn_words)
        # approach 1,利用均值和标准差过滤非重要句子
        avg = np.mean([s[1] for s in scored_sentences])    # 均值
        std = np.std([s[1] for s in scored_sentences])     # 标准差
        mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_sentences if score > (avg + 0.5 * std)]

        # approach 2，返回top n句子
        top_n_scored = sorted(scored_sentences, key=lambda s: s[1])[-topK:]
        top_n_scored = sorted(top_n_scored, key=lambda s: s[0])
        return [sentences[idx] for (idx,score) in top_n_scored]
