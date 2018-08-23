#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

from keyword_extraction.JiebaKeywordExtractor import JiebaKeywordExtractor
from utils.text_helper import split_sentences

from .TextSummarizerBase import TextSummarizerBase


class SimpleSummarizer(TextSummarizerBase):
    """
    对文本进行关键词提取，将关键词第一次出现的分句作为关键句
    """
    def summarize(self, topK=3):
        # 提取关键词（分值从高到低排序）
        keywords = [item[0] for item in JiebaKeywordExtractor().extract_keywords(self.doc_path)]
        # 对每个关键词，遍历所有分句找出其第一次出现的分句放入集合中
        # 直到集合中关键句为topK时结束循环
        top_set = set()
        sentences = split_sentences(self.doc_path)
        for keyword in keywords:
            for sentence in sentences:
                if keyword in sentence:
                    top_set.add(sentence)
                    break

            if len(top_set) == topK:
                break
        # 将关键句按照其在文本中出现的顺序输出
        summary = []
        for sentence in sentences:
            if sentence in top_set:
                summary.append(sentence)
            if len(summary) == topK:
                break

        return summary
