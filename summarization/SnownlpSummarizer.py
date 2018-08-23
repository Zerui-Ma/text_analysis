#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

from snownlp import SnowNLP

from utils.text_helper import split_sentences

from .TextSummarizerBase import TextSummarizerBase


class SnownlpSummarizer(TextSummarizerBase):
    """
    使用SnowNLP实现的文本自动摘要
    """
    def summarize(self, topK=3):
        """
        SnowNLP内置了文本自动摘要的方法，可以直接调用
        但是SnowNLP默认使用逗号作为分句的符号之一，为了不使用逗号分句，
        修改了SnowNLP的源代码
        .../Lib/site-packages/snownlp/normal/__init.py__第34行：
        delimiter = re.compile('[，。？！；]') ==> delimiter = re.compile('[。？！；]')
        """
        sentences = split_sentences(self.doc_path)
        # 将分句合为一个字符串，调用SnowNLP.summary()生成关键句
        s = SnowNLP('。'.join(sentences))
        top_sentences = s.summary(topK)
        # 将关键句按照其在文本中出现的顺序输出
        summary = []
        for sentence in sentences:
            for ts in top_sentences:
                if ts in sentence:
                    summary.append(ts)

            if len(summary) == topK:
                break

        return summary
