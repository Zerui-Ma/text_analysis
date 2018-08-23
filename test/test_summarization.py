#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author: mazr

import os
import sys
import unittest

from summarization.ClusterSummarizer import ClusterSummarizer
from summarization.PagerankSummarizer import PagerankSummarizer
from summarization.SimpleSummarizer import SimpleSummarizer
from summarization.SnownlpSummarizer import SnownlpSummarizer
from utils.file_helper import ROOT_DIR


test_file = os.path.join(ROOT_DIR, 'data', 'raw', 'summarization_test.txt')

class TestSummarization(unittest.TestCase):
    def test_simple_summarizer(self):
        sp_summarizer = SimpleSummarizer(test_file)
        summary = sp_summarizer.summarize(3)
        self.assertIsNotNone(summary)
        print('{}:\n\n{}\n============================'.format('SimpleSummarizer', str(summary)))

    def test_snownlp_summarizer(self):
        sn_summarizer = SnownlpSummarizer(test_file)
        summary = sn_summarizer.summarize(3)
        self.assertIsNotNone(summary)
        print('{}:\n\n{}\n============================'.format('SnownlpSummarizer', str(summary)))

    def test_cluster_summarizer(self):
        cl_summarizer = ClusterSummarizer(test_file)
        summary = cl_summarizer.summarize(3)
        self.assertIsNotNone(summary)
        print('{}:\n\n{}\n============================'.format('ClusterSummarizer', str(summary)))

    def test_pagerank_summarizer(self):
        pr_summarizer = PagerankSummarizer(test_file)
        summary = pr_summarizer.summarize(3)
        self.assertIsNotNone(summary)
        print('{}:\n\n{}\n============================'.format('PagerankSummarizer', str(summary)))


if __name__ == '__main__':
    unittest.main()
