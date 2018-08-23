#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author: mazr

import os
import sys
import unittest

from sklearn import ensemble

from keyword_extraction.GensimKeywordExtractor import GensimKeywordExtractor
from keyword_extraction.JiebaKeywordExtractor import JiebaKeywordExtractor
from utils.file_helper import ROOT_DIR


test_file = os.path.join(ROOT_DIR, 'data', 'raw', 'keyword_test.txt')

class TestKeywordExtraction(unittest.TestCase):
    def test_gensim_extraction(self):
        g_extractor = GensimKeywordExtractor()
        g_keywords = [item[0] for item in g_extractor.extract_keywords(test_file, 3)]
        self.assertListEqual(g_keywords, ['航线', '航空', '芝加哥'])

    def test_jieba_extraction(self):
        j_extractor = JiebaKeywordExtractor()
        j_keywords = [item[0] for item in j_extractor.extract_keywords(test_file, 3)]
        self.assertListEqual(j_keywords, ['航线', '芝加哥', '航空'])


if __name__ == '__main__':
    unittest.main()
