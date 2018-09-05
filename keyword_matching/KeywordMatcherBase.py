#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author: mazr

import re


class KeywordMatcherBase():
    def __init__(self, keyword_files, extra_pattern=None):
        self.keyword_list = []
        for kw_file in keyword_files:
            with open(kw_file, 'r', encoding='utf-8') as fp:
                for line in fp:
                    self.keyword_list.append(line.strip())

        self.keyword_pattern = '|'.join(self.keyword_list)
        if extra_pattern:
            self.keyword_pattern = '{}|{}'.format(self.keyword_pattern, extra_pattern)

    def find_keywords(self, text):
        return re.findall(self.keyword_pattern, text)

    def count_keywords(self, text):
        keywords_found = self.find_keywords(text)
        keyword_dict = {}
        for kw in keywords_found:
            if kw not in keyword_dict:
                keyword_dict[kw] = 1
            else:
                keyword_dict[kw] += 1

        return keyword_dict


if __name__ == '__main__':
    matcher = KeywordMatcherBase(['D:/Python/smartvision/smartvision/algorithm/text_analysis/data/userdict/sensitive_names.txt',\
                          'D:/Python/smartvision/smartvision/algorithm/text_analysis/data/userdict/positions.txt'], r'受贿罪')
    with open('D:/Python/smartvision/smartvision/algorithm/text_analysis/data/raw/corruption_news.txt',\
              'r', encoding='utf-8') as fp:
        text = fp.read()

    kw_list = matcher.find_keywords(text)
    kw_dict = matcher.count_keywords(text)
    print(kw_list)
    print(kw_dict)
