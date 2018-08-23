#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author: mazr

import os
import jieba

from utils.file_helper import ROOT_DIR


DATA_DIR = os.path.join(ROOT_DIR, 'data')

jieba.load_userdict(os.path.join(DATA_DIR, 'userdict', 'sensitive_names.txt'))
jieba.load_userdict(os.path.join(DATA_DIR, 'userdict', 'positions.txt'))

organizations = set()
sensitive_names = []
positions = []
with open(os.path.join(DATA_DIR, 'userdict', 'sensitive_names.txt'), 'r', encoding='utf-8') as f_sn:
    for line in f_sn:
        sensitive_names.append(line.strip())

with open(os.path.join(DATA_DIR, 'userdict', 'positions.txt'), 'r', encoding='utf-8') as f_pos:
    for line in f_pos:
        positions.append(line.strip())

with open(os.path.join(DATA_DIR, 'raw', 'leaders.txt'), 'r', encoding='utf-8') as f_leaders:
    for line in f_leaders:
        start = False
        possible = []
        tokens = jieba.lcut(line.strip())
        for t in tokens[::-1]:
            if t in sensitive_names:
                start = True

            if start:
                if t not in sensitive_names and t not in positions and t != '、':
                    possible.insert(0, t)

        organizations.add(' '.join(possible))
