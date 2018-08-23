#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author: mazr

import os

import jieba
import jieba.posseg as pseg

from utils.file_helper import ROOT_DIR


DATA_DIR = os.path.join(ROOT_DIR, 'data')

jieba.load_userdict(os.path.join(DATA_DIR, 'userdict', 'sensitive_names.txt'))
sensitive_names = []
positions = set()
with open(os.path.join(DATA_DIR, 'userdict', 'sensitive_names.txt'), 'r', encoding='utf-8') as f_sn:
    for line in f_sn:
        sensitive_names.append(line.strip())

with open(os.path.join(DATA_DIR, 'raw', 'leaders.txt'), 'r', encoding='utf-8') as f_leaders:
    for line in f_leaders:
        tokens = pseg.lcut(line.strip())
        for i in range(len(tokens)):
            if tokens[i].word in sensitive_names + ['、']:
                target = tokens[i-1]
                if target.flag == 'n':
                    positions.add(target.word)

main_positions = set()
for p in positions:
    if p[0] == '副':
        main_positions.add(p[1:])

positions = positions | main_positions
print(positions)
remove_set = set(['公安系统', '总局局长', '局局长', '公司', '总公司', '开除党籍', '职务', '厅', '官', '总', '室主任'])
add_set = set(['主任', '风险官', '常委', '信息官', '处长', '参谋', '参事'])
positions = positions - remove_set
positions = positions | add_set

with open(os.path.join(DATA_DIR, 'userdict', 'positions.txt'), 'w', encoding='utf-8') as f_pos:
    for p in positions:
        f_pos.write(p + '\n')

with open(os.path.join(DATA_DIR, 'raw', 'leaders.txt'), 'r', encoding='utf-8') as f_leaders:
    for line in f_leaders:
        need_printing = True
        for p in positions:
            if p in line:
                need_printing = False
                break
        if need_printing:
            print(line)
