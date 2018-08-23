#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author: mazr

import os
import re

import xlrd

from utils.file_helper import ROOT_DIR


DATA_DIR = os.path.join(ROOT_DIR, 'data')

data = xlrd.open_workbook(os.path.join(DATA_DIR, 'raw', '全国省市县列表.xlsx'))
table = data.sheets()[0]

province_list = []
for cell in table.col_values(0, 1, 2860):
    if cell != '':
        province_list.append(cell)
        if re.match(r'.*(市|省)$', cell):
            province_list.append(cell[:-1])

province_list.extend(['内蒙古自治区', '广西壮族自治区', '西藏自治区', '宁夏回族自治区', '新疆维吾尔自治区'])
with open(os.path.join(DATA_DIR, 'userdict', 'province.txt'), 'w', encoding='utf-8') as f_p:
    for province in province_list:
        f_p.write(province + '\n')

city_list = []
for cell in table.col_values(2, 1, 2860):
    if cell != ''and cell != '省直辖行政单位':
        cell = re.sub(r' ', '', cell)
        city_list.append(cell)
        if re.match(r'.*市$', cell):
            city_list.append(cell[:-1])
        elif re.match(r'.*地区$', cell):
            city_list.append(cell[:-2])

with open(os.path.join(DATA_DIR, 'userdict', 'city.txt'), 'w', encoding='utf-8') as f_c:
    for city in city_list:
        f_c.write(city + '\n')

district_list= []
for cell in table.col_values(3, 1, 2860):
    if cell != '':
        cell = re.sub(r' |\u3000', '', cell)
        district_list.append(cell)
        if re.match(r'.*市$', cell):
            district_list.append(cell[:-1])

with open(os.path.join(DATA_DIR, 'userdict', 'district.txt'), 'w', encoding='utf-8') as f_d:
    for district in district_list:
        f_d.write(district + '\n')
