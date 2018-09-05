#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import codecs
import json
import os
import urllib

from bs4 import BeautifulSoup

from utils.file_helper import ROOT_DIR

RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
JSON_IN = os.path.join(RAW_DIR, 'cpc_raw.json')
JSON_OUT = os.path.join(RAW_DIR, 'cpc_leaders.json')

def get_resume(url):
        res = urllib.request.urlopen(url)
        soup = BeautifulSoup(res, 'lxml')
        resume = ''
        for line in soup.find_all('p', attrs={'style': 'text-indent: 2em;'}):
            if line.string:
                resume += line.string

        return resume


def generate_cpc_leaders():
    with open(JSON_IN, 'r', encoding='utf-8') as f_in:
        raw_info = json.load(f_in)

    cpc_leaders = {}
    for leader in raw_info:
        name = leader['name']
        position = leader['position']
        organization = leader['organization']
        resume_url = leader['resume']
        if resume_url:
            resume = get_resume(resume_url)
        else:
            resume = None

        # 多个人名
        if '\u3000' in name:
            name_list = name.split('\u3000')

            for name in name_list:
                if name not in cpc_leaders:
                    cpc_leaders[name] = {'position': [position], 'organization': [organization], 'resume': resume}

                else:
                    cpc_leaders[name]['position'] = list(set(cpc_leaders[name]['position'] + [position]))
                    cpc_leaders[name]['organization'] = list(set(cpc_leaders[name]['organization'] + [organization]))

        # 单个人名
        else:
            if name not in cpc_leaders:
                    cpc_leaders[name] = {'position': [position], 'organization': [organization], 'resume': resume}

            else:
                cpc_leaders[name]['position'] = list(set(cpc_leaders[name]['position'] + [position]))
                cpc_leaders[name]['organization'] = list(set(cpc_leaders[name]['organization'] + [organization]))
                if cpc_leaders[name]['resume'] == None:
                    cpc_leaders[name]['resume'] = resume

    f_out = codecs.open(JSON_OUT, 'w', encoding='utf-8')
    f_out.write(json.dumps(cpc_leaders, indent=4, ensure_ascii=False))
    f_out.close()


if __name__ == '__main__':
    # generate_cpc_leaders()
    print(get_resume('http://cpc.people.com.cn/n1/2018/0319/c64094-29875773.html'))
