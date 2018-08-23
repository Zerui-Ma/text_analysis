#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr


def split_sentences(file_path):
    """
    对文本按照指定分句符进行分句
    :return: [分句1, 分句2, 分句3, ...]
    """
    # 读取文本
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
        text = fp.read()

    # 定义分句符
    delimiter = '!?。！？'
    # 遍历文本，每遇到一个分句符便进行分句
    start_idx = 0
    delimiter_idx = 0
    sentences = []
    for i in range(len(text)):
        if text[i] in delimiter:
            delimiter_idx = i
            splitted = text[start_idx:i]
            # 分句时删去句子中的换行符
            sentences.append(''.join([s for s in splitted if s != '\n']))
            start_idx = i + 1
    # 若文本结尾没有分句符，则将剩余文本作为最后一个分句
    if delimiter_idx != len(text) - 1:
        sentences.append(text[start_idx:])

    return sentences