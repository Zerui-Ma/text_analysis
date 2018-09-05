#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import scrapy

from ..items import CpcSpiderItem


class CpcSpider(scrapy.Spider):
    name = 'cpc'
    start_urls = ['http://cpc.people.com.cn/GB/64162/394696/index.html']

    def parse(self, response):
        tab_names = response.xpath('//li/text()').extract()
        tab_selectors = response.xpath('//div[@class="p_tab"]')
        for i in range(len(tab_names)):
            t_name = tab_names[i]
            t_selector = tab_selectors[i]
            end_while = False
            position = t_selector.xpath('./h2[1]/text()').extract_first()
            following_siblings = t_selector.xpath('./h2[1]/following-sibling::*')
            while True:
                for j in range(len(following_siblings)):
                    f_sibling = following_siblings[j]
                    if f_sibling.re(r'^<table'):
                        name_resume = []
                        items = f_sibling.xpath('.//p')
                        if i == 10:
                            for k in range(len(items)):
                                if k % 2 == 1:
                                    cur = items[k]
                                    pre = items[k-1]
                                    if cur.re(r'<p><a href'):
                                        link = response.urljoin(cur.xpath('./a/@href').extract_first().strip())
                                        name = cur.xpath('./a/text()').extract_first()
                                        org = ''.join(pre.xpath('./text()').extract())
                                        name_resume.append([(name, org), link])

                                    else:
                                        name = cur.xpath('./text()').extract_first()
                                        org = ''.join(pre.xpath('./text()').extract())
                                        name_resume.append([(name, org), None])

                        else:
                            for item in items:
                                if item.re(r'<p><a href'):
                                    link = response.urljoin(item.xpath('./a/@href').extract_first().strip())
                                    if i == 3 or i == 9:
                                        name_resume.append([item.xpath('./a/text()').extract(), link])

                                    else:
                                        name_resume.append([item.xpath('./a/text()').extract_first(), link])

                                else:
                                    name_resume.append([item.xpath('./text()').extract_first(), None])

                        for name, resume in name_resume:
                            if i == 3:
                                if len(name) == 2:
                                    position = name[0]
                                    name = name[1]

                                else:
                                    name = name[0]

                            if i == 9:
                                t_name = name[0]
                                position = '主席'
                                name = name[1][2:]

                            if i == 10:
                                t_name = name[1]
                                if name[0] == '书记处第一书记':
                                    position = '书记处第一书记'
                                    name = '贺军科'

                                else:
                                    position = ''
                                    name = name[0]

                            if '（' not in name:
                                if ':' in name:
                                    tokens = name.split(':')
                                    # yield {'name': tokens[1], 'position': position + tokens[0], 'organization': t_name, 'resume': resume}
                                    yield CpcSpiderItem({'name': tokens[1], 'position': position + tokens[0], 'organization': t_name, 'resume': resume})

                                elif '：' in name:
                                    tokens = name.split('：')
                                    # yield {'name': tokens[1], 'position': position + tokens[0], 'organization': t_name, 'resume': resume}
                                    yield CpcSpiderItem({'name': tokens[1], 'position': position + tokens[0], 'organization': t_name, 'resume': resume})

                                else:
                                    # yield {'name': name, 'position': position, 'organization': t_name, 'resume': resume}
                                    yield CpcSpiderItem({'name': name, 'position': position, 'organization': t_name, 'resume': resume})

                    elif f_sibling.re(r'^<h2>'):
                        position = f_sibling.xpath('./text()').extract_first()
                        following_siblings = f_sibling.xpath('./following-sibling::*')
                        break

                    elif i == 0 and f_sibling.re(r'<p>常务委员会委员'):
                        names = f_sibling.xpath('./text()').extract()[1]
                        # yield {'name': names, 'position': position + '常务委员会委员', 'organization': t_name, 'resume': None}
                        yield CpcSpiderItem({'name': names, 'position': position + '常务委员会委员', 'organization': t_name, 'resume': None})

                    if j == len(following_siblings) - 1:
                        end_while = True
                        break

                if end_while:
                    break
