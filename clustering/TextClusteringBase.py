#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: mazr

import os
import pickle
import re

import jieba
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpld3
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.externals import joblib
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

from utils.file_helper import DICT_DIR, MODEL_DIR, PLOT_DIR, ROOT_DIR


# 指定默认字体
mpl.rcParams['font.sans-serif'] = ['FangSong']
# 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False

class TextClusteringBase():
    '''
    文本聚类基类
    '''
    def __init__(self, userdict=None):
        self.DATA_DIR = os.path.join(ROOT_DIR, 'data', 'clustering')
        stopwords_path = os.path.join(DICT_DIR, 'stopwords.pk')
        with open(stopwords_path, 'rb') as sw:
            self.stopwords = pickle.load(sw)
        self.estimator = None
        self.frame = None
        self.num_clusters = None
        self.symbols = []           # 每条文本数据的记号（与标签相区别），如电影的标题
        self.original = []          # 原始文本数据
        self.tokenized = []         # 预处理后文本数据
        self.feature_matrix = []    # 特征向量矩阵
        self.dist_matrix = []       # 距离矩阵
        self.groups = None          # 按标签分组的二维化数据
        if userdict:
            jieba.load_userdict(os.path.join(DICT_DIR, userdict))

    def preprocess(self, text):
        '''
        对单条文本数据进行预处理
        '''
        tokens = [t for t in jieba.lcut(text.strip()) if t not in self.stopwords\
             and re.search('[a-zA-Z]', t) is None and re.search('[0-9]', t) is None]
        return ' '.join(tokens)

    def load_data(self, filename):
        '''
        载入每条文本数据后进行预处理
        文本记号存入self.symbols
        原始数据存入self.original
        处理后数据存入self.tokenized
        '''
        raise NotImplementedError

    def generate_feature_vectors(self, data):
        '''
        将每条文本数据转化为向量
        '''
        raise NotImplementedError

    def load_estimator(self, pickle_name):
        '''
        载入聚类模型，打印聚类信息并输出每条文本数据的标签
        '''
        file_path = os.path.join(MODEL_DIR, pickle_name)
        if os.path.exists(file_path):
            self.feature_matrix = self.generate_feature_vectors(self.tokenized)
            self.dist_matrix = 1 - cosine_similarity(self.feature_matrix)    # 从相似度矩阵生成距离矩阵
            self.estimator = joblib.load(file_path)
            clusters = self.estimator.labels_.tolist()
            self.num_clusters = len(set(clusters))

            dataset = {}
            dataset['symbols'] = self.symbols
            dataset['original'] = self.original
            dataset['clusters'] = clusters
            self.frame = pd.DataFrame(dataset, index = [clusters] , columns = ['symbols', 'clusters'])
            print('General clustering info (labels --> num of samples):')
            self.print_clusters_general()
            return clusters
        else:
            print('Pickle does not exist.')

    def generate_clusters(self, estimator, pickle_name):
        '''
        使用指定的聚类算法进行聚类，导出聚类模型，打印聚类信息并输出每条文本数据的标签
        '''
        pickle_path = os.path.join(MODEL_DIR, pickle_name)
        if os.path.exists(pickle_path):
            print('Pickle already exists. Please call "load_estimator" or use another name.')
        else:
            self.feature_matrix = self.generate_feature_vectors(self.tokenized)
            self.dist_matrix = 1 - cosine_similarity(self.feature_matrix)    # 从相似度矩阵生成距离矩阵
            estimator.fit(self.feature_matrix)
            self.estimator = estimator
            joblib.dump(estimator, pickle_path)
            clusters = estimator.labels_.tolist()
            self.num_clusters = len(set(clusters))

            dataset = {}
            dataset['symbols'] = self.symbols
            dataset['original'] = self.original
            dataset['clusters'] = clusters
            self.frame = pd.DataFrame(dataset, index = clusters, columns = ['symbols', 'clusters'])
            print('General clustering info (labels --> num of samples):')
            self.print_clusters_general()
            return clusters

    def print_clusters_general(self):
        '''
        打印聚类概况，显示每个类别包含的数据个数
        '''
        print(self.frame['clusters'].value_counts())

    def print_clusters_detail(self):
        '''
        打印聚类详情，显示每个类别中的数据的记号
        '''
        for i in range(self.num_clusters):
            symbols = ', '.join([symbol for symbol in self.frame.ix[i]['symbols'].values.tolist()])
            print('Cluster {} symbols:\n\n{}\n--------------------------------'.format(i, symbols))

    def plot_clusters_static(self, cluster_colors, cluster_names, plot_name):
        '''
        绘制静态聚类散点图
        cluster_colors, cluster_names均为字典形式，从标签映射到颜色/聚类名
        例如：
        cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
        cluster_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
        '''
        if not self.groups:
            # 将数据点之间的距离降至二维表示以便绘图
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
            pos = mds.fit_transform(self.dist_matrix)  # shape: (n_samples, n_components)
            xs, ys = pos[:, 0], pos[:, 1]
            coordinates = pd.DataFrame(dict(x=xs, y=ys, labels=self.estimator.labels_.tolist(),\
                                       original=self.original, symbols=self.symbols))
            self.groups = coordinates.groupby('labels')

        fig, ax = plt.subplots(figsize=(17, 9))
        ax.margins(0.05)

        # 按标签分组后分别绘图
        for label, group in self.groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                    label=cluster_names[label], color=cluster_colors[label],
                    mec='none')
            ax.set_aspect('auto')
            ax.tick_params(\
                axis= 'x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)
            ax.tick_params(\
                axis= 'y',
                which='both',
                left=False,
                top=False,
                labelleft=False)

        ax.legend(numpoints=1)

        # 散点上标注记号
        for i in range(len(coordinates)):
            ax.text(coordinates.ix[i]['x'], coordinates.ix[i]['y'], coordinates.ix[i]['symbols'], size=8)

        plt.savefig(os.path.join(PLOT_DIR, plot_name + '.png'), dpi=200)
        plt.show()

    def plot_clusters_interactive(self, cluster_colors, cluster_names, plot_name):
        '''
        绘制动态聚类散点图
        '''
        class TopToolbar(mpld3.plugins.PluginBase):
            '''
            将toolbar在图表顶层显示
            '''
            JAVASCRIPT = \
            '''
            mpld3.register_plugin("toptoolbar", TopToolbar);
            TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
            TopToolbar.prototype.constructor = TopToolbar;
            function TopToolbar(fig, props){
                mpld3.Plugin.call(this, fig, props);
            };

            TopToolbar.prototype.draw = function(){
            // the toolbar svg doesn't exist
            // yet, so first draw it
            this.fig.toolbar.draw();

            // then change the y position to be
            // at the top of the figure
            this.fig.toolbar.toolbar.attr("x", 150);
            this.fig.toolbar.toolbar.attr("y", 400);

            // then remove the draw function,
            // so that it is not called again
            this.fig.toolbar.draw = function() {}
            }
            '''
            def __init__(self):
                self.dict_ = {"type": "toptoolbar"}

        if not self.groups:
            # 将数据点之间的距离降至二维表示以便绘图
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
            pos = mds.fit_transform(self.dist_matrix)  # shape: (n_samples, n_components)
            xs, ys = pos[:, 0], pos[:, 1]
            coordinates = pd.DataFrame(dict(x=xs, y=ys, labels=self.estimator.labels_.tolist(),\
                                       original=self.original, symbols=self.symbols))
            self.groups = coordinates.groupby('labels')

        # 定义css样式
        css = \
        '''
        text.mpld3-text, div.mpld3-tooltip {
        font-family:Arial, Helvetica, sans-serif;
        }

        g.mpld3-xaxis, g.mpld3-yaxis {
        display: none; }

        svg.mpld3-figure {
        margin-left: -200px;}
        '''

        fig, ax = plt.subplots(figsize=(14,6))
        ax.margins(0.03)

        # 按标签分组后分别绘图
        for label, group in self.groups:
            points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18,
                            label=cluster_names[label], mec='none',
                            color=cluster_colors[label])
            ax.set_aspect('auto')
            labels = []
            for i in range(group.original.size):
                labels.append('{}:\n{}'.format(group.symbols.iloc[i], group.original.iloc[i]))

            # 按照配置生成tooltip
            tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                            voffset=10, hoffset=10, css=css)

            mpld3.plugins.connect(fig, tooltip, TopToolbar())

            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)


        ax.legend(numpoints=1)

        # 以html格式导出图表
        html = mpld3.fig_to_html(fig)
        with open(os.path.join(PLOT_DIR, plot_name + '.html'), 'w') as f_html:
            f_html.write(html)

        mpld3.display()

    def plot_clusters_dendrogram(self, plot_name, color_threshold=None):
        '''
        绘制聚类树状图
        '''
        linkage_matrix = ward(self.dist_matrix)

        fig, ax = plt.subplots(figsize=(15, 20))
        ax = dendrogram(linkage_matrix, orientation="right", labels=self.symbols, color_threshold=color_threshold)

        plt.tick_params(\
            axis= 'x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)

        plt.tight_layout()

        plt.savefig(os.path.join(PLOT_DIR, plot_name + '.png'), dpi=200)
        plt.show()

    def predict_clusters(self, data):
        '''
        输出给定文本数据的标签
        '''
        # tokenized = [self.preprocess(d) for d in data]
        # feature_vectors = self.generate_feature_vectors(tokenized)
        pass
