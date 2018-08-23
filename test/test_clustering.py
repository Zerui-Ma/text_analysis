#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author: mazr

import os
import sys
import unittest

from sklearn.cluster import KMeans

from clustering.MovieClusteringTfidf import MovieClusteringTfidf
from clustering.NewsClusteringTfidf import NewsClusteringTfidf
from utils.file_helper import ROOT_DIR


PLOT_DIR = os.path.join(ROOT_DIR, 'data', 'plot')

class TestClustering(unittest.TestCase):
    def test_movie_clustering(self):
        # 电影聚类测试
        mc = MovieClusteringTfidf()
        mc.load_data('douban_250.pk')
        self.assertEqual(mc.symbols[0], '肖申克的救赎')
        self.assertEqual(len(mc.original), len(mc.symbols))
        self.assertEqual(len(mc.tokenized), len(mc.symbols))

        cluster_path = os.path.join(ROOT_DIR, 'data', 'model', 'movie_clusters.pk')
        if os.path.exists(cluster_path):
            os.remove(cluster_path)

        estimator = KMeans(n_clusters=5, max_iter=500)
        labels_g = mc.generate_clusters(estimator, 'movie_clusters.pk')
        self.assertIsNotNone(labels_g)

        labels_l = mc.load_estimator('movie_clusters.pk')
        self.assertEqual(labels_g, labels_l)

        cluster_colors = {0: 'Red', 1: 'Yellow', 2: 'Green', 3: 'Blue', 4: 'Purple'}
        cluster_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
        mc.plot_clusters_static(cluster_colors, cluster_names, 'movies_static')
        mc.plot_clusters_interactive(cluster_colors, cluster_names, 'movies_interactive')
        mc.plot_clusters_dendrogram('movies_dendrogram')
        self.assertTrue(os.path.exists(os.path.join(PLOT_DIR, 'movies_static.png')))
        self.assertTrue(os.path.exists(os.path.join(PLOT_DIR, 'movies_interactive.html')))
        self.assertTrue(os.path.exists(os.path.join(PLOT_DIR, 'movies_dendrogram.png')))

    def test_news_clustering(self):
        # 新闻聚类测试
        nc = NewsClusteringTfidf()
        nc.load_data('news')
        self.assertTrue(os.path.exists(os.path.join(ROOT_DIR, 'data', 'clustering', 'news.pk')))
        self.assertNotEqual(0, len(nc.symbols))
        self.assertEqual(len(nc.original), len(nc.symbols))
        self.assertEqual(len(nc.tokenized), len(nc.symbols))

        cluster_path = os.path.join(ROOT_DIR, 'data', 'model', 'news_clusters.pk')
        if os.path.exists(cluster_path):
            os.remove(cluster_path)
        estimator = KMeans(n_clusters=7, max_iter=500)
        labels_g = nc.generate_clusters(estimator, 'news_clusters.pk')
        self.assertIsNotNone(labels_g)

        labels_l = nc.load_estimator('news_clusters.pk')
        self.assertEqual(labels_g, labels_l)

        nc.print_clusters_detail()
        cluster_colors = {0: 'Red', 1: 'Orange', 2: 'Yellow', 3: 'Green', 4: 'Blue', 5: 'Aqua', 6: 'Purple'}
        cluster_names = {0: '军事', 1: '教育', 2: '经济', 3: '政治', 4: '体育', 5: '艺术', 6: '交通'}
        nc.plot_clusters_static(cluster_colors, cluster_names, 'news_static')
        nc.plot_clusters_interactive(cluster_colors, cluster_names, 'news_interactive')
        nc.plot_clusters_dendrogram('news_dendrogram', 5.0)
        self.assertTrue(os.path.exists(os.path.join(PLOT_DIR, 'news_static.png')))
        self.assertTrue(os.path.exists(os.path.join(PLOT_DIR, 'news_interactive.html')))
        self.assertTrue(os.path.exists(os.path.join(PLOT_DIR, 'news_dendrogram.png')))


if __name__ == '__main__':
    unittest.main()
