#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：c5626 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：lrj
@Date    ：2023/10/12 11:34 
'''
import os
import csv
import re
import jieba
from gensim.models import FastText
from tqdm import tqdm

# 函数：加载并预处理文本数据
def preprocess_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = re.sub(r'\d+', '', line)
            yield from jieba.cut(line,cut_all=True, use_paddle=True)


root_folder = './data'  # 请替换为您的文件夹路径
years = [str(year) for year in range(2010, 2023)]
companies = os.listdir(root_folder)


# 使用生成器构建语料
class CorpusIterable:
    def __init__(self, root_folder, companies, years):
        self.root_folder = root_folder
        self.companies = companies
        self.years = years

    def __iter__(self):
        for company in tqdm(self.companies,desc = 'company'):
            for year in self.years:
                filename = os.path.join(self.root_folder, company, f'{year}.txt')
                if os.path.exists(filename):
                    yield from preprocess_text(filename)
        yield ['数据资产', '数据资源']

corpus = CorpusIterable(root_folder, companies, years)


# 训练模型
model = FastText(sentences=corpus, vector_size=100, window=5, min_count=1, workers=8,epochs=4)
model.save("wordvecmodel.bin")
