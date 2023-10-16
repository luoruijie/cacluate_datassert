#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：c5626 
@File    ：cacluate_dataassert.py
@IDE     ：PyCharm 
@Author  ：lrj
@Date    ：2023/10/12 17:43 
'''
import os
import csv
import re
import jieba
from gensim.models import FastText
from tqdm import tqdm


def preprocess_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = re.sub(r'\d+', '', line)
            yield from jieba.cut(line, cut_all=True, use_paddle=True)


root_folder = './data'  # 请替换为您的文件夹路径
years = [str(year) for year in range(2010, 2023)]
companies = os.listdir(root_folder)

model = FastText.load("wordvecmodel.bin")

# 获取与种子词相似的词
seed_words = ['数据资产', '数据资源']
similar_words = set(seed_words)
for word in seed_words:
    similarities = model.wv.most_similar(word, topn=None)
    similar_words.update([model.wv.index_to_key[i] for i, similarity in enumerate(similarities) if similarity > 0.5])

    # similar_words.update([w for w, _ in model.wv.most_similar(word, topn=100) if model.wv.similarity(word, w) > 0.5]) 效果差，不要用。
target_words = list(similar_words)
print("target_words的长度",len(target_words))
# 计算词频
results = []
for company in tqdm(companies, desc="Processing companies"):
    for year in years:
        filename = os.path.join(root_folder, company, f'{year}.txt')
        if os.path.exists(filename):
            text_words = list(preprocess_text(filename))
            target_frequency = sum(text_words.count(word) for word in target_words)   #公式中的分子
            total_frequency = len(text_words)                                         #公式中的分母
            results.append((company, year, target_frequency, total_frequency))

# 将结果写入CSV文件
csv_filename = "output.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["公司名", "时间", "目标词的总词频", "总词频", "比值"])

    for result in results:
        ratio = (result[2] / result[3] if result[3] != 0 else 0) *100
        writer.writerow([result[0], result[1], result[2], result[3], f"{ratio:.4f}"])
