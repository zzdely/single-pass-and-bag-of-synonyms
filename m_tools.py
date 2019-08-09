# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：single-pass -> tools
@IDE    ：PyCharm
@Author ：Zhang zhe
@Date   ：2019/6/29 21:38
=================================================='''
import gc
import numpy as np
import pandas as pd
from numpy import math
from numpy import zeros
from collections import Counter
from sklearn.preprocessing import normalize
from w2v import Word2VecModel
from matplotlib import pyplot as plt
import seaborn as sns
# average-link策略（即取平均相似度作为向量与簇的相似度）
def getaveragelink(cos_list):
    sum_v = 0
    for i in cos_list:
        sum_v = sum_v + i
    sum_v = sum_v / len(cos_list)
    return sum_v


# 归一化：分量/模
def guiyihua(word_v):
    # sum = 0
    new_word_v = []
    x_norm = np.linalg.norm(word_v)
    # 傻瓜归一化
    # for i in word_v:
    #     sum=sum+i*i
    for i in word_v:
        i = i / x_norm
        new_word_v.append(i)
    y_norm = np.linalg.norm(new_word_v)
    return new_word_v


def zero_vec(dimension):
    return zeros([1,dimension])[0].tolist()


# 分词结果去停用词
def cut_without_stop(sentence, s_list):
    stopwords = s_list  # 这里加载停用词的路径
    outstr = ''
    for word in sentence.split(" "):
        # print(word)
        if word not in stopwords:
            if word != " ":
                outstr = outstr + word
                outstr = outstr + " "
    return outstr


# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 计算欧氏距离
def getEuclidean(point1, point2):
    vec1 = np.array(point1)
    vec2 = np.array(point2)
    dist = np.linalg.norm(vec1-vec2)
    return dist


# cos计算2
def get_cossim(vector_a, vector_b):
    vector1 = np.array(vector_a)
    vector2 = np.array(vector_b)
    sim = np.dot(vector1, vector2) / \
        (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return sim


def compute_idf(news_list):
    all_dict = {}
    idf_dict = {}
    total = 0
    for line in news_list:
        temp_dict = {}
        total += 1
        # cut_line = jieba.cut(line, cut_all=False)
        for word in line:
            temp_dict[word] = 1
        for key in temp_dict:
            num = all_dict.get(key, 0)
            all_dict[key] = num + 1
    for key in all_dict:
        # w = key.encode('utf-8')
        p = '%.10f' % (math.log(total / (all_dict[key] + 1)))
        idf_dict[key] = float(p)
    # print(sorted(Counter(all_dict).items(), key=lambda d: d[1], reverse=True)[0:10])
    return idf_dict

def heatmap_draw(n_list):
    m_dict = {}
    a = Word2VecModel()
    a.load_embedding()
    w2vmodel = a.get_embedding()

    # L2正则化
    x_normalized = normalize(w2vmodel, norm='l2', axis=0)
    w2vvocabu = a.get_vocabu()

    for weight, w in zip(x_normalized, w2vvocabu):
        if w in ["_pad_", "_nil_", "_ukn_", "_sos_", "_eos_"] and w != "":
            continue
        else:
            m_dict[w] = weight
    del x_normalized, w2vvocabu, w2vmodel
    gc.collect()
    news_vec_list = []
    values=[]
    for i in n_list:
        news_vec_list.append(m_dict[i])
    for i in news_vec_list:
        for j in i:
            values.append(j)
    region =list(range(0,50))*37  #10个
    sns.set(font="simhei")
    kind=[]
    for i in n_list:#2600个
        for j in range(0,50):
            kind.append(i)

    f,ax= plt.subplots(figsize=(20, 50))
    np.random.seed(20180316)
    #arr_region = np.random.choice(region, size=(50,))
    list_region = region

    #arr_kind = np.random.choice(kind, size=(50,))
    list_kind = kind

    #values = np.random.randint(-1, 1, 2600)
    #print(values)
    #values=np.array()
    list_values = list(values)
    cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)

    df = pd.DataFrame({'region':list_region,'kind': list_kind,'values':list_values})
    pt = df.pivot_table(index='kind', columns='region', values='values', aggfunc=np.sum)   #数据透视表

    sns.heatmap(pt, linewidths = 0.05,ax = ax, center=None,cmap=cmap)
    ax.set_title('cubehelix map')
    ax.set_xlabel('')
    ax.set_xticklabels([])  # 设置x轴图例为空值
    ax.set_ylabel('kind')
    ax.tick_params(axis='y', labelsize=6)


    plt.show()
    
if __name__=="__main__":
    n_list=[]
    n_list2=[]
    with open("new.txt","r",encoding="utf_8_sig") as f:
        for line in f:
            n_list.append(line.strip().split())
            for i in line.strip().split():
                n_list2.append(i)

    # Get an example dataset from seaborn



    heatmap_draw(n_list2)
# # cos计算1
# def cossim(vector_a, vector_b):
#     vector_a = np.mat(vector_a)
#     vector_b = np.mat(vector_b)
#     num = float(vector_a * vector_b.T)
#     denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
#     sim = num / denom
#     return sim
#
#
# # cos计算2
# def cos(vector1, vector2):
#     dot_product = 0.0
#     normA = 0.0
#     normB = 0.0
#     for a, b in zip(vector1, vector2):
#         dot_product += a * b
#         normA += a ** 2
#         normB += b ** 2
#     if normA == 0.0 or normB == 0.0:
#         return None
#     else:
#         return dot_product / ((normA * normB) ** 0.5)