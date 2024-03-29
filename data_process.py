# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：single-pass -> data_process
@IDE    ：PyCharm
@Author ：Zhang zhe
@Date   ：2019/6/29 21:23
=================================================='''
import pymysql
import re
import operator
import jieba
import jieba.analyse
import multiprocessing
import pickle
import pandas as pd
from tqdm import tqdm
from m_tools import *
from w2v import Word2VecModel
from gensim import corpora
from gensim.models import LdaModel

jieba.load_userdict("./userdict/user_dict.txt")
jieba.load_userdict("./userdict/p2p.txt")
jieba.load_userdict("./userdict/电脑.txt")
jieba.load_userdict("./userdict/手机.txt")
jieba.load_userdict("./userdict/通信.txt")
jieba.load_userdict("./userdict/人工智能.txt")
jieba.load_userdict("./userdict/经济.txt")
jieba.load_userdict("./userdict/交通运输.txt")
jieba.load_userdict("./userdict/飞机航空.txt")
jieba.load_userdict("./userdict/公司.txt")
jieba.load_userdict("./userdict/叠词.txt")
jieba.load_userdict("./userdict/tongyici.txt")


# 读取数据并排序
def getdata_train():
    # 获取数据库游码
    conn = pymysql.Connect(host='127.0.0.1', port=3306, user='root', passwd='root', db='bishe3', charset='utf8')
    cursor = conn.cursor()
    sub_list = ["雷锋网按","证券时报","投资者报","本报","本文记者","&nbsp;", "<br>", "&#8212", "&#", "文章导读：","新浪财经讯", "讯记者", "每经编辑", "南方都市报","澎湃新闻"
                "TechWeb报道", "北京晨报", "新闻晨报", "AI科技评论", "AI 科技评论", "相关新闻：", "午间消息", "早间消息", "相关专题：", "北京商报", "摘要：",
                "本文来自", "北京时间", "来源：", "（：", "原标题：", "原标题", "自媒体：", "消息，", "AI金融评论","【早报】",  "上午消息","【晚报】","中国青年报"
                "下午消息", "参考消息网", "　", "猎云网", "环球网报道", "环球网科技综合报道","环球时报","环球网综合报道","环球时报综合报道", "21世纪经济报道", "每日经济新闻",
                "中国信息产业网", "IT时报", "晚间消息", "一线丨", "腾讯《一线》", "网易智能讯", "编者按：", "公众号/", "公众号：", "雷锋网：", "雷锋网", "本报记者",
                "消息，", "报道，", "导语：", "按：", "消息：", "新浪科技","观察者网综合报道","TechWeb","新闻纵横","金融时报","IT时报","　","，，","本报","每经","编者按","【早报】"]
    info1 = []
    info2 = []
    info3 = []
    # 读取新闻标题
    aa1 = cursor.execute("select news_title from news_2_3_4_copy")
    infoo1 = cursor.fetchmany(aa1)
    for i in infoo1:
        j = re.sub(r"\n", "", i[0])
        j = re.sub(r"\r", "", j)
        j = re.sub(r"\r\n", "", j)
        j = re.sub(r"\n\r", "", j)
        info1.append(j)
    # 读取新闻内容
    aa2 = cursor.execute("select news_content from news_2_3_4_copy")
    infoo2 = cursor.fetchmany(aa2)
    count=0
    for i in infoo2:
        j = re.sub(r"\n", "", i[0])
        j = re.sub(r"\r", "", j)
        j = re.sub(r"\r\n", "", j)
        j = re.sub(r"\n\r", "", j)
        j = re.sub(r"[（）【】()■▲◆★▽●▼♦]+", "", j)
        j = re.sub(r'((?:据)[\u4e00-\u9fa5]{2,10}?(?:报道))', "", j)
        j = re.sub(r'([\d]{1,2}月[\d]{1,2}日消息)', "", j)
        j = re.sub(r'([\d]{2,4}年{1,2}月[\d]{1,2}日)', "", j)
        j = re.sub(r'([\d]{1,2}月[\d]{1,2}日)', "", j)
        j = re.sub(r'([\d]{2,4}年[\d]{1,2}月)', "", j)
        j = re.sub(r'([\d]{2,4} 年 [\d]{1,2} 月 [\d]{1,2} 日)', "", j)
        j = re.sub(r'([\d]{1,2} 月 [\d]{1,2} 日)', "", j)
        j = re.sub(r'([\d]{2,4} 年 [\d]{1,2} 月)', "", j)
        j = re.sub(r'(作者 [\u4e00-\u9fa5]{2,3})', "", j)
        j = re.sub(r'(作者：[\u4e00-\u9fa5]{2,3})', "", j)
        j = re.sub(r'([\u4e00-\u9fa5]{2,3}科技讯)', "", j)
        j = re.sub(r'(ID：[\w]{2,10})', "", j)
        j = re.sub(r'(　　原标题 )', "", j)
        j = re.sub(r'(　　来源 )', "", j)
        j = re.sub(r'(文/[\u4e00-\u9fa5]{2,3})', "", j)
        j = re.sub(r'(记者 [\u4e00-\u9fa5]{2,3})', "", j)
        j = re.sub(r'((?:　){2,4}[\u4e00-\u9fa5]{2,3}(?:　){2,4})', "", j)
        j = re.sub(r'((?:　){2,4}[\u4e00-\u9fa5] [\u4e00-\u9fa5](?:　){2,4})', "", j)
        j = re.sub(r'(微信公众号“[\u4e00-\u9fa5]{2,10}”)', "", j)
        j = re.sub(r'((?:本报记者 )[\u4e00-\u9fa5]{2,3})', "", j)
        j = re.sub(r'([\u4e00-\u9fa5]{2,4}(?:日报))', "", j)
        j = re.sub(r'(网 [\u4e00-\u9fa5]{2,3})', "", j)
        j = re.sub(r'(北京[\u4e00-\u9fa5]{1,1}报)', "", j)

        for i in sub_list:
            j = re.sub(i, "", j)
        info2.append(j)
        j = re.sub(r'上午', "", j, 1)
        j = re.sub(r'下午', "", j, 1)
        j = re.sub(r'晚间', "", j, 1)
        j = re.sub(r'凌晨', "", j, 1)
    # print(count)

    # 读取新闻发布时间（排序依据）
    aa3 = cursor.execute("select news_pubtime from news_2_3_4_copy")
    infoo3 = cursor.fetchmany(aa3)
    for i in infoo3:
        j = re.sub(r"\n", "", i[0])
        j = re.sub(r"\r", "", j)
        j = re.sub(r"\r\n", "", j)
        j = re.sub(r"\n\r", "", j)
        j = re.sub(r"(?:年|月)", "-", j)
        j = re.sub(r"日", "", j)
        if j[0]==" ":
            j = re.sub(r" ", "", j,1)
        info3.append(j)

    # 构建dataframe
    dataframe_data = pd.DataFrame({"news_title": info1, "news_content": info2, "news_pubtime": info3})
    dataframe_data.sort_values(by='news_pubtime', axis=0, ascending=True, inplace=True,
                               na_position='first')  # ascending倒序
    # 索引问题通过reset完美解决
    dataframe_data = dataframe_data.reset_index(drop=True)

    # 构建news_id
    id_list = []
    for i in range(dataframe_data.shape[0]):
        id_list.append(i)
    # 格式转换list——>Series
    news_id_Series = pd.Series(list(id_list))
    dataframe_data = pd.DataFrame(
        {"news_id": news_id_Series, "news_title": dataframe_data["news_title"],
         "news_content": dataframe_data["news_content"], "news_pubtime": dataframe_data["news_pubtime"]})
    # 以pubtime排序
    order = ["news_id", "news_pubtime", "news_title", "news_content"]
    dataframe_data = dataframe_data[order]
    dataframe_data["news_title"] = dataframe_data["news_title"].apply(lambda x: x.lower())
    dataframe_data["news_content"] = dataframe_data["news_content"].apply(lambda x: x.lower())
    dataframe_data.to_csv('./news_train_234.csv', sep=',', index=None,encoding='utf_8_sig')  # index=None,
    cursor.close()
    conn.close()
    return dataframe_data


def getdata_test():
    # 获取数据库游码
    conn = pymysql.Connect(host='127.0.0.1', port=3306, user='root', passwd='root', db='test', charset='utf8')
    cursor = conn.cursor()
    sub_list = ["雷锋网按","证券时报","投资者报","本报","本文记者","&nbsp;", "<br>", "&#8212", "&#", "文章导读：","新浪财经讯", "讯记者", "每经编辑", "南方都市报","澎湃新闻"
                "TechWeb报道", "北京晨报", "新闻晨报", "AI科技评论", "AI 科技评论", "相关新闻：", "午间消息", "早间消息", "相关专题：", "北京商报", "摘要：",
                "本文来自", "北京时间", "来源：", "（：", "原标题：", "原标题", "自媒体：", "消息，", "AI金融评论","【早报】",  "上午消息","【晚报】","中国青年报"
                "下午消息", "参考消息网", "　", "猎云网", "环球网报道", "环球网科技综合报道","环球时报","环球网综合报道","环球时报综合报道", "21世纪经济报道", "每日经济新闻",
                "中国信息产业网", "IT时报", "晚间消息", "一线丨", "腾讯《一线》", "网易智能讯", "编者按：", "公众号/", "公众号：", "雷锋网：", "雷锋网", "本报记者",
                "消息，", "报道，", "导语：", "按：", "消息：", "新浪科技","观察者网综合报道","TechWeb","新闻纵横","金融时报","IT时报","　","，，","本报","每经","编者按","【早报】"]
    info1 = []
    info2 = []
    info3 = []
    info4 = []
    info5 = []
    # 读取新闻标题
    aa1 = cursor.execute("select news_title from news_onlabel_451")
    infoo1 = cursor.fetchmany(aa1)
    ## news_title_Series = pd.Series(list(infoo1))
    for i in infoo1:
        j = re.sub(r"\n", "", i[0])
        j = re.sub(r"\r", "", j)
        j = re.sub(r"\r\n", "", j)
        j = re.sub(r"\n\r", "", j)
        j = j.lower()
        #j = re.sub(r"[a-zA-Z0-9]+", "", j)
        info1.append(j)
    # 读取新闻内容
    aa2 = cursor.execute("select news_content from news_onlabel_451")
    infoo2 = cursor.fetchmany(aa2)
    count=0
    for i in infoo2:
        j = re.sub(r"\n", "", i[0])
        j = re.sub(r"\r", "", j)
        j = re.sub(r"\r\n", "", j)
        j = re.sub(r"\n\r", "", j)
        j = re.sub(r"[（）()■▲◆★▽●▼♦]+", "", j)
        j = re.sub(r'((?:据)[\u4e00-\u9fa5]{2,10}?(?:报道))', "", j)
        j = re.sub(r'([\d]{1,2}月[\d]{1,2}日消息)', "", j)
        j = re.sub(r'([\d]{2,4}年{1,2}月[\d]{1,2}日)', "", j)
        j = re.sub(r'([\d]{1,2}月[\d]{1,2}日)', "", j)
        j = re.sub(r'([\d]{2,4}年[\d]{1,2}月)', "", j)
        j = re.sub(r'([\d]{2,4} 年 [\d]{1,2} 月 [\d]{1,2} 日)', "", j)
        j = re.sub(r'([\d]{1,2} 月 [\d]{1,2} 日)', "", j)
        j = re.sub(r'([\d]{2,4} 年 [\d]{1,2} 月)', "", j)
        j = re.sub(r'(作者 [\u4e00-\u9fa5]{2,3})', "", j)
        j = re.sub(r'(作者：[\u4e00-\u9fa5]{2,3})', "", j)
        j = re.sub(r'([\u4e00-\u9fa5]{2,3}科技讯)', "", j)
        j = re.sub(r'(ID：[\w]{2,10})', "", j)
        j = re.sub(r'(　　原标题 )', "", j)
        j = re.sub(r'(　　来源 )', "", j)
        j = re.sub(r'(文/[\u4e00-\u9fa5]{2,3})', "", j)
        j = re.sub(r'(记者 [\u4e00-\u9fa5]{2,3})', "", j)
        j = re.sub(r'((?:　){2,4}[\u4e00-\u9fa5]{2,3}(?:　){2,4})', "", j)
        j = re.sub(r'((?:　){2,4}[\u4e00-\u9fa5] [\u4e00-\u9fa5](?:　){2,4})', "", j)
        j = re.sub(r'(微信公众号“[\u4e00-\u9fa5]{2,10}”)', "", j)
        j = re.sub(r'((?:本报记者 )[\u4e00-\u9fa5]{2,3})', "", j)
        j = re.sub(r'([\u4e00-\u9fa5]{2,4}(?:日报))', "", j)
        for i in sub_list:
            j = re.sub(i, "", j)
            j = j.lower()
        j = re.sub(r'上午',"",j,1)
        j = re.sub(r'下午', "", j, 1)
        j = re.sub(r'晚间', "", j, 1)
        j = re.sub(r'凌晨', "", j, 1)
        info2.append(j)

    # 读取新闻发布时间（排序依据）
    aa3 = cursor.execute("select news_pubtime from news_onlabel_451")
    infoo3 = cursor.fetchmany(aa3)
    for i in infoo3:
        j = re.sub(r"\n", "", i[0])
        j = re.sub(r"\r", "", j)
        j = re.sub(r"\r\n", "", j)
        j = re.sub(r"\n\r", "", j)
        j = re.sub(r"(?:年|月)", "-", j)
        j = re.sub(r"日", "", j)
        if j[0]==" ":
            j = re.sub(r" ", "", j,1)
        info3.append(j)

    aa4 = cursor.execute("select news_label from news_onlabel_451")
    infoo4 = cursor.fetchmany(aa4)
    for i in infoo4:
        j = re.sub(r"\n", "", i[0])
        j = re.sub(r"\r", "", j)
        j = re.sub(r"\r\n", "", j)
        j = re.sub(r"\n\r", "", j)
        j = j.lower()
        info4.append(j)

    aa5 = cursor.execute("select news_labelid from news_onlabel_451")
    infoo5 = cursor.fetchmany(aa5)
    for i in infoo5:
        info5.append(i[0])

    # 构建dataframe
    dataframe_data = pd.DataFrame({"news_title": info1, "news_content": info2, "news_pubtime": info3, "news_label": info4, "news_labelid": info5})

    dataframe_data.sort_values(by='news_pubtime', axis=0, ascending=True, inplace=True,
                               na_position='first')  # ascending倒序
    # 索引问题通过reset完美解决
    dataframe_data = dataframe_data.reset_index(drop=True)

    # 构建news_id
    id_list = []
    for i in range(dataframe_data.shape[0]):
        id_list.append(i)
    # 格式转换list——>Series
    news_id_Series = pd.Series(list(id_list))

    dataframe_data = pd.DataFrame(
        {"news_id": news_id_Series, "news_title": dataframe_data["news_title"],
         "news_content": dataframe_data["news_content"], "news_pubtime": dataframe_data["news_pubtime"],
         "news_label": dataframe_data["news_label"], "news_labelid": dataframe_data["news_labelid"]})
    # 以pubtime排序
    order = ["news_id", "news_pubtime", "news_title", "news_content", "news_label", "news_labelid"]
    dataframe_data = dataframe_data[order]
    dataframe_data.to_csv('./test_set_451.csv', sep=',', index=None, encoding='utf_8_sig')  # index=None,
    cursor.close()
    conn.close()
    return dataframe_data


# 生成新闻列表
def make_news_list(pd_d,stopwords_path,result_path):
    print("………………………………………………………………\n\n")
    print("加载数据并分词：")
    stopwords_list = stopwordslist(stopwords_path)
    pd_d["news_title"]=pd_d["news_title"].astype(str)
    pd_d["news_content"]=pd_d["news_content"].astype(str)
    pd_d["title_cut"] = pd_d["news_title"].apply(lambda x: ' '.join(jieba.cut(x)))
    pd_d["content_cut"] = pd_d["news_content"].apply(lambda x: ' '.join(jieba.cut(x)))
    # 去停用词
    pd_d["title_cut"] = pd_d["title_cut"].apply(lambda x: ''.join(cut_without_stop(x, stopwords_list)))
    pd_d["content_cut"] = pd_d["content_cut"].apply(lambda x: ''.join(cut_without_stop(x, stopwords_list)))
    news_list = []
    print("………………………………………………………………\n\n")
    print("生成分词结果：")
    for index in tqdm(pd_d.index):
        x = pd_d.loc[index]
        text = 3*x["title_cut"] + x["content_cut"]
        news_list.append(list(text.split()))
    # 固化，便于之后的使用
    f_f=open(result_path,"wb")
    pickle.dump(news_list,f_f)
    f_f.close()
    return news_list


# 生成候选词并保存
def vocubu_all(all_docs):
    all_docs["news_title"]=all_docs["news_title"].astype(str)
    all_docs["news_content"]=all_docs["news_content"].astype(str)
    all_docs['title_cut'] = all_docs['news_title'].apply(lambda x: ''.join(filter(lambda ch: ch not in ' \t◆#%', x)))
    all_docs['content_cut'] = all_docs['news_content'].apply(lambda x: ''.join(filter(lambda ch: ch not in ' \t◆#%', x)))
    # all_docs=all_docs.sample(frac=0.3,random_state=100)
    # print(part_dacs['title_cut'][0:10])
    m_dict_POS_word = {}
    m_dict_weight_word = {}
    print("构建词典：")
    for index in tqdm(all_docs.index):
        x = all_docs.loc[index]
        text = 3*(x['title_cut'] + "。") + x['content_cut']
        # jieba_tags = jieba.analyse.extract_tags(sentence=text, topK=10, allowPOS=('ns', 'nr', 'nz', 'nt', 'nl', 'n', 'vn', 'v', 'vd','vi','vl','a', 'an', 'x'),withWeight=True,withFlag=True)
        jieba_tags = jieba.analyse.extract_tags(sentence=text, topK=10, allowPOS=(
        'ns', 'nr', 'nz', 'nt', 'nl', 'n', 'x'), withWeight=True, withFlag=True)
        # jieba_tags = jieba.analyse.textrank(sentence=text, topK=10, allowPOS=(
        #     'ns', 'nr', 'nz', 'nt', 'nl', 'n', 'x'), withWeight=True, withFlag=True)

        for tag in jieba_tags:
            if tag[0].word not in m_dict_POS_word:
                m_dict_POS_word[tag[0].word]=tag[0].flag
            if tag[0].word not in m_dict_weight_word:
                m_dict_weight_word[tag[0].word]=tag[1]
            elif tag[0].word in m_dict_weight_word and tag[1]>m_dict_weight_word[tag[0].word]:
                m_dict_weight_word[tag[0].word] = tag[1]
            else:
                continue

    vocabu = []
    tem_dict1 = sorted(m_dict_weight_word.items(), key=lambda d: d[1], reverse=True)
    # print(len(tem_dict1))
    vocabu_list = open("./vocabu/vocabu_list_all.txt",mode="w",encoding="utf-8")
    vocabu_dict = open("./vocabu/vocabu_dict_all.txt", mode="w", encoding="utf-8")
    for i in range(len(tem_dict1)):
        vocabu.append(tem_dict1[i][0])
        vocabu_list.write(tem_dict1[i][0])
        vocabu_list.write("\n")
        vocabu_dict.write(tem_dict1[i][0]+"   ")
        vocabu_dict.write(m_dict_POS_word[tem_dict1[i][0]])
        vocabu_dict.write("\n")
    vocabu_dict.close()
    vocabu_list.close()


# 生成候选词
def vocabu2(len_list,stopwords_path,houxuanci_path):
    f_all = open("./vocabu/vocabu_list_all.txt", "r", encoding="utf_8_sig")
    word_list = []
    stopwords = [l.strip() for l in
                 open(stopwords_path, 'r', encoding='utf_8_sig').readlines()]
    for line in f_all.readlines():
        if line.strip() not in stopwords:
            word_list.append(line.strip())
    f_all.close()
    list_sample=word_list[0:len_list]

    # 将候选词固化
    f1 = open(houxuanci_path, "w", encoding="utf_8_sig")
    for l in list_sample:
        f1.write(l)
        f1.write("\n")
    f1.close()


def train_embedding(news_list,dimension):
    print("Train embedding")
    w2vModel = Word2VecModel(size=dimension, min_count=5, workers=(multiprocessing.cpu_count()-2))
    w2vModel.train_model(news_list)
    w2vModel.save_model()


# 单词查表向量化
def get_word_list(stopwords_path2,houxuanci_path):
    m_dict={}
    a = Word2VecModel()
    a.load_embedding()
    w2vmodel = a.get_embedding()
    x_normalized = normalize(w2vmodel, norm='l2', axis=0)
    w2vvocabu = a.get_vocabu()
    stopwords = [line.strip() for line in
                 open(stopwords_path2, 'r',encoding='utf_8_sig').readlines()]
    for weight, w in zip(x_normalized, w2vvocabu):
        if w in stopwords or w == "":
            continue
        else:
            m_dict[w] = weight
    del x_normalized,w2vvocabu,w2vmodel
    gc.collect()
    f = open(houxuanci_path,"r",encoding="utf_8_sig")
    word_list = {}
    for line in f.readlines():
        if line.strip() in m_dict:
            word_list[line.strip()]=m_dict[line.strip()].tolist()
        else:
            continue
    f.close()
    return word_list

# 对单词进行single-pass聚类
def single_pass_word(word_vec_dict,zero_vector,boundary):
    cluster_num = 0
    count_i = 0
    cluster_all = {}
    cluster_center = {}
    for term, word_vec in tqdm(word_vec_dict.items()):
        cluster_file = {}
        cos_dict = {}
        # 排除零向量的干扰
        now_words = word_vec_dict[term]
        if (operator.eq(now_words, zero_vector)):
            continue
        if (count_i == 0):  # 当前新闻为第一个
            cluster_file[term] = now_words  # 加入聚类结果
            cluster_center[cluster_num] = now_words
            cluster_all[cluster_num] = cluster_file
            cluster_num = cluster_num + 1
            count_i = count_i + 1
            continue
        for c_id in cluster_all.keys():
            cos1 = get_cossim(now_words, cluster_center.get(c_id))
            cos_dict[c_id] = cos1
        cos_dict = sorted(cos_dict.items(), key=lambda x: x[1], reverse=True)
        max_cos = cos_dict[0][1]
        max_cos_id = cos_dict[0][0]
        if max_cos > boundary:
            cluster_center[max_cos_id] = hebing(cluster_center[max_cos_id], len(cluster_all[max_cos_id]), now_words)
            cluster_file = cluster_all.get(max_cos_id)
            cluster_file[term] = now_words  # 加入聚类结果
            cluster_all[max_cos_id] = cluster_file
            count_i = count_i + 1
            continue
        else:
            cluster_file[term] = now_words  # 加入聚类结果
            cluster_all[cluster_num] = cluster_file
            cluster_center[cluster_num] = now_words
            cluster_num = cluster_num + 1
            count_i = count_i + 1
    return cluster_all

# 打印词聚类结果
def printout(cluster_all,result_path):
    f = open(result_path, mode="w", encoding="utf-8")
    for i in cluster_all.keys():
        for j in cluster_all.get(i):
            f.write(str(j)+" ")
        f.write("\n")
    f.close()

def LDA_train(train,vec_dimension,LDA_path):
    print(len(train))
    # print(' '.join(train[2]))
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=vec_dimension)
    lda.save(LDA_path)

if __name__=="__main__":
    # 设置文档向量长度
    vec_dimension = 500
    # 词向量长度
    w2v_dimensin = 50
    # 停用词1路径
    stopwords_path1='./stopwords/stopwords.txt'
    # 停用词2路径
    stopwords_path2='./stopwords/stopword_chuli.txt'
    # 从数据库获取训练数据
    # train_data = getdata_train()
    # 读取生成的结果
    train_data = pd.read_csv('./news_train_234.csv', encoding='utf_8_sig')
    # 从数据库获取测试数据
    # test_data = getdata_test()
    # 读取生成的结果
    test_data = pd.read_csv('./test_set_451.csv', encoding='utf_8_sig')

    # 生成训练新闻列表
    result_path="news_list_after.txt"
    # news_list = make_news_list(train_data,stopwords_path1,result_path)

    # 读取固化结果
    fr = open(result_path, 'rb')
    news_list = pickle.load(fr)
    fr.close()

    # 训练词向量
    # train_embedding(news_list,w2v_dimensin)


    # 训练LDA
    # LDA_path='./lda/model_' + str(vec_dimension) + '.model'
    # LDA_train(news_list,vec_dimension,LDA_path)

    # 构建所有候选词词典，固化
    # vocubu_all(train_data)

    # 挑选一定数量的候选词构建词典并固化
    houxuanci_path = "./vocabu/5_27_vocabu_list_"+str(vec_dimension)+".txt" # 路径
    # vocabu2(vec_dimension,stopwords_path1,houxuanci_path)

    # 以下为BOS方式必要步骤
    # 词聚类
    m_boundary = 0.7
    word_list=get_word_list(stopwords_path2,houxuanci_path)
    result_path="./result/5_27_"+str(vec_dimension)+"_words_cluster_result" + str(m_boundary) + ".txt"
    cluster_all = single_pass_word(word_list, zero_vec(w2v_dimensin), m_boundary)
    printout(cluster_all,result_path)






