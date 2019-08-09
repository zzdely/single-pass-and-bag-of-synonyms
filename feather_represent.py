# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：single-pass -> feather_represent
@IDE    ：PyCharm
@Author ：Zhang zhe
@Date   ：2019/7/1 14:53
=================================================='''
import gc
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from m_tools import *
from w2v import Word2VecModel
from sklearn.preprocessing import normalize
from tqdm import tqdm
from gensim import corpora
from gensim.models import LdaModel
from bocmodel import BOCModel

class Represent(object):
    def __init__(self, news_list, method, dimension,houxuancci_path):
        '''
        :param news_list: the news_list waiting process
        :param method: can choose TF-IDF
        :param dimension: the excepted dimension of the documents
        '''
        self.news_list = news_list
        self.method = method
        self.dimension = dimension
        self.houxuanci=houxuancci_path

    def get_tf_idf_sklearn(self):
        '''
        use sklearn to get the feather
        :return: the array of the news_list
        '''
        news_list2 = []
        for i in self.news_list:
            news_list2.append(" ".join(i))
        tf_idf_vec = TfidfVectorizer()
        tf_idf_matrix = tf_idf_vec.fit_transform(news_list2)
        #print(tfidf_matrix.toarray())
        return tf_idf_matrix.toarray()

    # 获取文档的特征表示_word2vec累加法
    def get_word2vec_vec(self):
        m_dict = {}
        a = Word2VecModel()
        a.load_embedding()
        w2vmodel = a.get_embedding()

        # L2正则化
        x_normalized = normalize(w2vmodel, norm='l2', axis=0)

        # maxmin归一化
        # w2vdict=(w2vmodel - w2vmodel.min(axis=0)) / (w2vmodel.max(axis=0) - w2vmodel.min(axis=0))
        # w2vdict = w2vmodel / w2vmodel.max(axis=0)

        w2vvocabu = a.get_vocabu()

        for weight, w in zip(x_normalized, w2vvocabu):
            if w in ["_pad_", "_nil_", "_ukn_", "_sos_", "_eos_"] and w != "":
                continue
            else:
                m_dict[w] = weight
        del x_normalized, w2vvocabu, w2vmodel
        gc.collect()
        news_vec_list = []

        for i in tqdm(self.news_list):
            word_list = []
            for word in i:
                if word != '':
                    if word in m_dict:
                        word_list.append(m_dict[word])
                    else:
                        continue
            word_sum = zero_vec(self.dimension)
            for words in word_list:
                word_sum += words
            word_sum = word_sum / len(word_list)
            news_vec_list.append(word_sum.tolist())

        return news_vec_list

    # tf-idf提特征
    def tf_idf_vectors(self):
        '''
        use the words chose from the pre-news to get feather
        :return: the array of the news_list
        '''
        vocabu_l = []
        with open(self.houxuanci, mode="r", encoding="utf-8") as f:
            for i in f:
                i = i.strip()
                vocabu_l.append(i)

        idf_dict = compute_idf(self.news_list)

        tf_idf_newslist = []
        for i in tqdm(self.news_list):
            after_process=[]
            voc_dict = [0.] * self.dimension
            # 压缩单个文本长度（只保留候选词，减少总词频）
            for j in i:
                if j in vocabu_l:
                    after_process.append(j)
            tem_dict1 = dict(Counter(after_process))
            # 保留所有词的方式
            # tem_dict1 = dict(Counter(i))
            for w, tf in tem_dict1.items():
                if w in vocabu_l:
                    voc_dict[vocabu_l.index(w)] = tf * 1.0 / len(after_process) * idf_dict[w]
                    # 另一种方式
                    # voc_dict[vocabu_l.index(w)] = tf*1.0 / len(i) * idf_dict[w]
            tf_idf_newslist.append(voc_dict)
        return tf_idf_newslist

    def get_lda_vec(self):
        lda = LdaModel.load('./lda/model_100_5_24.model')
        news_lda_list = []
        dictionary = corpora.Dictionary(self.news_list)
        corpus = [dictionary.doc2bow(text) for text in self.news_list]
        doc_lda = lda[corpus]
        for topic in doc_lda:
            news_vec_lda = [0] * self.dimension
            for i in topic:
                news_vec_lda[i[0]] = float(i[1])
            news_lda_list.append(news_vec_lda)
        return news_lda_list

    def get_feather_boc(self,w_dimension):
        corpus=[]
        for i in self.news_list:
            corpus.append(" ".join(i))
        print("load embedding")
        a = Word2VecModel()
        a.load_embedding()
        w2vmodel = a.get_embedding()
        wv=[]
        idx_to_vocab2=[]
        print("load vocabu")
        idx_to_vocab=a.get_vocabu()
        stopwords_list = stopwordslist('./stepwords/stopword_chuli.txt')
        for w,voc in zip(w2vmodel,idx_to_vocab):
            if voc not in stopwords_list and voc != "":
                wv.append(w)
                idx_to_vocab2.append(voc)
        wv=np.array(wv)
        k_num=self.dimension
        print("获取boc-icf表示")
        model = BOCModel(wv, idx_to_vocab=idx_to_vocab2,n_concepts=k_num,min_count=5)
        # get icf represent
        boc = model.transform(corpus,apply_icf=True)

        boc_ndnarry=boc.todense()
        f_feather = open("./boc_icf_"+str(w_dimension)+"_w2v/boc_icf_"+str(k_num)+"_feather_5_26.txt", "wb")
        pickle.dump(boc_ndnarry, f_feather)
        f_feather.close()
        return boc_ndnarry.tolist()

    # tf-idf提特征
    # def expand_tf_idf(self):
    #     vocabu_l = []
    #     concept_l = []
    #     with open("./vocabu/5_27_vocabu_list_5500.txt", mode="r", encoding="utf-8") as f:
    #         for i in f:
    #             i = i.strip()
    #             vocabu_l.append(i)
    #     with open("./result/5_27_5500_cijuleijieguo——yuan0.7.txt", mode="r", encoding="utf-8") as f2:
    #         for i in f2:
    #             i = i.strip().split()
    #             concept_l.append(i)
    #
    #     # doc frequency calculate
    #     total = 0
    #     wendangpin_dict = {}
    #     for line in self.news_list:
    #         temp_dict = {}
    #         total += 1
    #         for word in line:
    #             temp_dict[word] = 1
    #         for key in temp_dict:
    #             num = wendangpin_dict.get(key, 0)
    #             wendangpin_dict[key] = num + 1
    #
    #     tf_idf_newslist = []
    #
    #     for i in tqdm(self.news_list):
    #         after_process = []
    #         voc_dict = [0] * len(concept_l)
    #         # 压缩单个文本长度（只保留候选词，减少总词频）
    #         for j in i:
    #             if j in vocabu_l:
    #                 after_process.append(j)
    #         tem_dict1 = dict(Counter(after_process))
    #         for j in range(len(concept_l)):
    #             w_tf = 0
    #             idf = 1
    #             wendangp = 0
    #             ww = concept_l[j]
    #             if len(ww) == 1:
    #                 if ww[0] in tem_dict1:
    #                     w_tf = tem_dict1[ww[0]]
    #                     idf = math.log(len(self.news_list) * 1.0 / (wendangpin_dict[ww[0]] + 1))
    #             else:
    #                 for k in ww:
    #                     if k in tem_dict1:
    #                         #wendangpin_dict[k] = 0
    #                         #w_tf = 0
    #                         w_tf = w_tf + tem_dict1[k]
    #                         wendangp = wendangp + wendangpin_dict[k]
    #                 idf = math.log(len(self.news_list) * 1.0 / (wendangp + 1))
    #             voc_dict[j] = w_tf / (len(after_process) + 1) * idf
    #         tf_idf_newslist.append(voc_dict)
    #     return tf_idf_newslist

    # tf-idf提特征
    def expand_tfidf_vectors(self):
        vocabu_l = []
        concept_l=[]
        with open("./vocabu/5_27_vocabu_list_5500.txt",mode="r",encoding="utf-8") as f:
            for i in f:
                i = i.strip()
                vocabu_l.append(i)
        with open("./result/5_27_5500_cijuleijieguo——yuan0.7.txt", mode="r", encoding="utf-8") as f2:
            for i in f2:
                i = i.strip().split()
                concept_l.append(i)
        after_news_list = []
        str_word = []
        for i in self.news_list:
            t = []
            for j in i:
                if j in vocabu_l and j != "":
                    t.append(j)
            # print(t)
            after_news_list.append(t)

        for i in after_news_list:
            for j in i:
                if j != '':
                    str_word.append(j)

        str_word = list(set(str_word))
        # 计算idf
        idf_dict = {}
        wendangpin_dict = {}
        n_allnews = len(after_news_list)
        print("计算IDF值")
        for w in tqdm(str_word):
            c = 0
            for d in after_news_list:
                d = ' '.join(d)
                if w in d:
                    c = c + 1
            wendangpin_dict[w]=c
        # 计算袋中的IDF
        for j in range(len(concept_l)):
            wendangp=0
            w=concept_l[j]
            if len(w) == 1:
                if w[0] not in str_word:
                    idf = 0
                else:
                    idf = math.log(n_allnews * 1.0 / (wendangpin_dict[w[0]] + 1))
            else:
                for i in w:
                    if i not in str_word:
                        wendangpin_dict[i]=0
                    wendangp=wendangp+wendangpin_dict[i]
                if wendangp==0:
                    idf=0
                else:
                    idf = math.log(n_allnews * 1.0 / (wendangp + 1))
            idf_dict[j]=idf

        tf_idf_newslist=[]
        for i in tqdm(after_news_list):
            #tfidf_dict = {}
            voc_dict=[0] * len(concept_l)
            tem_dict1 = dict(Counter(i))
            for j in range(len(concept_l)):
                w_tf = 0
                ww = concept_l[j]
                if len(ww) == 1:
                    if ww[0] in tem_dict1:
                        w_tf = tem_dict1[ww[0]]
                else:
                    for k in ww:
                        if k in tem_dict1:
                            w_tf = w_tf+tem_dict1[k]
                voc_dict[j] = w_tf / (len(i)+1) * idf_dict[j]
            tf_idf_newslist.append(voc_dict)
        return tf_idf_newslist

def getL2(Represent):
    print("………………………………………………………………\n")
    print("获取文档特征表示：")
    news_represent=[]
    w_dimension=100
    if Represent.method == "TF-IDF":
        news_represent=Represent.tf_idf_vectors()

    elif Represent.method == "W2V":
        news_represent=Represent.get_word2vec_vec()

    elif Represent.method == "LDA":
        news_represent = Represent.get_lda_vec()

    elif Represent.method == "BOC":
        news_represent=Represent.get_feather_boc(w_dimension)

    elif Represent.method == "E-TF-IDF":
        news_represent = Represent.expand_tf_idf()

    else:
        print("Please input the right method")

    news_represent_l2 = normalize(np.array(news_represent), norm='l2', axis=0).tolist()

    return news_represent_l2
