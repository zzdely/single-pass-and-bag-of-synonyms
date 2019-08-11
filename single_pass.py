# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：single-pass -> single_pass
@IDE    ：PyCharm
@Author ：Zhang zhe
@Date   ：2019/7/1 14:42
=================================================='''
import time

from feather_represent import*
from data_process import make_news_list

from m_tools import*



if __name__ == "__main__":
    # 设置向量长度
    vec_dimension=3500
    w2v_dimension=200

    # 设置零向量
    zero_vector = zero_vec(vec_dimension)

    # cluster_id = {}
    # 读取测试数据
    pd_data = pd.read_csv('./test_set_451.csv', encoding='utf_8_sig')

    #分词、拼接，生成二维数组
    stopwords_path='./stepwords/stopword_chuli.txt'

    # 为便于二次使用可进行固化
    news_list = make_news_list(pd_data,stopwords_path)

    houxuanci_path = "./vocabu/5_27_vocabu_list_"+str(vec_dimension)+".txt"

    boc_icf_path="./boc_icf_" + str(w2v_dimension) + "_w2v/boc_icf_" + str(vec_dimension) + "_feather.txt", "wb"

    LDA_path='./lda/model_'+str(vec_dimension)+'_5_24.model'
    #文档表示
    method = "W2V"
    a = time.clock()
    news_represent = Represent(news_list,method,vec_dimension,w2v_dimension,houxuanci_path,LDA_path,boc_icf_path)
    after_news_list_l2 = getL2(news_represent)
    b = time.clock()
    print(b - a)

    # expand TF-IDF
    # a = time.time()
    # word_vec_dict_tfidf = expand_tfidf_vectors(news_list)
    # b = time.time()
    # print(b - a)
    # eboc 文档表示
    # eboc_vectors=get_feather_eboc(news_list)

    # 读取eboc结果
    # fr = open('eboc_feather_xiaoyangebn.txt', 'rb')
    # corpus = pickle.load(fr)
    # fr.close()
    # eboc_vec=corpus

    #LDA文档表示
    # word_vec_dict_LDA=LDA_use(news_list,m_dimension)

    # doc2vec训练文档表示
    # 训练
    # doc2vec_train(news_list)
    # model=Doc2Vec.load('./doc2vec_models/doc_test_3.13')
    # train_arrays= np.zeros((len(news_list),m_dimension))


    # for i in range(len(news_list)):
    #     train_arrays[i]=model.docvecs[i]
    # # doc_vec=normalize(train_arrays, norm='l2',axis=0).tolist()
    # doc_vec=train_arrays.tolist()

    # boc/icf-boc获取文档表示
    # #cboc_vec=get_feather_boc(news_list)
    # fr = open('./boc_icf_500_w2v/boc_icf_400_feather_5_26.txt', 'rb')
    # boc_vec = pickle.load(fr)
    # fr.close()

    # average w2v文档表示
    # word_dict_w2v = get_feather_word2vec(news_list, zero_vector)
    # 文章列表正则化
    #print(eboc_vectors[0])
    #after_news_list_l2=boc_vec.tolist()

    '''
    after_news_list_l2 = normalize(np.array(word_vec_dict_tfidf), norm='l2', axis=0).tolist()
    #print(after_news_list_l2[0])
    ##########################################################################
    f_4_5result = open("./result/5_27_TF_IDF_100r_2500d_L2结果.txt", "w", encoding="utf_8_sig")#
    for boundary in float_range.range(0.1, 0.7, 0.05):
        print("boundary = ")
        print(boundary)
        f_4_5result.write("boundary = "+str(boundary))
        f_4_5result.write("\n")
        # f_result.write("boundary = ")
        a=time.time()
        cluster_all = single_pass(after_news_list_l2, zero_vector, boundary)
        b=time.time()
        print(b-a)
        if len(cluster_all) >= 20:
            show_result(cluster_all, pd_data)
            #printout(boundary)
        else:
            print("\n簇数目为："+str(len(cluster_all))+"小于话题数")
            f_4_5result.write("簇数目为："+str(len(cluster_all))+"小于话题数\n")
    f_4_5result.close()
    '''
