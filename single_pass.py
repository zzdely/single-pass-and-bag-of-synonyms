# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：single-pass -> single_pass
@IDE    ：PyCharm
@Author ：Zhang zhe
@Date   ：2019/7/1 14:42
=================================================='''
import time
import float_range
import operator
from feather_represent import*
from data_process import make_news_list
from m_tools import*


# 对文章进行single-pass聚类
def single_pass(news_vec_list,zero_vector,boundary):
    cluster_num = 0
    cluster_all={}
    print("single-pass聚类")
    cluster_center = {}
    for i in range(len(news_vec_list)):
        cluster_file = {}
        cos_dict = {}
        # 排除零向量的干扰
        now_news = news_vec_list[i]
        if (operator.eq(now_news, zero_vector)):
            continue
        if (i == 0):  # 当前新闻为第一个
            cluster_file[i] = now_news  # 加入聚类结果
            cluster_center[i]= now_news
            cluster_all[cluster_num] = cluster_file
            cluster_num = cluster_num + 1
            continue
        # 更新簇中心版
        for c_id in cluster_all.keys():
            cos1 = get_cossim(now_news, cluster_center.get(c_id))
            cos_dict[c_id] = cos1
        cos_dict = sorted(cos_dict.items(), key=lambda x: x[1], reverse=True)
        max_cos = cos_dict[0][1]
        max_cos_id = cos_dict[0][0]
        if max_cos > boundary:
            cluster_center[max_cos_id] = hebing(cluster_center[max_cos_id],len(cluster_all[max_cos_id]),now_news)
            cluster_file = cluster_all.get(max_cos_id)
            cluster_file[i] = now_news  # 加入聚类结果
            cluster_all[max_cos_id] = cluster_file
            continue
        else:
            cluster_file[i] = now_news  # 加入聚类结果
            cluster_all[cluster_num] = cluster_file
            cluster_center[cluster_num] = now_news
            cluster_num = cluster_num + 1
    return cluster_all

def show_result(cluster_all,pd_data):
    labelid_lenlist = [27, 26, 19, 21, 38, 28, 14, 10, 13, 22, 28, 12, 20, 22, 34,26,30,16,25,21]
    label_lenlist = []
    labelid_dict = {}
    label_list = []
    for i in cluster_all.keys():
        label_list = []
        labelid_list = []
        for j in cluster_all.get(i):
            pd_index = pd_data.index[pd_data['news_id'] == j].tolist()[0]
            labelid_list.append(pd_data.iloc[pd_index]["news_labelid"])
        labelid_dict[i]=labelid_list
    for i in range(len(labelid_lenlist)):
        label_list.append(0)
        label_lenlist.append(0)

    for i in labelid_dict.keys():
        res1 = Counter(labelid_dict.get(i)).most_common(1)[0][0]
        res2 = Counter(labelid_dict.get(i)).most_common(1)[0][1]
        if label_list[res1] < res2:
            label_list[res1] = res2
            label_lenlist[res1] = len(labelid_dict.get(i))
        else:
            continue
    miss_rate = []
    error_rate = []
    recall_rate = []
    precision_rate = []
    f1 = []
    cost_rate1 = []
    cost_rate2 = []
    for a, ac, ab in zip(label_list, labelid_lenlist, label_lenlist):
        all_news = 451
        c = ac - a
        b = ab - a
        d = all_news - ac - b
        miss_rate.append(c / ac)
        error_rate.append(b / (b + d))
        recall_rate.append(a / ac)
        if ab == 0:
            precision_rate.append(0)
        else:
            precision_rate.append(a / ab)
        f1.append(2 * a / (2 * a + c + b))
        cost_rate1.append(0.02 * 1 * (c / ac) + 0.98 * 0.2 * (b / (b + d)))
        cost_rate2.append(0.02 * 1 * (c / ac) + 0.98 * 0.1 * (b / (b + d)))
    f_4_5result.write("各个簇大小：      " + str(label_lenlist))
    f_4_5result.write("\n")
    f_4_5result.write("簇内相关文档数目： " + str(label_list))
    f_4_5result.write("\n")
    f_4_5result.write("漏检率：          " + str(np.mean(miss_rate)) + "\n")
    f_4_5result.write("错检率：          " + str(np.mean(error_rate)) + "\n")
    f_4_5result.write("准确率：          " + str(np.mean(precision_rate)) + "\n")
    f_4_5result.write("召回率：          " + str(np.mean(recall_rate)) + "\n")
    f_4_5result.write("F值：             " + str(np.mean(f1)) + "\n")
    f_4_5result.write("损耗函数(错检)0.2：" + str(np.mean(cost_rate1)) + "\n")
    f_4_5result.write("损耗函数(错检)0.1：" + str(np.mean(cost_rate2)) + "\n\n")
    print(label_lenlist)
    print(label_list)
    print("漏检率：", np.mean(miss_rate))
    print("错检率：", np.mean(error_rate))
    print("准确率：", np.mean(precision_rate))
    print("召回率：", np.mean(recall_rate))
    print("F值：", np.mean(f1))
    print("损耗函数1：", np.mean(cost_rate1))
    print("损耗函数2：", np.mean(cost_rate2))

if __name__ == "__main__":
    # 设置向量长度
    vec_dimension=500
    w2v_dimension=50

    # 设置零向量
    zero_vector = zero_vec(vec_dimension)

    # cluster_id = {}
    # 读取测试数据
    pd_data = pd.read_csv('./test_set_451.csv', encoding='utf_8_sig')

    #分词、拼接，生成二维数组
    stopwords_path='./stepwords/stopword_chuli.txt'
    result_path="news_list_test.txt"
    # 为便于二次使用可进行固化
    news_list = make_news_list(pd_data,stopwords_path,result_path)

    # 读取测试集固化结果
    fr = open(result_path, 'rb')
    news_list = pickle.load(fr)
    fr.close()

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
    cluster_result_path="./result/"+method+"_"+str(vec_dimension)+"结果.txt"
    f_4_5result = open(cluster_result_path, "w", encoding="utf_8_sig")#
    for boundary in float_range.range(0.1, 0.7, 0.05):
        print("boundary = ")
        print(boundary)
        f_4_5result.write("boundary = "+str(boundary))
        f_4_5result.write("\n")
        a=time.time()
        cluster_all = single_pass(after_news_list_l2, zero_vector, boundary)
        b=time.time()
        print(b-a)
        if len(cluster_all) >= 20:
            show_result(cluster_all, pd_data)
        else:
            print("\n簇数目为："+str(len(cluster_all))+"小于话题数")
            f_4_5result.write("簇数目为："+str(len(cluster_all))+"小于话题数\n")
    f_4_5result.close()
