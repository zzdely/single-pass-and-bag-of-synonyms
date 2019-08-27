# single-pass-and-bag-of-synonyms
The sample code of the research of the bag-of-synonyms


流程：
一、预处理阶段：
1.从数据库中读取训练数据和测试数据并清洗保存
2.分词，处理，固化
3.训练词向量
4.训练LDA
5.构建完整词典，保存在"./vocabu/vocabu_list_all.txt"
6.选词构建候选词典

二、文档表示阶段
1.word2vec average pooling表示文档
method="W2V"

2.tf-idf+词典表示文档
method="TF-IDF"

3.LDA表示文档
method="LDA"

4.BOD-ICF表示文档
method="BOC"

三、聚类

其他说明：
模型由于较大，未放置在作业提交中，相关训练代码在data_process.py中，需要可自行训练。
