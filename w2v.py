# encoding=utf-8
# This Program is written by Victor Zhang at 2016-08-01 23:04:21
# Modified at 2018-12-04 16:30:21
# version 1.5
#
import logging
import os
import numpy as np
from gensim.models.word2vec import Word2Vec,LineSentence
from gensim.models.keyedvectors import KeyedVectors
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Word2VecModel():
    def __init__(self, size=200, window=5, min_count=5, workers =5, isSkipGram = False ,modelFolder="./models/", isLoadModel=False, **kwargs):
        """initialize the model

        Keyword Arguments:
            size {int} -- [the dimension of embeddings, usually 50,100,200,300] (default: {200})
            window {int} -- [the window size when training the embeddings] (default: {5})
            min_count {int} -- [the minimum count that the word is considered when training the model] (default: {5})
            workers {number} -- [number of workers to train the embeddings] (default: {5})
            isSkipGram {bool} -- [use skip-gram of CBOW] (default: {False})
            modelFolder {str} -- [the folder that store the trained model] (default: {"./models/"})
            isLoadModel {bool} -- [whether load the model when initializing] (default: {False})
        """

        # For masking purpose, 0 for padding
        self.index2word = {}
        self.word2index = {}
        self.model = None
        self.embedding = None
        self.dicSize = 0
        self.length = 0
        self.size = size
        self.window = window
        self.min_count = min_count
        self.modelFolder = modelFolder
        self.workers = workers
        self.sg = 1 if isSkipGram else 0
        self.uknStr = "_ukn_"
        self.nulStr = "_nil_"
        self.padStr = "_pad_"
        self.sosStr = "_sos_"
        self.eosStr = "_eos_"
        self.speStrs = [self.padStr,self.nulStr,self.uknStr,self.sosStr,self.eosStr]
        self.kwargs = kwargs

        if isLoadModel:
            self.load_model()

    def read_file(self, filename):
        """read string from a file

        the file should have one sentence each line, and separate words with blank space.

        Arguments:
            filename {[string]} -- [the file name]

        Returns:
            [list] -- [list of split sentences]
        """
        import jieba
        ifile = open(filename, 'r',encoding = 'utf-8')
        st = []
        for line in ifile:
            words = jieba.lcut(line.strip())
            st.append(words)
        return st
    
    def train_model_from_file(self, filename, is_cut=False):
        """train model from a text file

        the file should have one sentence each line, and separate words with blank space.

        Arguments:
            filename {[string]} -- [the file name]
            is_cut {[bool]} -- [whether to use the word segmentation tool]
  
        """
        st = []
        if is_cut:
            st = self.read_file(filename)
        else:
            st = LineSentence(filename)
        self.train_model(st)

    def get_word2vec_model(self, st):
        """get the word2vec model

        return the trained model, if the model is not exist, load the model or train the model.

        Arguments:
            st {[list]} -- [list of split sentences]

        Returns:
            [word2vec model] -- [a trained word2vec model]
        """
        if self.model is None:
            if os.path.exists(self.modelFolder + "embedding.txt") and os.path.exists(self.modelFolder + "vocab.txt") and os.path.exists(self.modelFolder + "word2vec.model"):
                self.model = self.load_model()
            else:
                self.model = self.train_model(st)
        return self.model

    def train_model(self, st):
        """train the word2vec model

        a satisfactory 'st' is a list of split sentences, for example
        st = [["i","like","apples"],["tom","likes","bananas"],["jane","hate","durian"]] is a suitable list, it is highly recommended that the word should be in lower case.

        Arguments:
            st {[list]} -- [list of split sentences]
        """
        logging.info('Trainning Word2Vec model')
        self.model = Word2Vec(st, size=self.size, window=self.window, min_count=self.min_count, workers = self.workers, sg = self.sg)

        self.set_embeddings()
        self.set_save_dict()
        self.save_model()
        return self.model

    def load_dict(self):
        """only load the dictionary

        the dictionary is one word each line.

        note: the first line in vocab.txt is the second word in vocabulary. The first one is 'Nil'

        """

        i = 0
        ifile = open(self.modelFolder + "vocab.txt", 'r', encoding='utf-8')
        for line in ifile:
            word = line.strip()
            self.index2word[i] = word
            self.word2index[word] = i
            i += 1
        ifile.close()

    def load_embedding(self):
        """only load the embeddings

        each line is the embedding of each word according to the dictionary.

        the first line in embedding.txt is the embedding of 'Nil'
        """
        self.embedding = np.loadtxt(self.modelFolder + "embedding.txt")
        self.dicSize, self.length = self.embedding.shape
        print("embedding_size", self.dicSize,self.length)

    def get_embedding(self):
        return self.embedding

    def set_embeddings(self):
        self.embedding = self.model.wv.vectors
        # print("vector size",self.model)
        self.dicSize, self.length = self.embedding.shape
        # print(self.dicSize, self.length)
        speStrsSize = len(self.speStrs)
        zeros = np.zeros((speStrsSize, self.length))
        # print(zeros.shape)
        self.embedding = np.r_[zeros, self.embedding]
        self.dicSize += speStrsSize

    def get_vocabu(self):
        '''
        :return:
        all the words in vocabu
        '''
        self.load_dict()
        w2vvocabu=[]
        for i in range(len(self.index2word)):
            w2vvocabu.append(self.index2word[i])
        return w2vvocabu

    def set_save_dict(self):
        """set index2word and word2index

        index2word -- from index to word
        word2index -- from word to index

        """
        i = 0
        for word in self.speStrs:
            self.index2word[i] = word
            self.word2index[word] = i
            i += 1

        for word in self.model.wv.index2word:
            self.index2word[i] = word
            self.word2index[word] = i
            i += 1

    def get_index(self, word):
        """get the index of a word


        Arguments:
            word {[string]} -- [word]

        Returns:
            [int] -- [return the index of the word]
        """
        word = word.lower()
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index[self.uknStr]

    def get_seq(self, words):
        """get the index of a list which consists of words


        Arguments:
            words {[list]} -- [list of words]

        Returns:
            [list of int] -- [return the index of words]
        """
        return [self.get_index(word) for word in words]

    def get_sen_vector(self, sent):
        """get the index of a sentence

        the sentence is separated by blank space

        Arguments:
            sent {[string]} -- [a sentence]

        Returns:
            [list of int] -- [return the index of words]
        """
        return self.get_seq(sent.split(' '))

    def get_word(self, index):
        """get the word by index

        Arguments:
            index {[int]} -- [index]

        Returns:
            [string] -- [word]
        """
        if index in self.index2word:
            return self.index2word[index]
        else:
            return ""
    '''
        获得从句子中单词索引到单词的方法，可选择忽略0索引
    '''
    def seq_to_sent(self, seq, neglect_zero=True):
        """from sequence of index to list of words

        e.g. seq = [25,35,45]  --> ["I","like","apple"]

        Arguments:
            seq {[list of int]} -- [sequence]

        Keyword Arguments:
            neglect_zero {bool} -- [if is True, do not translate 0 into 'Nil'] (default: {True})

        Returns:
            [list of string] -- [list of words]
        """
        if neglect_zero:
            return [self.get_word(index) for index in seq if index != 0]
        else:
            return [self.get_word(index) for index in seq]

    def get_vector(self, word, case_sensitive=False):
        """get the word embedding of a word

        Arguments:
            word {[string]} -- [word]

        Returns:
            [vector] -- [the embedding of the word]
        """
        if not case_sensitive:
            word = word.lower()
        if word in self.word2index:
            return self.embedding[self.word2index[word]]
        else:
            ######修改了一下，否则的话得到的不是零向量是[[0.,0.,……0.]]
            return np.zeros((1, self.length))[0]

    def get_avg_vector(self, sent):
        """get the average embedding of a sentence

        Arguments:
            sent {[string]} -- [the sentence is separated by blank space]

        Returns:
            [vector] -- [the embedding of the word]
        """
        isum = np.zeros((self.length))
        cnt = 0
        for word in sent.split(' '):
            vec = self.get_vector(word)
            if vec is not None:
                isum += vec
                cnt += 1
        if cnt != 0:
            isum /= cnt
        return isum

    def save_model(self):
        """save the model

        """
        logging.info('Saving Word2Vec model')
        if not os.path.exists(self.modelFolder):
            os.makedirs(self.modelFolder)
        self.model.wv.save_word2vec_format(self.modelFolder + "word2vec.model", binary=False)
        self.model.wv.save_word2vec_format(self.modelFolder + "b_word2vec.model", binary=True)
        np.savetxt(self.modelFolder + "embedding.txt", self.embedding)
        ifile = open(self.modelFolder + "vocab.txt", 'w' ,encoding = 'utf-8' )
        for i in range(self.dicSize):
            ifile.write("%s\n" % (self.index2word[i]))
        ifile.close()

    def load_model(self,loadCModel = True):
        """load the model
        Returns:
            [word2vec model] -- [the word2vec model]
        """
        logging.info('Loading Word2Vec model')
        if loadCModel:
            self.model = KeyedVectors.load_word2vec_format(self.modelFolder + "word2vec.model", binary=False,encoding="utf-8")
        self.load_dict()
        self.load_embedding()
        return self.model



if __name__ == '__main__':
    import multiprocessing
    model = Word2VecModel(min_count=5,size = 300, workers=multiprocessing.cpu_count())
    model.train_model_from_file("total.txt", is_cut=True)
