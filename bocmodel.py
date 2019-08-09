import multiprocessing
from collections import Counter
from gensim.models import Word2Vec
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot


class BOCModel:
    """
    :param input: 'List of str' or 'numpy.ndarray'
        就像下面这样的二维数组
        corpus = ['corpus is list of str format', 'each document is str']
        或者训练好的词向量
        List of documents. Document is represented with str
        Or trained word vector representation.
    :param n_concepts: int
        概念簇的数量
        Number of concept.
    :param min_count: int
        单词出现的最小频率，小于它的会被丢弃
        Minumum frequency of word occurrence
    :param embedding_dim: int
        词向量维度
        Word embedding dimension
    :param embedding_method: str
        词向量方法：'word2vec', 'nmf', 'svd'
        Embedding method. Choose from ['word2vec', 'nmf', 'svd']
    :param concept_method: str
        概念簇聚类方法，k-means
        Concept method. Choose from ['kmeans']
    :param tokenizer: callable
        返回类型应该是str的可迭代集合。
        Return type should be iterable collection of str.
        Default is lambda x:x.split()
    :param idx_to_vocab: list of str
        str word列表。每个字对应输入字向量矩阵的行
        Word list. Each word corresponds row of input word vector matrix

    Attributes
    ----------
    wv : numpy.ndarray
        Word vector representation
    idx_to_vocab : list of str
        Word list. Each word corresponds row of input word vector matrix
    idx_to_concept : numpy.ndarray
        Word to concept index
    idx_to_concept_weight : numpy.ndarray
        Word to concept weight
    x_tc : scipy.sparse.csr_matrix
        Term to concept sparse matrix
    """

    def __init__(self, input=None, n_concepts=100, min_count=10, embedding_dim=100,
        embedding_method='word2vec', concept_method='kmeans', tokenizer=None,
        idx_to_vocab=None, verbose=True):

        if not embedding_method in ['word2vec', 'nmf', 'svd']:
            raise ValueError("embedding_method should be ['word2vec', 'nmf' ,'svd']")
        if not concept_method in ['kmeans']:
            raise ValueError("concept_method should be ['kmeans']")
        if min_count < 1 and isinstance(min_count, int):
            raise ValueError('min_count should be positive integer')

        self.n_concepts = n_concepts
        self.min_count = min_count
        self.embedding_dim = embedding_dim
        self.embedding_method = embedding_method
        self.concept_method = concept_method
        self.verbose = True

        if tokenizer is None:
            tokenizer = lambda x:x.split()
        if not callable(tokenizer):
            raise ValueError('tokenizer should be callable')
        self.tokenizer = tokenizer

        self.wv = None
        self.idx_to_concept = None
        self.idx_to_concept_weight = None

        if isinstance(input, np.ndarray):
            if idx_to_vocab is None:
                raise ValueError('idx_to_vocab should be inserted '\
                                 'when input is word vector')
            if len(idx_to_vocab) != input.shape[0]:
                a = len(idx_to_vocab)
                b = input.shape[0]
                raise ValueError('Length of idx_to_vocab is different '\
                                 'with input matrix %d != %d' % (a, b))
            self.idx_to_vocab = idx_to_vocab
            self.wv = input
            self._train_concepts(input)
        elif input is not None:
            self.idx_to_vocab = None
            self.fit(input)
        else:
            self.idx_to_vocab = None

    def fit_transform(self, corpus, apply_icf=False):
        if isinstance(corpus, np.ndarray):
            raise ValueError('Input corpus should not be word vector')

        if ((self.wv is None)
             or (self.idx_to_concept is None)
             or (self.idx_to_vocab is None)):
            self.fit(corpus)
        return self.transform(corpus, apply_icf)

    def fit(self, corpus):
        self._train_word_embedding(corpus)
        self._train_concepts(self.wv)

    def _train_word_embedding(self, corpus):
        # tokenization
        if self.embedding_method != 'word2vec':
            self._bow, self.idx_to_vocab = corpus_to_bow(
                corpus, self.tokenizer, self.idx_to_vocab, self.min_count)

        # word embedding
        if self.embedding_method == 'word2vec':
            self.wv, self.idx_to_vocab = train_wv_by_word2vec(corpus,
                self.min_count, self.embedding_dim, self.tokenizer)
        elif self.embedding_method == 'svd':
            self.wv = train_wv_by_svd(self._bow, self.embedding_dim)
        elif self.embedding_method == 'nmf':
            self.wv = train_wv_by_nmf(self._bow, self.embedding_dim)
        else:
            raise ValueError("embedding_method should be ['word2vec', 'svd']")

    def _train_concepts(self, wv):
        idx_to_c, idx_to_cw = train_concepts_by_kmeans(wv, self.n_concepts)
        self.idx_to_concept = idx_to_c
        self.idx_to_concept_weight = idx_to_cw

        cols = idx_to_c
        rows = np.arange(idx_to_c.shape[0])
        data = idx_to_cw
        n_terms = rows.shape[0]

        self.x_tc = csr_matrix((data, (rows, cols)),
            shape=(n_terms, self.n_concepts))

    def transform(self, corpus_or_bow=None, apply_icf=False, remain_bow=False):
        if corpus_or_bow is None and hasattr(self, '_bow'):
            return self.transform(self._bow, remain_bow)

        # use input bow matrix
        if sp.sparse.issparse(corpus_or_bow):
            if corpus_or_bow.shape[1] != len(self.idx_to_vocab):
                a = corpus_or_bow.shape[1]
                b = len(self.idx_to_vocab)
                raise ValueError('The vocab size of input is different '\
                    'with traind vocabulary size {} != {}'.format(a, b))
            self._bow = corpus_or_bow
        # use only trained vocabulary
        else:
            self._bow, _ = corpus_to_bow(corpus_or_bow, self.tokenizer,
                self.idx_to_vocab, min_count=-1)

        # concept transformation
        boc = bow_to_boc(self._bow, self.x_tc)

        if apply_icf:
            self.icf = train_icf(boc)
            boc = safe_sparse_dot(boc, sp.sparse.diags(self.icf))

        if not remain_bow and hasattr(self, '_bow'):
            del self._bow

        return boc

def corpus_to_bow(corpus, tokenizer, idx_to_vocab=None, min_count=-1):
    if idx_to_vocab is not None:
        vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    else:
        idx_to_vocab, vocab_to_idx = scan_vocabulary(
            corpus, tokenizer, min_count)
    bow = vectorize_corpus(corpus, tokenizer, vocab_to_idx)
    return bow, idx_to_vocab

def scan_vocabulary(corpus, tokenizer, min_count):
    counter = Counter(word for doc in corpus
        for word in tokenizer(doc))
    counter = {vocab:count for vocab, count in counter.items()
        if count >= min_count}
    idx_to_vocab = [vocab for vocab in sorted(counter, key=lambda x:-counter[x])]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx

def vectorize_corpus(corpus, tokenizer, vocab_to_idx):
    rows = []
    cols = []
    data = []
    for i, doc in enumerate(corpus):
        terms = tokenizer(doc)
        terms = Counter([vocab_to_idx[t] for t in terms if t in vocab_to_idx])
        for j, c in terms.items():
            rows.append(i)
            cols.append(j)
            data.append(c)
    n_docs = i + 1
    ########################################
    n_terms = len(vocab_to_idx)
    return csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms))

def train_wv_by_word2vec(corpus, min_count, embedding_dim, tokenizer):
    class CorpusDecorator:
        def __init__(self, corpus, tokenizer):
            self.corpus = corpus
            self.tokenizer = tokenizer
        def __iter__(self):
            for doc in self.corpus:
                terms = self.tokenizer(doc)
                if terms:
                    yield terms

    decorated_corpus = CorpusDecorator(corpus, tokenizer)
    word2vec = Word2Vec(decorated_corpus,
        size=embedding_dim, min_count=min_count)
    wv = word2vec.wv.vectors
    idx_to_vocab = word2vec.wv.index2word
    return wv, idx_to_vocab

def train_wv_by_nmf(bow, embedding_dim):
    nmf = NMF(n_components=embedding_dim, verbose=False, tol=0.015)
    wv = nmf.fit_transform(bow.transpose())
    return wv

def train_wv_by_svd(bow, embedding_dim):
    svd = TruncatedSVD(n_components=embedding_dim)
    wv = svd.fit_transform(bow.transpose())
    return wv

def train_concepts_by_kmeans(wv, n_concepts):
    wv_ = normalize(wv)
    #设置了时间种子
    kmeans = KMeans(n_clusters=n_concepts,init='k-means++',
         max_iter=5, n_init=1,random_state=300,n_jobs=(multiprocessing.cpu_count()-2))
    #原版
    # kmeans = KMeans(n_clusters=n_concepts,
    #     init='random', max_iter=20, n_init=1)
    idx_to_concept = kmeans.fit_predict(wv_)
    idx_to_concept_weight = np.ones(wv_.shape[0])
    return idx_to_concept, idx_to_concept_weight

def bow_to_boc(bow, x_tc):
    boc = safe_sparse_dot(bow, x_tc)
    return boc

def train_icf(boc):
    _, cf_array = boc.nonzero()
    num_docs, num_concepts = boc.shape
    cf = np.bincount(cf_array, minlength=num_concepts)
    # 做了非零处理
    icf = num_docs / (cf+1)
    icf[np.isinf(icf)] = 0
    return icf
