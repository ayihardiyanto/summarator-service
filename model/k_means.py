# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# importing the libraries

import nltk
from nltk import corpus
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from scipy.spatial import distance
# nltk.download('punkt')   # one time execution
# nltk.download('stopwords')  # one time execution
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# print('SENTENCE', sentence)

# cleaning the sentences

# corpus = summary


def performKMeans(text):
    sentence = sent_tokenize(text)
    corpus = defineCorpus(sentence)
    sentenceVector = vectorize(corpus)
    result = clusterize(sentenceVector)
    summary = []
    for i in sorted(result):
        summary.append(sentence[i])
    summary = " ".join(summary)
    return summary


def defineCorpus(sentence):
    corpus = []
    for i in range(len(sentence)):
        sen = re.sub('[^a-zA-Z]', " ", sentence[i])
        sen = sen.lower()
        sen = sen.split()
        sen = ' '.join([i for i in sen if i not in stopwords.words('english')])
        corpus.append(sen)
    return corpus


def vectorize(corpus):

    # creating word vectors
    n = 300
    all_words = [i.split() for i in corpus]
    model = Word2Vec(all_words, min_count=1, size=n)

    # creating sentence vectors
    sen_vector = []
    for i in corpus:

        plus = 0
        for j in i.split():
            plus += model.wv[j]
        plus = plus/len(plus)
        sen_vector.append(plus)
    return sen_vector

# performing k-means


def clusterize(sen_vector):
    n_clusters = 3
    kmeans = KMeans(3, init='k-means++', random_state=43)
    y_kmeans = kmeans.fit_predict(sen_vector)

    # finding and printing the nearest sentence vector from cluster centroid
    my_list = []
    for i in range(n_clusters):
        my_dict = {}

        for j in range(len(y_kmeans)):
            if y_kmeans[j] == i:
                my_dict[j] = distance.euclidean(
                    kmeans.cluster_centers_[i], sen_vector[j])
        my_list.append(min(my_dict, key=my_dict.get))

    return my_list
