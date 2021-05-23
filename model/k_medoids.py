
from nltk.corpus import stopwords
import re
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import sent_tokenize
from sklearn_extra.cluster import KMedoids
from scipy.spatial import distance

def performKMedoids(text, cluster):
    sentence = sent_tokenize(text)
    corpus = defineCorpus(sentence)
    sentenceVector = vectorize(corpus)
    result = kMedoidsClustering(sentenceVector, cluster)
    summary = []
    for i in sorted(result):
        summary.append(sentence[i])
    summary = " ".join(summary)
    return summary


def kMedoidsClustering(sen_vector, cluster):
    n_clusters = cluster
    kMedoids = KMedoids(n_clusters, random_state=43, method="pam")
    y_kmedoids = kMedoids.fit_predict(sen_vector)

    # finding and printing the nearest sentence vector from cluster centroid
    my_list = []
    for i in range(n_clusters):
        my_dict = {}

        for j in range(len(y_kmedoids)):
            if y_kmedoids[j] == i:
                my_dict[j] = distance.euclidean(
                    kMedoids.labels_[i], sen_vector[j])
        my_list.append(min(my_dict, key=my_dict.get))

    return my_list


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