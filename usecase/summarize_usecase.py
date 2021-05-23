
# import asyncio
from model.cosine_similarities import performCosineSimilarity
from model.k_medoids import performKMedoids
from model.k_means import performKMeans
import operator
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from model.tf_idf import Stopwords, freq, lemmatize_words, remove_special_characters, sentence_importance


async def getSummaryResult(text, useClustering):
    summary = []
    sentence_no = []

    tokenized_sentence = sent_tokenize(text.rstrip('\n'))
    text = remove_special_characters(str(text))
    text = re.sub(r'\d+', '', text)
    tokenized_words_with_stopwords = word_tokenize(text)
    tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(tokenized_words)


    sentence_with_importance = getSentenceImportance(tokenized_sentence, tokenized_words)

    sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1),reverse=True)

    addSentenceNumber(tokenized_sentence, sentence_with_importance, sentence_no)
    
    sentence_no.sort()

    summary = assembleWords(tokenized_sentence, sentence_no)
    if(useClustering is True and len(summary) > 3):
        kMedoidsResult = performKMedoids(summary, 3)
        return performCosineSimilarity(kMedoidsResult)
    return performCosineSimilarity(summary)
    


def getSentenceImportance(tokenized_sentence, tokenized_words):
    sentence_with_importance = {}
    c = 1
    word_freq = freq(tokenized_words)
    for sent in tokenized_sentence:
        sentenceimp = sentence_importance(sent,word_freq,tokenized_sentence)
        sentence_with_importance[c] = sentenceimp
        c += 1
    return sentence_with_importance


def addSentenceNumber(tokenized_sentence, sentence_with_importance, sentence_no):
    cnt = 0
    no_of_sentences = int((50 * len(tokenized_sentence))/100)
    for word_prob in sentence_with_importance:
        if cnt < no_of_sentences:
            sentence_no.append(word_prob[0])
            cnt = cnt+1
        else:
            break

def assembleWords(tokenized_sentence, sentence_no):
    cnt = 1
    summary = []
    for sentence in tokenized_sentence:
        if cnt in sentence_no:
            summary.append(sentence)
        cnt = cnt+1
    
    summary = " ".join(summary)
    return summary