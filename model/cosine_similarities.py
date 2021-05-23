from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer 
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def find_similarities(text):
    # tokenize sentences
    sentences = sent_tokenize(text)
    #set stop words
    stops = list(set(stopwords.words('english'))) + list(punctuation)
    
    #vectorize sentences and remove stop words
    vectorizer = TfidfVectorizer(stop_words = stops)
    #transform using TFIDF vectorizer
    trsfm=vectorizer.fit_transform(sentences)
    
    #creat df for input article
    text_df = pd.DataFrame(trsfm.toarray(),columns=vectorizer.get_feature_names(),index=sentences)
    
    #declare how many sentences to use in summary
    num_sentences = text_df.shape[0]
    num_summary_sentences = int(np.ceil(num_sentences**.5))
        
    #find cosine similarity for all sentence pairs
    similarities = cosine_similarity(trsfm, trsfm)
    
    #create list to hold avg cosine similarities for each sentence
    avgs = []
    for i in similarities:
        avgs.append(i.mean())
     
    #find index values of the sentences to be used for summary
    top_idx = np.argsort(avgs)[-num_summary_sentences:]
    
    return top_idx

def performCosineSimilarity(text):
    #find sentences to extract for summary
    sents_for_sum = find_similarities(text)
    #sort the sentences
    sort = sorted(sents_for_sum)
    #display which sentences have been selected
    # print(sort)
    
    sent_list = sent_tokenize(text)
    #print number of sentences in full article
    # print(len(sent_list))
    
    
    #extract the selected sentences from the original text
    sents = []
    for i in sort:
        sents.append(sent_list[i].replace('\n', ''))
    
    #join sentences together for final output
    summary = ' '.join(sents)
    return summary