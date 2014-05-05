import twitter
import json
import csv
import nltk as nk
import gensim as gsm
from gensim import corpora, models, similarities
import numpy as np
from numpy import linalg as LA
import nltk as nltk
import itertools
from nltk.stem.lancaster import LancasterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import math as m

##########################################################################
##########################################################################

#Main function to do LSA
def main():
    reader = csv.reader(open("ColoradoTweets.csv", "rb"))
    tweets = []
    for row in reader:
        tweets.append(row)
    [U, S, Vt] = corpusGen(tweets)
##########################################################################
##########################################################################
       
#Forms the corpus of tf-idf vectors
##########################################################################
##########################################################################
def corpusGen(tweet_clean_text):
    dictionary = corpora.Dictionary(tweet_clean_text)
    #print(dictionary)
    
    corpus = [dictionary.doc2bow(tweet) for tweet in tweet_clean_text]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    tf_idf_MATRIX = gsm.matutils.corpus2dense(corpus_tfidf, num_terms = len(dictionary)) 
    
    U, S, Vt = np.linalg.svd(tf_idf_MATRIX)
    print(np.shape(U))
    
   
  
  
    np.savetxt("TFIDF-Matrix.csv", tf_idf_MATRIX, delimiter=",")
    np.savetxt("U.csv", U, delimiter=",")
    np.savetxt("S.csv", S, delimiter=",")
    np.savetxt("Vt.csv", Vt, delimiter=",")
    
    print('worked')
    
    return U, S, Vt 
    
    
    
    
    
##########################################################################
##########################################################################

main()
