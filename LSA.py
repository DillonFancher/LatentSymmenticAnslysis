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
    
    k = len(S)
    
    for p in range(0, 10):
        trunc = k-p
        S = np.delete(S, (trunc-1), axis = 0)
   
    I = np.identity(len(S))
    print(len(S))
    
    
    Ulen = np.shape(U)[1]
    Utrunk = Ulen-len(S)
    for i in range(0, Utrunk):
        j = Ulen - i
        U = np.delete(U, (int(j)-1) , axis = 1)
    
    Vtlen = np.shape(Vt)[1]
    Vtrunk = Vtlen - len(S) 
    for m in range(0, Vtrunk):
        n = Vtlen - m
        Vt = np.delete(Vt, (n-1), axis = 0)
    
    print(np.shape(U))
    print(np.shape(Vt))
    print(np.shape(I))
    inter = np.dot(U, I)
    A = np.dot(inter, Vt)
    
    error = tf_idf_MATRIX - A
    print(np.shape(error))
    norm = LA.norm(error, 2)
    print(norm)
   
  
  
    #np.savetxt("TFIDF-Matrix.csv", tf_idf_MATRIX, delimiter=",")
    #np.savetxt("U.csv", U, delimiter=",")
    #np.savetxt("S.csv", S, delimiter=",")
    #np.savetxt("Vt.csv", Vt, delimiter=",")
    
    
    return U, S, Vt 
    
    
    
    
    
##########################################################################
##########################################################################

main()
