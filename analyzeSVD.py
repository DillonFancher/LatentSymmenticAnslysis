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


U = np.genfromtxt("U.csv", delimiter = ",")
S = np.genfromtxt("S.csv", delimiter = ",")
Vt = np.genfromtxt("Vt.csv", delimiter = ",")
tfIDF = np.genfromtxt("TFIDF-MATRIX.csv", delimiter = ",")


#Truncate the SVD system
k = len(S)
for p in range(0, 97):
    trunc = k-p
    S = np.delete(S, (trunc-1), axis = 0)

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

#Re-Construct A from the truncated SVD system
inter = np.dot(U, np.diag(S))
A = np.dot(inter, Vt)
error = tfIDF - A
norm = LA.norm(error, 2)

print(np.column(A,2))
   