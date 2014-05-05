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
for p in range(0, 1):
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
Sk = np.diag(S)
inter = np.dot(U, Sk)
A = np.dot(inter, Vt)
#error = tfIDF - A

#Check the error of the newly truncated matrix 
#agianst the original
#norm = LA.norm(error, 2)
#print(np.column(A,2))
docRel = np.dot(A.T, A)
np.savetxt("docRel2.csv", docRel, delimiter = ",")

print(np.shape(Sk))
print(np.shape(Vt[:,0]))

doc1 = np.dot(Sk, Vt[:, 1])
doc2 = np.dot(Sk, Vt[:, 2])
nd1 = LA.norm(doc1, 2)
nd2 = LA.norm(doc2, 2)

f1 = LA.norm(A, 'fro')
f2 = LA.norm(tfIDF, 'fro')
#print(f1/f2)
#print(LA.norm(tfIDF-A, 'fro'))
sim = np.dot(doc1, doc2)/(nd1*nd2)
print(sim)








   