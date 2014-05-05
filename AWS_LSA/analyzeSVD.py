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


U = np.genfromtxt("U_Big.csv", delimiter = ",")
S = np.genfromtxt("S_Big.csv", delimiter = ",")
Vt = np.genfromtxt("Vt_Big.csv", delimiter = ",")
tfIDF = np.genfromtxt("TFIDF-MATRIX_Big.csv", delimiter = ",")

#Truncate the SVD system
k = len(S)
for p in range(0, 1):
    trunc = k-p
    S = np.delete(S, (trunc-1), axis = 0)
Sk = np.diag(S)
print("strunk =")
print(np.shape(Sk))

Ulen = np.shape(U)[1]
Utrunk = Ulen-len(S)
for i in range(0, Utrunk):
    j = Ulen - i
    U = np.delete(U, (int(j)-1) , axis = 1)
print("Utrunk shape")
print(np.shape(U))

Vtlen = np.shape(Vt)[1]
Vtrunk = Vtlen - len(S) 
for m in range(0, Vtrunk):
    n = Vtlen - m
    Vt = np.delete(Vt, (n-1), axis = 0)
print("Vtrunk shape =")
print(np.shape(Vt))

#Re-Construct A from the truncated SVD system
inter = np.dot(U, Sk)
A = np.dot(inter, Vt)
error = tfIDF - A
norm1 = LA.norm(error, 'fro')
norm2 = LA.norm(A, 'fro')/LA.norm(tfIDF, 'fro')
print(norm1)
print(norm2)

docRel = np.dot(A.T, A)
print(docRel[17,10])
print(docRel[16,9])
print(docRel[18, 11])
doc1 = np.dot(Sk, Vt[:, 17])
doc2 = np.dot(Sk, Vt[:, 10])
nd1 = LA.norm(doc1, 2)
nd2 = LA.norm(doc2, 2)
sim = np.dot(doc1, doc2)/(nd1*nd2)
print('simil')
print(sim)
np.savetxt("docRel.csv", docRel, delimiter = ",")
np.savetxt("Sk.csv", S, delimiter=",")
np.savetxt("Uk.csv", U, delimiter=",")
np.savetxt("Vtk.csv", Vt, delimiter=",")
np.savetxt("Ak.csv", A, delimiter=",")

   
