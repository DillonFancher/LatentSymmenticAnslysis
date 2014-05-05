import twitter
import json
import csv
import nltk as nk
import gensim as gsm
from gensim import corpora, models, similarities
import numpy as np
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
    Tweets = 'BigTweets.json'
    tweet_text, tweet_geo = TweetParser(Tweets)
    tweet_clean_text = tweetClean(tweet_text)
    
    with open('ColoradoTweets.csv', 'wb') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(tweet_clean_text)
##########################################################################
##########################################################################

##########################################################################
##########################################################################

def TweetParser(Tweets):
    
    #Read in the full json for each tweet into python dicts
    tweet_json = []
    with open(Tweets) as f:
        for line in f:
	    tweet_json.append(json.loads(line))
    print(len(tweet_json))
    #Parse the json file to extract the text object and the lattitude and
    #longitude coordinates from where the tweet came from  
    tweet_text = []
    tweet_geo = []
    ColoradoTweets = [] 
    lat1 = 39.0
    lon1 = -105.5  
    for obj in tweet_json:
        lat2 = obj['coordinates']['coordinates'][1]
        lon2 = obj['coordinates']['coordinates'][0]
        d = LatLongDist(lat1, lon1, lat2, lon2)
        if d < 300:
            tweet_text.append(obj['text'])#.encode('utf-8'))
           # print(tweet_text.encode('utf-8'))
          #  tweet_geo.append(obj['coordinates']['coordinates'])
          #  ColoradoTweets.append(obj['text'], obj['coordinates']['coordinates'])
     
    
    return tweet_text, tweet_geo        
    
##########################################################################           
##########################################################################            


###########################################################################
###########################################################################
def LatLongDist(lat1, lon1, lat2, lon2):
   
        
        #Convert latitude and longitude into radians
        la1 = m.radians(lat1)
        la2 = m.radians(lat2)
        lo1 = m.radians(lon1)
        lo2 = m.radians(lon2)
       
        #Calculate the distance between [lat1, long1] and [lat2, long2]
        dlon = abs(lo1-lo2)
        dlat = abs(la1-la2)
        a = (m.sin(dlat/2)**2) + (m.cos(la1)*m.cos(la2)*(m.sin(dlon/2)**2))
        c = 2*m.atan2(m.sqrt(a), m.sqrt(1-a))
        d = 3961*c
    
        return(d)
###########################################################################
###########################################################################


#Tokenizes the text of each tweet and de-stems each word
##########################################################################           
##########################################################################            
def tweetClean(tweet_text): 
    tweet_clean_text = []
  
    for doc in tweet_text:
        twitterWords = doc.split()
        
        #remove stop words using NLTK corpus
        twitterWords = [word.lower() for word in twitterWords]
        a = '"'
        count = -1
        for word in twitterWords:
            count = count+1
            if word[:1] == a or word[:1] == '[' or word[:1] == '(' or word [:1] == ']' or word[:1] == ')':
                    print(word)
                    twitterWords.pop(count)
        twitterWords = [w for w in twitterWords if not w in stopwords.words('english')]
        twitterWords = [w for w in twitterWords if repr(w[:2]) != repr(w[:3])]
        twitterWords = [w for w in twitterWords if w[:1] != '@' and w[:1] != '#' and w[:4] != 'http']
        twitterWords = [w for w in twitterWords if repr(w).find('\\') == -1]
        #twitterWords = [w.encode("utf-8") for w in twitterWords]
        #remove custom list of stop words using experimentation
        noiseWords = ["i'm", "like", "get", "don't", "it's", "go", "lol", "got", 
                      "one", "know", "@", "good", "want", "can't", "need", "see", 
                      "people", "going", "back", "really", "u", "think", "right",
                      "never", "day", "time", "never", "that's", "even", ",", "."
                      "make", "wanna", "you're", "come", "-", "still", "much", "someone",
                      "today", "gonna", "new", "would", "take", "always", "im", "i'll",
                      "best", "'", "feel", "getting", "say", "tonight", "last", "ever",
                      "better", "i've", "look", "fucking", "way", "could", "!", "oh"
                      "tomorrow", "night", "first", "miss", "ain't", "thank", "2", "bad"
                      "little", "thanks", "something", "wait", "&amp;", "`", "oh", "make", 
                      "bad", "let","stop", "well", "tell"]
    
        twitterWords = [w for w in twitterWords if not w in noiseWords]
        twitterWords = [w.encode('utf-8') for w in twitterWords]
        twitterWords = [w.rstrip('.') for w in twitterWords]
        twitterWords = [w.rstrip('!') for w in twitterWords]
        twitterWords = [w.rstrip('?') for w in twitterWords]
        twitterWords = [w.rstrip(',') for w in twitterWords]
        #twitterWords = [st.stem(w) for w in twitterWords]
        if len(twitterWords) != 0:
            tweet_clean_text.append(twitterWords)
    print(tweet_clean_text)
    return tweet_clean_text
            
##########################################################################           
##########################################################################            
main()
