import tweepy
import re
import pickle
    
from tweepy import OAuthHandler
#       
consumer_key = 'UdTCXYKRioTKpqPgiCrhvWErM'
consumer_secret = '9N7WZWlrdNGuVK6DRnrkKShGbZMCJ0XlyotMUZABdUY6KbEuLn'
access_token = '1052781121811824641-fJ2dLdrjCtdqvwBjgaSZuigsagQoSB'
access_secret = 'r86x7L4ltUugVDHPjuTczwb6LVX5ITWsmFVmjHK9iqbgg'
      
auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
api = tweepy.API(auth,timeout=10)
       
list_tweets=[]
processed_tweets=[]
args=['google']
query = args[0]
if len(args)==1:
for status in tweepy.Cursor(api.search,q=query+"-filter:retweets",lang='en',result_type = 'recent').items(100):
    list_tweets.append(status.text)
  
       
       
       
with open('tfidfmodel.pickle','rb') as f:
vectorizer = pickle.load(f)
with open('classifier.pickle','rb') as f:
clf = pickle.load(f)
           
pos,neg = 0,0
      
for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$"," ",tweet)
    tweet = tweet.lower()

       
       
    tweet = re.sub(r"n't"," not ",tweet)
    tweet = re.sub(r"'s"," is ",tweet)
#           
    tweet = re.sub(r"\W"," ",tweet)#remove punctuation
    tweet = re.sub(r"\d"," ",tweet)#remove digits
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)#remove single characters
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)#remove single characters
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)#remove single characters
    tweet = re.sub(r"\s+"," ",tweet)#remove extra space
    processed_tweets.append(tweet)
    sent = clf.predict(vectorizer.transform([tweet]).toarray())
    print(tweet,":",sent)
    if sent==1: pos=pos+1
    else: neg = neg+1
               
           
        

#import matplotlib.pyplot as plt
#import numpy as np
#objects = ['positive','negative']
#y_pos=np.arange(len(objects))
#
#plt.bar(y_pos,[pos,neg],alpha=.5)
#plt.xticks(y_pos,objects)
#plt.ylabel('height of numbers')
#plt.title('comparison between negative and positive comments')
#plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    