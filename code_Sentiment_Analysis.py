import sys
import pandas as pd
from datetime import datetime
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os, sys
import re

start_time = datetime.now()

def sentiment_scores(sentence):

	# Creating a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    # Creating final list containing number of positive, negative and neural tweets
    y=[]
    if sentiment_dict['compound'] >= 0.05 :
        print("Positive")
        y.append(1)
        y.append(0)
        y.append(0)
    elif sentiment_dict['compound'] <= - 0.05 :
        print("Negative")
        y.append(0)
        y.append(1)
        y.append(0)
    else :
        print("Neutral")
        y.append(0)
        y.append(0)
        y.append(1)
    return y


def get_wordnet_pos(POS_TAG):
    # return the wordnet object value corresponding to the POS tag

    if POS_TAG.startswith('J'):
        return wordnet.ADJ
    elif POS_TAG.startswith('R'):
        return wordnet.ADV
    elif POS_TAG.startswith('N'):
        return wordnet.NOUN
    elif POS_TAG.startswith('V'):
        return wordnet.VERB
    else:
        return wordnet.NOUN


def clean_text(text):

    # Removing URLs from the text
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # Removing puncutation marks
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # Removing words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # Removing stop words as they do not contribute in the analysis
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # Removing empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(tag[0], get_wordnet_pos(tag[1])) for tag in pos_tags]
    # Removing words with only one letter
    text = [t for t in text if len(t) > 1]
    # Join all
    text = " ".join(text)
    return(text)

# Path of the directory where complete dataset is stored
path = '   '
dirs = sorted(os.listdir(path))
for f in dirs:
    dirr = sorted(os.listdir(path+"\\"+f))
    for fil in dirr:
        if str(fil) == str(f+".csv") : continue
        ua=pd.read_csv( path+"\\"+f+"\\"+fil ,names=["no","id","user","date","tweet","none1","place1"])
        #initialising number of positive, negative , neutral , average negative ,average neutral , average positive and total tweet respectively
        pos = neg = neut = neg_avg = neutral_avg = pos_avg = total = 0

        for i in range(len(ua)):
            sentiment = ua["tweet"][i]
            try:
              # cleaning the tweet
              sentiment=clean_text(sentiment)
              #printing cleaned tweet
              print(sentiment)
              k = sentiment_scores(sentiment)
              pos=pos+k[0]
              neg=neg+k[1]
              neut=neut+k[2]
            except: continue

        total = pos + neg + neut
        pos_avg = pos/total
        neg_avg = neg/total
        neutral_avg  = neut/total
        print(total,neg,neut,pos,neg_avg,neutral_avg,pos_avg)
        #saving the above results in a text file
        with open(path+"\\"+f+".txt", "a") as text_file:
            print(f"{fil},{total},{neg},{neut},{pos},{neg_avg},{neutral_avg},{pos_avg} \n", file=text_file)

end_time = datetime.now()
#printing the total duration taken by the algorithm
print('Duration: {}'.format(end_time - start_time))
