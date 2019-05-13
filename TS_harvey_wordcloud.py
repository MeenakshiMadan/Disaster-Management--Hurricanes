#####TS harvey visualizations######

###importing all the packages#####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

df=pd.read_csv("TS_Harvey_Tweets.csv")
df.head()


#Removing the  punctutations and unuseful characters
df['Tweet']=df['Tweet'].str.replace('[^\w\s]','')
df['Tweet'].head()
df['Tweet'] =df['Tweet'].str.replace(r"\b[a-zA-Z]\b",'')
df['Tweet'] =df['Tweet'].str.replace("\_",'')
df['Tweet']=df['Tweet'].str.strip()
df['Tweet'] = df['Tweet'].str.replace(r"[0-9]",'')
df['Tweet'] = df['Tweet'].str.replace('https?://[A-Za-z0-9./]+','')
df['Tweet'] = df['Tweet'].str.replace(r'RT\w+','')
df['Tweet']=df['Tweet'].str.replace(r'@[A-Za-z0-9]+','')
df['Tweet']=df['Tweet'].str.replace('[^\w\s]','')

##Removing the stopwords from the Tweets
from nltk.corpus import stopwords
stop=stopwords.words('english')
df['Tweet']=df["Tweet"].apply(lambda x:" ".join(x.lower() for  x in str(x).split() if x not in stop))
df['Tweet'].head()

###generating the word cloud
wordcloud = WordCloud(stopwords=STOPWORDS,max_words=500,
                      background_color='white',width=3000,
                      collocations=False,min_font_size=6,
                      height=3000
                     ).generate(cleaned_word)
                     
                     
plt.figure(1,figsize=(20, 20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
####Saving the file to the png
wordcloud.to_file("Tropical_storm_Harvey_Keywords.png")
