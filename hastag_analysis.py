#######################################
#########                      ########
#########                      ########
#########  Project AIT 582     ########
#########  Author:Meenakshi    ########
#########  Date:10th May 2019  ########
#########                      ########
#######################################

#######################################
###importing all the packages#####
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline


####Reading the file
df=pd.read_csv('Hurricane_Harvey.csv', encoding='windows-1254')
df.head()

from nltk.corpus import stopwords
stop=stopwords.words('english')
df['Tweet']=df["Tweet"].apply(lambda x:" ".join(x.lower() for  x in str(x).split() if x not in stop))
df['Tweet'].head()

#####Finding the hashtags in a tweet
def hashtag(tweet):
    hashtags = " ".join([word for word in tweet.split() if word.startswith('#')])
    hashtags = hashtags.lower().split()
    return hashtags
df['Tweettag'] = df['Tweet'].map(lambda x: hashtag(x))
df['Tweettag']=df['Tweettag'].astype(str)
df['Tweettag']=df['Tweettag'].str.replace('[','')
df['Tweettag']=df['Tweettag'].str.replace(']','')
df2=df['Tweettag']
df2.replace('',np.nan,inplace=True)
df2.dropna()

#######Splitting the values by comma
df2=df2.str.split(',', expand=True)

####COnverting all split columns to seperate dataframes #######
df3=df2[0]
df3.dropna()
df4=df2[1]
df5=df2[2]
df6=df2[3]
df7=df2[4]
df8=df2[5]
df9=df2[6]
df10=df2[7]
df11=df2[8]
df12=df2[9]
df13=df2[10]
df14=df2[11]
df15=df2[12]
df16=df2[13]
df17=df2[14]
df18=df2[15]

#####Dropping the null values#######
df4.dropna()
df5.dropna()
df6.dropna()
df7.dropna()
df8.dropna()
df9.dropna()
df10.dropna()
df11.dropna()
df12.dropna()
df13.dropna()
df14.dropna()
df15.dropna()
df16.dropna()
df17.dropna()
df18.dropna()

#####Concatenating the dataframes#######
final=pd.concat([df3, df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18])
final
final = final.replace('\'', np.nan)
final=final.dropna()
final.isnull().sum()
final.head()
#####converting this to the csv file for visualization in the Tableau
final.to_csv('out.csv')


#####Top 15 hashtags
bs_cnt = pd.DataFrame(final.value_counts()[:15])
bs_cnt[0]
