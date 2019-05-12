import pandas as pd
import numpy as np
data = pd.read_csv('C:\\Users\Meenakshi\Desktop\hurricane.csv')
data.head()

data['shelter'] = np.where(data['shelter']=='shelter',1,0)
print(data.shape)
data.head(10)

from sklearn.model_selection import train_test_split
data['Tweet']=data['Tweet'].str.replace(r'@[A-Za-z0-9]+','')
data['Tweet']=data['Tweet'].str.replace('[^\w\s]','')
data['Tweet'] = data['Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
data['Tweet']

import re
data['Tweet'] =data['Tweet'].str.replace(r"\b[a-zA-Z]\b",'')
data['Tweet'] =data['Tweet'].str.replace("\_",'')
data['Tweet']=data['Tweet'].str.strip()
data['Tweet'] = data['Tweet'].str.replace(r"[0-9]",'')
data['Tweet'] = data['Tweet'].str.replace('https?://[A-Za-z0-9./]+','')
data['Tweet'] = data['Tweet'].str.replace(r'RT\w+','')

from nltk.corpus import stopwords
stop=stopwords.words('english')
data['Tweet']=data["Tweet"].apply(lambda x:" ".join(x.lower() for  x in str(x).split() if x not in stop))
data['Tweet']

X_train, X_test, y_train, y_test = train_test_split(data['Tweet'],data['shelter'],random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)
clfrNB = MultinomialNB(alpha = 0.1)
clfrNB.fit(X_train_vectorized, y_train)
preds = clfrNB.predict(vect.transform(X_test))
score = roc_auc_score(y_test, preds)

print(score)
print(preds)


from sklearn.metrics import confusion_matrix, classification_report
# suppose your predictions are stored in a variable called preds
# and the true values are stored in a variable called y
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
