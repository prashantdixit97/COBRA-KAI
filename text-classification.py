import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files


reviews = load_files('txt_sentoken/')
X,y = reviews.data,reviews.target
# pickling files
with open('X.pickle','wb') as f:
    pickle.dump(X,f)

with open('y.pickle','wb') as f:
    pickle.dump(y,f)
# unpickling files    
with open('X.pickle','rb') as f:
    X = pickle.load(f)

with open('y.pickle','rb') as f:
    y = pickle.load(f)


corpus=[]
for i in range(0,len(X)):
    review = re.sub(r'\W',' ',str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+', ' ',review)
    review = re.sub(r'^[a-z]\s+' , ' ',review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
                   # unpickling the classifier and vectorizER
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)

with open('tfidfmodel.pickle','rb') as f:
    tfdf = pickle.load(f)
    
smpl = ['people are worst shit']
smpl = tfdf.transform(smpl).toarray()
print(clf.predict(smpl))
    
    
    


