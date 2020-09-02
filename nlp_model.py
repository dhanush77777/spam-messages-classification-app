import pandas as pd
import pickle
df=pd.read_csv(r'C:/Users/SAIDHANUSH/spam-ham.csv')

df['Category'].replace('spam',0,inplace=True)
df['Category'].replace('ham',1,inplace=True)

x=df['Message']
y=df['Category']

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x=cv.fit_transform(x)

pickle.dump(cv,open('transform.pkl','wb'))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
pr=clf.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pr))

pickle.dump(clf,open('nlp_model1.pkl','wb'))