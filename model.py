### import libraries
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

### lode the iris data
data = pd.read_csv("iris.csv")
### label encoder
from sklearn import preprocessing 
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
### Encode labels in column 'species'. 
data['species']= label_encoder.fit_transform(data['species']) 

## 0 = Iris-setosa
## 1 = Iris-versicolor
## 2 = Iris-virginica

### inform X & Y
X = data.drop(columns='species')
Y = data.species
###Train the model
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=4,weights='uniform',algorithm='ball_tree', p=1)
model_knn.fit(X,Y)
# Saving model to disk
pickle.dump(model_knn,open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5.1,3.5,1.4,0.2]]))
