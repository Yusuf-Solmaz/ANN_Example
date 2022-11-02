import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Churn_Modelling.csv')


X = dataset.iloc[:, 3:-1].values
y = dataset['Exited'].values

#print(dataset['Geography'].unique())   # ['France' 'Spain' 'Germany']

labelEncoder = LabelEncoder()
X[:, 2] = labelEncoder.fit_transform(X[:, 2])
#X[:, 1] = labelEncoder.fit_transform(X[:, 1])


ct= ColumnTransformer (transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))



X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)



#Train ve test'i ölçeklendiriyor.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

ann.add(tf.keras.layers.Dense(units='1',activation='sigmoid'))
                                        #Output non-binary ise(1 den fazla binary output) activation='softmax'
                            #Output non-binary ise: loss='categorical_crossentropy'
ann.compile(optimizer='adam',loss='binary_crossentropy' , metrics=['accuracy'])

ann.fit(X_train,y_train, batch_size=32, epochs=20)

    #Predict methodu her zaman 2D array olmalı
#print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))>0.5)


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
