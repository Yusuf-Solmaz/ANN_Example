import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,minmax_scale
import matplotlib.pyplot as plt



#                             ANN Regression Example 1

#Dataset'imizi bağımlı ve bağımsız değişkenleri matrix haline getirdik. X bağımsız değişken, y bağımlı değişken.
dataset = pd.read_excel('Folds5x2_pp.xlsx')

X = dataset.drop(["PE"],axis=1).values # Better way axis = sütun
y = dataset.iloc[:,-1].values

X_scaled = minmax_scale(X)
X=pd.DataFrame(X_scaled)



#X1 = dataset.iloc[:,:-1].values
#print(X1)
#print(y)

# Değişkenlerimizi (X ve y) test ve train (yani işlem yapılan kısım) olarak ayırdık. Test size kısmı Datanın %20 lik kısmını test olarak belirlendiğini gösterir.
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Artificial Neural Network için obje oluşturduk.
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(1548,input_shape=(4,),activation='relu'))
# Hidden Layer oluşturduk.


ann.add(tf.keras.layers.Dense( units=6,activation='relu'))
ann.add(tf.keras.layers.Dense( units=6,activation='relu'))
ann.add(tf.keras.layers.Dense( units=6,activation='relu'))




# Output Layer oluşturduk.
ann.add(tf.keras.layers.Dense(units=1))

#Train yani verimizin işlenip tahmin yapıldığı, makine öğrenmesinin gerçekleştiği kısım.
ann.compile(optimizer='adam',loss='mean_squared_error')
results=ann.fit(X_train,y_train,batch_size=32,epochs=100)

#Tahminlerin bulunması.
y_pred = ann.predict(X_test)

#Örnek Output 1

"""
print (y_pred)
print("***************")
print(y_test)

[[430.18]
 [461.17]
 [464.63]
 ...
 [471.88]
 [438.81]
 [457.95]]
***************
[431.23 460.01 461.14 ... 473.26 438.   463.28]

"""


# Burada tahmin ve gerçek değerlerimizi aynı mattrixte yan yana yazmış oluyoruz.
np.set_printoptions(precision=2)
finalResult=np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

print(ann.predict(sc.fit_transform([[15.32,28.96,1023.85,90.47]])))





#Örnek Output 2
"""
 Tahmin - Gerçek Data
[[440.24 431.23]
 [466.65 460.01]
 [472.97 461.14]
 ...
 [479.67 473.26]
 [445.66 438.  ]
 [463.51 463.28]]
"""