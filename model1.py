import numpy as np
import pandas as pd
import os

df = pd.read_csv('Churn_Modelling.csv')

# Drop useless columns
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

print(df.shape)
#print(df.head())
print('###')
#print(df.info())
print('###')
#print(df.describe())
print('###')
#print(df.isnull().sum())
print('###')
#print(df['Exited'].value_counts())
print("")
#print(df.duplicated().sum())
#print(df['Exited'].value_counts())
#print(df['Gender'].value_counts())
df=pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)
#print(df.head())
x=df.drop(columns=['Exited'])
y=df['Exited']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)#used to make test data and trauning dat using this module
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
#print(x_train_scaled)
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(3,activation='relu',input_dim=11))
model.add(Dense(1,activation='sigmoid'))
#print(model.summary())
model.compile(loss='binary_crossentropy',optimizer='Adam')
model.fit(x_train_scaled,y_train,epochs=15)
#print(model.layers[0].get_weights())# to get value of weights
#prdiction
pred=model.predict(x_test_scaled)
#print(pred)
p=np.where(pred>0.5,1,0)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,p)
print(acc*100)






















































































































































































































