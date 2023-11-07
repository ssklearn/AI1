# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:47:50 2023

@author: hama
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("D:/ai middle class/data/iris3.csv")
print(df.head())

# sns.pairplot(df, hue='species')
#직접 컬럼을 선택하는 방법
# sns.pairplot(df[['petal_length', 'petal_width']])
#vars옵션을 사용해서 컬럼 선택 후 입력하는 방법
# sns.pairplot(df, vars=['petal_length', 'petal_width'], hue='species')
# plt.show()
#히스토그램
# sns.histplot(data=df, x='sepal_length', kde=True, hue='species')
#jointplot
# sns.jointplot(x='petal_length', y='petal_width',data=df, kind='kde')
# plt.show()

x = df.iloc[:,0:4]
y = df.iloc[:,4]
print(x.isnull().sum())
y_encode = pd.get_dummies(y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#모델 설정
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

#환경설정(컴파일)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#학습
history = model.fit(x, y_encode, epochs=50, batch_size=5)

#plt.plot(history.history['loss])
# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()