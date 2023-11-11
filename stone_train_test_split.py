# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:55:24 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:11:00 2023

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('D:/s_night/data/sonar3.csv', header=None)
# print(df.head(10))
# print(df.info())
# print(df.describe())
# print(df[60].value_counts())

# plt.figure(figsize=(12, 12))
# sns.heatmap(df.corr(), vmax=0.7, cmap='coolwarm', linewidths=0.5)
# 46, 47번 컬럼 간의 히스토그램 그래프
# plt.hist(x=[df[47], df[46]], bins=50, histtype='stepfilled')

# 45번과 47번 컬럼에 대한 상관관계 계산
# correlation = df[[46, 47]].corr()

# 상관관계 히트맵 그리기
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation, annot=True, cmap='coolwarm', square=True)
# plt.legend()
# plt.tight_layout()

#산점도
# sns.pairplot(df[[46, 47, 60]], hue=60, plot_kws={'alpha':0.3})
# plt.show()

x = df.iloc[:, 0:60]
y = df.iloc[:,60]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
# print(y_train.value_counts())
# print(x_train[50].value_counts())
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim=60, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x, y, epochs=50, batch_size=16)

loss, accuracy = model.evaluate(x_test, y_test)
print('test test:', loss)
print('test accuracy:', accuracy)
# plt.plot(history.history['loss'])
# plt.plot(score)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()