# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:53:14 2023

@author: hama
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:\AI middle class\data\pima-indians-diabetes3.csv')
print(df.head())

# print(df["diabetes"].value_counts())
# #print(df.info())
# #print(df.isnull().sum())
# #print(df.describe())
# print(df.corr())
# colormap = plt.cm.autumn
# plt.figure(figsize=(12, 12))
# sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap, linecolor='white', annot=True)
# plt.show()
# plt.hist(x=[df.plasma[df.diabetes==0], df.plasma[df.diabetes==1]], bins=30, histtype='barstacked', label=['normal','diabetes'])
# plt.legend()
#산점도
# plt.scatter(df.plasma.values, df.bmi.values, alpha=0.5)
# plt.title('scatter')
# plt.xlabel('plasma')
# plt.ylabel('bmi')
# plt.show()
#-------------------------------------------

#데이터 분류(x, y)
x = df.iloc[:,:8]
y = df.iloc[:,8]

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 1. 모델 객체 생성
model = Sequential()

# 2. 모델에 Dense 층 추가
# 2-1. 첫 번째 Dense 층
model.add(Dense(12, input_dim=8, activation='relu', name='input_1'))

# 2-2. 두 번째 Dense 층
model.add(Dense(8, activation='relu', name='output_1'))

# 2-3. 세 번째 Dense 층
model.add(Dense(1, activation='sigmoid', name='output_2'))
print(model.summary())

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
history = model.fit(x, y, epochs=10, batch_size=16)

# 그래프 크기 설정
plt.figure(figsize=(12, 4))

# Loss 그래프
plt.subplot(1, 2, 1)
sns.lineplot(x=range(1, len(history.history['loss']) + 1), y=history.history['loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Accuracy 그래프
plt.subplot(1, 2, 2)
sns.lineplot(x=range(1, len(history.history['accuracy']) + 1), y=history.history['accuracy'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()