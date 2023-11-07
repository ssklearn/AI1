# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:48:07 2023

@author: hama
"""

#환경준비
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#데이터 불러오기 및 준비
data_set = np.loadtxt("D:\AI middle class\data\ThoraricSurgery3.csv", delimiter=",")
# D:\AI middle class\data\ThoraricSurgery3.csv
#data_set = pd.read_csv("D:\AI middle class\data\ThoraricSurgery3.csv", header=None)

x = data_set[:,:16]
y = data_set[:,16]

#데이터 전처리
#print(x.isnull().sum())
#new_dataset = data_set.dropna()
#new_dataset = data_set.fillna(0)
#new_dataset = data_set.fillna(data_set.mean())
#new_dataset = data_set.fillna(method='ffill')
#new_dataset = data_set.fillna(method='bfill')
#new_dataset = data_set.interpolate() #선형 보간법

# 모델 생성
model = Sequential()
model.add(Dense(30, input_dim=16, activation='relu', name='input_hidden_1'))
model.add(Dense(60, activation='relu', name='input_hidden_2'))
model.add(Dense(20, activation='relu', name='input_hidden_3'))
model.add(Dense(1, activation='sigmoid', name='output'))
model.summary()

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
history = model.fit(x, y, epochs=10, batch_size=16)

# 학습 과정 시각화
#plt.figure(figsize=(12, 4))

# Loss 그래프
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'])
# plt.title('Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')

# # Accuracy 그래프
# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'])
# plt.title('Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')

# plt.show()

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