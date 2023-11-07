# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:13:51 2023

@author: hama
"""

import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import numpy as np
accuracy_scores = []

digits = datasets.load_digits()
#plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

x_train,x_test,y_train,y_test = train_test_split(data, digits.target, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

for i in range(1, 101):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    #테스트 데이터로 예측해본다.
    y_pred = knn.predict(x_test)
    #정확도를 계산한다.
    scores = metrics.accuracy_score(y_test,y_pred)
    accuracy_scores.append(scores)


plt.plot(range(1, 101), accuracy_scores, 'r')
plt.ylabel('Accuracy')
plt.xlabel('number of neighbors (k)')
plt.tight_layout()
plt.show()

print("The best accuracy was with ",
      max(accuracy_scores),
      np.argmax(accuracy_scores)+1)

#이미지를 출력하기 위하여 평탄화된 이미지를 다시 8x8 형상으로 만든다.
#plt.imshow(x_test[10].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
#y_pred = knn.predict([x_test[10]]) #입력은 항상 2차원 행렬이어야 한다.
#print("예측값은 %s이고 실제값은 %s입니다." % (y_pred, y_test[10]))

#print(f"{metrics.classification_report(y_test, y_pred)}\n") #혼동 행렬