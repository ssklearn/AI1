# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = datasets.load_iris()
# a = iris.data
# print(iris.data)

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

print(x_train.shape)
print(y_test.shape)

accuracy_data = []
for i in range(2, 100) :
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores = metrics.accuracy_score(y_test, y_pred)
    accuracy_data.append(scores)
print(accuracy_data)
print(max(accuracy_data))
#그래프 시각화(k값 기반 정확도 그래프)
import matplotlib.pyplot as plt
import numpy as np
plt.plot(range(2,100),accuracy_data)
plt.ylabel('Accuracy')
plt.xlabel('number of Neighbors (K)')
plt.tight_layout()
plt.show()

#최대 정확도 값 및 위치번호(인덱스)출력
print("The best accuracy was with", 
      max(accuracy_data), "with k =", 
      np.argmax(accuracy_data)+1)

classes = {0:'setosa',1:'versicolor',2:'virginica'}

# 전혀 보지 못한 새로운 데이터를 제시해보자.
x_new=[[3,4,5,2],[5,4,2,2]]
y_predict = knn.predict(x_new)
print(classes[y_predict[0]])
print(classes[y_predict[1]])