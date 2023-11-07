
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import copy

#데이터를 불러 옵니다.
df = pd.read_csv("/content/drive/MyDrive/house_train.csv")

#데이터를 미리 살펴 보겠습니다. 
print(df)
print(df.dtypes)
#데이터 전체보기
pd.set_option('display.max_rows', None)

print(df.isnull().sum().sort_values(ascending=False).head(50))
#원핫인코딩
pre_df = pd.get_dummies(df)

#원핫인코딩 값이 True, False로 보일경우
# pre_df = pd.get_dummies(df, dtype=int)

#전처리(빈값 채우기)
fill_pre_df = pre_df.fillna(df.mean())

copy_df = copy.deepcopy(df) #깊은 복사(deepcopy)

dtypes_s = df.isnull().sum().sort_values(ascending=False).head(50)

#결측치 컬럼 제거 코드 *****
count = 0
for i in dtypes_s.index :
    if count < 19 :
        copy_df.drop(i, axis=1, inplace=True)
        count += 1
print(copy_df.isnull().sum().sort_values(ascending=False).head(50))

#컬럼명 변경 방법
df.rename(columns={"HouseStyle" : 'hs'})

#데이터 사이의 상관 관계를 저장합니다.
df_corr=df.corr()

#집 값과 관련이 큰 것부터 순서대로 저장합니다.
df_corr_sort=df_corr.sort_values('SalePrice', ascending=False)

#집 값과 관련도가 가장 큰 10개의 속성들을 출력합니다. 
df_corr_sort['SalePrice'].head(10)

#집 값과 관련도가 가장 높은 속성들을 추출해서 상관도 그래프를 그려봅니다.
cols=['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
sns.pairplot(df[cols])
plt.show()

pd.set_option('display.max_columns', None)
print(df.describe())

def findOutliers(x, column) :
    q1 = x[column].quantile(0.25)
    q3 = x[column].quantile(0.75)
    
    iqr = 1.5 * (q3-q1)
    
    y = x[(x[column] > (q3 + iqr)) | (x[column] < (q1 - iqr))]
    
    return len(y)

print(findOutliers(pre_df, '2ndFlrSF'))

plt.figure(figsize=(10, 7))

plt.subplot(2, 3, 1)
df[['SalePrice']].boxplot()


plt.subplot(2, 3, 2)
df[['OverallQual']].boxplot()


plt.subplot(2, 3, 3)
df[['GrLivArea']].boxplot()


plt.subplot(2, 3, 4)
df[['GarageCars']].boxplot()


plt.subplot(2, 3, 5)
df[['GarageArea']].boxplot()


plt.subplot(2, 3, 6)
df[['2ndFlrSF']].boxplot()
plt.tight_layout()
plt.show()

#집 값을 제외한 나머지 열을 저장합니다. 
cols_train=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
x_train_pre = df[cols_train]

#집 값을 저장합니다.
y = df['SalePrice'].values

x_train, x_test, y_train, y_test = train_test_split(x_train_pre, y, test_size=0.2)

model = Sequential()
model.add(Dense(10, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

modelpath = "./data/model/Ch15-house.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

history = model.fit(x_train, y_train, validation_split=0.25, epochs=2000, batch_size=32, callbacks=[early_stopping_callback,checkpointer])

real_prices = []
pred_prices = []
X_num = []

n_iter = 0
Y_prediction = model.predict(x_test).flatten()
for i in range(25):
    real = y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.2f}, 예상가격: {:.2f}".format(real, prediction))
    real_prices.append(real)
    pred_prices.append(prediction)
    n_iter = n_iter + 1
    X_num.append(n_iter)