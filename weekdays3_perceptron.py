# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:03:49 2023

@author: hama
"""

import numpy as np
#가중치 설정
w1 = np.array([-2, -2])
w2 = np.array([2, 2])
w3 = np.array([1, 1])

#bias 설정
b1 = 3
b2 = -1
b3 = -1

def MLP(x, w, b):
    #가중합
    y = np.sum(x * w) + b
    #활성화 함수
    if y <= 0 :
        return 0
    else :
        return 1
    
#NAND 게이트
def NAND(x1, x2):
    return MLP(np.array([x1, x2]), w1, b1)

#OR 게이트
def OR(x1, x2):
    return MLP(np.array([x1, x2]), w2, b2)

#AND 게이트
def AND(x1, x2):
    return MLP(np.array([x1, x2]), w3, b3)

#XOR 게이트
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    y = XOR(x[0], x[1])
    print(f"입력 값 : {x}, 출력 값 : {y}")