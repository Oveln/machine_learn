#梯度下降法求函数f(x)=x**2+x的最小值
#求函数f(x,y)的偏导 2x+1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def targetFun(x):
    return  x**2+x
def gradientFun(x):
    return 2*x+1
listx=[]
def gradientDescent(guessX,learningRate=0.1,epsilon=0.000001):
    newGuess=targetFun(guessX)
    grad= gradientFun(guessX)
    newX=guessX-learningRate * grad  # 沿梯度方向下降
    newResult=targetFun(newX)
    subResult = np.abs(newResult - newGuess)
    while subResult>=epsilon:
        guessX = newX
        newguessX = newResult
        listx.append(guessX)
        newX = newX - learningRate * gradientFun(newX)
        newResult = targetFun(newX)
        subResult = np.abs(newguessX - newResult)
    return guessX


if __name__ == '__main__':
    print(gradientDescent(2))
    x = np.arange(-10, 11, 1)
    y = x ** 2 + x
    plt.plot(x, y)
    plt.grid(linestyle='--')
    plt.scatter(np.array(listx), np.array(listx) ** 2, s=20)
    plt.show()
    print(listx)



