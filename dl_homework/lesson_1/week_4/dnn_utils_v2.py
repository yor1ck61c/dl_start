import numpy as np
 
# sigmoid激活函数
def sigmoid(Z):
    """
    param :
    Z: 任意维度的numpy数组
    return:
    A -- sigmoid(Z)的输出，与Z的维度相同
    cache -- 返回Z，存储起来在反向传播时使用
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
 
    return A,cache
 
# ReLU激活函数
def relu(Z):
    """
    param :
     Z -- 任意维度,线性层的输出
    return:
    A -- 激活后的结果, 与Z的维度相同
    cache -- python字典包含"A" ,存储起来以便高效执行反向传播
    """
    A = np.maximum(0,Z)
 
    # Python中的assert断言用起来非常简单，可在assert后面跟上任意判断条件
    # 如果断言失败则会抛出异常
    assert (A.shape == Z.shape)
    cache = Z
 
    return A,cache
 
# 利用ReLU单元计算反向传播
def relu_backward(dA,cache):
    """
    param :
    dA -- 任意维度,损失函数对A的梯度
    cache -- Z，存储起来在反向传播时使用
    return:
    dZ -- 损失函数对Z的梯度 = 损失函数对A的梯度 * relu激活函数对Z的梯度
    """
    Z = cache
    dZ = np.array(dA,copy=True)     #relu激活函数在Z>0时梯度为1
 
    #当Z<=0时，将dZ设置为0
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
 
    return dZ
 
# 利用sigmoid单元计算反向传播
def sigmoid_backward(dA,cache):
    """
    param :
    dA -- 任意维度,损失函数对A的梯度
    cache --  Z，存储起来在反向传播时使用
    return:
    dZ -- 损失函数对Z的梯度 = 损失函数对A的梯度 * sigmoid激活函数对Z的梯度
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
 
    dZ = dA * s * (1 - s)
 
    assert (dZ.shape == Z.shape)
    return dZ