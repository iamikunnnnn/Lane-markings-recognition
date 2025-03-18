import numpy as np

a = np.array([0, 0, 1, 0])
b = np.array([0, 1, 3, 0])



print(a & b)
print(a | b)
"""
####与运算:---------(&)

在python和numpy中,
元素类型为布尔值或者可以隐式转换为布尔值,
整数:  非0  视为True ;  0  视为False,
按照元素逐个进行逻辑与的规则.

##逻辑与的规则:
只有两个操作数都为True,结果为Ture,
只要两个操作数中有一个为False,结果为False




####逻辑或---------(|)
整数:  非0  视为true ;  0  视为False,
按照元素逐个进行逻辑与的规则.

##逻辑或的规则:
只有当两个操作数都是False时结果是False
只要两个操作数中有一个为True,结果为True,

"""