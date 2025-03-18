"""
二项式拟合介绍：




"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 4)

# a = 2 ,b = 3 ,c = 4
y = 2 * x ** 2 + 3 * x + 4
plt.plot(x, y, 'ro')
plt.show()

# 拟合出二元多项式,deg代表多项式的最高次幂
(a, b, c) = np.polyfit(x, y, 2)
print(a, b, c)  # 越接近原值,代表越准
