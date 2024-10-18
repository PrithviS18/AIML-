x1 = [[1,2,3],[4,5,6],[7,8,9]]
print(x1)
import numpy as np
new_x1 = np.array(x1)
print(new_x1)
y = new_x1.ravel()
print(y)
a = np.random.randint(1,8,size = (1,8))
a.reshape(2,4)
