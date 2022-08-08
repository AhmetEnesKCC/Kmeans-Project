import numpy as np

arr = np.array([[1, 2, 3], [4, 5 , 6], [7, 8, 9]])

mean = np.sum(arr, axis = 0) / 3
print(mean)