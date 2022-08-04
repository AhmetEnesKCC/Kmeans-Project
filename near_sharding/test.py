import numpy as np

arr = np.array([[2, 3], [4, 5]])

arr2 = arr[1]

arr[1] = [7, 8]

arr2 = [6, 7]


print(arr)
