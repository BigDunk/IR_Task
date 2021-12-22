import numpy as np

nums = [1, 2, 3, 4]
li = np.array(nums)
print(li)
# out = np.sort(li)
# print(out)
out = np.argsort(-li)
print(out)
