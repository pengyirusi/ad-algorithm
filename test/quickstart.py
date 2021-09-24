
a = [[1, 2, 3],
     [3, 4, 5]]

import numpy as np

a = np.array(a)

print(a[..., 0:3:2])
print(a[:, 0:3:2])
