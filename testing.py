
import numpy as np 

grid = [[0,0,3],[0,2,0],[1,0,0]]
grid = np.array(grid)

thing = np.divide(0.001, np.sqrt(grid))

param = 3.234

print(param - thing)
