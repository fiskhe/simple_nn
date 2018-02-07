
# Alrighty
print('working')

import numpy as np

# input
x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# output
y = np.array([[0,0,2,2]]).T

np.random.seed(1)

rand = np.random.random((3,1))

# weights
syn0 = 2*np.random.random((3,1)) - 1

# sigmoid as activation function, I believe
def nonlin(var, deriv = False):
    if(deriv == True):
        return var*(1-var)
    return 1/(1+np.exp(-var))

l1 = None
for i in range(300):

    # 0th layer is input layer
    l0 = x
    l1 = nonlin(np.dot(l0,syn0))

    l1_err = y - l1

    l1_delta = l1_err * nonlin(l1, True)
    # print(l1_delta)

    syn0 += np.dot(l0.T, l1_delta)

print('ootpoot')
print(l1)
print(syn0)
