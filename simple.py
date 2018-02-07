
# Alrighty
print('working')

import numpy as np

# input
x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# output
y = np.array([[0,1,1,0]]).T

np.random.seed(1)

rand = np.random.random((3,1))

# weights
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

# sigmoid as activation function, I believe
def nonlin(var, deriv = False):
    if(deriv == True):
        return var*(1-var)
    return 1/(1+np.exp(-var))

for i in range(60000):
    # 0th layer is input layer
    l0 = x
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    
    l2_err = y - l2
    l2_delta = l2_err * nonlin(l2, True)
    
    print(syn1)
    print(l2_delta)
    l1_err = l2_delta.dot(syn1.T)
    l1_delta = l1_err * nonlin(l1, True)
    quit()
    # print(l1_delta)
    # print('')

    syn0 += np.dot(l0.T, l1_delta)
    syn1 += np.dot(l1.T, l2_delta)

print('ootpoot')
print(l2)
