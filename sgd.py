
import iris

quit()

# Alrighty
print('working')

import numpy as np

# input
x = np.array([[[0,0,1],[0,1,1],[1,0,1],[1,1,1]],
                [[0,1,0],[1,1,1],[1,0,1],[0,1,1]],
                [[1,1,0],[1,0,0],[0,0,1],[0,0,1]],
                [[0,1,1],[1,1,0],[0,1,1],[1,0,1]]
])
# output
y = [np.array([[0,1,1,0]]).T,np.array([[1,0,1,1]]).T,np.array([[0,1,0,0]]).T,np.array([[1,0,1,1]]).T]
# number of times to run through data set
num_epochs = 60000
# number of datapoints in batch
batch_size = 1

num_datapts = len(x)

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

for epoch in range(num_epochs):

    for data_entry in range(len(x)):

        # 0th layer is input layer
        l0 = x[data_entry]
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        
        # Essentially the loss function
        l2_err = y[batch] - l2
        l2_delta = l2_err * nonlin(l2, True)
        
        l1_err = l2_delta.dot(syn1.T)
        l1_delta = l1_err * nonlin(l1, True)

        # param = param - learn_rate * grad
        syn0 += np.dot(l0.T, l1_delta)
        syn1 += np.dot(l1.T, l2_delta)

print('ootpoot')
print(l2)
