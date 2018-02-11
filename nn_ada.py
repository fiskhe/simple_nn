import pandas as pd
import numpy as np

dataframe = pd.read_csv('irises_processed.csv')

data = dataframe['Data']
iris_type = dataframe['Class']

learn_rate = 0.01
epsilon = 1e-8
num_epochs = 1000
sum_grad0 = 0
sum_grad1 = 0

# Ensure "randomness" is a controlled variable
np.random.seed(1)

# Initialize weights with mean of zero
weights0 = 2*np.random.random((4,4)) - 1
weights1 = 2*np.random.random((4,3)) - 1

# Activation function
def sigmoid(x, deriv = False):
    if deriv:
        return x*(1-x)
    else:
        return 1/(1 + np.exp(x))

def adagrad(lr, sum_g, epsilon):
    return np.divide(lr, np.sqrt(sum_g + epsilon))

for epoch in range(num_epochs):
    
    total_err = np.array([[0.,0.,0.]])
    
    for ind in range(120):
        layer0 = data[ind].split('|')
        layer0 = [float(x) for x in layer0]
        layer0 = np.array([layer0])
        true_output = iris_type[ind].split('|')
        true_output = [float(x) for x in true_output]
        true_output = np.array([true_output])

        layer1 = sigmoid(np.dot(layer0, weights0)) # 1 x 4 matrix
        layer2 = sigmoid(np.dot(layer1, weights1)) # 1 x 3 matrix
        # layer 2 is the output

        lay2_err = true_output - layer2
        lay1_err = np.dot(lay2_err, weights1.T)
        
        total_err += lay2_err
        
        # Use of the backpropagation algorithm
        w1_delt = lay2_err * sigmoid(layer2, True)
        w0_delt = lay1_err * sigmoid(layer1, True)
        # backprop continued
        w1_grad = np.dot(layer1.T, w1_delt)
        w0_grad = np.dot(layer0.T, w0_delt)
        
        sum_grad0 += np.square(w0_grad)
        sum_grad1 += np.square(w1_grad)

        weights0 -= adagrad(learn_rate, sum_grad0, epsilon) * w0_grad
        weights1 -= adagrad(learn_rate, sum_grad1, epsilon) * w1_grad

    # Print out total error over epoch every 100 epochs
    if epoch % 100 == 0:
        print(total_err)
        
        
