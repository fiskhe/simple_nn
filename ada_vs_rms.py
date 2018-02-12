import pandas as pd
import numpy as np
import time

dataframe = pd.read_csv('irises_processed.csv')

data = dataframe['Data']
iris_type = dataframe['Class']

ada_learn_rate = 0.01
rms_learn_rate = 0.001
epsilon = 1e-8
gamma = 0.9
num_epochs = 30000 # thirty-thousand
sum_grad0 = 0
sum_grad1 = 0

metrics = []

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

def adagrad(lr, sum_g, grad, eps):
    sum_g += np.square(grad)
    new_lr = np.divide(lr, np.sqrt(sum_g + eps))
    return [sum_g, new_lr]

def rmsprop(lr, sum_g, grad, eps, gam):
    sum_g = gam * sum_g + (1 - gam) * np.square(grad) 
    new_lr = np.divide(lr, np.sqrt(sum_g + eps))
    return [sum_g, new_lr]

def eval_acc(real, pred):
    accurate = False
    # The endes of the greatest value int he actual vector
    r_index = np.where(real == real.max())[1]
    # The index of the greatest value in predicted vector
    p_index = np.where(pred == pred.max())[1]
        
    if r_index == p_index:
        accurate = True
            
    return accurate
    
start_time = time.monotonic()

for epoch in range(num_epochs): 
    total_train_acc = 0
    total_test_acc = 0
    
    # Training
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

        ######## Gathering data ########
        ######## A.K.A. Metrics ########
        if eval_acc(true_output, layer2):
            total_train_acc += 1
        ################################
        
        # Use of the backpropagation algorithm
        w1_delt = lay2_err * sigmoid(layer2, True)
        w0_delt = lay1_err * sigmoid(layer1, True)
        # backprop continued
        w1_grad = np.dot(layer1.T, w1_delt)
        w0_grad = np.dot(layer0.T, w0_delt)
        
        # Using Adagrad
        # changes0 = adagrad(ada_learn_rate, sum_grad0, w0_grad, epsilon)
        # changes1 = adagrad(ada_learn_rate, sum_grad1, w1_grad, epsilon)
        
        # Using RMSprop
        changes0 = rmsprop(rms_learn_rate, sum_grad0, w0_grad, epsilon, gamma)
        changes1 = rmsprop(rms_learn_rate, sum_grad1, w1_grad, epsilon, gamma)

        sum_grad0 = changes0[0]
        sum_grad1 = changes1[0]

        new_lr0 = changes0[1]
        new_lr1 = changes1[1]

        weights0 -= new_lr0 * w0_grad
        weights1 -= new_lr1 * w1_grad

    # Testing set
    for ind in range(30):
        i = 120 + ind
        layer0 = data[i].split('|')
        layer0 = [float(x) for x in layer0]
        layer0 = np.array([layer0])
        true_output = iris_type[i].split('|')
        true_output = [float(x) for x in true_output]
        true_output = np.array([true_output])

        layer1 = sigmoid(np.dot(layer0, weights0)) # 1 x 4 matrix
        layer2 = sigmoid(np.dot(layer1, weights1)) # 1 x 3 matrix

        if eval_acc(true_output, layer2):
            total_test_acc += 1

        
    # Print out total error over epoch every 500 epochs
    if epoch % 500 == 0:
        # print('total training accurate')
        # print(total_train_acc)
        # print(total_train_acc/120)
        # print('total testing accurate')
        # print(total_test_acc)
        # print(total_test_acc/30)
        time_taken = np.round(time.monotonic() - start_time, 2) # in seconds
        metric_entry = [total_train_acc/120, total_test_acc/30, time_taken]
        metrics.append(metric_entry)


metrics_df = pd.DataFrame(metrics, columns = ['Training Accuracy', 'Testing Accuracy', 'Time Taken'])
metrics_df.to_csv('rmsprop_out.csv', index = False)
        
        
