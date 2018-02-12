import pandas as pd
import matplotlib.pyplot as plt

epoch_interval = 500



############# ADAGRAD ###############
file_name = 'adagrad_out.csv'

data = pd.read_csv(file_name)
train_acc = data['Training Accuracy']
test_acc = data['Testing Accuracy']
time = data['Time Taken']

line = plt.plot(time, test_acc)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('Adagrad Test Accuracy')
plt.xlabel('Time (s)')
plt.ylabel('Testing Accuracy')
plt.show()


line = plt.plot(time, train_acc)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('Adagrad Training Accuracy')
plt.xlabel('Time (s)')
plt.ylabel('Training Accuracy')
plt.show()



############# RMSPROP ###############

file_name = 'rmsprop_out.csv'

data = pd.read_csv(file_name)
train_acc = data['Training Accuracy']
test_acc = data['Testing Accuracy']
time = data['Time Taken']

line = plt.plot(time, test_acc)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('RMSprop Test Accuracy')
plt.xlabel('Time (s)')
plt.ylabel('Testing Accuracy')
plt.show()


line = plt.plot(time, train_acc)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('RMSprop Training Accuracy')
plt.xlabel('Time (s)')
plt.ylabel('Training Accuracy')
plt.show()
