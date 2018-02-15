import pandas as pd
import matplotlib.pyplot as plt

epoch_interval = 500



############# ADAGRAD ###############
file_name = 'adagrad_out.csv'

data = pd.read_csv(file_name)
train_acc = data['Training Accuracy']
train_err = data['Training Error']
test_acc = data['Testing Accuracy']
test_err = data['Testing Error']
time = data['Time Taken']

line = plt.plot(time, test_acc)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('Adagrad Testing Accuracy Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Testing Accuracy')
# plt.show()
plt.savefig('graphs/ada_test_acc.png')
plt.close()

line = plt.plot(time, test_err)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('Adagrad Testing Error Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Testing Error')
# plt.show()
plt.savefig('graphs/ada_test_err.png')
plt.close()

line = plt.plot(time, train_acc)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('Adagrad Training Accuracy Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Training Accuracy')
# plt.show()
plt.savefig('graphs/ada_train_acc.png')
plt.close()

line = plt.plot(time, train_err)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('Adagrad Training Error Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Testing Accuracy')
# plt.show()
plt.savefig('graphs/ada_train_err.png')
plt.close()


############# RMSPROP ###############

file_name = 'rmsprop_out.csv'

data = pd.read_csv(file_name)
train_acc = data['Training Accuracy']
train_err = data['Training Error']
test_acc = data['Testing Accuracy']
test_err = data['Testing Error']
time = data['Time Taken']

line = plt.plot(time, test_acc)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('RMSprop Test Accuracy Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Testing Accuracy')
# plt.show()
plt.savefig('graphs/rms_test_acc.png')
plt.close()

line = plt.plot(time, test_err)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('RMSprop Testing Error Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Testing Error')
# plt.show()
plt.savefig('graphs/rms_test_err.png')
plt.close()

line = plt.plot(time, train_acc)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('RMSprop Training Accuracy Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Training Accuracy')
# plt.show()
plt.savefig('graphs/rms_train_acc.png')
plt.close()

line = plt.plot(time, train_err)
plt.setp(line, color = 'k', linestyle = '-', linewidth = '2.0', marker = '.', mec = 'b')
plt.title('RMSprop Training Error Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Training Error')
# plt.show()
plt.savefig('graphs/rms_train_err.png')
plt.close()


