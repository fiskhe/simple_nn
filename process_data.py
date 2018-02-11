import pandas as pd
import numpy as np

file_name = 'irises_raw.csv'

dataframe = pd.read_csv(file_name)
sepal_length = dataframe['Sepal_Length']
sepal_width = dataframe['Sepal_Width']
petal_length = dataframe['Petal_Length']
petal_width = dataframe['Petal_Width']
iris_classes = dataframe['Class']

train_set = []
test_set = []

for ind in range(50):
    for i in range (3):
        # Put in the 4 values that determine type of iris
        data_entry = []
        sep_l = sepal_length[ind + i*50]
        sep_w = sepal_width[ind + i*50]
        pet_l = petal_length[ind + i*50]
        pet_w = petal_width[ind + i*50]

        data = str(sep_l) + '|' + str(sep_w) + '|' + str(pet_l) + '|' + str(pet_w)
        data_entry.append(data)

        iris = iris_classes[ind + i*50]
        
        if iris == 'Iris-setosa':
            data_entry.append('1|0|0')
        elif iris == 'Iris-versicolor':
            data_entry.append('0|1|0')
        elif iris == 'Iris-virginica':
            data_entry.append('0|0|1')
        else:
            print('Houston, we have a problem.')

        if ind < 40:
            train_set.append(data_entry)
        else:
            test_set.append(data_entry)
       
np.random.shuffle(train_set)
np.random.shuffle(test_set)

full_set = train_set + test_set

dataset = pd.DataFrame(full_set, columns = ['Data', 'Class'])
dataset.to_csv('irises_processed.csv', index = False)
