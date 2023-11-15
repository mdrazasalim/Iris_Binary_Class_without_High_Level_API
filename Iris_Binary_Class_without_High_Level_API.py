""" ----------Binary Classification without using  High Level API (like Tensorflow, Keasr, Sklearn) ---------------------"""

# import required library
import numpy as np
import pandas as pd

#%%
# Load Iris data set [link: https://www.kaggle.com/datasets/saurabh00007/iriscsv/]

iris_data = pd.read_csv('C:\\Users\\Salim Raza\\Desktop\\UTEP Research\\Code\\Iris Data Classification\\Iris.csv')
pd.set_option('display.max_columns', None) # Set display options to show all columns and rows
pd.set_option('display.max_rows', None) # Set display options to show all columns and rows 
print('Raw Iris Data:');
print(iris_data)

#%% 
# Preprocessing dataset to fit them with the model (1)

species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2} # A mapping for 'Species' labels
iris_data['Species'] = iris_data['Species'].map(species_mapping)

print('Iris Data after lebeling of classes:');
print(iris_data)

#%% 
# Preprocessing dataset to fit them with the model (2)

iris_data = iris_data[iris_data['Species'] != 2] # Keep only two classes (0 and 1)
#[NOTE: For binary classification, only two classes (0, and 1) have been fitered in Iris Data]

print('Iris Data with two clases [0, 1]:');
print(iris_data)

#%%
# Preprocessing dataset to fit them with the model (3)

iris_data = iris_data.sample(frac=1, random_state=42).reset_index(drop=True) # To shuffle the DataFrame

X= iris_data.drop(columns=['Species', 'Id']) #Fatures separatation 
y = iris_data['Species'] # Target separatation 

print('Input Fatures of Iris Data:');
print(X)
print('Output Target of Iris Data:');
print(y)

#%%
# Preprocessing dataset to fit them with the model (4)
# -------Split the data into training and testing sets

test_size = 0.2 # 20% of Total sample data
split_index = int(len(iris_data) * (1 - test_size))


X_train = X.iloc[:split_index, :] 
X_test = X.iloc[split_index:, :]
y_train= y.iloc[:split_index]
y_test= y.iloc[split_index:]

print('X_train:')
print(X_train)
print('y_train:')
print(y_train)
print('X_test:')
print(X_test)
print('y_test:')
print(y_test)

#%%
#Add a bias to the input feature of Iris data

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test] 

print('X_train with Bias:')
print(X_train)
print('X_test with Bias:')
print(X_test)

#%%
# Initialize weights

w = np.zeros(X_train.shape[1]) # Weights are intialized based on input feature and single neuron

#%%
# Define activation function (sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#%%
# Set hyperparameters
learning_rate = 0.01
epochs = 100

#%%
# Training Process regarding single neuron based ANN (step 01 to step 03)
for epoch in range(epochs):
    # Step 01: Forward Propagation 
    predictions = sigmoid(np.dot(X_train, w))
   
    
   # Step 02: Back Propagation
    gradient = np.dot(X_train.T, y_train - predictions) # [cross-entropyloss and Gradient Descent are implicitly used] 
    
    #[NOTE: The above gradient equation is found by the two operations
            #1. Log loss, aka logistic loss or cross-entropyloss is calculated
            #2. Then, find out gradient of the loss function
            
            # [Link to Clarify the Loss concept: https://www.youtube.com/watch?v=ar8mUO3d05w]
            # [Link to Clarify the gradient concept: https://www.youtube.com/watch?v=0VMK18nphpg]
            
        
    # Step 03: Update weights
    w = w+ learning_rate * gradient


#%% 
# Accuracy Measurement on Training Data

predictions = sigmoid(np.dot(X_train, w))
y_train_pred = (predictions >= 0.5).astype(int) # The predicted probability >= 0.5, it will be converted to 1; otherwise, 0;

correct_predictions = np.sum(y_train == y_train_pred)
total_samples = len(y_train)
 
train_data_accuracy= correct_predictions / total_samples

print('Accuracy score on Train Data:')
print (train_data_accuracy)

#%% 
# Accuracy Measurement on Testing Data

predictions = sigmoid(np.dot(X_test, w))
y_test_pred = (predictions >= 0.5).astype(int) # The predicted probability >= 0.5, it will be converted to 1; otherwise, 0;

correct_predictions = np.sum(y_test == y_test_pred)
total_samples = len(y_test)
 
test_data_accuracy= correct_predictions / total_samples

print('Accuracy score on Test Data:')
print (test_data_accuracy)

""" ------------------------------------------------- End----------------------------------------------------------------"""