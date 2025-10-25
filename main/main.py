# Workflow: Diabetes data -> preprocess(standradize) -> split data(train and test)
# -> the model(we will use SVM(support vector machine)) is trained otn the data 
# -> the tested with new unseen data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Standardization centers all data around 0 and scales it based on its variance, 
# preventing high-magnitude features from disproportionately influencing the model's training process.
from sklearn import svm # the model works on something called, maximizing the margin between 2 classes
# in our class those classes will be either diabetic or non diabetic
from sklearn.metrics import accuracy_score

import os
print("Pwd", os.getcwd()) # to show the currert working directory

# now lets bring in the dataset, 
print("Initializing data")
diabetes_data = pd.read_csv("diabetes.csv")
# it has features like: pregnancis, Gluccose, BP, skinThicknesn, insulin, BMI, diabetesPedigreeFunction, Age
# and the result is in: Outcome, 0: non-diabetic - 1: diabetic
# right now the data has no null values

# the size of dataset is,
print(f"dataset contains: {diabetes_data.shape}")
print(f"done with data initialization \n Starting data pre processing")

# Now the values in this data is having a very vast range of values, we are gonna standardise to birng it as close as possible to 0
# now before that lets divide our data for features and label
X = diabetes_data.drop(columns="Outcome", axis=1)
Y = diabetes_data["Outcome"]

# now lets split data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=42)

# lets bring in the class that we will use to standardise,
scaler = StandardScaler()
# we created an object scaler, which has mean and standarsh deviation as its parmeters
scaler.fit(X_train)
# now we the fit function, finds out mean and std of every feature and then stores each in the parameter of scaler object
# but the data of X_train is still the same now we are gonna use transform to changes these values based on their mean and std
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# this performs a process called Z-score Normalization
# Z-score Normalization, here Z-score = (x- mean)/std, where x is values of features
# note: NEVER PERFORM THIS WITH THE TEST DATA, YOU WILL CAUSE DATA LEAKAGE..

print(f"Data preprocessing over \n starting model training")
# the model we are going to use is svm 
classifier = svm.SVC(kernel="linear") # a linear model

# Now we will train it with X_train and Y_train, which will be fit into classifier's parameters
classifier.fit(X_train, Y_train)
# here we have to send both X and Y, and the amount of time needed to train is proportional to size of dataset
# here the values are of X and Y are being stored as parameters, which are used to find out optimal hyperplan(for this model)
# optimal hyperplan: the one that has the largest margin—the greatest distance to the nearest training data point of any class.
# The SVM algorithm doesn't use all the training data to define this boundary. Instead, it relies on a small subset of critical data points: the Support Vectors.
# It Stores the Hyperplane Parameters: It calculates and stores the mathematical definition of the hyperplane itself (its weight vector and bias term).

print(f"Model training is over \n checking model accuracy")
X_test_prediction = classifier.predict(X_test)
# When you later call classifier.predict(X_test), the model takes a new data point from X_text and determines which side of the learned hyperplane it falls on
accuracy = accuracy_score(X_test_prediction, Y_test) # this will find out the accuracy score by comparing the predicted data with labelled data 
print(f"The accuracy score is: {accuracy}")
while True:
    Ch = input(f"\n Do you have any personal data to test? (y/n)").lower().strip()
    if Ch == 'n':
        print("Tanks for watching")
        break
    elif Ch == 'y':
        print(f" \t The is a basic accuray test for the model \t")
        data = []
        num = int(input("Enter the number of times you got pregnant: "))
        data.append(num)
        num = int(input("Enter the amount of Glucose in your blood: "))
        data.append(num)
        num = int(input("Enter you blood pressure: "))
        data.append(num)
        num = int(input("Enter your skin Thickness: "))
        data.append(num)
        num = float(input("Enter your the amount of insuline: "))
        data.append(num)
        num = float(input("Enter your IBM: "))
        data.append(num)
        num = float(input("Enter your Diabetes Pedigree Function: "))
        data.append(num)
        num = int(input("Enter your age: "))
        data.append(num)
        # now all the input data is in a list, we will convert it to an array
        data = np.array(data)
        # but now this is an 1D array, want an 2D 
        data = data.reshape(1,-1)
        # now this is an 2D array with (1,8)

        #Now we have to standardise the data
        data = scaler.transform(data)
        # since this is same as the range, X had so we wont need to do fit again
        prediction = classifier.predict(data)
        if prediction[0] ==1:
            print("Congratulation you have diabetes")
        elif prediction[0] == 1:
            print("Congratulations you dont have diabetes")
    else:
        print("Nigga you fumbled yes or no question ??😭😭")