import sys
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz


#load the .env variables
load_dotenv()



#If - Else to control whre we load data from: URL or local file

#Local filepath to use
my_drug_csv_path = "C:\Programming Projects\Machine Learning\IBM Ai Engineering\Machine Learning with Python\Support_Files\drug200.csv"

print("\n")
print("\n")

if os.path.exists(my_drug_csv_path):
    df = pd.read_csv(my_drug_csv_path, delimiter=",")
else:
    #URL filepath to use
    #use os.environ or os.getenv to access environment variables as if they are from ACTUAL enviornment and not .env file
    print(os.getenv("DEC_TREE_DATA_URL"))
    DEC_TREE_DATA_URL = os.getenv("DEC_TREE_DATA_URL")
    df = pd.read_csv(DEC_TREE_DATA_URL, delimiter=",")

#Get the Size of the Dataset
print("The Dataset Shape and Size (Rows, Columns):", df.shape)
print("\n")
print("\n")

print(df.dtypes)
print("\n")
print("\n")

#Print the top lines of the CSV file we turned into a data frame to confrim its loaded

print(df.head())
print("\n")
print("\n")

#Use the dataframe function 'describe()' to give us a quick desciptive summary of the data

print(df.describe())
print("\n")
print("\n")


# y is our target column and is categorical data
# make x our features of our data - excluding the target column and just the values
x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(x[0:5])
print("\n")
print("\n")


"""
Some of our feature variables are categorical - BP, Sex.
SkLearn Decision Trees cannot handle categorical data. However, there is a function that we can use to assing numerical values to our categorical data.

We will use LabelEncoder() to convert categorical variables to numerical
"""

enc = preprocessing.LabelEncoder()

# Pass in the 'Sex' category values to the encoder
enc.fit(['F','M'])

# Transform our 'Sex' column in our dataset - column is Index 1
# Syntax of array = [Start_Row : End_Row , Start_Column : End_Column]
x[:,1] = enc.transform(x[:,1])

print(x[0:5])
print("\n")
print("\n")


# Encode our BP column
enc.fit(['LOW', 'NORMAL', 'HIGH'])

# Transform our BP data
x[:,2] = enc.transform(x[:,2])

print(x[0:5])
print("\n")
print("\n")


# Encode our Cholestrol column
enc.fit(['NORMAL', 'HIGH'])

# Transform our Cholestrol data
x[:,3] = enc.transform(x[:,3])

print(x[0:5])
print("\n")
print("\n")


# create our y variable with our target column
y = df["Drug"]
print(y[0:5])
print("\n")
print("\n")

"""
We will use the train/test split approach with our dataset
We will use the train_test_split function of the sklearn.model_selection library

train_test_split function returns four variables: x_train, x_test, y_train, y_test
It takes the params: x,y, test_size, random_state
- x and y are the arrays before the splitting of our dataset
- test_size = the ratio of the data we want in our test set
- random_state = random seed that allows us to reproduce our results
"""
X_trainset, X_testset, y_trainset, y_testset = train_test_split(x, y, test_size=0.3, random_state=3)

# Print out the shape of our train and test sets
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))
print("\n")
print("\n")


"""
///////////////////////////////////////////////////////////////
                  Modeling - DecisionTreeClassifier
///////////////////////////////////////////////////////////////
"""

# Create an instance of the DecisionTreeClassifier
# criterion = "entropy" - see information gain of each node
TreeClass = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

# Print our classifer default params
print(TreeClass)
print("\n")
print("\n")

# Fit our model using our training splits
TreeClass.fit(X_trainset,y_trainset)

"""
/*//////////////////////////////////////////////////////////////
                        Predicting With Model
//////////////////////////////////////////////////////////////*/
"""

# Make some predictions using test set and store results in variable

test_pred_results = TreeClass.predict(X_testset)

# We can print out the predicted Y values and the actual Y values for a quick comparison
print("Predicted Y Values: ")
print(test_pred_results[0:5])
print("\n")
print("\n")
print("Actual Y Values: ")
print(y_testset[0:5])
print("\n")
print("\n")



"""
**Accuracy classification score** computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.

In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly matches with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0
"""

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, test_pred_results))


# Create a diagram of our decision tree
export_graphviz(TreeClass, out_file='tree.dot', filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])