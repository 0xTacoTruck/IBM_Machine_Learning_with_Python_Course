from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Import the load_dotenv to allow us to use environment variables from the .env file of the project
from dotenv import load_dotenv
import os
from sklearn import model_selection
#Import the KNN classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#load the .env variables
load_dotenv()

#If - Else to control whre we load data from: URL or local file

#Local filepath to use
                    #Use double slashes to escape the special characters being detected
knn_dataset_filepath = "C:\\Programming Projects\\Machine Learning\\IBM Ai Engineering\\Machine Learning with Python\\Support_Files\\teleCust1000t.csv"

if os.path.exists(knn_dataset_filepath):
    df = pd.read_csv(knn_dataset_filepath)
else:
    #URL filepath to use
    #use os.environ or os.getenv to access environment variables as if they are from ACTUAL enviornment and not .env file
    print(os.getenv("KNN_DATASET_URL"))
    knn_dataset_url = os.getenv("KNN_DATASET_URL")
    df = pd.read_csv(knn_dataset_url)

print(df.dtypes)
print("\n")
print("\n")

#Print the top lines of the CSV file we turned into a data frame to confrim its loaded

print(df.head())
print("\n")
print("\n")



#Get a count of the different labels rom  the 'custcat' column
print("Count of unique labels in 'custcat' field: ",df['custcat'].value_counts())
"""RESULTS:
    3    281
    1    266
    4    236
    2    217
    Name: custcat, dtype: int64"""
print("\n")
print("\n")

#Print columns of dataframe/dataset
print("Dataframe Colums: ",df.columns)
print("\n")
print("\n")

# Get descriptive statistics of income column
print(df['income'].describe())
print("\n")
print("\n")

# Display histogram to show the data distribution - column is the column to be plotted, bins is the size of the interval/width of the bars in the histogram
df.hist(column='income', grid=True,bins=50)
plt.show()
"""FINDINGS:
    The histogram showed skewed RIGHT distributiuon of income amounts
"""

# To use the Scikit-learn library, we need to convert the Pandas dataframe to a Numpy array
X_values = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)

#------------------------------------------------------------------------------------------------------------------
#TESTING WITH ADJUSTED COLUMNS
#X_values = df[['region','age', 'marital', 'ed', 'income', 'employ', 'gender', 'reside']] .values

#WAS INTERSTING TO SEE HOW THE IDEAL K VALUE CHANGED BASED UPON THE COLUMNS, AS WELL AS HOW THE ACCURACY CHANGED - SUPRISINGLY, THE DIFFERENCE IN ACCURACY WAS LOWER THAN WHAT I THOUGHT AS I TESTED DIFFERENT COLUMNS
#------------------------------------------------------------------------------------------------------------------


#Print the first 5 rows to confirm the conversion and ensure the data is in the correct format
print(X_values[0:5])
print("\n")


# Get our datset's Y values (Target values)
Y_values = df['custcat'].values

#Print the first 5 rows to confirm the conversion and ensure the data is in the correct format
print(Y_values[0:5])
print("\n")


# Normalise Data
# Standardise features/columns by removing the mean and scaling to unit variance
#Data Standardization gives the data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on the distance of data points:

"""
----------------------------------------------------------------------------------
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler

fit() ---> Compute the mean and std to be used for later scaling
transform() ---> Perform standardization by centering and scaling

Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using transform.

Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data
----------------------------------------------------------------------------------
"""

X_values = preprocessing.StandardScaler().fit(X_values).transform(X_values.astype(float))

#Display the first 5 rows to confirm the conversion and ensure the data is in the correct format
print(X_values[0:5])
print("\n")


"""
SPLIT OUR DATASET UP INTO TRAIN AND TEST SETS

sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split

"""
X_train, X_test, y_train, y_test = model_selection.train_test_split( X_values, Y_values, test_size=0.2, random_state=4)

print ('Train set: ', X_train.shape,  y_train.shape)
print ('Test set: ', X_test.shape,  y_test.shape)

#Define K value - the number of neighbours
k_value = 4

"""
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn-neighbors-kneighborsclassifier
"""

#Train Model
trained_neighbours = KNeighborsClassifier(n_neighbors = k_value).fit(X_train,y_train)

print("Trained Model: ", trained_neighbours)
print("\n")
print("\n")

#Predict Y values of test set
predicted_y_values = trained_neighbours.predict(X_test)

print("Predicted Y values: ", predicted_y_values)
print("Actual Y values: ", y_test)

#VISUALISE THE PREDICTINS VS THE ACTUAL VALUES
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predicted_y_values, color='blue', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')

"""enumerate() is used to iterate over the pairs of actual and predicted values along with their indices.
plt.annotate() is called within the loop to annotate each point on the scatter plot with its corresponding index label (str(i)).
textcoords="offset points", xytext=(0,10) specifies that the annotation text should be offset by 10 points above the point being annotated for better readability."""
# Annotate every nth point with its index label ----> Change this value to adjust the frequency of index labels
n = 20
# Annotate each point with its index label
for i, (actual, predicted) in enumerate(zip(y_test, predicted_y_values)):
    if i % n == 0:
        plt.annotate(str(i), (actual, predicted), textcoords="offset points", xytext=(0,10), ha='center')
    
plt.title('Actual vs Predicted Values of y')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.legend()
plt.grid(True)
plt.show()


"""Accuracy evaluation

In multilabel classification, **accuracy classification score** is a function that computes subset accuracy. This function is equal to the jaccard_score function. Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.
"""
print("Train set Accuracy: ", metrics.accuracy_score(y_train, trained_neighbours.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, predicted_y_values))

#----------------------------------------------------------------------------------------------------------------------
"""
TO THIS POINT WE ONLY USED A HARDCODED VALUE FOR K AND WE WOULD HAVE TO MANUALLY GO BACK THROUGH AND CHANGE THIS AND RECORD THE ACCURACY FOR EACH K VALUE TO FIND THE BEST K VALUE TO USE

WE CAN DO THIS PROGRAMMATICALLY WITH A FOR LOOP
"""
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print("MEAN ACCURACY: ",mean_acc)

#Create a plot of the accuracy values WITH STANDARD DEVIATION
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

#Based on the console prinout and the visualisation, we can see the best value for K is:
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 