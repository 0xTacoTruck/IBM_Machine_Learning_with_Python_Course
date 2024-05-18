import sys
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from dotenv import load_dotenv
import os


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



