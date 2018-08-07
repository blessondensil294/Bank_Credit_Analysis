# -*- coding: utf-8 -*-
#"""
#Created on Wed Jul 25 22:11:29 2018

#@author: BLAZIN
#"""

#Home Credit Group
#Home Credit Default Risk

#Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by 
#untrustworthy lenders.

#Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure 
#this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional 
#information--to predict their clients' repayment abilities.

#While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them 
#unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, 
#maturity, and repayment calendar that will empower their clients to be successful.

#TARGET indicating 0: the loan was repaid or 1: the loan was not repaid.

#importing the data
#NUmpy and Panda as Data Manipulation
import numpy as np
import pandas as pd

#sklearn for categorical variables
from sklearn.preprocessing import LabelEncoder

#File system Management
import os

#supress the warnings
import warnings
warnings.filterwarnings('ignore')

#ploting
import matplotlib.pyplot as plt
import seaborn as sns

print(os.listdir("E:\Kaggle\Home Credit Kaggle\Data"))

#training data
app_train = pd.read_csv('E:\\Kaggle\\Home Credit Kaggle\\Data\\application_train.csv')
print("Training Data Dimension: ",app_train.shape)
app_train.head()

#testing data
app_test = pd.read_csv('E:\\Kaggle\\Home Credit Kaggle\\Data\\application_test.csv')
print("Test Data Dimension: ", app_test.shape)
app_test.head()

#Exploratory Data Analysis

#Examin the Target Column
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist()

#Function to find the missing values based on the columns
def missing_values_table(df):
    #total missing values in the columns
    mis_val = df.isnull().sum()
    
    #percentage of missing values
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    
    #make table with results and store row wise
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    #Rename the columns 
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    #Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending = False).round(1)
    
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
        
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

#Missing values for the dataset
missing_values_train = missing_values_table(app_train)
missing_values_train.head(20)

# Number of each type of column
app_train.dtypes.value_counts()