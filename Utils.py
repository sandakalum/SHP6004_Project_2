#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 00:21:59 2020

"""
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 



#===============================================================================
#                  Data Visualization
#===============================================================================  

def PlotDistribution(data, a, label):    
    print(np.min(data))
    print(np.max(data))
#    plt.figure(1)
    plt.subplot(a)
    plt.hist(data, bins=100,  color='c', edgecolor='k', alpha=0.65)
    plt.gca().set(title=label, ylabel='Frequency')
    
    
    
    
def plotNewDis(y):
    N, bins, patches = plt.hist(y.x, 3, ec="k")
    
    cmap = plt.get_cmap('jet')
    low = cmap(0.1, 0.4               )
    medium =cmap(0.1, 0.6)
    high = cmap(0.1, 0.8)
    
    
    for i in range(0,1):
        patches[i].set_facecolor(low)
    for i in range(1,2):
        patches[i].set_facecolor(medium)
    for i in range(2,3):
        patches[i].set_facecolor(high)
    from matplotlib.patches import Rectangle
    #create legend
    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium, high]]
    labels= ["1-3 Days","4-7 Days", "8-30 Days"]
    plt.legend(handles, labels)
    plt.xticks([])
    plt.title('Time To Discharge Interval Distribution')


#===============================================================================
#                  Load Data Functions 
#===============================================================================    

def loadData(mode='discharge'):
    #get train data   

    p = '/home/kwaygo/Documents/NUS/SPH6004/P2/data/FinalData'
    if mode=='discharge':
        x_path = os.path.join(p, 'x_discharge_train.csv')
        y_path = os.path.join(p, 'y_discharge_train.csv')    
    elif mode=='death':
         x_path = os.path.join(p, 'x_death_train.csv')
         y_path = os.path.join(p, 'y_death_train.csv')   
    x = pd.read_csv(x_path,index_col=0)    
    y = pd.read_csv(y_path,index_col=0)
    return x, y


def loadTestData(mode='discharge'):

    p = '/home/kwaygo/Documents/NUS/SPH6004/P2/data/FinalData'
    
    if mode=='discharge':
        x_path = os.path.join(p, 'x_discharge_test.csv')
        y_path = os.path.join(p, 'y_discharge_test.csv')    
    elif mode=='death':
         x_path = os.path.join(p, 'x_death_test.csv')
         y_path = os.path.join(p, 'y_death_test.csv') 
    x = pd.read_csv(x_path,index_col=0)    
    y = pd.read_csv(y_path,index_col=0)
    return x, y



#===============================================================================
#                  Data Processing Functions 
#=============================================================================== 


def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type : 
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans    
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans



def oneHotEncode(df,colNames):
    '''
    onehot encode categorical features 
    Args: 
        df:      dataframe containing features in columns
        coNames: names of categorical features to be trnasformed 
                 to onehot encoding 
    '''
    for col in colNames:
        if( df[col].dtype == np.dtype('object')):
            #print(col)
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)
            #drop the encoded column
            df.drop([col],axis = 1 , inplace=True)
    return df


def Classify_ICD9(data):
    """
    this function classifies ICD9 codes into 18 categories as defined in 
    -> https://en.wikipedia.org/wiki/List_of_ICD-9_codes
    
    Args:
        data: Dataframe with a column named 'Primary_ICD9_CODE' containing all 
        ICD_9 codes
    
    Return: 
        The same dataframe as passed to the fundtion with 
        classified ICD9-Codes 
    """
    data = data.reset_index(drop=True)           
    print('There are {} unique ICD9 codes in this dataset.'.format(data['Primary_ICD9_CODE'].value_counts().count()))    
    # extract only first 3 characters from string 
    data_icd9 = data['Primary_ICD9_CODE'].str.slice(start=0, stop=3, step=1)
    #data_icd9 = data.Primary_ICD9_CODE
    data_enco = pd.Series([])
    #codes = pd.to_numeric(data['Primary_ICD9_CODE'])
    data_enco[1] = 12
    # set E and V codes to 0 to identify them later 
    data_icd9  = data_icd9.replace(to_replace=r'^V', value=0, regex=True)
    data_icd9  = data_icd9.replace(to_replace=r'^E', value=0, regex=True)
    data_icd9 = pd.to_numeric(data_icd9)
    data_icd9 = data_icd9.to_frame()
    # define rangers to classify 
    icd9_ranges = [(1, 140), (140, 240), (240, 280), (280, 290), (290, 320), (320, 390), 
               (390, 460), (460, 520), (520, 580), (580, 630), (630, 680), (680, 710),
               (710, 740), (740, 760), (760, 780), (780, 800), (800, 1000)]     
    # Encode icd9 codes 
    for num, cat_range in enumerate(icd9_ranges, 1):
        data_icd9['Primary_ICD9_CODE'] = np.where(data_icd9['Primary_ICD9_CODE'].between(cat_range[0],cat_range[1]), 
            num, data_icd9['Primary_ICD9_CODE'])
  
    data['Primary_ICD9_CODE']  = data_icd9.Primary_ICD9_CODE.astype(str)
    #ev = np.unique(data['Primary_ICD9_CODE'] )
    return data 


def convertRange(values, new_min, new_max, old_min, old_max):
    '''
    converts a range of values to a new range 
    Args:
        new_min:
        new_max:
    '''
    new_values = np.zeros(len(values))
    #old_max = np.max(values)
    #old_min = np.min(values)
    for i, old_value in enumerate(values):
        new_values[i] =  ((old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    return new_values 