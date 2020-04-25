#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:48:59 2020

Authors: Yeshe/Sanda

This script trains a classifier on MIMIC III data for prediction the length 
of stay for individuals for defined time intervals 

Intervals are: (1-3 days, 4-7 days, 8-30 days)

"""

# imports 
import tensorflow 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings 
import os
from Utils import loadData, loadTestData
from Utils import plotNewDis
from Utils import get_cols_with_no_nans, oneHotEncode
from Utils import Classify_ICD9, convertRange

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import BatchNormalization


# custum loss 
def mean_squared_logarithmic_error(y_true, y_pred):    
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)    
    return K.mean(K.square(first_log - second_log), axis=-1)

# custom R2-score metrics for keras backend
from sklearn.metrics import r2_score
from keras import backend as K
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#==============================================================================
#                   Display functions
#============================================================================== 

def plotHist(history):
    plt.plot(history.history['loss'], label='CCE (testing data)')
    plt.plot(history.history['val_loss'], label='CCE (validation data)')
    plt.title('Categrorical Cross Entropy (CCE) over Epochs')
    plt.ylabel('CCE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()


def plotHistR(history):
    plt.figure()
    plt.plot(history.history['r2_keras'], label='R² (testing data)')
    plt.plot(history.history['val_r2_keras'], label='R² (validation data)')
    plt.title('R²')
    plt.ylabel('R²value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

#===============================================================================
#                     Data Handeling and Preprocessing Functions 
#=============================================================================== 
    
def checkCollinearity(x):
    """
    plot collinearity for all features
    Args: 
        x: dataframe with features stored in columns 
    """
    C_mat = x.corr()
    fig = plt.figure(figsize = (15,15))
    sb.heatmap(C_mat, vmax = .8, square = True)
    plt.show()
    
    
def preprocess(x_train, y):
    '''
    This function does all the data preparation for neural network training, 
    including: encoding, normalization of sevirity scores and definition of 
    classes (time intervals)
    '''
    
    x_train= x_train.reset_index(drop=True)
    y = y.reset_index(drop=True)
    # convert hours to days 
    y.x = y.x/24
    
    # exclude patinets shorter than 1 day in ICU 
    x_train= x_train[y.x>=2]
    y = y[y.x>=2]
    
    # define time intervals for classification 
    y.loc[(y.x <=3)] = 0
    y.loc[(y.x > 3) & (y.x <=7)] = 1
    y.loc[(y.x > 7) & (y.x <=30)] = 2


    x_train= x_train.reset_index(drop=True)   
    y = y.reset_index(drop=True)
    
    # normalize scores 
    x_train['OASIS']  = x_train['OASIS'].div(83)
    x_train['SAPSII']  = x_train['SAPSII'].div(163) 
#    x_train['elixhauser_vanwalraven']  = x_train['elixhauser_vanwalraven'].div(89) 
    x_train['elixhauser_vanwalraven']  =  convertRange(x_train['elixhauser_vanwalraven'], 0, 1, -19, 89)
    
    # encode all CID9 codes to 18 more general classes 
    x_train = Classify_ICD9(x_train)

    # check data for nans and count categrical and numerical features in dataset 
    num_cols = get_cols_with_no_nans(x_train , 'num')
    cat_cols = get_cols_with_no_nans(x_train , 'no_num')
    print ('Number of numerical columns with no nan values :',len(num_cols))
    print ('Number of nun-numerical columns with no nan values :',len(cat_cols))

    # One Hot Encode Categorical Features 
    print('There were {} columns before encoding categorical features'.format(x_train.shape[1]))
    combined = oneHotEncode(x_train, cat_cols)
    print('There are {} columns after encoding categorical features'.format(combined.shape[1]))
    return  combined, y


#===============================================================================
#                     Neural Network Functions
#===============================================================================    
# ------------------ build models --------------------------------------------
#import keras
def create_model_1(input_dim, classes): 
    """
    Define the neural network architecture 
    Args:
        train: input data (X)
    """
#    global input_dim
    n = 64
    model = tensorflow.keras.models.Sequential()    
    model.add(Dense(n, input_dim=input_dim, kernel_initializer='uniform'))
    model.add(BatchNormalization())

    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n*2, input_dim=input_dim, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    model.add(Dense(classes, kernel_initializer='uniform',activation='softmax')) 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

import keras

# same model as above implemneted just for Gridsearch training 
def create_model(): 
    """
    Define the neural network architecture 
    Args:
        train: input data (X)
    """
    global input_dim
    classes = 3
    n = 64
    model = tensorflow.keras.models.Sequential()    
    model.add(Dense(n, input_dim=input_dim, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n*2, input_dim=input_dim, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(classes, kernel_initializer='uniform',activation='softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model  

#------------------------- training functions ---------------------------------
    
from sklearn.model_selection import cross_val_score, KFold
def Train_CV(X_train, y_train, X_test, y_test, k=5, epochs=1000, batchsize=200, seed = 100):
    '''
    Crossvalidation training function 
    '''
    estimator = KerasRegressor(build_fn=create_model, nb_epoch=epochs, batch_size=batchsize, verbose=False)
    kfold = KFold(n_splits=k, random_state=seed)
    results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    print("Results: %.2f (%.2f) MAE" % (results.mean(), results.std()))
    estimator.fit(X_train, y_train)
    
    # evaluate model on test set 
    prediction = estimator.predict(X_test)
    train_error =  np.abs(y - prediction)
    mean_error = np.mean(train_error)
   # min_error = np.min(train_error)
    #max_error = np.max(train_error)
    std_error = np.std(train_error)
    print('-'*30)
    print('Evaluation Results')
    print("Results (mean, std): %.2f (%.2f) MSE" % (mean_error , std_error))
    
    
from sklearn.utils import class_weight
def train_val_training(X_train, y_train, model, class_weights):
    """
    Train Neural Network model base on train validation set 
    Args:
        X_train: traininng data 
        y_train: corrisponding labels 
        model: compiled neural network model     
    """
    # set pach where trained  models will be saved to 
    savepath = Path('/home/kwaygo/Documents/NUS/SPH6004/P2/SPH6004_P2/models/Regression')
    checkpoint_name = os.path.join(savepath, 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' )      
    # define callbacks
    cp = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    es = EarlyStopping(monitor='val_loss', patience= 4, verbose=1)
    callbacks_list = [es, cp]
    
    # start training
    hist = model.fit(X_train, y_train, epochs=30, batch_size=500, validation_split = 0.2, callbacks=callbacks_list, class_weight=class_weights)
#    hist = model.fit(X_train, y_train, epochs=500, batch_size=3000, validation_split = 0.2, callbacks=callbacks_list)
    
    print("[INFO] avg. ICU LOS of train set: {}, std ICU LOS of test set: {}".format(np.mean(y_train), np.std(y_train)))
    # plot training History 
    plotHist(hist)
    return model

# ------------------ model evaluation -----------------------------------------

from sklearn.metrics import auc  
from sklearn.metrics import roc_curve
def evaluate_plotROC(model, X_test, y_test):
    '''
    This function plots the ROC corves for all classes and claculates their AUC
    and also plots the confusion matrix 
    '''
    y_score = model.predict(X_test)
    n_classes =  y_score.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    labels= ["1-3 Days","4-7 Days", "8-30 Days"]
    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
#        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(labels[i] + 'ROC curve')
        plt.legend(loc="lower right")
        plt.show()
    score = y_score.argmax(axis=1)
    test = y_test.argmax(axis=1)
    
    from mlxtend.evaluate import confusion_matrix

    y_target =    test
    y_predicted = score
    
    cm = confusion_matrix(y_target=y_target, 
                          y_predicted=y_predicted, 
                          binary=False)

#    target_names = ['0-3', '3-7', '7-14', '14-21', '21-30']     
    s1 =np.sum(y_test[:,0])
    s2 =np.sum(y_test[:,1])
    s3 =np.sum(y_test[:,2])
    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    plt.show()

def PlotDistribution(data, a, label):    
    print(np.min(data))
    print(np.max(data))
#    plt.figure(1)
    plt.subplot(a)
    plt.hist(data, bins=100, color='c', edgecolor='k',alpha=0.65)
    plt.gca().set(title=label, ylabel='Frequency')

 
def GridSearchTraining(X_train, y_train, param_grid,k, class_weights):
    """
    Train Neural Network model with k-fold validation tuning batch size 
    and numers of epochs 
    
    Args:
        X_train: traininng data 
        y_train: corrisponding labels 
        param_grid: contains the ranges for batch size and number of epochs 
        k: number of CV folds 
    """                                                                                                              
    # create model
    Kmodel = KerasRegressor(build_fn=create_model, verbose=2)
    # start grid search 
    grid = GridSearchCV(estimator=Kmodel, param_grid=param_grid, n_jobs=1,scoring='neg_mean_absolute_error', cv=k)
    grid_result = grid.fit(X_train, y_train, class_weight= class_weights)
    # summarize results
    print('='*50)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param)) 
    
#===============================================================================
#                       Main Function 
#===============================================================================  0 

import tensorflow as tf
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model 


def main():
    
#------------------- load data ------------------------------------------------
  
   X_train, y_train = loadData()   
   # load test data 
   X_test, y_test = loadTestData()
   # combine data 
   X_all = X_train.append(X_test)
   y_all = y_train.append(y_test)   
   X_train_d, y_train_d = loadData('death')
   X_test_d, y_test_d = loadTestData('death')  
   X_all_d = X_train.append(X_test_d)
   y_all_d = y_train.append(y_test_d)     
   X_all = X_train.append(X_all_d)
   y_all = y_train.append(y_all_d)
#---------------  preprocess data    ------------------------------------------
 
   X_all, y_all = preprocess(X_all, y_all)
   # compute class weights 
   class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_all.x),
                                                 y_all.x)
   y_all = keras.utils.to_categorical(y_all) 
   c1 = np.sum(y_all[:,0])
   c2 = np.sum(y_all[:,1])
   c3 = np.sum(y_all[:,2])
   # split data 
   X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state=42)
   global input_dim
   input_dim = X_train.shape[1]
   classes = y_all.shape[1]
#  get model and 
   model = create_model_1(input_dim, classes)
   model.summary() 

##------------------- normal train validation split ----------------------------
   model = train_val_training(X_train, y_train, model, class_weights)
   evaluate_plotROC(model, X_test, y_test)  
   
##------------------- Grid Search training ----------------------------
   # grid search training 
   #define the grid search parameters
#   batch_range = [200, 500, 1000]
#   epoch_range = [15, 20, 30]
#   param_grid = dict(batch_size=batch_range, epochs=epoch_range)
#   k = 5
#   GridSearchTraining(X_train, y_train, param_grid, k, class_weights)

        
#from sklearn.metrics import confusion_matrix, classification_report

if __name__ == "__main__":
    main()
    
    
    