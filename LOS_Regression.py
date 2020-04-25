#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:48:59 2020
"""
import tensorflow 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings 
import json
import os
from Utils import Classify_ICD9, get_cols_with_no_nans, convertRange, oneHotEncode
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
#from keras.optimizers import adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.layers import BatchNormalization

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
    plt.plot(history.history['loss'], label='MAE (testing data)')
    plt.plot(history.history['val_loss'], label='MAE (validation data)')
    plt.title('MAE')
    plt.ylabel('MAE value')
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
#                              Load Data 
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
#                     Data Handeling and Preprocessing Functions 
#=============================================================================== 
    
def checkCollinearity(x):
    """
    plot collinearity for all variables in x 
    """
    C_mat = x.corr()
    fig = plt.figure(figsize = (15,15))
    sb.heatmap(C_mat, vmax = .8, square = True)
    plt.show()
    

def preprocess(x_train, y):
    '''
    preprocess data, feature enncoding (onehot encoding) and normalitazion 
    of sevitŕity scores , double checking for missing data .. 
    '''
    x_train= x_train.reset_index(drop=True)
    y = y.reset_index(drop=True)
    # convert hours to days
    y.x = y.x/24
    
    # define time interval
    x_train= x_train[y.x>=2]
    y = y[y.x>=2]    
    x_train= x_train.reset_index(drop=True)   
    y = y.reset_index(drop=True)
    
    #============  remove features based on lasso  ============================
    # sodium min sodim max correlated with bun_min and chloride_min
#    x_train = x_train.drop(['MARITAL_STATUS', 'CREATININE_max', 'WBC_min', 'SpO2_Mean', 'LIVER_DISEASE', 'VALVULAR_DISEASE'], axis=1)
    
#    x_train = x_train.drop(['SysBP_Max', 'SysBP_Min', 'SysBP_Mean'], axis=1)#
#    x_train= x_train.drop(['DiasBP_Max', 'DiasBP_Min', 'HeartRate_Max', 'HeartRate_Min', 'MeanBP_Max', 'MeanBP_Min', 'RespRate_Max', 'RespRate_Min', 'SpO2_Max', 'SpO2_Min', 'TempC_Max', 'TempC_Min'], axis=1)    
    #==========================================================================
    
    # normalize scores 
    x_train['OASIS']  = x_train['OASIS'].div(83)
    x_train['SAPSII']  = x_train['SAPSII'].div(163) 
#    x_train['elixhauser_vanwalraven']  = x_train['elixhauser_vanwalraven'].div(89) 
    x_train['elixhauser_vanwalraven']  =  convertRange(x_train['elixhauser_vanwalraven'], 0, 1, -19, 89)
    
    #x_train = x_train.drop(['Primary_ICD9_CODE'], axis=1)
    x_train = Classify_ICD9(x_train)

    num_cols = get_cols_with_no_nans(x_train , 'num')
    cat_cols = get_cols_with_no_nans(x_train , 'no_num')
    print ('Number of numerical columns with no nan values :',len(num_cols))
    print ('Number of nun-numerical columns with no nan values :',len(cat_cols))
   
    #Visualize each feature histogram
    #x = x[num_cols + cat_cols]
    #x.hist(figsize = (12,10))
    #plt.show()
    # One Hot Encode Categorical Features 
    print('There were {} columns before encoding categorical features'.format(x_train.shape[1]))
    combined = oneHotEncode(x_train, cat_cols)
    print('There are {} columns after encoding categorical features'.format(combined.shape[1]))
    return  combined, y


#===============================================================================
#                     Neural Network Functions
#===============================================================================    
 
# ------------------ build models --------------------------------------------

def create_model(input_dim): 
    """
    Define the neural network architecture 
    Args:
        train: input data (X)
    """
    n = 64
#    global input_dim
    model = tensorflow.keras.models.Sequential()
    
    model.add(Dense(n, input_dim=input_dim, kernel_initializer='uniform'))
    model.add(BatchNormalization())
#    model.add(LeakyReLU(alpha=0.1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n, input_dim=input_dim, kernel_initializer='uniform'))
    model.add(BatchNormalization())
#    model.add(LeakyReLU(alpha=0.1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, kernel_initializer='uniform',activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])    
    #model.compile(loss=r2_keras, optimizer='adam', metrics=['mse', 'mae', 'accuracy', r2_keras])    
    return model

#------------------------- training functions ---------------------------------

def GridSearchTraining(X_train, y_train, param_grid,k):
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
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    print('='*50)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param)) 
    
    
def Train_CV(X_train, y_train, X_test, y_test, k=5, epochs=1000, batchsize=200, seed = 100):
    
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
    
    
def train_val_training(X_train, y_train, model):
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
    hist = model.fit(X_train, y_train, epochs=500, batch_size=500, validation_split = 0.2, callbacks=callbacks_list) 
    
    print("[INFO] avg. ICU LOS of train set: {}, std ICU LOS of test set: {}".format(np.mean(y_train), np.std(y_train)))
    # plot training History 
    plotHist(hist)
    return model


# ------------------ model evaluation -----------------------------------------
def PlotDistribution(data, a, label):    
    print(np.min(data))
    print(np.max(data))
#    plt.figure(1)
    plt.subplot(a)
    plt.hist(data, bins=100, color='c', edgecolor='k',alpha=0.65)
    plt.gca().set(title=label, ylabel='Frequency')

import locale

def evaluate_model(X_test, y_test, model):
    '''
    Evaluation on test set:     
        -compute difference between the predicted LOS and the
        actual LOS
        -plot absolute difference in histogram 
        -then compute the percentage difference and the absolute percentage 
        difference
    '''    
    testY = y_test.x
    preds = model.predict(X_test)
#    preds = preds*30
#    testY = testY*30

#    preds = np.exp(preds)
#    testY = np.exp(testY)
    pred = preds.flatten()    
    
    # plot distribution of prediction and ground truth 
    plt.figure(2)
    PlotDistribution(testY, 211, 'Test Distribution')
    PlotDistribution(pred, 212, 'Prediction Distribution')
    
    
    print('-'*30)
    print('mean absolute error on test set: ' + str(mean_absolute_error(testY,pred)))
    print('-'*30)    

    plt.figure(3)
#    plt.title('Absolute Error')
    plt.hist(np.abs(pred  - testY), bins=100,  color='c', edgecolor='k', alpha=0.65)    
    plt.gca().set(title='Absolute Error', ylabel='Frequency', xlabel='Error (Days)');

        
    diff = pred  - testY    
    # Percentage difference and Absolute percentage difference
    percentDiff = (diff / testY) * 100
    absPercentDiff = np.abs(percentDiff)
    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)
    
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    print("[INFO] avg. ICU LOS of test set: {}, std ICU LOS of test set: {}".format(np.mean(testY), np.std(testY)))
    print("[INFO] absolute percentage diff. mean: {:.2f}%, std: {:.2f}%".format(mean, std))
    print('-'*30)
    r2s = r2_score(testY, pred)
    print('r2_score: ' + str(r2s))
    

    slope, intercept, r_value, p_value, std_err = stats.linregress(testY,pred)
    print(r_value**2)

#===============================================================================
#                       Main Function 
#===============================================================================  

import tensorflow as tf

'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''
def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

def quantile_loss(y_true, y_pred):
    q = 0.7
    e = y_true - y_pred
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)


import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot
#keras.utils.vis_utils.pydot = pydot
#import tensorflow as tf
from tensorflow.keras.utils import plot_model 
#plot_model(model, to_file='model.png')
#Visualize Model
#import tf.keras

def main ():
    
   # load train data 
   X_train, y_train = loadData()
   # load test data 
   X_test, y_test = loadTestData()
   # combine data 
   X_all = X_train.append(X_test)
   y_all = y_train.append(y_test)
   # preprocess data 
   X_all, y_all = preprocess(X_all, y_all)

#   PlotDistribution(y_all.x, 211, 'Test Distribution')
   # split data 
   X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state=42)

##   y_train = np.log(y_train)
##   y_test = np.log(y_test)

##   global input_dim
   input_dim = X_train.shape[1]
  
#   get model and plot architecture 
   model = create_model(input_dim)
   model.summary()
#------------------- normal train validation split ----------------------------

   model = train_val_training(X_train, y_train, model)
   evaluate_model(X_test, y_test, model) 

   #X_test,y_test  = loadTestData()
   #Train_CV(X_train,y_train,X_test,y_test)
   
#    grid search training 
#   define the grid search parameters
#   batch_range = [200, 500, 1000]
#   epoch_range = [100, 500, 900]
#   param_grid = dict(batch_size=batch_range, epochs=epoch_range)
#   k = 5
#   GridSearchTraining(X_train, y_train, param_grid, k)

    
if __name__ == "__main__":
    main()
    
    

# extra
    

#from tensorflow.keras.constraints import non_neg, max_norm
#from keras import backend as K
## R square loss 
#def coeff_determination(y_true, y_pred):
#    from keras import backend as K
#    SS_res =  K.sum(K.square( y_true-y_pred ))
#    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
##    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
#    return (  SS_res/(SS_tot + K.epsilon()) )
#import keras
#
#
#def create_model(input_dim): 
#    """
#    Define the neural network architecture 
#    Args:
#        train: input data (X)
#    """
#    n = 64
##    global input_dim
#    model = tensorflow.keras.models.Sequential()    
#    model.add(Dense(n, input_dim=input_dim, kernel_initializer='uniform'))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(n, input_dim=input_dim, kernel_initializer='uniform'))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
#    
#    model.add(Dense(1,activation='linear' , kernel_constraint=max_norm(10.)))
#    model.compile(loss=coeff_determination, optimizer='adam', metrics=['mse', 'mae', 'accuracy'])    
#    model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae', 'accuracy']) 
#    #model.compile(loss=r2_keras, optimizer='adam', metrics=['mse', 'mae', 'accuracy', r2_keras])    
#    return model  