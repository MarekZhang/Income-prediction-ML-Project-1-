# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:29:36 2019

@author: Admin
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeCV


# =============================================================================
# DEFING THE FUNCTIONS
# =============================================================================


def set_directory():
    path = 'C:\\Users\\Admin\\Desktop\\Year 5\\Machine Learning\\Assignment1'
    os.chdir(path)
    print("\n current directory: \n", os.getcwd())
    
def parse_csv(csv_name):
    #csv_name = 'tcd ml 2019-20 income prediction training (with labels).csv' 
    data = pd.read_csv(csv_name)
    return data


def feat_hash(train_data, test_data, feats):

    print('\n',type(train_data[5]))
    hashed = FeatureHasher(n_features=feats, input_type='string')
    hashed.fit(train_data)
    trans_train = hashed.transform(train_data)
    trans_test = hashed.transform(test_data)
    return trans_train,trans_test


def clean_test_data(train_data,test_data):
    
    #Clean income data
    mean_inc = train_data['Income in EUR'].mean()
    train_data[train_data['Income in EUR'] < 0] = mean_inc
    ytrain = train_data['Income in EUR'].values
    
    #CLEAN YEAR DATA
    mean_year = train_data['Year of Record'].mean()
    train_data['Year of Record'] = train_data['Year of Record'].replace( np.nan,mean_year)
    train_data['Year of Record'] = train_data['Year of Record'].replace('unknown', mean_year)
    
    test_data['Year of Record'] = test_data['Year of Record'].replace( np.nan,mean_year)
    test_data['Year of Record'] = test_data['Year of Record'].replace('unknown', mean_year)
    
    xtrain_year = train_data['Year of Record'].values
    xtest_year = test_data['Year of Record'].values
    
    #CLEAN AGE DATA
    mean_age = train_data['Age'].mean()
    print("\n mean age = ",mean_age,"\n")
    train_data['Age'] = train_data['Age'].replace( np.nan,mean_age)
    train_data['Age'] = train_data['Age'].replace('unknown', mean_age)
    
    test_data['Age'] = test_data['Age'].replace( np.nan,mean_age)
    test_data['Age'] = test_data['Age'].replace('unknown', mean_age)
    
    xtrain_age = train_data['Age'].values
    xtest_age = test_data['Age'].values
    
    #% Gender Vecor Encoding
    
    train_data['Gender'] = train_data['Gender'].replace( np.nan,'other')
    train_data['Gender'] = train_data['Gender'].replace('unknown', 'other')
    train_data['Gender'] = train_data['Gender'].replace(0, 'other')
    #train_data[train_data['Gender'].str.isnumeric() ] = 'other'
    
    
    test_data['Gender'] = train_data['Gender'].replace( np.nan,'other')
    test_data['Gender'] = train_data['Gender'].replace('unknown', 'other')
    test_data['Gender'] = train_data['Gender'].replace(0, 'other')
    #test_data[test_data['Gender'].str.isnumeric() ] = 'other'
    
    xtrain_gender, xtest_gender = feat_hash(train_data['Gender'], test_data['Gender'], 3)

    #% Country Vecor Encoding      
    
    train_data['Country'] = train_data['Country'].replace( np.nan,'other')
    train_data['Country'] = train_data['Country'].replace('unknown', 'other')
    
    test_data['Country'] = test_data['Country'].replace( np.nan,'other')
    test_data['Country'] = test_data['Country'].replace('unknown', 'other')
    
    #test_data['Country'] = train_data['Country'].astype('category',categories=train_data["Country"].unique())
    #train_data = pd.concat([train_data,pd.get_dummies(train_data['Country'], prefix='Country',dummy_na=True)],axis=1).drop(['Country'],axis=1)
    #test_data = pd.concat([test_data,pd.get_dummies(test_data['Country'], prefix='Country',dummy_na=True)],axis=1).drop(['Country'],axis=1)
    xtrain_country, xtest_country = feat_hash(train_data['Country'], test_data['Country'], 50)
    
    #% Profession Vecor Encoding
   
    train_data['Profession'] = train_data['Profession'].replace( np.nan,'other')
    train_data['Profession'] = train_data['Profession'].replace('unknown', 'other')
    
    test_data['Profession'] = test_data['Profession'].replace( np.nan,'other')
    test_data['Profession'] = test_data['Profession'].replace('unknown', 'other')
    xtrain_prof, xtest_prof = feat_hash(train_data['Profession'], test_data['Profession'], 100)
   
    #% university Vecor Encoding
   
    train_data['University Degree'] = train_data['University Degree'].replace( np.nan,'No')
    train_data['University Degree'] = train_data['University Degree'].replace('unknown', 'No')
    #train_data[test_data['University Degree'].str.isnumeric() ] = 'No'
    
    test_data['University Degree'] = test_data['University Degree'].replace( np.nan,'No')
    test_data['University Degree'] = test_data['University Degree'].replace('unknown', 'No')
    #test_data[test_data['University Degree'].str.isnumeric() ] = 'No' 
    xtrain_uni, xtest_uni = feat_hash(train_data['University Degree'], test_data['University Degree'], 50)
    
    # Hair vector encoding
   
    train_data['Hair Color'] = train_data['Hair Color'].replace( np.nan,'other')
    train_data['Hair Color'] = train_data['Hair Color'].replace('unknown', 'other')
    
    test_data['Hair Color'] = test_data['Hair Color'].replace( np.nan,'other')
    test_data['Hair Color'] = test_data['Hair Color'].replace('unknown', 'other') 
    xtrain_hair, xtest_hair = feat_hash(train_data['Hair Color'], test_data['Hair Color'], 6)
    
    xtrain_city = train_data['Size of City'].values
    xtest_city = test_data['Size of City'].values
    
    xtrain_glasses = train_data['Wears Glasses'].values
    xtest_glasses = test_data['Wears Glasses'].values
    
    xtrain_height = train_data['Body Height [cm]'].values
    xtest_height = test_data['Body Height [cm]'].values
    
    xtrain = np.concatenate([xtrain_year,xtrain_age,xtrain_gender,xtrain_country,xtrain_prof,xtrain_uni,xtrain_hair,xtrain_city,xtrain_glasses,xtrain_height])
    xtest = np.concatenate([xtest_year,xtest_age,xtest_gender,xtest_country,xtest_prof,xtest_uni,xtest_hair,xtest_city,xtest_glasses,xtest_height])
    return train_data,test_data,xtrain,xtest,ytrain


def make_xy(data):
    y = data.iloc[:,6:7].values
    data = data.drop(['Income in EUR'],axis=1)
    X = data.iloc[:,1:1533].values
    type(X)
    type(y)
    return X , y,data

def make_xy2(data):
    y = data.iloc[:,6:7].values
    data = data.drop(['Income'],axis=1)
    X = data.iloc[:,1:1533].values
    type(X)
    type(y)
    return X , y,data

def normalise_data(train_X,test_X): 
    #sc = StandardScaler()
    #X = sc.fit_transform(X)
    scaler = StandardScaler()
    train_X = scaler.fit_transform( train_X )
    test_X = scaler.transform( test_X )
    return train_X, test_X


def y_csv_maker(y):
    print('\n Saving as y_pred.csv  \n ')
    y_csv_template = 'tcd ml 2019-20 income prediction submission file.csv' 
    data = parse_csv(y_csv_template)
    print(data.head())
    data['Income'] = y
    print(data.head())
    #data[1:,1] = y
    y_new = data.values
    np.savetxt("y_pred.csv", y_new, delimiter=",")
    return y_new

#MODEL
def why_male_models(x,y,x_test):  
    alphas = [0.1, 1, 10, 100, 1e3, 1e4, 2e4, 5e4, 8e4, 1e5, 1e6, 1e7, 1e8]
    reg = linear_model.RidgeCV(alphas=alphas, store_cv_values=True)
    reg.fit(x, y)
    cv_mse = np.mean(reg.cv_values_, axis=0)
    #model_cv = regr_cv.fit(x, y)
    print("alphas: %s" % alphas)
    print("CV MSE: %s" % cv_mse)
    print("Best alpha using built-in RidgeCV: %f" % reg.alpha_)
    
    alpha = reg.alpha_
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(x,y)
    y_pred = reg.predict(x_test)
    return y_pred


def do_stuff(df,string):
    hasher = ce.HashingEncoder(cols = [string])
    
    
    
    
    
    
# =============================================================================
# =============================================================================
# # DOING THE SHTUFF
# =============================================================================
# =============================================================================

#%%
set_directory()

csv_train = 'tcd ml 2019-20 income prediction training (with labels).csv' 
data_train = parse_csv(csv_train)
csv_test = 'tcd ml 2019-20 income prediction test (without labels).csv' 
data_test = parse_csv(csv_test)

data_train2,data_test2,xtrain,xtest,ytrain = clean_test_data(data_train,data_test)
# =============================================================================
# print('\n finished f1')
# x_train , y_train,data_train2 = make_xy(data_train2)
# print('\n finished f2')
# x_test , y_test,data_test2 = make_xy2(data_test2)
# print('\n finished f3')
# =============================================================================
xtrain,xtest = normalise_data(xtrain, xtest)
print('\n finished f4')

y_pred = why_male_models(x_train,ytrain, xtest)
y_csv = y_csv_maker(y_pred)

# =============================================================================
# TESTIN
# =============================================================================
#%%
data_test2.isnull().values.any()

#%%
data_test["Country"].unique()

#%%
alpha = 0.1
reg = linear_model.Ridge(alpha=alpha)
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

#%%
data_test2 = data_test2.drop(['Income'],axis=1,)

#%%




ce_hash = ce.HashingEncoder(cols = ['color'])
ce_hash.fit_transform(X, y)



