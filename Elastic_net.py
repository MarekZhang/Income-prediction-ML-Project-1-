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
    
def set_directory():
    path = 'C:\\Users\\Admin\\Desktop\\Year 5\\Machine Learning\\Assignment1'
    os.chdir(path)
    print("\n current directory: \n", os.getcwd())
    
def parse_csv(csv_name):
    #csv_name = 'tcd ml 2019-20 income prediction training (with labels).csv' 
    data = pd.read_csv(csv_name)
    return data
    

def clean_test_data(train_data,test_data):
    
    #Clean income data
    mean_inc = train_data['Income in EUR'].mean()
    train_data[train_data['Income in EUR'] < 0] = mean_inc
    #test_data[test_data['Income in EUR'] < 0] = mean_inc
    
    #CLEAN YEAR DATA
    mean_year = train_data['Year of Record'].mean()
    train_data['Year of Record'] = train_data['Year of Record'].replace( np.nan,mean_year)
    train_data['Year of Record'] = train_data['Year of Record'].replace('unknown', mean_year)
    
    test_data['Year of Record'] = test_data['Year of Record'].replace( np.nan,mean_year)
    test_data['Year of Record'] = test_data['Year of Record'].replace('unknown', mean_year)
    
    #CLEAN AGE DATA
    mean_age = train_data['Age'].mean()
    print("\n mean age = ",mean_age,"\n")
    train_data['Age'] = train_data['Age'].replace( np.nan,mean_age)
    train_data['Age'] = train_data['Age'].replace('unknown', mean_age)
    
    test_data['Age'] = test_data['Age'].replace( np.nan,mean_age)
    test_data['Age'] = test_data['Age'].replace('unknown', mean_age)
    
    #% Gender Vecor Encoding
    
    train_data['Gender'] = train_data['Gender'].replace( np.nan,'other')
    train_data['Gender'] = train_data['Gender'].replace('unknown', 'other')
    train_data['Gender'] = train_data['Gender'].replace(0, 'other')
    #train_data[train_data['Gender'].str.isnumeric() ] = 'other'
    
    
    test_data['Gender'] = train_data['Gender'].replace( np.nan,'other')
    test_data['Gender'] = train_data['Gender'].replace('unknown', 'other')
    test_data['Gender'] = train_data['Gender'].replace(0, 'other')
    #test_data[test_data['Gender'].str.isnumeric() ] = 'other'
    
    test_data['Gender'] = train_data['Gender'].astype('category',categories=train_data["Gender"].unique())
    train_data = pd.concat([train_data,pd.get_dummies(train_data['Gender'], prefix='Gender',dummy_na=True)],axis=1).drop(['Gender'],axis=1)
    test_data = pd.concat([test_data,pd.get_dummies(test_data['Gender'], prefix='Gender',dummy_na=True)],axis=1).drop(['Gender'],axis=1)
    

    #% Country Vecor Encoding      
    
    train_data['Country'] = train_data['Country'].replace( np.nan,'other')
    train_data['Country'] = train_data['Country'].replace('unknown', 'other')
    
    test_data['Country'] = test_data['Country'].replace( np.nan,'other')
    test_data['Country'] = test_data['Country'].replace('unknown', 'other')
    
    test_data['Country'] = train_data['Country'].astype('category',categories=train_data["Country"].unique())
    train_data = pd.concat([train_data,pd.get_dummies(train_data['Country'], prefix='Country',dummy_na=True)],axis=1).drop(['Country'],axis=1)
    test_data = pd.concat([test_data,pd.get_dummies(test_data['Country'], prefix='Country',dummy_na=True)],axis=1).drop(['Country'],axis=1)
    
    
    #% Profession Vecor Encoding
   
    train_data['Profession'] = train_data['Profession'].replace( np.nan,'other')
    train_data['Profession'] = train_data['Profession'].replace('unknown', 'other')
    
    test_data['Profession'] = test_data['Profession'].replace( np.nan,'other')
    test_data['Profession'] = test_data['Profession'].replace('unknown', 'other')
    
    test_data['Profession'] = train_data['Profession'].astype('category',categories=train_data["Profession"].unique())
    train_data = pd.concat([train_data,pd.get_dummies(train_data['Profession'], prefix='Profession',dummy_na=True)],axis=1).drop(['Profession'],axis=1)
    test_data = pd.concat([test_data,pd.get_dummies(test_data['Profession'], prefix='Profession',dummy_na=True)],axis=1).drop(['Profession'],axis=1)
     
    #% university Vecor Encoding
   
    train_data['University Degree'] = train_data['University Degree'].replace( np.nan,'No')
    train_data['University Degree'] = train_data['University Degree'].replace('unknown', 'No')
    #train_data[test_data['University Degree'].str.isnumeric() ] = 'No'
    
    test_data['University Degree'] = test_data['University Degree'].replace( np.nan,'No')
    test_data['University Degree'] = test_data['University Degree'].replace('unknown', 'No')
    #test_data[test_data['University Degree'].str.isnumeric() ] = 'No'
    
    test_data['University Degree'] = train_data['University Degree'].astype('category',categories=train_data["University Degree"].unique())
    train_data = pd.concat([train_data,pd.get_dummies(train_data['University Degree'], prefix='University Degree',dummy_na=True)],axis=1).drop(['University Degree'],axis=1)
    test_data = pd.concat([test_data,pd.get_dummies(test_data['University Degree'], prefix='University Degree',dummy_na=True)],axis=1).drop(['University Degree'],axis=1)
    
    
    # Hair vector encoding
   
    train_data['Hair Color'] = train_data['Hair Color'].replace( np.nan,'other')
    train_data['Hair Color'] = train_data['Hair Color'].replace('unknown', 'other')
    
    test_data['Hair Color'] = test_data['Hair Color'].replace( np.nan,'other')
    test_data['Hair Color'] = test_data['Hair Color'].replace('unknown', 'other')
    
    test_data['Hair Color'] = train_data['Hair Color'].astype('category',categories=train_data["Hair Color"].unique())
    train_data = pd.concat([train_data,pd.get_dummies(train_data['Hair Color'], prefix='Hair Color',dummy_na=True)],axis=1).drop(['Hair Color'],axis=1)
    test_data = pd.concat([test_data,pd.get_dummies(test_data['Hair Color'], prefix='Hair Color',dummy_na=True)],axis=1).drop(['Hair Color'],axis=1)
    
    return train_data,test_data


def make_xy(data):
    y = data.iloc[:,6:7].values
    data.drop(['Income in EUR'],axis=1)
    X = data.iloc[:,1:1533].values
    type(X)
    type(y)
    return X , y, data

def make_xy2(data):
    y = data.iloc[:,6:7].values
    data.drop(['Income'],axis=1)
    X = data.iloc[:,1:1533].values
    type(X)
    type(y)
    return X , y, data


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
    
def val_set(data , labels):
    # Split the data
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.33, shuffle= True)
    return x_train, x_val, y_train, y_val 

def train_net(x,y):
    model = ElasticNet()
    model.fit(x,y)
    return model

def kfold(data):
    # prepare cross validation
    kfold = KFold(10, True, 1)
    db = kfold.split(data)
#%%   

    set_directory()
    csv_train = 'tcd ml 2019-20 income prediction training (with labels).csv' 
    csv_test = 'tcd ml 2019-20 income prediction test (without labels).csv'
    
    # SORTING OUT TRAIN DATA
    data_train = parse_csv(csv_train)
    
    # SORTING OUT TEST DATA
    data_test = parse_csv(csv_test)
    
    # CLEANING THE DATA
    data_train2,data_test2 = clean_test_data(data_train,data_test)
    
    #SPLITTING INTO TRAINING AND TARGET PARAMETERS
    x_train , y_train,data_train = make_xy(data_train)
    x_test , y_test,data_test = make_xy2(data_test)
    
    # NORMALISING THE DATA SETS
    x_train,x_test = normalise_data(x_train, x_test)
    
    # SPLIT TRAIN INTO TRAIN AND VAL SETS
    x_train, x_val, y_train, y_val = val_set(x_train,y_train)

    
    model = train_net(x_train,y_train)
    y_pred = model.predict(x_test)

    y_csv = y_csv_maker(y_pred)

    


#%%
y,ycsv = main()

#%%
str1 = "Instance"
str2 = "Income"
arr = [str1,str2]
X = np.insert(y, 0, arr, axis=0)


#%%
np.savetxt("y_pred3.csv", ycsv, delimiter=",")
