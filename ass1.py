# -*- coding: utf-8 -*-
"""

Spyder Editor

Colin Comey
15320035
Machine Learnig Project 1

"""

#%%

# =============================================================================
# SETTING DIRECTORY AND IMPORTING DATA
# =============================================================================

NUM_EPOCHS = 3
BATCH_SIZE = 30000
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder

    
def set_directory():
    path = 'C:\\Users\\Admin\\Desktop\\Year 5\\Machine Learning\\Assignment1'
    os.chdir(path)
    print("\n current directory: \n", os.getcwd())
    
def parse_csv(csv_name):
    #csv_name = 'tcd ml 2019-20 income prediction training (with labels).csv' 
    data = pd.read_csv(csv_name)
    return data
    

def clean_train_data(data):
    data = data[~data['Year of Record'].isnull()]
    data = data[~data['Gender'].isnull()]
    data = data[~data['Age'].isnull()]
    data = data[~data['Country'].isnull()]
    data = data[~data['Size of City'].isnull()]
    data = data[~data['Profession'].isnull()]
    data = data[~data['University Degree'].isnull()]
    data = data[~data['Hair Color'].isnull()]
            

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

def model_define() :# define the keras model
    model = Sequential()
    model.add(Dense(1532, input_dim=1532, activation='linear'))
    model.add(Dense(1200, activation='linear'))
    model.add(Dense(800, activation='linear'))
    model.add(Dense(400, activation='linear'))
    model.add(Dense(150, activation='linear'))
    model.add(Dense(50, activation='linear'))
    model.add(Dense(10, activation='linear'))
    model.add(Dense(1, activation='linear'))
    return model

def normalise_data(train_X,test_X): 
    #sc = StandardScaler()
    #X = sc.fit_transform(X)
    scaler = StandardScaler()
    train_X = scaler.fit_transform( train_X )
    test_X = scaler.transform( test_X )
    return train_X, test_X

def train_model(x_train, y_train, model, x_test): #%% compile and train the keras model
    #opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mse', optimizer='adam',metrics=['mae', 'acc'])
    history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    y_pred = model.predict(X_test)
    return y_pred, model, history

def print_history_shtuff(history):
    print(history.history.keys())
    # list all data in history

# =============================================================================
#     # summarize history for accuracy
#     fig = plt.figure()
#     plt.plot(history.history['mean_absolute_error'])
#     plt.plot(history.history['val_mean_absolute_error'])
#     plt.title('model mean absolute error')
#     plt.ylabel('mae')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     fig.show()
#     input("Press Enter to continue...")
#     
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
# =============================================================================
    
    
    train_mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(NUM_EPOCHS)

    plt.figure()
    plt.plot(epochs, train_mae, 'r', label='Training MAE')
    plt.plot(epochs, val_mae, 'b', label='Validation MAE')

    plt.title('Training and validation MAE')
    plt.legend()
    plt.show()
    
def y_csv_maker(y):
    print('\n Saving as y_pred.csv  \n ')
    y_csv_template = 'tcd ml 2019-20 income prediction submission file.csv' 
    data = parse_csv(y_csv_template)
    #print(data.head())
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


def main():
    set_directory()
    csv_train = 'tcd ml 2019-20 income prediction training (with labels).csv' 
    data_train = parse_csv(csv_train)
    data_train = clean_test_data(data_train)
    print(data_train.head())
    #input("Press Enter to continue...")
    
    x_train , y_train = make_xy(data_train)
    
    csv_test = 'tcd ml 2019-20 income prediction test (without labels).csv' 
    data_test = parse_csv(csv_test)
    data_test = clean_test_data(data_test)
    X_test , y_test = make_xy(data_test)
    x_train,X_test = normalise_data(x_train, X_test)
    
      
    x_train, x_val, y_train, y_val = val_set(x_train,y_train)
    model = model_define()
    y_p, model, history = train_model(x_train, x_val, y_train, y_val, model)
    
    #input("Press Enter to continue...")
    print("\n Printing history shtuff \n")
    #print_history_shtuff(history)
    

    
    y_pred = model.predict(X_test)

    y_csv = y_csv_maker(y_pred)
    return y_pred, y_csv
    


# =============================================================================
# =============================================================================
# # #%% THE REAL MVP
# =============================================================================
# =============================================================================
set_directory()

csv_train = 'tcd ml 2019-20 income prediction training (with labels).csv' 
data_train = parse_csv(csv_train)
csv_test = 'tcd ml 2019-20 income prediction test (without labels).csv' 
data_test = parse_csv(csv_test)

data_train2,data_test2 = clean_test_data(data_train,data_test)
x_train , y_train,data_train2 = make_xy(data_train2)
x_test , y_test,data_test2 = make_xy2(data_test2)
x_train,x_test = normalise_data(x_train, x_test)
model = model_define()
y_p, model, history = train_model(x_train, y_train,  model, x_test)

y_pred = model.predict(x_test)
y_csv = y_csv_maker(y_pred)
#print(data_train.head())









#%%
    
x_train , y_train = make_xy(data_train)
X_test , y_test = make_xy(data_test)
x_train,X_test = normalise_data(x_train, X_test)
    
      
x_train, x_val, y_train, y_val = val_set(x_train,y_train)
model = model_define()
#
model, history = train_model(x_train, x_val, y_train, y_val, model)
    
y_pred = model.predict(x_test)
y_csv = y_csv_maker(y_pred)
#%% finding negatives 
#num = data_train['University'].str.isnumeric().sum()
num = data_train['Size of City'].lt(0).sum()
print('\n no of negs: ',num )
#%%
y,ycsv = main()
#%%
#Clean income data
    #mean_inc = data_train['Size of City'].mean()
    data_train[data_train['Size of City'] < 0] = mean_inc

#%%
np.savetxt("y_pred3.csv", ycsv, delimiter=",")


#%%
nans = []
for i in range(len(y)):
    if np.isnan(y_train[i]) == True:
        nans.append(i)

#%%   gender_onehot_encoder = OneHotEncoder()
    from sklearn.preprocessing import OneHotEncoder
    #from sklearn.feature_extraction import FeatureHasher
    #h = FeatureHasher(input_type='string')
    gender_onehot_encoder = OneHotEncoder(categories='auto')
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data_train['Gender'])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = gender_onehot_encoder.fit_transform(integer_encoded)
    #%%
    data_train = data_train.drop('Gender',axis=1)
    
    #m,n = onehot_encoded.shape
    #%%
    data_train['Gender'] = data_test['Gender'].astype('category',categories=data_train["Gender"].unique())
    #uniqueId = data_train["Gender"].unique() 
    #data_train = pd.concat([data_train,pd.get_dummies(data_test['Gender'], prefix='Gender',dummy_na=True)],axis=1).drop(['Gender'],axis=1)
    
    
    
    
    #%%
    
    
    
    
data_test['Country'] = data_test['Country'].replace( np.nan,'other')


#%%
data_test['Country'] = data_train['Country'].astype('category',categories=data_train["Country"].unique())
data_train = pd.concat([data_train,pd.get_dummies(data_train['Country'], prefix='Country',dummy_na=True)],axis=1).drop(['Country'],axis=1)
data_test = pd.concat([data_test,pd.get_dummies(data_test['Country'], prefix='Country',dummy_na=True)],axis=1).drop(['Country'],axis=1)




