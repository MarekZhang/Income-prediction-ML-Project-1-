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

NUM_EPOCHS = 500
BATCH_SIZE = 5000
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
    #train_data['Gender'] = train_data['Gender'].replace(0, 'other')
    #train_data[train_data['Gender'].str.isnumeric() ] = 'other'
    
    
    test_data['Gender'] = train_data['Gender'].replace( np.nan,'other')
    test_data['Gender'] = train_data['Gender'].replace('unknown', 'other')
    test_data['Gender'] = train_data['Gender'].replace(0, 'other')
    #test_data[test_data['Gender'].str.isnumeric() ] = 'other'
    
    
    gender_label_encoder = LabelEncoder()
    train_data['Gender'] = gender_label_encoder.fit_transform(train_data['Gender'].astype(str))
    test_data['Gender'] = gender_label_encoder.transform(test_data['Gender'].astype(str))


    #% Country Vecor Encoding       
    train_data['Country'] = train_data['Country'].replace( np.nan,'other')
    train_data['Country'] = train_data['Country'].replace('unknown', 'other')
    
    test_data['Country'] = test_data['Country'].replace( np.nan,'other')
    test_data['Country'] = test_data['Country'].replace('unknown', 'other')
    
    country_label_encoder = LabelEncoder()
    train_data['Country'] = country_label_encoder.fit_transform(train_data['Country'].astype(str))
    test_data['Country'] = country_label_encoder.transform(test_data['Country'].astype(str))

    
    #% Profession Vecor Encoding
    train_data['Profession'] = train_data['Profession'].replace( np.nan,'other')
    train_data['Profession'] = train_data['Profession'].replace('unknown', 'other')
    
    test_data['Profession'] = test_data['Profession'].replace( np.nan,'other')
    test_data['Profession'] = test_data['Profession'].replace('unknown', 'other')
    
    prof_label_encoder = LabelEncoder()
    train_data['Profession'] = prof_label_encoder.fit_transform(train_data['Profession'].astype(str))
    test_data['Profession'] = prof_label_encoder.transform(test_data['Profession'].astype(str))

     
    #% university Vecor Encoding
    train_data['University Degree'] = train_data['University Degree'].replace( np.nan,'No')
    train_data['University Degree'] = train_data['University Degree'].replace('unknown', 'No')

    
    test_data['University Degree'] = test_data['University Degree'].replace( np.nan,'No')
    test_data['University Degree'] = test_data['University Degree'].replace('unknown', 'No')

    uni_label_encoder = LabelEncoder()
    train_data['University Degree'] = uni_label_encoder.fit_transform(train_data['University Degree'].astype(str))
    test_data['University Degree'] = uni_label_encoder.transform(test_data['University Degree'].astype(str))
    
    # Hair vector encoding
    train_data['Hair Color'] = train_data['Hair Color'].replace( np.nan,'other')
    train_data['Hair Color'] = train_data['Hair Color'].replace('unknown', 'other')
    
    test_data['Hair Color'] = test_data['Hair Color'].replace( np.nan,'other')
    test_data['Hair Color'] = test_data['Hair Color'].replace('unknown', 'other')
    
    hair_label_encoder = LabelEncoder()
    train_data['Hair Color'] = hair_label_encoder.fit_transform(train_data['Hair Color'].astype(str))
    test_data['Hair Color'] = hair_label_encoder.transform(test_data['Hair Color'].astype(str))

    return train_data,test_data


def make_xy(data):
    y = data.iloc[:,11:12].values
    X = data.iloc[:,1:11].values
    type(X)
    type(y)
    return X , y

def model_define() :# define the keras model
    model = Sequential()
    model.add(Dense(16, input_dim=10, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='relu'))
    return model

def normalise_data(train_X,test_X): 
    #sc = StandardScaler()
    #X = sc.fit_transform(X)
    scaler = StandardScaler()
    train_X = scaler.fit_transform( train_X )
    test_X = scaler.transform( test_X )
    return train_X, test_X

def train_model(x_train, x_val, y_train, y_val,model): #%% compile and train the keras model
    #opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mse', optimizer='adm',metrics=['mae', 'acc'])
    history = model.fit(x_train, y_train,validation_data=(x_val,y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    return model, history


    
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



def main():
    set_directory()
    csv_train = 'tcd ml 2019-20 income prediction training (with labels).csv' 
    train_data = parse_csv(csv_train)
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
    model, history = train_model(x_train, x_val, y_train, y_val, model)
    
    #input("Press Enter to continue...")
    print("\n Printing history shtuff \n")
    #print_history_shtuff(history)
    

    
    y_pred = model.predict(X_test)

    y_csv = y_csv_maker(y_pred)
    return y_pred, y_csv
#%%

set_directory()

csv_train = 'tcd ml 2019-20 income prediction training (with labels).csv' 
train_data = parse_csv(csv_train)
csv_test = 'tcd ml 2019-20 income prediction test (without labels).csv' 
data_test = parse_csv(csv_test)

train_data2,data_test2 = clean_test_data(train_data,data_test)

x_train , y_train = make_xy(train_data)
X_test , y_test = make_xy(data_test)



x_train,x_test = normalise_data(x_train, x_test)
model = model_define()
y_p, model, history = train_model(x_train, y_train,  model, x_test)

y_pred = model.predict(x_test)
y_csv = y_csv_maker(y_pred)
#print(data_train.head())



#%%%%%%

 #% Gender Vecor Encoding
    
    train_data['Gender'] = train_data['Gender'].replace( np.nan,'other')
    train_data['Gender'] = train_data['Gender'].replace('unknown', 'other')
    train_data['Gender'] = train_data['Gender'].replace(0, 'other')

    
    
# =============================================================================
#     test_data['Gender'] = train_data['Gender'].replace( np.nan,'other')
#     test_data['Gender'] = train_data['Gender'].replace('unknown', 'other')
#     #test_data['Gender'] = train_data['Gender'].replace(0, 'other')
# =============================================================================

    
    gender_label_encoder = LabelEncoder()
    train_data['Gender'] = gender_label_encoder.fit_transform(train_data['Gender'].astype(str))
    #test_data['Gender'] = gender_label_encoder.transform(test_data['Gender'])
   # data['Gender'] = gender_label_encoder.fit_transform(data['Gender'])












































    


#%%
y,ycsv = main()

#%%
str1 = "Instance"
str2 = "Income"
arr = [str1,str2]
X = np.insert(y, 0, arr, axis=0)


#%%
np.savetxt("y_pred3.csv", ycsv, delimiter=",")

#%%
nans = []

#%%
for i in range(len(y)):
    if np.isnan(y[i]) == True:
        nans.append(i)
#%%
        
print(len(ycsv))
#%%

nans = lambda df: df[df.isnull().any(axis=1)]

nans(your_dataframe) 

