# -*- coding: utf-8 -*-
#Importing train and test data
import pandas as pd

train_data = pd.read_csv('./train.csv')
train_data.shape

test_data = pd.read_csv('./test.csv')
test_data.shape

#Find missing values in train and test data
train_missing_values = train_data.isnull().sum()
test_missing_values = test_data.isnull().sum()

#Drop 55% more missing value columns from train and test data

def drop_missing_value_columns(data):
    for col in data.columns:
        count = data[col].isnull().sum()
        if(count>((55/100)*(len(data.index)))):
            data.drop([col], axis=1, inplace=True)
            
drop_missing_value_columns(train_data)
train_data.drop(['GarageYrBlt'], axis=1, inplace=True)
test_data.drop(['GarageYrBlt'], axis=1, inplace=True)
drop_missing_value_columns(test_data)

#Fill mode for categorical missing values
def fill_mode(data):
    categorical_data = data.select_dtypes(include='object')
    for col in categorical_data:
        count = categorical_data[col].isnull().sum()
        if(count > 0):
            data[col] = data[col].fillna(data[col].mode()[0])

fill_mode(train_data)
fill_mode(test_data)
train_missing_values = train_data.isnull().sum()
test_missing_values = test_data.isnull().sum()

#Fill mean for Integer missing values
def fill_mean(data):
    Integer_data = data.select_dtypes(exclude='object')
    for col in Integer_data:
        count = Integer_data[col].isnull().sum()
        if(count > 0):
            data[col] = data[col].fillna(data[col].mean())
            
fill_mean(train_data)
fill_mean(test_data) 
train_missing_values = train_data.isnull().sum()
test_missing_values = test_data.isnull().sum()    

#Concat train and test data
final_data = pd.concat([train_data, test_data], axis=0)
final_data.drop(['Id'], axis=1, inplace=True)      

#Encoding categorical values in the final dataset
final_data = pd.get_dummies(final_data) 

#Seperate our final train and test data

final_train_data = final_data.iloc[:1460,:]
final_test_data = final_data.iloc[1460:,:]
final_test_data.drop(['SalePrice'], axis=1, inplace=True)

#Seperating train features and train predict
x_train = final_train_data.drop(['SalePrice'], axis=1)
y_train = final_train_data['SalePrice']

#Training a model with hyper tunning n_estimator as 200
import xgboost
xgb_classifier_model = xgboost.XGBRegressor(n_estimators=200)
xgb_classifier_model.fit(x_train, y_train)

#Predicting sale price for final test data
predict=xgb_classifier_model.predict(final_test_data)

#write our predictions in sample_submission csv
get_id = pd.read_csv('./sample_submission.csv')
sale_prediction_dict = {'Id': get_id['Id'], 'SalePrice': predict}
house_sale_predict_df = pd.DataFrame(sale_prediction_dict)
house_sale_predict_df.to_csv('./sample_submission.csv', index=False)
           