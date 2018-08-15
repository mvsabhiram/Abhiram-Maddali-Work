# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection
from sklearn.ensemble import RandomForestRegressor


# Importing the metrics we are going to be using to measure the results of the model.
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


### Performing RNN Required Libraries
import keras
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df= pd.read_csv("/Users/abhirammaddali/Downloads/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv")
df_lable=pd.read_csv("/Users/abhirammaddali/Downloads/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv")
df_final = pd.merge(df,df_lable, how='inner', left_on=['city','year','weekofyear'],right_on=['city','year','weekofyear'])
df_sj= df_final.loc[df_final['city'] == 'sj']
df_iq=df_final.loc[df_final['city'] == 'iq']df_sj_1990= df_final.loc[df_final['year'] == 1990]

df_sj_1991= df_final.loc[df_final['year'] == 1991]
df_sj.groupby('year')['weekofyear'].agg(['count'])

### 2. Explanatory data analysis
sns.heatmap(, annot=True, fmt="g", cmap='viridis')
plt.title("Correlation matrix")
plt.show()
plt.plot(df_sj_1990.weekofyear,df_sj_1990.total_cases)
plt.title("Weekofyear_1990 vs total_cases")
plt.show()
plt.plot(df_sj_1991.weekofyear,df_sj_1991.total_cases)
plt.title("Weekofyear_1991 vs total_cases")
plt.show()
plt.plot(df_sj_1990.weekofyear,df_sj_1990.ndvi_sw)
plt.title("Weekofyear_1990 vs Ndvi_sw")
plt.show()
plt.plot(df_sj_1991.weekofyear,df_sj_1991.ndvi_sw)
plt.title("Weekofyear_1991 vs Ndvi_sw")
plt.show()
plt.plot(df_sj_1990.weekofyear,df_sj_1990.station_avg_temp_c )
plt.title("Weekofyear_1990 vs avgtemp")
plt.show()
plt.plot(df_sj_1991.weekofyear,df_sj_1991.station_avg_temp_c )
plt.title("Weekofyear_1991 vs avgtemp")
plt.show()
plt.plot(df_sj.week_start_date,df_sj.ndvi_sw)
plt.title("Weekofyear_allyears vs ndvi_sw")
plt.show()
print (df_sj.describe())
### Data Cleaning 
print (df_sj.isnull().sum())

df_sj['week_start_date'] = pd.to_datetime(df_sj['week_start_date'])

df_st=df_sj[((df_sj['year']>(1990)) & (df_sj['year'] <(2005)))]

print ("\n",df_st.head())

df_st = df_st.drop(['weekofyear'], axis = 1)
df_st = df_st.drop(['year'], axis = 1)
df_st = df_st.drop(['city'], axis = 1)

#setting index to dates
df_st = df_st.set_index(pd.DatetimeIndex(df_st['week_start_date']))
del df_st['week_start_date']
df_st.head()

df_st.fillna(0,inplace=True)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
# put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df_st)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
#reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print (reframed)

# Splitting the data to training and testing
values = reframed.values
n_train_hours =24*7
train = values[n_train_hours:, :]
test = values[:n_train_hours, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

print(test_X)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

print(test_X)

# design LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()

# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

print (test_X)

# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:scaled.shape[1]]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]


# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:scaled.shape[1]]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


# calculate RMSE
rmse =np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

#### Testing the model with 2008 data 

#loading the data
df_sj= df_final.loc[df_final['city'] == 'sj']
testsj=df_sj.loc[df_sj['year'] == 2008]
print(testsj.head())

#setting the index
testsj = testsj.set_index(pd.DatetimeIndex(testsj['week_start_date']))
del testsj['week_start_date']
testsj.fillna(0,inplace=True)
print(testsj.head())

testsj= testsj.drop(['weekofyear'], axis = 1)
testsj = testsj.drop(['year'], axis = 1)
testsj = testsj.drop(['city'], axis = 1)
#testsj = test1iq.reset_index()
#test1iq = test1iq.drop(['index'], axis = 1)

#converting the data into required form
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_tsj = scaler.fit_transform(testsj)
# frame as supervised learning
reframed_tsj= series_to_supervised(scaled_tsj, 1, 1)
# drop columns we don't want to predict
#reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed_tsj.head())

valuessj = reframed_tsj.values
testsj= valuessj[:n_train_hours, :]
test_Xtsj, test_ytsj = testsj[:, :-1], testsj[:, -1]

test_Xtsj = test_Xtsj.reshape((test_Xtsj.shape[0], 1, test_Xtsj.shape[1]))
print(test_Xtsj.shape, test_ytsj.shape)

yhattesj = model.predict(test_Xtsj)
test_Xtesj = test_Xtsj.reshape((test_Xtsj.shape[0], test_Xtsj.shape[2]))

inv_yhattesj = np.concatenate((yhattesj, test_Xtesj[:, 1:scaled.shape[1]]), axis=1)
inv_yhattesj = scaler.inverse_transform(inv_yhattesj)
inv_yhattesj = inv_yhattesj[:,0]

test_ytesj = test_ytsj.reshape((len(test_ytsj), 1))
inv_yacsj = np.concatenate((test_ytesj, test_Xtesj[:, 1:scaled.shape[1]]), axis=1)
inv_yacsj = scaler.inverse_transform(inv_yacsj)
inv_yacsj = inv_yacsj[:,0]

# calculate RMSE
rmsetesj=np.sqrt(mean_squared_error(inv_yacsj, inv_yhattesj))
print('Test RMSE: %.3f' % rmsetesj)

#Predcition
plt.plot(inv_yhattesj,label='Predicted_Values')
plt.plot(inv_yacsj,label='Actual_Values')
plt.legend()
plt.title('Predicted data vs Actual data for 2008 year')
plt.show()


# Analysis on  Iquitos  city Data

df_iq_1990= df_iq.loc[df_iq['year'] == 2000]
df_iq_1991= df_iq.loc[df_iq['year'] == 2001]
df_iq.groupby('year')['weekofyear'].agg(['count'])

### 2. Explanatory data analysis

plt.plot(df_iq_1990.weekofyear,df_iq_1990.total_cases)
plt.show()
plt.plot(df_iq_1991.weekofyear,df_iq_1991.total_cases)
plt.show()
plt.plot(df_iq_1991.weekofyear,df_iq_1991.ndvi_sw)
plt.show()
plt.plot(df_iq_1990.weekofyear,df_iq_1990.station_avg_temp_c )
plt.show()
plt.plot(df_iq_1991.weekofyear,df_iq_1991.station_avg_temp_c )
plt.show()
plt.plot(df_iq.week_start_date,df_iq.ndvi_sw)
plt.show()
print(df_iq.describe())

df_iq=df_iq[((df_iq['year']>(2001)) & (df_iq['year'] <(2009)))]

df_iq = df_iq.drop(['weekofyear'], axis = 1)
df_iq = df_iq.drop(['year'], axis = 1)
df_iq = df_iq.drop(['city'], axis = 1)

df_iq = df_iq.set_index(pd.DatetimeIndex(df_iq['week_start_date']))
del df_iq['week_start_date']
df_iq.fillna(0,inplace=True)
print(df_iq.head())

scaler = MinMaxScaler(feature_range=(0, 1))
scalediq = scaler.fit_transform(df_iq)
# frame as supervised learning
reframediq = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
#reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframediq.head())

valuesiq = reframediq.values
n_train_hoursiq =24*5
trainiq = valuesiq[n_train_hoursiq:, :]
testiq = valuesiq[:n_train_hoursiq, :]
train_Xiq, train_yiq = trainiq[:, :-1], trainiq[:, -1]
test_Xiq, test_yiq = testiq[:, :-1], testiq[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_Xiq = train_Xiq.reshape((train_Xiq.shape[0], 1, train_Xiq.shape[1]))
test_Xiq = test_Xiq.reshape((test_Xiq.shape[0], 1, test_Xiq.shape[1]))
print(train_Xiq.shape, train_yiq.shape, test_Xiq.shape, test_yiq.shape)
# design network
model2 = Sequential()
model2.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model2.add(Dense(1))
model2.compile(loss='mse', optimizer='adam')
model2.summary()
# fit network
history1 = model2.fit(train_Xiq, train_yiq, epochs=100, batch_size=10, validation_data=(test_Xiq, test_yiq), verbose=2, shuffle=False)
# plot history
plt.plot(history1.history['loss'], label='train')
plt.plot(history1.history['val_loss'], label='test')
plt.legend()
plt.show()
# make a prediction
yhatiq = model.predict(test_Xiq)
test_Xiq = test_Xiq.reshape((test_Xiq.shape[0], test_Xiq.shape[2]))
# invert scaling for forecast
inv_yhatiq = np.concatenate((yhatiq, test_Xiq[:, 1:scaled.shape[1]]), axis=1)
inv_yhatiq = scaler.inverse_transform(inv_yhatiq)
inv_yhatiq = inv_yhatiq[:,0]
# invert scaling for actual
test_yiq = test_yiq.reshape((len(test_yiq), 1))
inv_yiq = np.concatenate((test_yiq, test_Xiq[:, 1:scaled.shape[1]]), axis=1)
inv_yiq = scaler.inverse_transform(inv_yiq)
inv_yiq = inv_yiq[:,0]
# calculate RMSE
rmseiq =np.sqrt(mean_squared_error(inv_yiq, inv_yhatiq))
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_yhatiq,label='Predicted_Values')
plt.plot(inv_yiq,label='Actual_Values')
plt.legend()
plt.show()
#### Testing with outside data
df_iq=df_final.loc[df_final['city'] == 'iq']
test1iq=df_iq.loc[df_iq['year'] == 2009]
print(test1iq.head())
test1iq = test1iq.drop(['weekofyear'], axis = 1)
test1iq = test1iq.drop(['year'], axis = 1)
test1iq = test1iq.drop(['city'], axis = 1)
test1iq = test1iq.reset_index()
test1iq = test1iq.drop(['index'], axis = 1)
test1iq = test1iq.set_index(pd.DatetimeIndex(test1iq['week_start_date']))
del test1iq['week_start_date']
test1iq.fillna(0,inplace=True)
print(test1iq.head())
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_tiq = scaler.fit_transform(test1iq)
# frame as supervised learning
reframed_tiq= series_to_supervised(scaled_tiq, 1, 1)
# drop columns we don't want to predict
#reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
reframed_tiq.head()
valuesiq = reframed_tiq.values
testt= valuesiq[:n_train_hours, :]
test_Xtiq, test_ytiq = testt[:, :-1], testt[:, -1]
print(test_Xtiq)
test_Xtiq = test_Xtiq.reshape((test_Xtiq.shape[0], 1, test_Xtiq.shape[1]))
print(test_Xtiq.shape, test_ytiq.shape)
yhatteiq = model2.predict(test_Xtiq)
test_Xteiq = test_Xtiq.reshape((test_Xtiq.shape[0], test_Xtiq.shape[2]))
inv_yhatteiq = np.concatenate((yhatteiq, test_Xteiq[:, 1:scaled.shape[1]]), axis=1)
inv_yhatteiq = scaler.inverse_transform(inv_yhatteiq)
inv_yhatteiq = inv_yhatteiq[:,0]
test_yteiq = test_ytiq.reshape((len(test_ytiq), 1))
inv_yaciq = np.concatenate((test_yteiq, test_Xteiq[:, 1:scaled.shape[1]]), axis=1)
inv_yaciq = scaler.inverse_transform(inv_yaciq)
inv_yaciq = inv_yaciq[:,0]
# calculate RMSE
rmseteiq =np.sqrt(mean_squared_error(inv_yaciq, inv_yhatteiq))
print('Test RMSE: %.3f' % rmseteiq)
plt.plot(inv_yhatteiq,label='Predicted_Values')
plt.plot(inv_yaciq,label='Actual_Values')
plt.legend()
plt.show()