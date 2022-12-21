import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#import waipy

#from datetime import datetime

#def parse(x):
#	return datetime.strptime(x, '%Y %m %d %H')
dataset = pd.read_csv('testset.csv',index_col=0)
#dataset.drop('No', axis=1, inplace=True)
# manually specify column names

#dataset=dataset.set_index('datetime')
#dataset['datetime']=pd.to_datetime(dataset['datetime'],format='%Y%m%d-%H:%M')

dataset.columns = ['condensation', 'dew', 'fog', 'hail', 'ht_index', 'hum', 'prec','pressure','rain','snow','temperature','thunder','tornado','vis','wspeed','wdir']
# mark all NA values with 0
dataset['ht_index'].fillna(0, inplace=True)
dataset['prec'].fillna(0, inplace=True)
dataset['vis'].fillna(0, inplace=True)
# drop the first 24 hours
dataset #= dataset[24:]
# summarize first 5 rows
print(dataset.head(30))
# save to file
dataset.to_csv('C:\\Users\\Atith\\Downloads\\pollution1.csv') 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
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
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('C:\\Users\\Atith\\Downloads\\pollution1.csv', header=0, index_col=0)
#dataset=dataset.drop('condensation',axis=1)
values = dataset.values
#dataset['condensation'] = dataset['condensation'].astype('|S')
dataset.dtypes
# integer encode direction
encoder = LabelEncoder()

print(values[:,0])
values[:,0] = encoder.fit_transform(values[:,0].astype(str))
print(values[:,0])
values[:,15] = encoder.fit_transform(values[:,15].astype(str))

# ensure all data is float
values = values.astype('float32')
#fft

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[16,17,18,19,20,21,22,23,24,25,27,28,29,30,31]], axis=1, inplace=True)
print(reframed.head())
x2=reframed['var1(t-1)']
reframed['var1(t-1)']=reframed['var11(t-1)']
reframed['var11(t-1)']=x2
# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

#p=model.predict(100)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)



pyplot.plot(inv_y[0:300], color = 'red', label = 'Temperature')
pyplot.plot(inv_yhat[0:300], color = 'blue', label = 'Predicted Temperature')
pyplot.title('Temperature Prediction')
pyplot.xlabel('Time')
pyplot.ylabel('Temperature')
pyplot.legend()
pyplot.show()



 