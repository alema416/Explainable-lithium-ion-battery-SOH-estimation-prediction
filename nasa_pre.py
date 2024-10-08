import datetime
import os, random
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn import metrics, ensemble, tree, inspection, model_selection
from matplotlib.colors import TwoSlopeNorm
from alibi.explainers import IntegratedGradients
import shap
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from pyswarm import pso
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from scipy.io import loadmat
from PyALE import ale
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, max_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pywt
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

rand = 42

os.environ['PYTHONHASHSEED']=str(rand)
np.random.seed(rand)
random.seed(rand)
corrs = []
# in: name
# out: SOH dataframe

def load_data_y(battery):
  mat = loadmat('battery_data/' + battery + '.mat')
  counter = 0
  capacity_data = []
  
  for i in range(len(mat[battery][0, 0]['cycle'][0])):
    row = mat[battery][0, 0]['cycle'][0, i]
    if row['type'][0] == 'discharge':
      data = row['data']
      capacity = data[0][0]['Capacity'][0][0]
      capacity_data.append([counter + 1, capacity / 2])
      counter = counter + 1
  return pd.DataFrame(data=capacity_data,
                       columns=['cycle', 'capacity'])
# in: name
# out: V, time dataframe
def load_data(battery):
  mat = loadmat('battery_data/' + battery + '.mat')
  counter = 0
  dataset = []
  
  for i in range(len(mat[battery][0, 0]['cycle'][0])):
    row = mat[battery][0, 0]['cycle'][0, i]
    if row['type'][0] == 'charge':
      date_time = datetime.datetime(int(row['time'][0][0]),
                               int(row['time'][0][1]),
                               int(row['time'][0][2]),
                               int(row['time'][0][3]),
                               int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
      data = row['data']
      for j in range(len(data[0][0]['Voltage_measured'][0])):
        voltage_measured = data[0][0]['Voltage_measured'][0][j]
        current_measured = data[0][0]['Current_measured'][0][j]
        temperature_measured = data[0][0]['Temperature_measured'][0][j]
        time = data[0][0]['Time'][0][j]
        dataset.append([counter + 1,
                        voltage_measured, time])
      counter = counter + 1
  return pd.DataFrame(data=dataset,
                       columns=['cycle',
                                'voltage_measured', 'time'])

# in: name, sample_rate, sigma, N
# out: train-ready dataframe
def pre_proc(name, sample_rate, sigma):
    dataset = load_data(name)
    capacity = load_data_y(name)

    labels = []
    dataf = []
    nones = []

    grouped = dataset.groupby('cycle', as_index=False)

    # Step 2: Iterate over groups or access specific groups
    for group_name, group_df in grouped:
        gk_volt = group_df.iloc[:, group_df.columns.get_loc('voltage_measured')].values
        gk_curr = group_df.iloc[:, group_df.columns.get_loc('time')].values
        row = []

        # discard outliers
        stopin = len(gk_volt)
        indexof800time = np.abs(gk_curr - 800).argmin()

        if len(gk_volt) < indexof800time or gk_volt[indexof800time] > 4.183:
            nones.append(int(group_name) - 1)
            continue

        for kk in range(len(gk_volt)):
            if abs(4.2 - gk_volt[kk]) < 0.01:
                stopin = kk
                break
        
        gk_volt = gk_volt[2:stopin]
        gk_curr = gk_curr[2:stopin]
        
        def average_difference(lst):
            differences = [abs(lst[i] - lst[i+1]) for i in range(len(lst)-1)]
            average_diff = sum(differences) / len(differences)
            return average_diff
        def resample_list(lst, old_interval, new_interval):
            factor = int(new_interval / old_interval)
    
            resampled_values = lst[::factor]
    
            return resampled_values
        for i in range(len(gk_curr)):
            gk_curr[i] /= 60 
        #plt.scatter(gk_curr, gk_volt, label=f'old: {len(gk_volt)}')
        gk_curr = resample_list(gk_curr, average_difference(gk_volt), sample_rate)

        gk_volt = resample_list(gk_volt, average_difference(gk_volt), sample_rate)
        #plt.scatter(gk_curr, gk_volt, label=f'new: {len(gk_volt)}')
        # calculate IC curve 
        I = 1.510
        new_voltage = []
        cumulative_capacity = np.zeros_like(gk_volt)

        for i in range(1, len(gk_volt)):
            dt = gk_curr[i] - gk_curr[i-1]  # Time step
            incremental_charge = I * dt  # Incremental charge (using trapezoidal rule)
            cumulative_capacity[i] = cumulative_capacity[i-1] + \
                                     incremental_charge  # Cumulative capacity
        intsum = []

        ind1 = np.abs(gk_volt - (3.9)).argmin()
        ind2 = np.abs(gk_volt - (4.1)).argmin()
        ind_end = np.abs(gk_volt - (4.1)).argmin()

        gk_volt = gaussian_filter1d(gk_volt, sigma)

        row.append(gk_curr[ind2] - gk_curr[ind1])   # ok

        for i in range(1, len(gk_volt)):
            intsum.append((cumulative_capacity[i] - cumulative_capacity[(i-1)]) / (gk_volt[i] - gk_volt[i-1]))

        for kkk in range(len(intsum)):
          intsum[kkk] /= 50
        #row.append(cumulative_capacity[ind2] - \
        #           cumulative_capacity[ind1])
        row.append(cumulative_capacity[ind_end])
        row.append(np.max(intsum[:ind_end]))   # ok
        #plt.plot(gk_volt[:ind_end], intsum[:ind_end])
        #plt.plot(gk_volt[:ind_end], intsum[:ind_end])
        #plt.plot(gk_volt, cumulative_capacity)
        row.append(sum(intsum[ind1:ind_end]))      # ok
        dataf.append(row)
    cols = []
    #plt.title('cumulative capacity curves')
    #plt.xlabel('voltage')
    #plt.ylabel('Q~')
    #plt.show()
    for kj in range(4):
      cols.append(f'f{kj+1}')
    
    out1 = pd.DataFrame(dataf, columns=['time', 'cap', \
                                        'ic_max', 'ic_sum'])

    out1['soh'] = capacity['capacity']
    aka = []
    return out1

sample = 0.016
sigma = 2
totest = 'B0005'
rand = 9
os.environ['PYTHONHASHSEED']=str(rand)
tf.random.set_seed(rand)
np.random.seed(rand)

trains = ['B0006', 'B0007', 'B0005']
trains.remove(totest)
tests = [totest]

ytrain_scaler6 = MinMaxScaler()
xtrain_scaler6 = MinMaxScaler()
ytest_scaler5 = MinMaxScaler()
xtest_scaler5 = MinMaxScaler()
ytrain_scaler7 = MinMaxScaler()
xtrain_scaler7 = MinMaxScaler()

Dtr = pd.DataFrame()
Dte = pd.DataFrame()

Dn = pre_proc('B0006', sample, sigma)
ytrain_scaler6.fit(Dn[['soh']])
xtrain_scaler6.fit(Dn.drop(['soh'], axis=1))

y_train6 = ytrain_scaler6.transform(Dn[['soh']])
X_train6 = xtrain_scaler6.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc('B0007', sample, sigma)
ytrain_scaler7.fit(Dn[['soh']])
xtrain_scaler7.fit(Dn.drop(['soh'], axis=1))

y_train7 = ytrain_scaler7.transform(Dn[['soh']])
X_train7 = xtrain_scaler7.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc('B0005', sample, sigma)
ytest_scaler5.fit(Dn[['soh']])
xtest_scaler5.fit(Dn.drop(['soh'], axis=1))

y_test5 = ytest_scaler5.transform(Dn[['soh']])
X_test5 = xtest_scaler5.transform(Dn.drop(['soh'], axis=1))
labelsol = ['time', 'cap', \
                      'ic_max', 'ic_sum']
'''
for jj in range(len(X_test5[0])):
  plt.plot(X_test5[:, jj], label=f'{labelsol[jj]}')
plt.ylabel('normalized value')
plt.xlabel('cycle')
plt.legend()
plt.show()
'''
lng = 7
gen_train = TimeseriesGenerator(X_train7,\
                                y_train7,length=lng,batch_size=1)
gen_test = TimeseriesGenerator(X_test5, \
                               y_test5, length=lng,\
                               batch_size=1)
print("gen_train:%s×%s→%s" % (len(gen_train),\
                              gen_train[0][0].shape, gen_train[0][1].shape))
print("gen_test:%s×%s→%s" % (len(gen_test),\
                             gen_test[0][0].shape, gen_test[0][1].shape))

grid_model = Sequential()
grid_model.add(LSTM(50,return_sequences=True,input_shape=(lng,4)))
grid_model.add(LSTM(50))
grid_model.add(Dropout(0.2))
grid_model.add(Dense(1))

grid_model.compile(loss = 'mse', optimizer = 'adam')
grid_model.fit(gen_train, epochs=100, verbose=0)

'''
grid_model = Sequential()
grid_model.add(SimpleRNN(50, input_shape=(lng, 4), return_sequences=False))
grid_model.add(Dense(1))

# Step 4: Compile the model
grid_model.compile(optimizer='adam', loss='mse')

# Step 5: Train the model
grid_model.fit(gen_train, epochs=20, batch_size=32, verbose=0)
'''
y_pred = ytest_scaler5.inverse_transform(grid_model.predict(gen_test, \
                                                    verbose=0))
testY = ytest_scaler5.inverse_transform(gen_test.targets)
'''
plt.plot(testY[lng:], label='real')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()
'''
rmse = 100 * np.sqrt(mean_squared_error(testY[lng:], y_pred))
mae = 100 * mean_absolute_error(testY[lng:], y_pred)
idd1 = (np.abs(testY[lng:] - 0.8)).argmin()
idd2 = (np.abs(y_pred - 0.8)).argmin()

print(f'{rmse}, {mae}, {idd1-idd2}')
ig = IntegratedGradients(grid_model, n_steps=25, \
                     internal_batch_size=lng)
images = []
for cyclen in [50, 70, 80]:
  print(gen_test[cyclen][0].dtype)

  explanation = ig.explain(gen_test[cyclen][0], target=None)
  attributions = explanation.attributions[0]
  attribution_img = np.transpose(attributions[0,:,:])
  plt.title('Integrated Gradient Attribution Map for cycle {} prediction'.\
            format(cyclen), fontsize=16)
  divnorm = TwoSlopeNorm(vmin=attribution_img.min(),vcenter=0,vmax=attribution_img.max())
  plt.imshow(attribution_img,interpolation='nearest' ,aspect='auto',cmap='coolwarm_r',norm=divnorm)

  plt.xticks([u for u in range(lng)], labels=\
             [f'lag_{lng-i}' for i in range(lng)])
  plt.yticks([*range(4)], labels=['time', 'cap', 'ic_max', 'ic_sum'])
  plt.colorbar(pad=0.01,fraction=0.02,anchor=(1.0,0.0))
  #plt.savefig(f'cycle_{j}.png')
  plt.show()
