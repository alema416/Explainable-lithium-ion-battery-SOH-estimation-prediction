from scipy.io import loadmat
import scipy.io 
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from alibi.explainers import IntegratedGradients
import datetime
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.io import loadmat
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
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import math
import shap
import torch
from scipy.ndimage import gaussian_filter1d
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Model
import torch.nn as nn
from scipy.stats import pearsonr
from scipy.stats import skew
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pandas as pd
import os
import numpy as np
from scipy.stats import mstats
import csv
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
corrs = []


def pre_proc(auta, sigma, factor):

    mat = loadmat('Oxford_Battery_Degradation_Dataset_1.mat')

    input_data_q = [[], [], [], [], [], [], [], []]         # ok

    pre_features = []
    dataf = []
    
    CELL_SIZE = [83, 78, 82, 52, 49, 51, 82, 82]

    for i in auta:#range(0, 8): 
        cell_num = "Cell{}".format(i + 1)
        for j in range(0, CELL_SIZE[i]):
            cyc_num = "cyc{:04d}".format(j * 100)
            try:
                cur_q = mat[cell_num][0][cyc_num][0][0]["C1ch"][0][0]['q'][0].tolist()
                flattened_list_q = [item for sublist in cur_q for item in sublist]

                cur_v = mat[cell_num][0][cyc_num][0][0]["C1ch"][0][0]['v'][0].tolist()
                flattened_list_v = [item for sublist in cur_v for item in sublist]

                cur_T = mat[cell_num][0][cyc_num][0][0]["C1ch"][0][0]['T'][0].tolist()
                flattened_list_T = [item for sublist in cur_T for item in sublist]

                cur_time = mat[cell_num][0][cyc_num][0][0]["C1ch"][0][0]['t'][0].tolist()
                flattened_list_time = [item for sublist in cur_time for item in sublist]

            except ValueError:
                curr = float("NaN")
            
            # soh
            row = []

            input_data_q[i].append(flattened_list_q[-1] / 740)
            #cycles
            first = flattened_list_time[0]

            for k in range(len(flattened_list_time)):
                flattened_list_time[k] = flattened_list_time[k] - first
            gk_volt = np.array(flattened_list_v)
            gk_curr = np.array(flattened_list_time)

            I = 0.740

            #print(average_difference(gk_volt))
            new_voltage = []
            cumulative_capacity = np.zeros_like(gk_volt)
        
            for k in range(1, len(gk_volt)):
                dt = gk_curr[k] - gk_curr[k-1]  # Time step
                incremental_charge = I * dt  # Incremental charge (using trapezoidal rule)
                cumulative_capacity[k] = cumulative_capacity[k-1] + incremental_charge  # Cumulative capacity
            gk_volt = gk_volt[::factor]
            gk_curr = gk_curr[::factor]
            

            intsum = []
            for kkk in range(len(gk_curr)):
                gk_curr[kkk] *= 1000000
            whattime = np.abs(gk_curr - (5000)).argmin()
            ind1 = np.abs(gk_volt - (3.75)).argmin()
            ind2 = np.abs(gk_volt - (3.95)).argmin()
            ind_end = np.abs(gk_volt - (3.95)).argmin()

            
            gk_volt = gaussian_filter1d(gk_volt, sigma)
            #row.append(gk_volt[whattime])
            row.append(gk_curr[ind2] - gk_curr[ind1])

            endol = np.abs(gk_volt - (4.0)).argmin()

            intsum = intsum[:endol]
            gk_volt = gk_volt[:endol]

            
            for k in range(1, len(gk_volt)):
                intsum.append((cumulative_capacity[k] - cumulative_capacity[(k-1)]) / (gk_volt[k] - gk_volt[k-1]))
            for kkk in range(len(intsum)):
                intsum[kkk] *= 200
            row.append(cumulative_capacity[ind_end])

            #for kk in range(N):
            #  ind = np.abs( gk_volt - (3.75 + kk * 0.2 / N) ).argmin()
            #  row.append(intsum[ind])
            ind1 = np.abs(gk_volt - (3.75)).argmin()
            ind2 = np.abs(gk_volt - (3.95)).argmin()
            row.append(np.max(intsum))
            row.append(sum(intsum[ind1:ind2]))
            dataf.append(row)
        
        cols = []
        for kj in range(4):
            cols.append(f'f{kj+1}')

        #out1 = pd.DataFrame(dataf, columns=cols)

        out1 = pd.DataFrame(dataf, columns=['time', 'cap', 'ic_max', 'ic_sum'])
        out1['soh'] = input_data_q[i]
        #print(auta[0]+1)
        #aka = []
        #for column in out1.columns:
        #  if column != 'soh':
            #print(column)
            #print(out1[column].corr(out1['soh']))
            #aka.append(out1[column].corr(out1['soh']))
        #globals()['corrs'].append(aka)

        #for column in out1.columns:
        #    if column != 'soh':
        #        plt.plot(out1.index, out1[column], label=out1[column].corr(out1['soh']))
        #plt.plot(out1.index, out1['soh'], label='soh')

        #plt.legend()
        #plt.show()
        return out1

maxes = {'0': 0.978541745147728, '1': 0.9712720242236986, \
         '2': 0.9713015399053075, '3': 0.9744817356524126, \
         '4': 0.9735213754237313, '5': 0.9684562377305221, \
         '6':0.9640287493534672, '7': 0.962103625422366 }

mins = {'0': 0.7154826193406617, '1': 0.6425156245451873, \
         '2': 0.724309780923847, '3': 0.750777605597799, \
         '4': 0.7843708316174206, '5': 0.7580770946478309, \
         '6':0.7474074160743461, '7': 0.7117930509141985 }

rand = 9
os.environ['PYTHONHASHSEED']=str(rand)
tf.random.set_seed(rand)
np.random.seed(rand)

sigma = 2
factor = 10

trains = [6, 7]
tests = [5]


ytrain_scaler6 = MinMaxScaler()
xtrain_scaler6 = MinMaxScaler()
ytest_scaler5 = MinMaxScaler()
xtest_scaler5 = MinMaxScaler()
ytrain_scaler7 = MinMaxScaler()
xtrain_scaler7 = MinMaxScaler()

Dtr = pd.DataFrame()
Dte = pd.DataFrame()

Dn = pre_proc([6], sigma, factor)
ytrain_scaler6.fit(Dn[['soh']])
xtrain_scaler6.fit(Dn.drop(['soh'], axis=1))

y_train6 = ytrain_scaler6.transform(Dn[['soh']])
X_train6 = xtrain_scaler6.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc([7], sigma, factor)
ytrain_scaler7.fit(Dn[['soh']])
xtrain_scaler7.fit(Dn.drop(['soh'], axis=1))

y_train7 = ytrain_scaler7.transform(Dn[['soh']])
X_train7 = xtrain_scaler7.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc([5], sigma, factor)
ytest_scaler5.fit(Dn[['soh']])
xtest_scaler5.fit(Dn.drop(['soh'], axis=1))

y_test5 = ytest_scaler5.transform(Dn[['soh']])
X_test5 = xtest_scaler5.transform(Dn.drop(['soh'], axis=1))

#train_X = np.vstack((X_train6, X_train7))
#train_y = np.vstack((y_train6, y_train7))
#test_X = X_test5
#test_y = y_test5





lng = 11
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

plt.plot(testY[lng:], label='real')
plt.plot(y_pred, label='pred')
plt.ylim(0, 1.1)
plt.legend()
plt.show()

rmse = 100 * np.sqrt(mean_squared_error(testY[lng:], y_pred))
mae = 100 * mean_absolute_error(testY[lng:], y_pred)
idd1 = (np.abs(testY[lng:] - 0.8)).argmin()
idd2 = (np.abs(y_pred - 0.8)).argmin()

print(f'{rmse}, {mae}, {idd1-idd2}')

ig = IntegratedGradients(grid_model, n_steps=25, \
                     internal_batch_size=lng)
images = []

for cyclen in [20, 25, 35]:
  input_data = np.array(gen_test[cyclen][0], dtype=np.float64)

    # Convert input data to TensorFlow tensor with the correct dtype
  input_data = tf.convert_to_tensor(input_data, dtype=tf.float64)

    # Check data type and shape
  print("Input data dtype:", input_data.dtype)
  print("Input data shape:", input_data.shape)

  explanation = ig.explain(gen_test[cyclen][0], target=None)
  attributions = explanation.attributions[0]
  attribution_img = np.transpose(attributions[0,:,:])
  plt.title('Integrated Gradient Attribution Map for cycle {} prediction'.\
            format(cyclen), fontsize=16)
  divnorm = TwoSlopeNorm(vmin=attribution_img.min(),\
                         vcenter=0,vmax=attribution_img.max())
  plt.imshow(attribution_img,interpolation='nearest' ,\
             aspect='auto',cmap='coolwarm_r',norm=divnorm)

  plt.xticks([u for u in range(lng)], labels=\
             [f'lag_{lng-i}' for i in range(lng)])
  plt.yticks([*range(4)], labels=['time', 'cap', 'ic_max', 'ic_sum'])
  plt.colorbar(pad=0.01,fraction=0.02,anchor=(1.0,0.0))
  #plt.savefig(f'cycle_{j}.png')
  plt.show()
