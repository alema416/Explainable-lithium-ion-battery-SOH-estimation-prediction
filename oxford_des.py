from scipy.io import loadmat
import scipy.io 
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
import shap
from PyALE import ale
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
            uu = []
            for kkr in gk_curr:
                uu.append(kkr/10)#kkr / 600)
            endol = np.abs(gk_volt - (4.0)).argmin()
            intsum = intsum[:endol]
            gk_volt = gk_volt[:endol]

            
            for k in range(1, len(gk_volt)):
                intsum.append((cumulative_capacity[k] - cumulative_capacity[(k-1)]) / (gk_volt[k] - gk_volt[k-1]))
            for kkk in range(len(intsum)):
                intsum[kkk] *= 200
            #row.append(cumulative_capacity[ind2] - cumulative_capacity[ind1])
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

Dn = pre_proc([5], sigma, factor)
ytrain_scaler6.fit(Dn[['soh']])
xtrain_scaler6.fit(Dn.drop(['soh'], axis=1))

y_train6 = ytrain_scaler6.transform(Dn[['soh']])
X_train6 = xtrain_scaler6.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc([6], sigma, factor)
ytrain_scaler7.fit(Dn[['soh']])
xtrain_scaler7.fit(Dn.drop(['soh'], axis=1))

y_train7 = ytrain_scaler7.transform(Dn[['soh']])
X_train7 = xtrain_scaler7.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc([7], sigma, factor)
ytest_scaler5.fit(Dn[['soh']])
xtest_scaler5.fit(Dn.drop(['soh'], axis=1))

y_test5 = ytest_scaler5.transform(Dn[['soh']])
X_test5 = xtest_scaler5.transform(Dn.drop(['soh'], axis=1))

labelsol = ['time', 'cap', \
                      'ic_max', 'ic_sum']
labelsol = ['F1', 'F2', \
                      'F3', 'F4']

for jj in range(len(X_test5[0])):
  plt.plot(X_test5[:, jj], label=f'{labelsol[jj]}')
plt.ylabel('normalized value')
plt.xlabel('cycle')
plt.legend()
plt.show()

train_X = np.vstack((X_train6, X_train7))
train_y = np.vstack((y_train6, y_train7))
test_X = X_test5
test_y = y_test5

print('{}, {}, {}, {}'.format(train_X.shape, train_y.shape, \
                test_X.shape, test_y.shape))

rf = RandomForestRegressor(
  n_estimators=500,           # Number of trees in the forest
  criterion='squared_error',  # Function to measure the quality of a split
  max_depth = 20, min_samples_split = 2,
  min_samples_leaf = 2)

rf.fit(train_X, train_y.ravel())

'''
rf = LinearRegression()
rf.fit(train_X, train_y.ravel())
'''
y_pred = ytest_scaler5.inverse_transform((rf.predict(test_X)).reshape(-1, 1))
testY = ytest_scaler5.inverse_transform(test_y)

plt.plot(testY, label='real')
plt.plot(y_pred, label='pred')
plt.ylim(0, 1.1)
plt.xlabel('cycle')
plt.ylabel('SOH')
plt.title('OXFORD results - descriptive')
plt.legend()
plt.show()

rmse = 100 * np.sqrt(mean_squared_error(testY, y_pred))
mae = 100 * mean_absolute_error(testY, y_pred)
idd1 = (np.abs(testY - 0.8)).argmin()
idd2 = (np.abs(y_pred - 0.8)).argmin()

print(f'{rmse}, {mae}, {idd1-idd2}')

features = ['time', 'cap', 'ic_max', 'ic_sum']
# start explaining
# shap values
test_X = pd.DataFrame(test_X, columns=features)

cb_explainer = shap.Explainer(rf)
cb_shap = cb_explainer(test_X)
shap.plots.bar(cb_shap, max_display=15)
plt.show()
shap.plots.beeswarm(cb_shap, max_display=15)
plt.show()


shap.plots.partial_dependence("ic_sum", rf.predict, \
        test_X, ice=False, model_expected_value=True,\
          feature_expected_value=True)
ale_effect = ale(X=test_X, model=rf, \
                 feature=['ic_sum'],\
                 feature_type='continuous', grid_size=80)
plt.show()


shap.plots.partial_dependence("ic_max", rf.predict, \
        test_X, ice=False, model_expected_value=True,\
          feature_expected_value=True)
ale_effect = ale(X=test_X, model=rf, \
                 feature=['ic_max'],\
                 feature_type='continuous', grid_size=80)
plt.show()


shap.plots.partial_dependence("cap", rf.predict, \
        test_X, ice=False, model_expected_value=True,\
          feature_expected_value=True)
ale_effect = ale(X=test_X, model=rf, \
                 feature=['cap'],\
                 feature_type='continuous', grid_size=80)
plt.show()


shap.plots.partial_dependence("time", rf.predict, \
        test_X, ice=False, model_expected_value=True,\
          feature_expected_value=True)
ale_effect = ale(X=test_X, model=rf, \
                 feature=['time'],\
                 feature_type='continuous', grid_size=80)
plt.show()


features_l = ["ic_sum", "ic_max"]
ale_effect = ale(X=test_X, model=rf, \
                 feature=features_l,\
                 feature_type='continuous', \
                 grid_size=50, include_CI=False)
plt.show()

