import pandas as pd
import os
import numpy as np
from scipy.stats import mstats
import csv
from scipy.stats import pearsonr
import datetime
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import shap
from PyALE import ale
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter
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
import pywt
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def min_subarray_length(arr):
    min_length = float('inf')  
    for sub_arr in arr:
        current_length = len(sub_arr)
        if current_length < min_length:
            min_length = current_length
    return min_length

def stop_voltage(voltage):
    voltage_n = []
    f = 1
    temp = 0
    for k in range(len(voltage)):
        if f == 1:
            if ( 4.2 - voltage[k] ) > 0.01:
                voltage_n.append(voltage[k])
            else:
                f = 0
                temp = voltage[k]
                voltage_n.append(temp)
        else:
            voltage_n.append(temp)
    return voltage_n

def index_closest_to_number(lst, target):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))

def pre_proc(num, factor):
    folder_path = 'cs2_3{}'.format(num)

    all_dfs = []
    dataf = []
    charge_n = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, sep=';')
        
            if all_dfs:
                prev_cycle_num = all_dfs[-1]['Cycle_Index'].max()
                df['Cycle_Index'] += prev_cycle_num
            else:
                prev_cycle_num = 0
        
            all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    grouped_merged_df = merged_df.groupby('Cycle_Index')

    for i in range(2, len(grouped_merged_df)):
        row = []
        cycle_data = grouped_merged_df.get_group(i)
        cycle_data.reset_index(drop=True, inplace=True)

        time = cycle_data['Test_Time(s)']
        voltage = cycle_data['Voltage(V)']
        current = cycle_data['Current(A)']     

        discharge = cycle_data['Discharge_Capacity(Ah)']

        time = time.values
        first = time[0]
        for j in range(len(time)):
            time[j] -= first

        stopin = len(voltage)

        indexof1000time = np.abs(time - 1000).argmin()

        if len(voltage) < indexof1000time or voltage[indexof1000time] > 4.1:
            continue
        charge = cycle_data['Charge_Capacity(Ah)']
        charge = charge.values

        first = charge[0]
        for j in range(len(charge)):
            charge[j] -= first

        for kk in range(len(voltage)):
            if abs(4.2 - voltage[kk]) < 0.01:
                stopin = kk
                break
        voltage = voltage[:stopin]
        time = time[:stopin]
        voltage = voltage[::factor]
        time = time[::factor]

        voltage = np.array(voltage)
        time = np.array(time)
        I = 0.55
        cumulative_capacity = np.zeros_like(voltage)

        for k in range(1, len(voltage)):
            dt = time[k] - voltage[k-1]  # Time step
            incremental_charge = I * dt / 3600# Incremental charge (using trapezoidal rule)
            cumulative_capacity[k] = cumulative_capacity[k-1] + incremental_charge  # Cumulative capacity
        intsum = []
        #voltage = gaussian_filter1d(voltage, sigma)

        for k in range(1, len(voltage)):
            intsum.append((cumulative_capacity[k] - cumulative_capacity[(k-1)]) / (voltage[k] - voltage[k-1]))

        ind1 = np.abs(voltage - (3.8) ).argmin()
        ind2 = np.abs(voltage - (4.0) ).argmin()
        intsum = savgol_filter(intsum, 9, 3)
        if np.isnan(intsum).any() == True:
            #print(f'{num}, {i}: {np.isnan(intsum).any()}')
            continue
        if len(charge_n) > 2 and abs(( np.max(charge) / 1.1 ) - charge_n[-1]) > 0.1:
            continue
        charge_n.append((np.max(charge)) / 1.1)

        row.append(time[ind2] - time[ind1])
        row.append(cumulative_capacity[ind2])
        #row.append((cumulative_capacity[ind2] - cumulative_capacity[ind1])/100000)
        row.append(np.max(intsum))
        row.append(sum(intsum[ind1:ind2]))
        dataf.append(row)
        #plt.plot(voltage[:-1], intsum)
    aka = []
    out1 = pd.DataFrame(dataf, \
            columns=['time', 'cap', 'ic_max', 'ic_sum'])
    out1['soh'] = charge_n
    return out1

sample = 1

rand = 9
os.environ['PYTHONHASHSEED']=str(rand)
tf.random.set_seed(rand)
np.random.seed(rand)

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

Dn = pre_proc(6, sample)
ytrain_scaler6.fit(Dn[['soh']])
xtrain_scaler6.fit(Dn.drop(['soh'], axis=1))

y_train6 = ytrain_scaler6.transform(Dn[['soh']])
X_train6 = xtrain_scaler6.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc(7, sample)
ytrain_scaler7.fit(Dn[['soh']])
xtrain_scaler7.fit(Dn.drop(['soh'], axis=1))

y_train7 = ytrain_scaler7.transform(Dn[['soh']])
X_train7 = xtrain_scaler7.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc(5, sample)

for f in ['time', 'cap', 'ic_max', 'ic_sum']:
  corr, _ = pearsonr(Dn[f], Dn['soh'])
  print(f'{f}: {corr} %')

ytest_scaler5.fit(Dn[['soh']])
xtest_scaler5.fit(Dn.drop(['soh'], axis=1))

y_test5 = ytest_scaler5.transform(Dn[['soh']])
X_test5 = xtest_scaler5.transform(Dn.drop(['soh'], axis=1))
labelsol = ['time', 'cap', \
                      'ic_max', 'ic_sum']
labelsol = ['F1', 'F2', \
                      'F3', 'F4']
'''
for jj in range(len(X_test5[0])):
  plt.plot(X_test5[:, jj], label=f'{labelsol[jj]}')
plt.ylabel('normalized value')
plt.xlabel('cycle')
plt.legend()
plt.show()
'''
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

'''
plt.plot(testY, label='real')
plt.plot(y_pred, label='pred')
plt.ylim(0, 1.1)
plt.xlabel('cycle')
plt.ylabel('SOH')
plt.title('CALCE results - descriptive')
plt.legend()
plt.show()
'''
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





features_l = ["time", "ic_max"]
ale_effect = ale(X=test_X, model=rf, \
                 feature=features_l,\
                 feature_type='continuous', \
                 grid_size=50, include_CI=False)
plt.show()
# end explaining


