import pandas as pd
import os
import numpy as np
from scipy.stats import mstats
import csv
from scipy.stats import pearsonr
import datetime
import numpy as np
import tensorflow as tf
from matplotlib.colors import TwoSlopeNorm
from alibi.explainers import IntegratedGradients
import shap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
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
        oxo = []
        #voltage = gaussian_filter1d(voltage, sigma)

        for k in range(1, len(voltage)):
            oxo.append((cumulative_capacity[k] - cumulative_capacity[(k-1)]) / (voltage[k] - voltage[k-1]))

        ind1 = np.abs(voltage - (3.8) ).argmin()
        ind2 = np.abs(voltage - (4.0) ).argmin()
        oxo = savgol_filter(oxo, 9, 3)
        if np.isnan(oxo).any() == True:
            #print(f'{num}, {i}: {np.isnan(oxo).any()}')
            continue
        if len(charge_n) > 2 and abs(( np.max(charge) / 1.1 ) - charge_n[-1]) > 0.1:
            continue
        charge_n.append((np.max(charge)) / 1.1)

        #plt.plot(voltage[:-1], oxo)
        row.append(time[ind2] - time[ind1])
        row.append(cumulative_capacity[ind2])
        #row.append((cumulative_capacity[ind2] - cumulative_capacity[ind1])/100000)
        row.append(np.max(oxo))
        row.append(sum(oxo[ind1:ind2]))
        dataf.append(row)
        #plt.plot(voltage[:-1], oxo)
    aka = []
    out1 = pd.DataFrame(dataf, \
            columns=['time', 'cap', 'ic_max', 'ic_sum'])
    out1['soh'] = charge_n
    return out1
    #plt.show()

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

Dn = pre_proc(7, sample)
ytrain_scaler6.fit(Dn[['soh']])
xtrain_scaler6.fit(Dn.drop(['soh'], axis=1))

y_train6 = ytrain_scaler6.transform(Dn[['soh']])
X_train6 = xtrain_scaler6.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc(5, sample)
ytrain_scaler7.fit(Dn[['soh']])
xtrain_scaler7.fit(Dn.drop(['soh'], axis=1))

y_train7 = ytrain_scaler7.transform(Dn[['soh']])
X_train7 = xtrain_scaler7.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc(6, sample)

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
lng = 4
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

