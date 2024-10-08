import datetime
import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn import metrics, ensemble, tree, inspection,\
model_selection
from scipy.stats import pearsonr
import shap
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from keras.optimizers import RMSprop
from PyALE import ale
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
            cumulative_capacity[i] = cumulative_capacity[i-1] + incremental_charge  # Cumulative capacity

        intsum = []

        ind1 = np.abs(gk_volt - (3.9)).argmin()
        ind2 = np.abs(gk_volt - (4.1)).argmin()
        ind_end = np.abs(gk_volt - (4.1)).argmin()

        gk_volt = gaussian_filter1d(gk_volt, sigma)

        row.append(gk_curr[ind2] - gk_curr[ind1])

        for i in range(1, len(gk_volt)):
            intsum.append((cumulative_capacity[i] - cumulative_capacity[(i-1)]) / (gk_volt[i] - gk_volt[i-1]))
        for kkk in range(len(intsum)):
          intsum[kkk] /= 50
        row.append(cumulative_capacity[ind_end])
        row.append(np.max(intsum[:ind_end]))
        row.append(sum(intsum[ind1:ind_end]))
        dataf.append(row)

    cols = []
    for kj in range(4):
      cols.append(f'f{kj+1}')
    
    out1 = pd.DataFrame(dataf, columns=['time', 'cap', 'ic_max', 'ic_sum'])

    out1['soh'] = capacity['capacity']
    return out1



maxes = {'B0007': 0.945526147695395, 'B0006': 1.017668795502799, \
         'B0005': 0.9282437104090787, 'B0018': 0.9275022603955408}
mins = {'B0007': 0.7002276199533257, \
        'B0006': 0.576909165798125, 'B0005': 0.6437262610689704, \
        'B0018': 0.6731154382004401}

sample = 0.016
sigma = 2

rand = 9
os.environ['PYTHONHASHSEED']=str(rand)
tf.random.set_seed(rand)
np.random.seed(rand)

trains = ['B0007', 'B0006']

tests = ['B0005']

ytrain_scaler6 = MinMaxScaler()
xtrain_scaler6 = MinMaxScaler()
ytest_scaler5 = MinMaxScaler()
xtest_scaler5 = MinMaxScaler()
ytrain_scaler7 = MinMaxScaler()
xtrain_scaler7 = MinMaxScaler()


Dtr = pd.DataFrame()
Dte = pd.DataFrame()

Dn = pre_proc('B0005', sample, sigma)
ytrain_scaler6.fit(Dn[['soh']])
xtrain_scaler6.fit(Dn.drop(['soh'], axis=1))

y_train6 = ytrain_scaler6.transform(Dn[['soh']])
X_train6 = xtrain_scaler6.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc('B0007', sample, sigma)
ytrain_scaler7.fit(Dn[['soh']])
xtrain_scaler7.fit(Dn.drop(['soh'], axis=1))

y_train7 = ytrain_scaler7.transform(Dn[['soh']])
X_train7 = xtrain_scaler7.transform(Dn.drop(['soh'], axis=1))

Dn = pre_proc('B0006', sample, sigma)
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
'''
rf = LinearRegression()
rf.fit(train_X, train_y.ravel())
'''


rf = RandomForestRegressor(
  n_estimators=500,           
  criterion='squared_error',  
  max_depth = 20, min_samples_split = 2,
  min_samples_leaf = 2)

rf.summary()

rf.fit(train_X, train_y.ravel())

y_pred = ytest_scaler5.inverse_transform((rf.predict(test_X)).reshape(-1, 1))
testY = ytest_scaler5.inverse_transform(test_y)
'''
plt.plot(testY, label='real')
plt.plot(y_pred, label='pred')
plt.xlabel('cycle')
plt.ylim(0, 1.1)
plt.ylabel('SOH')
plt.title('NASA results - descriptive ')
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


shap.plots.partial_dependence("ic_max", rf.predict, \
        test_X, ice=False, model_expected_value=True,\
          feature_expected_value=True)
ale_effect = ale(X=test_X, model=rf, \
                 feature=['ic_max'],\
                 feature_type='continuous', grid_size=80)
plt.show()
shap.plots.partial_dependence("ic_sum", rf.predict, \
        test_X, ice=False, model_expected_value=True,\
          feature_expected_value=True)
ale_effect = ale(X=test_X, model=rf, \
                 feature=['ic_sum'],\
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

features_l = ["ic_max", "cap"]
ale_effect = ale(X=test_X, model=rf, \
                 feature=features_l,\
                 feature_type='continuous', \
                 grid_size=50, include_CI=False)
plt.show()
# end explaining

