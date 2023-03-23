# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:15:09 2023

@author: Kiost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# excel file read
num = 30
f = f'IOP_AOP_Sun{num} - v2.xls'

df_basic = pd.read_excel(f, sheet_name='Basics', header=6, 
                         usecols='A:C')
df_basic = df_basic.dropna()
df_basic['Unnamed: 0'] = df_basic['Unnamed: 0'].astype(int)
df_basic = df_basic.set_index(keys=['Unnamed: 0'])

a_w = df_basic[['a_w']].transpose()
bb_w = df_basic[['bb_w']].transpose()

sheets = pd.ExcelFile(f).sheet_names[1:]

nms = [410, 440, 490, 510, 550, 620, 670, 680, 710]

for i, sheet in enumerate(sheets):
    globals()['df_' + sheet] = pd.read_excel(f, 
                                             sheet_name = sheet,
                                              # names = 'B:AP',
                                             # usecols=[410],
                                             header=8)
                                             # nrows=1500)
    globals()['df_' + sheet] = globals()['df_' + sheet].dropna(axis='columns')

# nms
df = pd.DataFrame(None)
for i, sheet in enumerate(sheets):
    data = globals()['df_' + sheet][nms]
    data.insert(0,'name',sheet)
    print(sheet)
    df = df.append(data)

df.set_index(df['name'], inplace=True, drop=True)


## Level-1 a_pg and bb_p
df_a_pg = df_a_ph + df_a_dm + df_a_g


# a = a_ph + a_g + a_dm + a_w
# print(df_a[400][0])
# a = df_a_ph[400][0] + df_a_g[400][0] + df_a_dm[400][0] + a_w.loc[400][0]

# bb = bb_p + bb_w

df_bb_p = pd.DataFrame(None)
for i in range(len(df_bb)):
    df_bb_p = df_bb_p.append(df_bb.loc[i] - bb_w)

df_bb_p.index = np.arange(0,len(df_bb_p))


## nm 443
df_a_pg_interp = pd.DataFrame(None, columns=(420,440,443,450,460))
cols = [420,440,450,460]
df_a_pg_interp[cols] = df_a_pg[cols]

df_a_pg_interp = df_a_pg_interp.apply(pd.to_numeric)
df_a_pg_interp = df_a_pg_interp.transpose().interpolate()


df_bb_p_interp = pd.DataFrame(None, columns=(420,440,443,450,460))
df_bb_p_interp[cols] = df_bb_p[cols]
df_bb_p_interp = df_bb_p_interp.apply(pd.to_numeric)
df_bb_p_interp = df_bb_p_interp.transpose().interpolate()

# model input and output data
# model input ; Rrs above-surface remote-sensing reflectance, Lw(0+)/Ed(0+)
# model output ; a_pg(443) and b_bp(443)

cols_input = [410,440,490,510,550,620,670,680,710]
df_input = df_Rrs[cols_input]

df_output_a_pg = df_a_pg_interp.loc[[443]].transpose()
df_output_a_pg = df_output_a_pg.rename(columns={443:'a_pg'})
df_output_b_bp = df_bb_p_interp.loc[[443]].transpose()
df_output_b_bp = df_output_b_bp.rename(columns={443:'b_bp'})

df_output = pd.concat([df_output_a_pg, df_output_b_bp], axis=1)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_input, df_output, test_size=0.2, shuffle=False)


# 데이터 정규화 추가 
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1))
# X_train_scaled = scaler.fit_transform(X_train)


# print(f'원본 train data:\n{X_train.loc[0].tolist()}')
# print(f'Scaled된 train data:\n{X_train_scaled[0]}')


# define the keras model
model = Sequential()
model.add(Input(shape=(9,)))
model.add(Dense(6, activation='tanh'))
model.add(Dense(2))

# compile the keras model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
history = model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=2)

# evaluate the keras model
_, accuracy = model.evaluate(X_train, y_train)
print('Train accuracy: %.2f' % (accuracy*100))

accuracy_train = history.history['accuracy']
loss_train = history.history['loss']
epochs = range(0,150)

fig, ax1 = plt.subplots()
line1 = ax1.plot(epochs, loss_train, 'g', label='loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epochs')


ax2 = ax1.twinx()
line2 = ax2.plot(epochs, accuracy_train, 'b', label='accuracy')
ax2.set_ylabel('Accuracy')


lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels)

plt.title('Training loss and accuracy')
plt.show()



# model test

# X_test_scaled = scaler.transform(X_test)
# print(f'원본 test data:\n{X_test.loc[400].tolist()}')
# print(f'Scaled된 test data:\n{X_test_scaled[0]}')

y_pred = model.predict(X_test)


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
RMSE = np.round(mean_squared_error(y_test, y_pred)**0.5, 4)
MAPE = np.round(mean_absolute_percentage_error(y_test, y_pred), 4)
print(f'Test RMSE: {RMSE}, MAPE: {MAPE}%')


plt.title(f'Test -- RMSE: {RMSE}, MAPE: {MAPE}%')
epochs = range(0,len(X_test))
plt.plot(epochs, y_test, label='Label data')
plt.plot(epochs, y_pred, linestyle='--', label='Pred data')
plt.grid()
plt.legend()
plt.show()






