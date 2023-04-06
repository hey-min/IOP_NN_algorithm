# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:15:09 2023

@author: Kiost
"""


import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.offsetbox as offsetbox
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def calc_pow(x):
    y = math.pow(10, x)
    return y


scaler = StandardScaler()

# train and test data load
X_train = pd.read_csv('data_train_input.csv', index_col=0)

X_train_log = np.log10(X_train)
scaled_X_train = scaler.fit_transform(X_train_log)

X_test = pd.read_csv('data_test_input.csv', index_col=0)
X_test_log = np.log10(X_test)
scaled_X_test = scaler.transform(X_test_log)

cols = ['a_pg', 'bb_p']
y_train = pd.read_csv('data_train_output.csv', index_col=0)
y_train = y_train[cols]
y_train_log = np.log10(y_train)
scaled_y_train = scaler.fit_transform(y_train_log)

y_test = pd.read_csv('data_test_output.csv', index_col=0)
y_test = y_test[cols]

# desnormalize and pow 10 
# y_test_log = np.log10(y_test)
# scaled_y_test = scaler.transform(y_test_log)
# rescaled_y_test = scaler.inverse_transform(scaled_y_test)
# df_y_test = pd.DataFrame(rescaled_y_test)
# df_y_test = df_y_test.applymap(calc_pow)


def accuracy(y_test, y_pred):

    r2 = np.round(r2_score(y_test, y_pred), 3)
    RMSE = np.round(mean_squared_error(y_test, y_pred)**0.5, 3)
    # MAPE = np.round(np.mean(np.abs((y_test - y_pred) / y_test)) * 100, 3)
    print(f'Test R2:{r2} RMSE: {RMSE}')

    return r2, RMSE


def NN1_model():

    # define the keras model
    model = Sequential()
    model.add(Input(shape=(9,)))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(2))

    # compile the keras model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    history = model.fit(scaled_X_train, scaled_y_train,
                        epochs=200, batch_size=10, verbose=2)

    # evaluate the keras model
    _, accuracy = model.evaluate(scaled_X_train, scaled_y_train)
    print('Train accuracy: %.2f' % (accuracy*100))

    y_pred = model.predict(scaled_X_test)
    

    # inverse transform and log
    rescaled_y_pred = scaler.inverse_transform(y_pred)
    df_y_pred = pd.DataFrame(rescaled_y_pred)
    df_y_pred = df_y_pred.applymap(calc_pow)
    
    df_rslt = pd.DataFrame({'label_a_pg': y_test.a_pg.values.reshape(-1),
                            'label_b_bp': y_test.bb_p.values.reshape(-1),
                            'pred_a_pg': df_y_pred.loc[:, 0].values.reshape(-1),
                            'pred_b_bp': df_y_pred.loc[:, 1].values.reshape(-1)})

    return df_rslt


# Plot - label and pred data
def plot_timeSeries(df_rslt):

    # plt.title(f'Test -- RMSE: {RMSE}, \nMAPE: a_pg: {MAPE.a_pg} b_bp: {MAPE.b_bp}%')
    # plt.title(f'$R^2$: {df_ac.r2} RMSE: {df_ac.rmse}, MAPE: {df_ac.mape}%')
    plt.title('NN-I')
    epochs = range(0, len(df_rslt.label_a_pg))
    plt.plot(epochs, df_rslt.label_a_pg, label='Label a_pg')
    plt.plot(epochs, df_rslt.pred_a_pg, linestyle='--', label='Pred a_pg')
    plt.plot(epochs, df_rslt.label_b_bp, label='Label b_bp')
    plt.plot(epochs, df_rslt.pred_b_bp, linestyle='--', label='Pred b_bp')
    plt.grid()
    plt.legend()
    plt.show()


def plot_scatter(title, labels, preds):

    v_max, v_min = max(labels), min(labels)

    fig, ax = plt.subplots()
    plt.title(title+' 443')

    r2, RMSE = accuracy(labels, preds)
    num = len(labels)
    txt = f'$R^2$: {r2}\nRMSE: {RMSE}\nN={num}'
    textbox = offsetbox.AnchoredText(txt, loc='upper left')
    ax.add_artist(textbox)
    # plt.text(0, (v_max-v_min)/2, f'$R^2$: {r2}\nRMSE: {RMSE}\nMAPE: {MAPE}%')

    plt.plot([0, max(labels)], [0, max(labels)], 'k--', color='r')
    plt.scatter(labels, preds, c='b', s=3)
    plt.xlabel("Measured")
    plt.ylabel('Predicted')
    plt.grid()

    plt.show()
    plt.close()


if __name__ == "__main__":

    df_rslt = NN1_model()

    plot_timeSeries(df_rslt)

    # a_pg
    plot_scatter('a_pg', df_rslt.label_a_pg, df_rslt.pred_a_pg)

    # b_bp
    plot_scatter('b_bp', df_rslt.label_b_bp, df_rslt.pred_b_bp)
