# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:15:09 2023

@author: Kiost
"""


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
import math
warnings.filterwarnings('ignore')

def calc_pow(x):
    y = math.pow(10, x)
    return y

def y_data(file):
    
    cols = ['a_dm', 'a_g']
    y_train = pd.read_csv(file, index_col=0)


    # 2. ratio a_ph / a_dg
    # ratio = y_train.a_dm / y_train.a_g
    
    return y_train
    
scaler = StandardScaler()

# train and test data load
X_train = pd.read_csv('data_train_input.csv', index_col=0)

X_train_log = np.log10(X_train)
scaled_X_train = scaler.fit_transform(X_train_log)

X_test = pd.read_csv('data_test_input.csv', index_col=0)
X_test_log = np.log10(X_test)
scaled_X_test = scaler.transform(X_test_log)

cols = ['a_dm', 'a_g']
y_train = pd.DataFrame(y_data('data_train_output.csv'))
y_train = y_train[cols]
y_train_log = np.log10(y_train)
scaled_y_train = scaler.fit_transform(y_train_log)

y_test = pd.DataFrame(y_data('data_test_output.csv')).reset_index(drop=True)
y_test = y_test[cols]


def accuracy(y_test, y_pred):

    r2 = np.round(r2_score(y_test, y_pred), 3)
    RMSE = np.round(mean_squared_error(y_test, y_pred)**0.5, 3)
    # MAPE = np.round(np.mean(np.abs((y_test - y_pred) / y_test)) * 100, 3)
    # print(f'Test R2:{r2} RMSE: {RMSE}, MAPE: {MAPE}%')
    
    return r2, RMSE

def NN3_model():

    # define the keras model
    model = Sequential()
    model.add(Input(shape=(9,)))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(125, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    model.add(Dense(2))

    # compile the keras model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    history = model.fit(scaled_X_train, scaled_y_train, epochs=200, batch_size=10, verbose=2)

    # evaluate the keras model
    _, accuracy = model.evaluate(scaled_X_train, scaled_y_train)
    print('Train accuracy: %.2f' % (accuracy*100))

    y_pred = model.predict(scaled_X_test)
    
    # df_ac = accuracy(y_test, y_pred)
    
    # inverse transform and log
    rescaled_y_pred = scaler.inverse_transform(y_pred)
    df_y_pred = pd.DataFrame(rescaled_y_pred)
    df_y_pred = df_y_pred.applymap(calc_pow)

    # df_rslt = pd.DataFrame({'label': y_test, 'R':y_pred.reshape(-1)})
    # df_rslt = pd.DataFrame(
    #     {'label': y_test.values.reshape(-1), 'R': df_y_pred.values.reshape(-1)})
    
    df_rslt = pd.DataFrame({'label_a_d': y_test.a_dm.values.reshape(-1),
                            'label_a_g': y_test.a_g.values.reshape(-1),
                            'pred_a_d': df_y_pred.loc[:, 0].values.reshape(-1),
                            'pred_a_g': df_y_pred.loc[:, 1].values.reshape(-1)})

    
    return df_rslt





def plot_timeSeries(df_rslt):

    # plt.title(f'Test -- RMSE: {RMSE}, \nMAPE: a_pg: {MAPE.a_pg} b_bp: {MAPE.b_bp}%')
    # plt.title(f'$R^2$: {df_ac.r2} RMSE: {df_ac.rmse}, MAPE: {df_ac.mape}%')
    plt.title('NN')
    epochs = range(0, len(df_rslt.label_a_d))
    plt.plot(epochs, df_rslt.label_a_d, label='Label a_d')
    plt.plot(epochs, df_rslt.pred_a_d, linestyle='--', label='Pred a_d')
    plt.plot(epochs, df_rslt.label_a_g, label='Label a_g')
    plt.plot(epochs, df_rslt.pred_a_g, linestyle='--', label='Pred a_g')
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
    
    df_rslt = NN3_model()

    plot_timeSeries(df_rslt)

    # a_pg
    plot_scatter('a_d', df_rslt.label_a_d, df_rslt.pred_a_d)

    # b_bp
    plot_scatter('a_g', df_rslt.label_a_g, df_rslt.pred_a_g)

    
    
    
