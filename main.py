# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:30:04 2023

@author: Kiost
"""

import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from sklearn.metrics import r2_score, mean_squared_error
from modelNN3 import NN3_model
from modelNN2 import NN2_model
from modelNN1 import NN1_model
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')



def plot_scatter(title, labels, preds):

    v_max, v_min = max(labels), min(labels)

    fig, ax = plt.subplots()
    plt.title(title+' 440')

    r2 = np.round(r2_score(labels, preds), 3)
    RMSE = np.round(mean_squared_error(labels, preds)**0.5, 3)
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





# NN-I
df_nn1 = NN1_model()
label_a_pg = df_nn1.label_a_pg
label_b_bp = df_nn1.label_b_bp

pred_a_pg = df_nn1.pred_a_pg
pred_b_bp = df_nn1.pred_b_bp

# Level 1
plot_scatter('a_pg', label_a_pg, pred_a_pg)
plot_scatter('b_bp', label_b_bp, pred_b_bp)


# NN-II
df_nn2 = NN2_model()
pred_ratio_1 = df_nn2.R
label_ratio_1 = df_nn2.label

# Level 2
label_a_ph = label_a_pg / (1 + 1/label_ratio_1)
label_a_dg = label_a_pg - label_a_ph

pred_a_ph = pred_a_pg / (1 + 1/pred_ratio_1)
pred_a_dg = pred_a_pg - pred_a_ph

plot_scatter('a_ph', label_a_ph, pred_a_ph)
plot_scatter('a_dg', label_a_dg, pred_a_dg)



# NN-III
df_nn3 = NN3_model()
pred_ratio_2 = df_nn3.R
label_ratio_2 = df_nn3.label

# Level 3
# return a_d and a_g
label_a_d = label_a_dg / (1 + 1/label_ratio_2)
label_a_g = label_a_dg - label_a_d

pred_a_d = pred_a_dg / (1 + 1/pred_ratio_2)
pred_a_g = pred_a_dg - label_a_g

plot_scatter('a_d', label_a_d, pred_a_d)
plot_scatter('a_g', label_a_g, pred_a_g)





