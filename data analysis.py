# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:09:15 2023

@author: Kiost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# excel file read
num = 60
f = f'IOP_AOP_Sun{num} - v2.xls'

sheets = pd.ExcelFile(f).sheet_names[1:]


for i, sheet in enumerate(sheets):
    globals()['df_' + sheet] = pd.read_excel(f, 
                                             sheet_name = sheet,
                                             usecols='B:AP',
                                             header=8)

def plot(ind=1):
    
    df = pd.DataFrame(None)
    for i, sheet in enumerate(sheets):
        df = pd.concat([df, globals()['df_' + sheet].loc[[ind]]])
    
    df.index = sheets

    # plot
    plt.title(f'IOP_AOP_Sun{num} - Point {ind}', pad=10)
    plt.plot(df.transpose())
    plt.grid(True)
    plt.xlabel('Wavelength (nm)')
    plt.legend(sheets, bbox_to_anchor=(1.05, 1.05), loc='upper left')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'figure/Sun{num}/IOP_AOP_Sun{num}_target{ind+1}.png')

for i in range(500):
    plot(i)
    
    
# df_a plot
def plot_500(var):
    
    plt.title(f'IOP_AOP_Sun{num} - {var}', pad=10)
    plt.plot(globals()['df_' + sheet][:500].transpose())
    plt.grid(True)
    plt.xlabel('Wavelength (nm)')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'figure/plot_all/IOP_AOP_Sun{num}_{var}.png')
    plt.show()
    

for i, var in enumerate(sheets):
    print(var)
    plot_500(var)
