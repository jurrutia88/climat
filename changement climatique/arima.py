#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 22:02:34 2023

@author: harry
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as geo
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import datetime
import warnings

''' on va utiliser geopandas pour la visualisation.'''
world = geo.read_file(geo.datasets.get_path('naturalearth_lowres'))
world=world.rename(columns={'name': 'country'})


''' on importe le dataset total.'''
data=pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\data_temperature_co2_anomalie.csv", index_col=0)
debut=data.year.min() # année 1960
fin=data.year.max() # année 2019


''' ARIMA'''

''' température '''

poly_température=PolynomialFeatures(degree=2)
model_température=LinearRegression()

temperature=data[['year', 'température']].groupby('year').agg('mean')

train_poly=poly_température.fit_transform(np.array(temperature.index).reshape(-1, 1))

model_température.fit(train_poly, temperature.température)


plt.figure(figsize=(20,10))
plt.title('Evolution de température moyenne mondiale entre 1962 et 2019')
plt.plot(temperature.index, temperature.température, label='courbe températures')
plt.plot(temperature.index, model_température.predict(train_poly), label='régression poly degree deux')
plt.legend()
plt.show() # l évolution semble prendre la courbure d'un polynome de degré deux, bref une légère hausse en s approchant de 2019.

''' on prepare le dataset en passant les données temporelles en index au format datetime '''

temperature.index=pd.date_range('1960-06-01', '2020-06-01', freq='y')

''' visualisation de la decomposition de la temperature '''

decompose_temperature=seasonal_decompose(temperature)
decompose_temperature.plot()
plt.show() # pas de saisonnalité car données par ans, pas par mois.


''' on va tester la stationnarité de la variable dans le temps'''

rolling_mean = temperature.rolling(window = 10).mean()
rolling_std = temperature.rolling(window = 10).std()
plt.plot(temperature, color = 'blue', label = 'Origine')
plt.plot(rolling_mean, color = 'red', label = 'Moyenne mobile')
plt.plot(rolling_std, color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Moyenne et Ecart-type mobiles')
plt.show() # si on obsereve la stationnarité par pas de dix ans on constate une moyenne légèrement montante et un ecart type en revanche stable.


''' verification avec le test ADF '''

result = adfuller(temperature.température)
 
print('Statistiques ADF : {}'.format(result[0]))
print('p-value : {}'.format(result[1]))
print('Valeurs Critiques :')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value)) # au-dessus de la p_value donc pas stationnaire, expliqué par la tendance de la variable qui monte, et la moyenne mobile en légère hausse.
    
''' on va donc utiliser le modele ARIMA pour pouvoir modeliser notre variable non stationnaire, sans saisonnalité.'''
  
''' on va utiliser diff pour voir si en différenciant on peut stationnariser'''

temperature_diff=temperature.diff().dropna()
plt.title(" Variable température différenciée ")
plt.plot(temperature_diff)
plt.show()

pd.plotting.autocorrelation_plot(temperature_diff) # autocorrelation de la variable temperature différencié ordre 1 tend vers zero.

result_diff = adfuller(temperature_diff.température)
 
print('Statistiques ADF : {}'.format(result_diff[0]))
print('p-value : {}'.format(result_diff[1]))
print('Valeurs Critiques :')
for key, value in result_diff[4].items():
    print('\t{}: {}'.format(key, value)) # pvalue en-dessous du seuil, donc en differentiant une fois c est stationnaire.


''' recherche des meilleurs parametres un et trois sur les 20 dernieres années.'''

train_data = temperature[1:len(temperature)-20]
test_data = temperature[len(temperature)-20:]

p_values = range(0, 2) # grille de recherche des trois parametres arima.
d_values = [1] # differentiation ordre 1.
q_values = range(0, 2)

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            warnings.filterwarnings("ignore")
            model = ARIMA(train_data, order=order).fit()
            predictions = model.predict(start=len(train_data), end=len(train_data) + len(test_data)-1)
            error = mean_squared_error(test_data, predictions)
            print('ARIMA%s MAE=%.3f' % (order,error)) # meilleurs parametres 1:1:0.
            


model = ARIMA(temperature, order=(1,1,0))
results = model.fit(disp=-1)
results.plot_predict(1,150)
plt.xticks([i for i in range(0, 140, 10)], [i for i in range(1960, 2100, 10)], rotation=90)
plt.title('Evolution température mondiale')
plt.show() # augmentation de temperature de 1,5 degrés en 2100 environ au niveau de la tendance.



''' co2 '''

poly_co2=PolynomialFeatures(degree=2)
model_co2=LinearRegression()

co2=data[['year', 'co2']].groupby('year').agg('mean')

train_poly=poly_co2.fit_transform(np.array(co2.index).reshape(-1, 1))

model_co2.fit(train_poly, co2.co2)


plt.figure(figsize=(20,10))
plt.title('Evolution de température moyenne mondiale entre 1960 et 2019')
plt.plot(co2.index, co2.co2, label='courbe co2')
plt.plot(co2.index, model_co2.predict(train_poly), label='régression poly degree deux')
plt.legend()
plt.show()


''' on va tester la stationnarité de la variable dans le temps'''

rolling_mean = co2.rolling(window = 10).mean()
rolling_std = co2.rolling(window = 10).std()
plt.plot(co2, color = 'blue', label = 'Origine')
plt.plot(rolling_mean, color = 'red', label = 'Moyenne mobile')
plt.plot(rolling_std, color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Moyenne et Ecart-type mobiles')
plt.show() # ecart type a l air stationnaire, mais pas moyenne mobile.

''' verification avec le test ADF '''

result_co2 = adfuller(co2.co2)
 
print('Statistiques ADF : {}'.format(result_co2[0]))
print('p-value : {}'.format(result_co2[1]))
print('Valeurs Critiques :')
for key, value in result_co2[4].items():
    print('\t{}: {}'.format(key, value)) # pas stationnaire, au dessus du seuil.
  
''' du coup on va differentier une fois '''

co2_diff=co2.diff().dropna()


result_co2_2 = adfuller(co2_diff.co2)
 
print('Statistiques ADF : {}'.format(result_co2_2[0]))
print('p-value : {}'.format(result_co2_2[1]))
print('Valeurs Critiques :')
for key, value in result_co2_2[4].items():
    print('\t{}: {}'.format(key, value)) # là c'est stationnaire.
  
''' recherche des meilleurs parametres sur les 20 dernieres années ''' 

train_data_co2 = co2[1:len(co2)-20]
test_data_co2 = co2[len(co2)-20:]

p_values = range(0, 2) # grille de recherche des trois parametres arima.
d_values = [1] # differeentiation ordre 1.
q_values = range(0, 2)

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            warnings.filterwarnings("ignore")
            model_co2 = ARIMA(train_data_co2, order=order).fit()
            predictions_co2 = model_co2.predict(start=len(train_data_co2), end=len(train_data_co2) + len(test_data_co2)-1)
            error = mean_squared_error(test_data_co2, predictions_co2)
            print('ARIMA%s MAE=%.3f' % (order,error)) # meilleurs parametres 1:1:1.
            
  
    

model_co2 = ARIMA(co2, order=(1,1,1))
results_co2 = model_co2.fit(disp=-1)
pred_co2=results_co2.plot_predict(1,140)
plt.xticks([i for i in range(0, 140, 10)],[i for i in range(1960, 2100, 10)], rotation=90)
plt.title('Evolution émission co2 mondiale')
plt.show()

