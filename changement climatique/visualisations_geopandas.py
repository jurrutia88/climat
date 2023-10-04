#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 19:26:33 2023

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
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

''' on va utiliser geopandas pour la visualisation.'''
world = geo.read_file(geo.datasets.get_path('naturalearth_lowres'))
world=world.rename(columns={'name': 'country'})


''' on importe le dataset total.'''
data=pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\data_temperature_co2_anomalie.csv", index_col=0)
debut=data.year.min() # année 1960
fin=data.year.max() # année 2019

"""
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

''' on va tester la stationnarité de la variable dans le temps'''

rolling_mean = temperature.rolling(window = 10).mean()
rolling_std = temperature.rolling(window = 10).std()
plt.plot(temperature, color = 'blue', label = 'Origine')
plt.plot(rolling_mean, color = 'red', label = 'Moyenne mobile')
plt.plot(rolling_std, color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Moyenne et Ecart-type mobiles')
plt.show() # si on obsereve la stationnarité par pas de dix ans on constate une moyenne t un ecart type stable.

''' verification avec le test ADF '''

result = adfuller(temperature.température)
 
print('Statistiques ADF : {}'.format(result[0]))
print('p-value : {}'.format(result[1]))
print('Valeurs Critiques :')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value)) # au-dessus de la p_value donc pas stationnaire, ça colle pas avec la visu d'avant.
    

temperature_log = np.log(temperature)
plt.plot(temperature_log)
plt.show()

''' on refait tests avec log'''

rolling_mean = temperature_log.rolling(window = 10).mean()
rolling_std = temperature_log.rolling(window = 10).std()
plt.plot(temperature_log, color = 'blue', label = 'Origine')
plt.plot(rolling_mean, color = 'red', label = 'Moyenne mobile')
plt.plot(rolling_std, color = 'black', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Moyenne et Ecart-type mobiles')
plt.show()

result = adfuller(temperature_log.température)
 
print('Statistiques ADF : {}'.format(result[0]))
print('p-value : {}'.format(result[1]))
print('Valeurs Critiques :')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value)) # tjrs au-dessus ?
    
#decomposition = seasonal_decompose(temperature_log) 
model = ARIMA(temperature, order=(2,1,1))
results = model.fit(disp=-1)
results.plot_predict(1,140)
plt.xticks([i for i in range(0, 140, 10)], [i for i in range(1960, 2100, 10)], rotation=90)
plt.title('Evolution température mondiale')
plt.show()

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
plt.show()

''' verification avec le test ADF '''

result = adfuller(co2.co2)
 
print('Statistiques ADF : {}'.format(result[0]))
print('p-value : {}'.format(result[1]))
print('Valeurs Critiques :')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value)) # pas stationnaire
    
co2_log=np.log(co2)

''' avec log '''

result = adfuller(co2_log.co2)
 
print('Statistiques ADF : {}'.format(result[0]))
print('p-value : {}'.format(result[1]))
print('Valeurs Critiques :')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value)) # là c'est stationnaire.
    
#decomposition = seasonal_decompose(temperature_log) 
model_2 = ARIMA(co2_log, order=(5,1,5))
results_2 = model_2.fit(disp=-1)
pred_co2=results_2.plot_predict(1,140)
plt.xticks([i for i in range(0, 140, 10)],[i for i in range(1960, 2100, 10)], rotation=90)
plt.yticks([i for i in range(4, 8)], [int(np.exp(i)) for i in range(4,8)])
plt.title('Evolution émission co2 mondiale')
plt.show()

"""

''' visualisations geopandas '''

''' calcul des pentes pour les variables causales du co2'''

''' pour chaque pays'''

nom_pays=[]
pente_co2=[]
pente_population=[]
pente_temperature=[]
pente_anomalie=[]
pente_ciment=[]
pente_gaz=[]
pente_petrole=[]
pente_torchage=[]
pente_terrain=[]
pente_charbon=[]

for pays in data.country.unique():
   
    nom_pays.append(pays)
    df=data.loc[data.country==pays,:]
    
    encoder_country=LabelEncoder()
    encoder_continent=LabelEncoder()
    
    scaler=StandardScaler()
    
    df['continent']=encoder_continent.fit_transform(df['continent'])
    df['country']=encoder_country.fit_transform(df['country'])
    
    scaled_df=scaler.fit_transform(df)
    
    co2_model=LinearRegression()
    population_model=LinearRegression()
    anomalie_model=LinearRegression()
    temperature_model=LinearRegression()
    ciment_model=LinearRegression()
    petrole_model=LinearRegression()
    charbon_model=LinearRegression()
    torchage_model=LinearRegression()
    terrain_model=LinearRegression()
    gaz_model=LinearRegression()
    
    co2_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.co2).reshape(-1, 1))
    population_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.population).reshape(-1, 1))
    anomalie_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.anomalie).reshape(-1, 1))
    temperature_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.température).reshape(-1, 1))
    ciment_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.cement_co2).reshape(-1, 1))
    petrole_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.oil_co2).reshape(-1, 1))
    charbon_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.coal_co2).reshape(-1, 1))
    torchage_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.flaring_co2).reshape(-1, 1))
    terrain_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.land_use_change_co2).reshape(-1, 1))
    gaz_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.gas_co2).reshape(-1, 1))



    pente_co2.append(co2_model.coef_[0][0])
    pente_population.append(population_model.coef_[0][0])
    pente_anomalie.append(anomalie_model.coef_[0][0])
    pente_temperature.append(temperature_model.coef_[0][0])
    pente_ciment.append(ciment_model.coef_[0][0])
    pente_petrole.append(petrole_model.coef_[0][0])
    pente_gaz.append(gaz_model.coef_[0][0])
    pente_charbon.append(charbon_model.coef_[0][0])
    pente_torchage.append(torchage_model.coef_[0][0])
    pente_terrain.append(terrain_model.coef_[0][0])



data_pentes_pays=pd.DataFrame({'country': nom_pays, 'co2': pente_co2, 'anomalie': pente_anomalie, 'temperature': pente_temperature, 'population': pente_population, 
                               'gaz': pente_gaz, 'petrole': pente_petrole, 'charbon': pente_charbon, 'ciment': pente_ciment, 
                               'affectation_terre': pente_terrain, 'torchage': pente_torchage})


data_pentes_pays=data_pentes_pays.merge(world, on='country', how='inner')
data_pentes_pays=geo.GeoDataFrame(data_pentes_pays, geometry='geometry')

'''top 5 pays par variable'''

top_co2=data_pentes_pays[['country', 'co2']].sort_values(by='co2', ascending=False).head(5)
top_temperature=data_pentes_pays[['country', 'temperature']].sort_values(by='temperature', ascending=False).head(5)
top_population=data_pentes_pays[['country', 'population']].sort_values(by='population', ascending=False).head(5)
top_petrole=data_pentes_pays[['country', 'petrole']].sort_values(by='petrole', ascending=False).head(5)
top_gaz=data_pentes_pays[['country', 'gaz']].sort_values(by='gaz', ascending=False).head(5)
top_charbon=data_pentes_pays[['country', 'charbon']].sort_values(by='charbon', ascending=False).head(5)
top_ciment=data_pentes_pays[['country', 'ciment']].sort_values(by='ciment', ascending=False).head(5)
top_affectation_terre=data_pentes_pays[['country', 'affectation_terre']].sort_values(by='affectation_terre', ascending=False).head(5)
top_torchage=data_pentes_pays[['country', 'torchage']].sort_values(by='torchage', ascending=False).head(5)



fig, ax = plt.subplots(1, 1)
plt.title('Evolution émission du co2 entre 1962 et 2019')
data_pentes_pays.plot(column='co2', ax=ax, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'})
plt.xticks([])
plt.yticks([])
plt.axes([0.13, 0.67, 0.1, 0.1])
plt.bar(top_co2.country, top_co2.co2)
plt.xticks( rotation=90)
plt.yticks([])
plt.show()

fig2, ax2=plt.subplots(1, 1)
plt.title('Evolution co2 charbon entre 1962 et 2019 ')
data_pentes_pays.plot(column='charbon', ax=ax2, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'})
plt.yticks([])
plt.xticks([])
plt.axes([0.13, 0.67, 0.1, 0.1])
plt.bar(top_charbon.country, top_charbon.charbon)
plt.xticks( rotation=90)
plt.yticks([])
plt.show()
        
fig3, ax3=plt.subplots(1, 1)  
plt.title('Evolution co2 gaz entre 1962 et 2019')
data_pentes_pays.plot(column='gaz', ax=ax3, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'}) 
plt.yticks([]) 
plt.xticks([])
plt.axes([0.13, 0.67, 0.1, 0.1])
plt.bar(top_gaz.country, top_gaz.gaz)
plt.xticks( rotation=90)
plt.yticks([])
plt.show()

fig4, ax4=plt.subplots(1, 1)
plt.title('Evolution co2 pétrole entre 1962 et 2019')
data_pentes_pays.plot(column='petrole', ax=ax4, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'})
plt.xticks([])
plt.yticks([])
plt.axes([0.13, 0.67, 0.1, 0.1])
plt.bar(top_petrole.country, top_petrole.petrole)
plt.xticks( rotation=90)
plt.yticks([])
plt.show()

fig5, ax5=plt.subplots(1, 1)
plt.title('Evolution co2 ciment entre 1962 et 2019')
data_pentes_pays.plot(column='ciment', ax=ax5, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'})
plt.xticks([])
plt.yticks([])
plt.axes([0.13, 0.67, 0.1, 0.1])
plt.bar(top_ciment.country, top_ciment.ciment)
plt.xticks( rotation=90)
plt.yticks([])
plt.show()

fig6, ax6=plt.subplots(1, 1)
plt.title('Evolution co2 torchage entre 1962 et 2019')
data_pentes_pays.plot(column='torchage', ax=ax6, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'})
plt.xticks([])
plt.yticks([])
plt.axes([0.13, 0.67, 0.1, 0.1])
plt.bar(top_torchage.country, top_torchage.torchage)
plt.xticks( rotation=90)
plt.yticks([])
plt.show()
    
fig7, ax7=plt.subplots(1, 1)
plt.title('Evolution co2 affectation terre entre 1962 et 2019')
data_pentes_pays.plot(column='affectation_terre', ax=ax7, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'})
plt.xticks([])
plt.yticks([])
plt.axes([0.13, 0.67, 0.1, 0.1])
plt.bar(top_affectation_terre.country, top_affectation_terre.affectation_terre)
plt.xticks( rotation=90)
plt.yticks([])
plt.show()

fig8, ax8=plt.subplots(1,  1)
plt.title('Evolution temperature entre 1962 et 2019')
data_pentes_pays.plot(column='temperature', ax=ax8, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'})
plt.yticks([])
plt.xticks([])
plt.axes([0.13, 0.67, 0.1, 0.1])
plt.bar(top_temperature.country, top_temperature.temperature)
plt.xticks( rotation=90)
plt.yticks([])
plt.show()

fig12, ax12=plt.subplots(1,  1)
plt.title('Evolution population entre 1962 et 2019')
data_pentes_pays.plot(column='population', ax=ax12, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'})
plt.yticks([])
plt.xticks([])
plt.axes([0.13, 0.67, 0.1, 0.1])
plt.bar(top_population.country, top_population.population)
plt.xticks( rotation=90)
plt.yticks([])
plt.show()

''' et maintenant par continents '''

data_pentes_continents=data_pentes_pays.dissolve(by='continent', aggfunc='mean')

fig9, ax9=plt.subplots(1, 1)
plt.title('Evolution co2 entre 1962 et 2019')
data_pentes_continents.plot(column='co2', ax=ax9, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'})
plt.xticks([])
plt.yticks([])
plt.show()

fig10, ax10=plt.subplots(1, 1)
plt.title('Evolution population entre 1962 et 2019')
data_pentes_continents.plot(column='population', ax=ax10, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'})
plt.xticks([])
plt.yticks([])
plt.show()

fig11, ax11=plt.subplots(1, 1)
plt.title('Evolution température entre 1962 et 2019')
data_pentes_continents.plot(column='temperature', ax=ax11, legend=True, legend_kwds={'label': 'Indice evolution', 'orientation': 'horizontal'})
plt.xticks([])
plt.yticks([])
plt.show()


