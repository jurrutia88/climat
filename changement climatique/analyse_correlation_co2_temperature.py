#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:16:46 2023

@author: harry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas 
from scipy.stats import pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

''' on va utiliser geopandas pour ajouter la variable continent pour la visualisation.'''
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world=world.rename(columns={'name': 'country'})
continent=world[['country', 'continent']]

''' on importe nos données.'''
co2=pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\data_co2_traité.csv", index_col=0)
temperature=pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\data_température_traitée.csv", index_col=0)
anomalie=pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\data_anomalie_kaggle_traité.csv", index_col=0)



temperature=temperature.rename(columns={'année': 'year'})
#anomalie=anomalie.rename(columns={'année': 'year'})

''' on les agences pour obtenir un dataframe complet pour notre analyse'''
data=co2.merge(temperature, on=['country', 'year'], how='inner')
data=data.merge(anomalie, on=['country', 'year'], how='inner')
data=data.merge(continent, on='country', how='inner')

''' on obtient un dataframe avec les variables co2, year, country, continent, anomalie, température, coal_co2, oil_co2, gas_co2, land_use_change_co2, cement_co2, flaring_co2 '''

data.loc[data.country=='Russia', 'continent']='Asia' # on classe la Russie en Asie pour être coherent avec le debut de notre analyse.
    

''' on va crée deux dataframes supplementaires pour analyser la correlation à differentes
échelles, mondiale puis par continents et enfin par pays.'''

data_world=pd.DataFrame(data[['year', 'température', 'anomalie', 'co2', 'population', 'coal_co2', 'oil_co2', 'gas_co2', 'land_use_change_co2', 'cement_co2', 'flaring_co2']].groupby('year').agg('mean')).reset_index()
data_continent=pd.DataFrame(data[['year', 'température', 'anomalie', 'co2', 'population', 'continent', 'coal_co2', 'oil_co2', 'gas_co2', 'land_use_change_co2', 'cement_co2', 'flaring_co2']].groupby(['continent', 'year']).agg('mean')).reset_index()

''' avec un pairplot, visualisation des correlations et leur forme (lineaire ou ou pas). '''

#data_world.columns=[name.upper() for name in data_world.columns]
sns.pairplot(data_world)
plt.show() # a l'échelle mondiale on voit bien que c'est lineaire pour l'ensemble des variables.



sns.pairplot(data_continent, hue='continent')
plt.show() # a l'échelle continentale on voit des petits changements, on retrouve de la linearité mais variables par continents.


sns.pairplot(data, hue='continent')
plt.show() # a l'échelle des pays c'est plus complexe.

'''test hypothese entre valeurs quantitatives. '''

''' tests de pearson échelle mondiale'''

coefficient_temperature, p_value_temperature=pearsonr(data_world.température, data_world.co2)
coefficient_anomalie, p_value_anomalie=pearsonr(data_world.anomalie, data_world.co2)
coefficient_population, p_value_population=pearsonr(data_world.population, data_world.co2)
coefficient_year, p_value_year=pearsonr(data_world.year, data_world.co2)

print(f"correlation mondiale co2/temperature coeff:{coefficient_temperature}, pvalue:{p_value_temperature}") # forte correlation positive entre co2 et temperature a 89%.
print(f"correlation mondiale co2/anomalie coeff:{coefficient_anomalie}, pvalue:{p_value_anomalie}") # très forte correlation positive..
print(f"correlation mondiale co2/population coeff:{coefficient_population}, pvalue:{p_value_population}") # très forte correlation positive.
print(f"correlation mondiale co2/year coeff:{coefficient_year}, pvalue:{p_value_year}") # très forte correlation positive.


coefficient_temperature, p_value_temperature=pearsonr(data_continent.température, data_continent.co2)
coefficient_anomalie, p_value_anomalie=pearsonr(data_continent.anomalie, data_continent.co2)
coefficient_population, p_value_population=pearsonr(data_continent.population, data_continent.co2)
coefficient_year, p_value_year=pearsonr(data_continent.year, data_continent.co2)

print(f"correlation continentale co2/temperature coeff:{coefficient_temperature}, pvalue:{p_value_temperature}") # pvalue en dessous du seuil mais coeff negatif.
print(f"correlation continentale co2/anomalie coeff:{coefficient_anomalie}, pvalue:{p_value_anomalie}") # correlation positive à 25%, pvalue sous le seuil.
print(f"correlation continentale co2/population coeff:{coefficient_population}, pvalue:{p_value_population}") # correlation positive à 50% pvalue sous le seuil.
print(f"correlation continentale co2/year coeff:{coefficient_year}, pvalue:{p_value_year}") # correlation positive à 31%, pvalue sous le seuil.


''' tests de pearson échelle par pays'''

coefficient_temperature, p_value_temperature=pearsonr(data.température, data.co2)
coefficient_anomalie, p_value_anomalie=pearsonr(data.anomalie, data.co2)
coefficient_population, p_value_population=pearsonr(data.population, data.co2)
coefficient_year, p_value_year=pearsonr(data.year, data.co2)

print(f"correlation par pays co2/temperature coeff:{coefficient_temperature}, pvalue:{p_value_temperature}") # correlation négative -17%, pvalue sous le seuil.
print(f"correlation par pays co2/anomalie coeff:{coefficient_anomalie}, pvalue:{p_value_anomalie}") # correlation positive 3%, pvalue sous le seuil.
print(f"correlation par pays co2/population coeff:{coefficient_population}, pvalue:{p_value_population}") # correlation positive 64%, pvalue 0.
print(f"correlation par pays co2/year coeff:{coefficient_year}, pvalue:{p_value_year}") # correlation positive 8%, pvalue en dessous seuil.

''' au niveau mondial la tendance est assez claire, ii y a de fortes correlations entre co2, temperature, anomalie, population et année ce qui
indique une augmentation du co2 fort probable pour une augmentation de l'année, population, temperature puisque les correlations sont positives.
Par contre quand on change d echelle geographique , bien que la pvalue soit inferieure au seuil, les correlation deviennent plus faibles voires 
negative pour la tempearture, ce qui suggere une forme des données devenant non lineaire, indiquant des changements selon les régions. En changeant 
d echelle on voit que les correlations les moins impactés sont le co2 et la population et le co2 et l'année, 
indiquant peut etre une tendance générale à une augmentation de la population et du co2.'''

''' visualisation par variable, pour les pays, pour leur evolution entre 1960 et 2019.'''

''' pour chaque pays'''

nom_pays=[]
pente_co2=[]
pente_population=[]
pente_temperature=[]
pente_anomalie=[]

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
    
    co2_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.co2).reshape(-1, 1))
    population_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.population).reshape(-1, 1))
    anomalie_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.anomalie).reshape(-1, 1))
    temperature_model.fit(np.array(df.year).reshape(-1, 1), np.array(df.température).reshape(-1, 1))

    pente_co2.append(co2_model.coef_[0][0])
    pente_population.append(population_model.coef_[0][0])
    pente_anomalie.append(anomalie_model.coef_[0][0])
    pente_temperature.append(temperature_model.coef_[0][0])
    
data_pentes_pays=pd.DataFrame({'country': nom_pays, 'co2': pente_co2, 'anomalie': pente_anomalie, 'temperature': pente_temperature, 'population': pente_population})


data_pentes_pays=data_pentes_pays.merge(world, on='country', how='inner')
data_pentes_pays=geopandas.GeoDataFrame(data_pentes_pays, geometry='geometry')

fig, ax = plt.subplots(1, 1)
plt.title('Evolution emission co2 entre 1960 et 2019')
data_pentes_pays.plot(column='co2', ax=ax, legend=True)
plt.xticks([])
plt.yticks([])
plt.show()


fig, ax = plt.subplots(1, 1)
plt.title('Evolution population entre 1960 et 2019')
data_pentes_pays.plot(column='population', ax=ax, legend=True)
plt.xticks([])
plt.yticks([])
plt.show()



fig, ax = plt.subplots(1, 1)
plt.title('Evolution temperature entre 1960 et 2019')
data_pentes_pays.plot(column='temperature', ax=ax, legend=True)
plt.xticks([])
plt.yticks([])
plt.show()

fig, ax = plt.subplots(1, 1)
plt.title('Evolution anomalie entre 1960 et 2019')
data_pentes_pays.plot(column='anomalie', ax=ax, legend=True)
plt.xticks([])
plt.yticks([])
plt.show()

''' on exporte le dataset total pour faire de la visualisation et du machine learning pour la suite.'''

#data.to_csv("data_temperature_co2_anomalie.csv")
