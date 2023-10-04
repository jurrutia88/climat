#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:26:13 2023

@author: harry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset
import geopandas 
from shapely.geometry import Point
from sklearn.linear_model import LinearRegression

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
countries = world[['geometry', 'name']]
countries = countries.rename(columns={'name':'country'})
#countries['country']=countries['country'].str.lower()

#co2=pd.read_csv('data_co2_traité.csv')
#co2=co2.drop('Unnamed: 0', axis=1)

data=Dataset('/home/harry/projet datascientest/fichiers csv/climate change knowledge portal/timeseries-tas-annual-mean_cru_annual_cru-ts4.06-timeseries_mean_1901-2021.nc')

print(data.variables.keys())

df_1=data.variables['time'][:]
df_2=data.variables['lat'][:]
df_3=data.variables['lon'][:]
df_4=data.variables['timeseries-tas-annual-mean'][:]
df_5=data.variables['bnds'][:]
df_6=data.variables['lon_bnds'][:]
df_7=data.variables['lat_bnds'][:]




échantillon=df_4[120] # une tranche du tenseur df_4 indique la température annuelle moyenne pour une année (ici 2020).

def tenseur_to_carts(array=df_4, dates=[1960, 2021], pas=1, zoom=[-90, 90, -180, 180], temperature_ou_anomalie='température'):
    
    
    if temperature_ou_anomalie=='anomalie':
        
    
        for an in range(array.shape[0]): # permet de convertir les températures en anomalies de températures.
            if an<30:
                pass
            else:
                array[an,:,:]= array[an,:,:]-np.mean(array[an-30:an,:,:], axis=0)
            
    
    
    elif temperature_ou_anomalie=='température':
        pass
    
    years=[] # va permettre de faire un dataframe à 4 variables.
    latitudes=[]
    longitudes=[]
    températures=[]
    
    lat_nan=[] # permet d avoir un dataframe de nan pour voir où sont les données manquantes.
    lon_nan=[]
    year_nan=[]
    
    for year in range(dates[0]-1901 , dates[1]-1901, pas): # chaque tranche du tenseur est convertie en dataframe indexé sur la latitude et colonnes longitudes.
        df=pd.DataFrame(array[year])
        df=df.set_index(df_2)
        df.columns=df_3
        
        
        for col in df.columns: # si la valeur n'est pas nan, elle est changée en échantillon dans un dataframe. Sinon va dans data_nan.
            for line in df[col].index:
                
                if pd.isna(df.loc[line, col])==True:
                    lat_nan.append(line)
                    lon_nan.append(col)
                    year_nan.append(1901+year)
                else:
                    longitudes.append(col)
                    latitudes.append(line)
                    years.append(1901+year)
                    températures.append(df.loc[line, col])
    
    data=pd.DataFrame({'année': years, 'latitude': latitudes, 'longitude': longitudes, 'anomalie_température': températures})
    data=data.loc[(data.latitude>=zoom[0]) & (data.latitude<=zoom[1]) & (data.longitude>=zoom[2]) & (data.longitude<=zoom[3])]
    geometry = [Point(xy) for xy in zip(data.longitude, data.latitude)]
    data=data.drop(['latitude', 'longitude'], axis=1)
    data = geopandas.GeoDataFrame(data, crs="EPSG:4326", geometry=geometry)
    data=data.sjoin(countries, how='inner', predicate='intersects')
    data=pd.DataFrame(data)
    data=data.drop(['geometry', 'index_right'], axis=1)
    data=pd.DataFrame(data.groupby(['country', 'année']).agg('mean')).reset_index()
    data=data.rename(columns={'anomalie_température': temperature_ou_anomalie})
    
    
    data_nan=pd.DataFrame({'année': year_nan,'latitude': lat_nan, 'longitude': lon_nan })
    data_nan=data_nan.loc[(data_nan.latitude>=zoom[0]) & (data_nan.latitude<=zoom[1]) & (data_nan.longitude>=zoom[2]) & (data_nan.longitude<=zoom[3])]
    geometry= [Point(xy) for xy in zip(data_nan.longitude, data_nan.latitude)]
    data_nan=data_nan.drop(['latitude', 'longitude'], axis=1)
    data_nan = geopandas.GeoDataFrame(data_nan, crs="EPSG:4326", geometry=geometry)
    data_nan=data_nan.sjoin(countries, how='inner', predicate='intersects')
    data_nan=pd.DataFrame(data_nan)
    data_nan=data_nan.drop(['geometry', 'index_right'], axis=1)
    data_nan=pd.DataFrame(data_nan.groupby(['country']).agg('count')).reset_index()
    data_nan=data_nan.rename(columns={'année': 'valeurs_manquantes'})
    
    contraste=data.sort_values(by=['country', 'année'], ascending=True)
    
    pays_nom=[]
    pays_pente=[]
    
    for pays in contraste.country.unique():
        
        évolution_tempé=contraste.loc[contraste.country==pays,['année', temperature_ou_anomalie]].sort_values(by='année', ascending=True)
        linear_model=LinearRegression()
        linear_model.fit(np.array(évolution_tempé.année).reshape(-1, 1) , np.array(évolution_tempé[temperature_ou_anomalie]))
        pays_nom.append(pays)
        pays_pente.append(linear_model.coef_[0])
         
    data_pentes=pd.DataFrame({'country': pays_nom, f'pente_{temperature_ou_anomalie}': pays_pente})
    
    return data, data_nan, data_pentes

un, deux, trois=tenseur_to_carts(temperature_ou_anomalie='température')
print(deux)

plt.figure(figsize=(10, 10)) # vérifier présence de valeurs aberrantes.
plt.title('distribution des températures')
plt.boxplot(un.température)
plt.show() # outliers mais pas de valeurs aberrantes à priori.
# nan surtout en Antarctique, puis en Russie, Canada en troisième position mais pas énorme, et nan absorbés par la moyenne par pays.
un.to_csv('data_température_traitée.csv')
#deux.to_csv('data_nan_temperatures.csv')
#trois.to_csv(f'data_pente_anomalie.csv')