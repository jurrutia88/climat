#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:19:58 2023

@author: harry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.impute
import geopandas 
import re

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
countries = world[['geometry', 'name']]
countries = countries.rename(columns={'name':'country'}) # cree un dataframe avec les polygones de coordonnées associés à chaque pays.


data_anomalie=pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\Temperature_change_Data (1).csv")
data_anomalie=data_anomalie.loc[data_anomalie.year>=1960, ['year','Country Name', 'tem_change']] # selectionne ce qui nous interesse.
data_anomalie=data_anomalie.rename(columns={'Country Name': 'country', 'tem_change': 'anomalie'})


print(data_anomalie.info()) 
print(data_anomalie.isna().sum()/len(data_anomalie)*100) # pourcentage de nan, ici 11% environ pour l'anomalie temperature.

liste_pays_sans_anomalie=[]

compteur_pays_sans_country=0

new_df=[] # traitement des nan et années manquantes.
années=pd.DataFrame([i for i in range(1960, 2019)], columns=['year'])
for pays in data_anomalie.country.unique():
    df=data_anomalie.loc[data_anomalie.country==pays,:]
    df=df.merge(années, on='year', how='outer')
    df=df.sort_values(by='year')
    df=df.fillna(df.mean(axis=0)) # on comble les nan par la moyenne par pays.
    df.country=df.country.fillna(pays)
    if df.country.isna().any():
        compteur_pays_sans_country+=1
    elif df.anomalie.isna().any():
        liste_pays_sans_anomalie.append(pays)
    
    
    if (pays not in liste_pays_sans_anomalie):
        new_df.append(df)

data_anomalie=pd.concat(new_df)   

country_no_match=[i for i in countries.country.unique() if i not in data_anomalie.country.unique()] # permet de voir les noms de pays qui devront etre changés.
anomalie_no_match=[i for i in data_anomalie.country.unique() if i not in countries.country.unique()]


match=[]
for i in country_no_match:
    for j in anomalie_no_match:
        i2=str(i).lower()
        r=re.compile(r'[a-zA-Z]{4,}')
        i2=r.findall(i2)
        
        j2=str(j).lower()
        q=re.compile(r'[a-zA-Z]{4,}')
        j2=q.findall(j2)
        
        for word in i2:
            for woword in j2:
                if word==woword:
                    match.append([i,j])
  


result=[0, 1, 2, 3, 4, 6, 7, 8, 12, 13, 14, 17, 23, 24, 25, 26, 27, 28, 29] 
keep={}
for i in result:
    keep.update({match[i][1]: match[i][0]})
    
data_anomalie.country=data_anomalie.country.replace(keep)
data_anomalie.country=data_anomalie.country.replace({'Russian Federation': 'Russia'})
 
print(data_anomalie.isna().sum())     
print(compteur_pays_sans_country)

plt.figure(figsize=(10, 10))
plt.title('distribution anomalies températures')
plt.boxplot(data_anomalie.anomalie)
plt.show() # présence de outliers mais ne semblent pas être des valeurs aberrantes pour autant.

data_anomalie.to_csv('data_anomalie_kaggle_traité.csv')
