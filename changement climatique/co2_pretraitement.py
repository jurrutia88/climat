#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:21:00 2023

@author: harry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.impute
import re



data_co2=pd.read_csv(r"C:\Users\jt_ur\OneDrive\Documents\DataScientest\Projet\Datasets\owid-co2-data.csv")

''' on va voir si on peut imputer correctement la variable méthane.'''

print(data_co2.methane.isna().sum()/len(data_co2.methane)) # pourcentage de nan pour le méthane égal à 86%. Si le peu de valeurs est dispersé on peut peut etre faire quelque chose.
distri_methane=data_co2[['methane', 'year']].groupby(['year']).agg('count').reset_index() # ce dataframe nous indique que la variable methane commence en 1990, pas avant.
pays_methane=data_co2.loc[data_co2.methane>0,:]
print(pays_methane.country.nunique()) # seulement 202 pays ont une variable methane exploitable. Les données inexistantes avant 1990 et les pays n'ayant pas du tout de données dessus, nous décidons de ne pas garder cette variable.
        

data_co2=data_co2.loc[data_co2.year>=1960, ['year','country','co2', 'population', 'coal_co2', 'oil_co2', 'gas_co2', 'land_use_change_co2', 'cement_co2', 'flaring_co2']] # selectionne ce qui nous interesse.




print(data_co2.info()) 
print(data_co2.isna().sum()/len(data_co2)*100) # pourcentage de nan par variable, entre 0 et 11% environ.

liste_pays_sans_co2=[]
liste_pays_sans_population=[]
liste_pays_sans_charbon=[]
liste_pays_sans_pétrole=[]
liste_pays_sans_gaz=[]
liste_pays_sans_affectation_terres=[]
liste_pays_sans_ciment=[]
liste_pays_sans_torchage=[]
compteur_pays_sans_country=0

new_df=[] # traitement des nan et années manquantes.
années=pd.DataFrame([i for i in range(1960, 2022)], columns=['year'])
for pays in data_co2.country.unique():
    df=data_co2.loc[data_co2.country==pays,:]
    df=df.merge(années, on='year', how='outer')
    df=df.sort_values(by='year')
    df=df.fillna(df.mean(axis=0)) # on comble les nan par la moyenne par pays.
    df.country=df.country.fillna(pays)
    if df.country.isna().any():
        compteur_pays_sans_country+=1
    elif df.co2.isna().any():
        liste_pays_sans_co2.append(pays)
    elif df.population.isna().any():
        liste_pays_sans_population.append(pays)
    elif df.coal_co2.isna().any():
        liste_pays_sans_charbon.append(pays)
    elif df.oil_co2.isna().any():
        liste_pays_sans_pétrole.append(pays)
    elif df.gas_co2.isna().any():
        liste_pays_sans_gaz.append(pays)
    elif df.land_use_change_co2.isna().any():
        liste_pays_sans_affectation_terres.append(pays)
    elif df.cement_co2.isna().any():
        liste_pays_sans_ciment.append(pays)
    elif df.flaring_co2.isna().any():
        liste_pays_sans_torchage.append(pays)
    if (pays not in liste_pays_sans_population) and (pays not in liste_pays_sans_co2) and (pays not in liste_pays_sans_affectation_terres)  and (pays not in liste_pays_sans_charbon) and (pays not in liste_pays_sans_ciment) and (pays not in liste_pays_sans_co2) and (pays not in liste_pays_sans_gaz) and (pays not in liste_pays_sans_pétrole) and (pays not in liste_pays_sans_torchage):
        new_df.append(df)

data_co2=pd.concat(new_df)   

country_no_match=[i for i in countries.country.unique() if i not in data_co2.country.unique()] # va nous permettre de voir les noms de pays qui ne coincident pas avec geopandas.
co2_no_match=[i for i in data_co2.country.unique() if i not in countries.country.unique()]


match=[]
for i in country_no_match:
    for j in co2_no_match:
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
  


result=[2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14] 
keep={}
for i in result:
    keep.update({match[i][1]: match[i][0]})
    
data_co2.country=data_co2.country.replace(keep)
 
print(data_co2.isna().sum())     
print(compteur_pays_sans_country)


plt.figure(figsize=(10, 10))
count=1
for val in data_co2.iloc[:,2:].columns:
    plt.subplot(4, 2, count)
    plt.title(f'distribution {val}')
    plt.boxplot(data_co2[val])
    count+=1
plt.show() # distributions très variables, certaines dépassent largement la médiane donc très déséquilibrées selon les pays.

print(data_co2.country.nunique()) # 211 pays enregistrés.

data_co2.to_csv('data_co2_traité.csv')

