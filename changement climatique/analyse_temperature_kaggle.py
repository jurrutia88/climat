#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 13:28:06 2023

@author: harry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

''' importation données et premières vérifs'''

data=pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\Temperature_change_Data (1).csv")
print(data.columns) #année, changement température, pays et code pays.
print(data.info) 
print(data.isna().sum()) #2714 nan chez les codes pays et 1781 pour les températures.
print(data.dtypes) #country code et country string et année et température int et float.
print(data.duplicated().sum()) # pas de doublon.
data=data.rename({'Country Code': 'code pays', 'Country Name': 'pays', 'tem_change': 'changement_température', 'year': 'années'}, axis=1)
data['pays']=data['pays'].str.lower()



''' on va voir les nan un peu '''

data_nan=data.loc[data.isna().any(axis=1),:]
data_nan=data_nan.sort_values(by=['pays', 'années'])

def nan_count(col):
    
    if col.isna().sum()!=0:
        return col.isna().sum()
    else:
        return 0
    

tableau_nan=data_nan.groupby('pays').agg({'changement_température': nan_count, 'code pays': nan_count, 'années': nan_count}) # ce tableau va nous aider à mieux voir les nan.
    
''' avec ce tableau on va chercher à classer les pays qu'on peut garder en interpolant les nan, et les pays à virer. '''

print(tableau_nan.sum(axis=0)) # on retrouve les nan du départ.

tableau_nan=tableau_nan.drop('années', axis=1) # pas besoin de la colonne années.

tableau_nan=tableau_nan.sort_values(by='changement_température', ascending=False) # on classe du pays le plus fort en nan niveau données températures au plus faible.

''' pour les valeurs code pays on laisse de côté, pour les valeurs température afin d'interpoler au mieux on va virer les pays à plus de dix années de données manquantes en température '''

pays_a_virer=list(tableau_nan[tableau_nan['changement_température']>10].index)

pays_restants=[pays for pays in data.pays.unique() if pays not in pays_a_virer] # il reste 232 pays.

''' on va mettre les données en forme en interpolant les nan '''

for val in range(len(pays_a_virer)):
    data=data[data.pays!=pays_a_virer[val]]
    
data=data.sort_values(by=['pays', 'années']).set_index('pays').drop('code pays', axis=1)

encoder=LabelEncoder()
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')



index=data.index
data=data.reset_index() # on remet index en variable pour interpoler, le pays est une info utile.
data.pays=encoder.fit_transform(data.pays)
data=pd.DataFrame(imputer.fit_transform(data))
data.index=index
data.columns=['pays', 'années', 'changement_temperature']
data=data.drop('pays', axis=1)

print(data.isna().sum())
print(data.head()) # vérif.

'''on va calculer la pente de l'évolution de la température
avec une régression linéaire pour classer les pays selon leur évolution de température.'''

model=LinearRegression()

coef_liste=[]
pays_liste=[]
for pays in data.index.unique():
    df=data[data.index==pays]
    années=np.array(df.années).reshape(-1, 1)
    changements=np.array(df.changement_temperature).reshape(-1, 1)
    model.fit(années, changements)
    coef_liste.append(model.coef_)
    pays_liste.append(pays)
    
tableau_pentes=pd.DataFrame({'pays': pays_liste, 'pentes': coef_liste}).sort_values(by='pentes', ascending=False)
print(tableau_pentes.head(10)) # affiche les 10 pays où ça évolue le plus.
print(tableau_pentes.tail(10)) # affiche pays ou sa évolue le moins.
   
    
    


    
    
  

    
