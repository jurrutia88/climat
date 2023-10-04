#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 08:16:56 2022

@author: harry
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_1880=pd.read_csv('/home/harry/projet datascientest/fichiers csv/gitsemp/1880_present/ZonAnn.Ts+dSST.csv', index_col='Year')
data_2002=pd.read_csv('/home/harry/projet datascientest/fichiers csv/gitsemp/2002_present/ZonAnn.Ts+dSST.csv', sep=',', header=1, index_col='Year')

''' traitement de base des datasets'''

print(data_1880.info()) # a priori pas de nan, pas de variable object.

print(data_2002.info()) # variables numériques mais object avec des nan.

# en fait data_2002 comprend 3 datasets, donc à diviser en trois.

durée=2021-2003

data_2002_airs_v6=data_2002.iloc[:19,:]
data_2002_airs_v7=data_2002.iloc[21:40,:]
data_2002_ghcnv4=data_2002.iloc[42:61,:]

# on revérifie séparément chaque dataset.

print(data_2002_airs_v6.info())
print(data_2002_airs_v7.info())
print(data_2002_ghcnv4.info())

# plus de nan, mais variables à convertir.

dataset_list=[data_2002_airs_v6, data_2002_airs_v7, data_2002_ghcnv4]
datasets=[]
for df in dataset_list:
    df=df.astype(float)
    datasets.append(df)
    
print(datasets[0].info())


# a priori sur les csv les données sont en degrés celsius mais pas sur les fichiers txt.

''' petit graphique '''

# évolution anomalie température par hémisphères entre 1880 et maintenant.

shapes=['-k', ':r', '-g']


plt.figure(figsize=(20, 10))
plt.title('Evolution anomalie entre les hémisphères')
plt.xlabel('Années')
plt.ylabel('Degrés Celsius')
for i in range(len(shapes)):
    plt.plot(data_1880.index, data_1880.iloc[:, i], shapes[i], label=data_1880.columns[i])
plt.legend()


# évolution anomalie température hémisphère nord par zone.

shapes_2=['-y*', '-ro', '--g', ':b', ':o']
Nhem_col=[i for i in data_1880.columns if 'N' in i and 'S' not in i and 'NHem' not in i]

plt.figure(figsize=(20, 10))
plt.title('Evolution anomalie entre zones hémisphère nord')
plt.xlabel('Années')
plt.ylabel('Degrés Celsius')
for i in range(len(shapes_2)):
    plt.plot(data_1880.index, data_1880.loc[:, Nhem_col[i]], shapes_2[i], label=Nhem_col[i])
plt.legend()


# évolution anomalie température hémisphère sud par zone.

shapes_3=['-y*', '-ro', '--g', ':b', ':o']
Shem_col=[i for i in data_1880.columns if 'S' in i and 'N' not in i and 'SHem' not in i]

plt.figure(figsize=(20, 10))
plt.title('Evolution anomalie entre zones hémisphère sud')
plt.xlabel('Années')
plt.ylabel('Degrés Celsius')
for i in range(len(shapes_3)):
    plt.plot(data_1880.index, data_1880.loc[:, Shem_col[i]], shapes_3[i], label=Shem_col[i])
plt.legend()
plt.show()
"""
# évolution anomalie température par hémisphères entre 2002 et maintenant(AIRSv6).

shapes=['-k', ':r', '-g']


plt.figure(figsize=(30, 20))
plt.title('évolution anomalie entre les hémisphères(AIRSv6)')
plt.xlabel('années')
plt.ylabel('degrés celsius')
for i in range(len(shapes)):
    plt.plot(data_2002_airs_v6.index, data_2002_airs_v6.iloc[:, i], shapes[i], label=data_2002_airs_v6.columns[i])
plt.legend()


# évolution anomalie température hémisphère nord par zone (AIRSv6).

shapes_2=['-y*', '-ro', '--g', ':b', ':o']
Nhem_col=[i for i in data_2002_airs_v6.columns if 'N' in i and 'S' not in i and 'NHem' not in i]

plt.figure(figsize=(30, 20))
plt.title('évolution anomalie entre zones hémisphère nord(AIRSv6)')
plt.xlabel('années')
plt.ylabel('degrés celsius')
for i in range(len(shapes_2)):
    plt.plot(data_2002_airs_v6.index, data_2002_airs_v6.loc[:, Nhem_col[i]], shapes_2[i], label=Nhem_col[i])
plt.legend()

# évolution anomalie température hémisphère sud par zone(AIRSv6).

shapes_3=['-y*', '-ro', '--g', ':b', ':o']
Shem_col=[i for i in data_2002_airs_v6.columns if 'S' in i and 'N' not in i and 'SHem' not in i]

plt.figure(figsize=(30, 20))
plt.title('évolution anomalie entre zones hémisphère sud(AIRSv6)')
plt.xlabel('années')
plt.ylabel('degrés celsius')
for i in range(len(shapes_3)):
    plt.plot(data_2002_airs_v6.index, data_2002_airs_v6.loc[:, Shem_col[i]], shapes_3[i], label=Shem_col[i])
plt.legend()
plt.show()




# évolution anomalie température par hémisphères entre 2002 et maintenant(AIRSv7).

shapes=['-k', ':r', '-g']


plt.figure(figsize=(30, 20))
plt.title('évolution anomalie entre les hémisphères(AIRSv7)')
plt.xlabel('années')
plt.ylabel('degrés celsius')
for i in range(len(shapes)):
    plt.plot(data_2002_airs_v7.index, data_2002_airs_v7.iloc[:, i], shapes[i], label=data_2002_airs_v7.columns[i])
plt.legend()


# évolution anomalie température hémisphère nord par zone (AIRSv7).

shapes_2=['-y*', '-ro', '--g', ':b', ':o']
Nhem_col=[i for i in data_2002_airs_v7.columns if 'N' in i and 'S' not in i and 'NHem' not in i]

plt.figure(figsize=(30, 20))
plt.title('évolution anomalie entre zones hémisphère nord(AIRSv7)')
plt.xlabel('années')
plt.ylabel('degrés celsius')
for i in range(len(shapes_2)):
    plt.plot(data_2002_airs_v7.index, data_2002_airs_v7.loc[:, Nhem_col[i]], shapes_2[i], label=Nhem_col[i])
plt.legend()

# évolution anomalie température hémisphère sud par zone(AIRSv7).

shapes_3=['-y*', '-ro', '--g', ':b', ':o']
Shem_col=[i for i in data_2002_airs_v7.columns if 'S' in i and 'N' not in i and 'SHem' not in i]

plt.figure(figsize=(30, 20))
plt.title('évolution anomalie entre zones hémisphère sud(AIRSv7)')
plt.xlabel('années')
plt.ylabel('degrés celsius')
for i in range(len(shapes_3)):
    plt.plot(data_2002_airs_v7.index, data_2002_airs_v7.loc[:, Shem_col[i]], shapes_3[i], label=Shem_col[i])
plt.legend()
plt.show()
"""