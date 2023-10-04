# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:58:28 2023

@author: jt_ur
"""

## Lancer la cellule pour importer les packages/fichiers nÃ©cessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme() # pour modifier le theme

# Pour éviter d'avoir les messages warning
import warnings
warnings.filterwarnings('ignore')

## Importation du dataset et premières analyses

df = pd.read_csv(r"C:\Users\jt_ur\OneDrive\Documents\DataScientest\Projet\Datasets\owid-co2-data.csv")

df.info()
print(df.shape)
print(df.country.unique())
print(df.columns)
df.columns.value_counts()

## Création de df par catégorie de variables

main_indicators= df[['country', 'year','population','gdp','cement_co2', 'co2', 'coal_co2', 
                     'consumption_co2', 'flaring_co2', 'gas_co2', 'land_use_change_co2',
                     'methane', 'nitrous_oxide', 'oil_co2', 'other_industry_co2', 'trade_co2' ]]

## Analyse des corrélations 

correlation_main = main_indicators.corr()
plt.figure(figsize=(10, 10))  # Taille de la figure
sns.heatmap(correlation_main, annot=True, cmap='viridis');

# Calculer le pourcentage de NaN par colonne
nan_percentage = main_indicators.isnull().mean() * 100

# Créer un diagramme à barres montrant le pourcentage de NaN par colonne
plt.figure(figsize=(10,5))
plt.bar(nan_percentage.index, nan_percentage.values)
plt.title('Pourcentage de NaN par colonne')
plt.xlabel('Colonnes')
plt.xticks(rotation=90)
plt.ylabel('Pourcentage de NaN')
plt.show()

### Couper les df par variable et par an

methanedf= main_indicators.drop(['trade_co2', 'cement_co2', 'coal_co2', 
                     'consumption_co2', 'flaring_co2', 'gas_co2', 'land_use_change_co2', 'nitrous_oxide', 'oil_co2', 'other_industry_co2'], axis=1) 
methanedf=methanedf.loc[methanedf.year >= 1962]

### Créer des df par zone

liste_zoom_afrique=['Algeria', 'Angola' ,'Benin','Botswana', 'Burkina Faso' ,'Burundi',
                    'Cameroon','Cape Verde', 'Central African Republic' ,'Chad' ,'Comoros', 
                    'Congo',"Cote d'Ivoire", 'Democratic Republic of Congo', 'Djibouti',
                    'Egypt' , 'Equatorial Guinea' ,'Eritrea','Eswatini' ,'Ethiopia', 'Gabon',
                    'Gambia','Ghana','Guinea' ,'Guinea-Bissau' ,'Kenya', 'Lesotho' ,'Liberia', 
                    'Libya','Madagascar','Malawi','Mali' ,'Mauritania' ,'Mauritius' ,'Mayotte', 
                    'Morocco', 'Mozambique' ,'Myanmar', 'Namibia','Niger', 'Nigeria','Reunion',
                    'Rwanda' ,'Sao Tome and Principe','Senegal', 'Sierra Leone' ,'Somalia' ,
                    'South Africa','South Sudan'  , 'Sudan' ,'Tanzania' , 'Togo','Tunisia',
                    'Uganda', 'Zambia' ,'Zimbabwe']

afrique= methanedf.loc[(methanedf.country.isin(liste_zoom_afrique))]

liste_zoom_ameriques=['Antigua and Barbuda','Argentina','Bahamas','Barbados','Belize','Bolivia', 'Bonaire Sint Eustatius and Saba',
                      'Brazil' , 'Canada', 'Chile','Colombia','Costa Rica', 'Cuba', 'Curacao',
                      'Dominica' ,'Dominican Republic' ,'Ecuador', 'El Salvador', 'French Guiana',
                      'Grenada' ,'Guadeloupe', 'Guatemala','Guyana' ,'Haiti', 'Honduras', 'Jamaica',
                      'Martinique' , 'Mexico', 'Montserrat' ,'Netherlands Antilles','Nicaragua'
                      'Panama','Paraguay' ,'Peru', 'Puerto Rico', 'Saint Kitts and Nevis'
                      ,'Saint Lucia','Saint Pierre and Miquelon' ,'Saint Vincent and the Grenadines',
                      'Suriname', 'Trinidad and Tobago', 'United States', 'United States Virgin Islands',
                      'Uruguay', 'Venezuela' ]

ameriques= methanedf.loc[(methanedf.country.isin(liste_zoom_ameriques))]

liste_zoom_asie=['Afghanistan', 'Azerbaijan','Bahrain' ,'Bangladesh',  'Bhutan', 'Brunei', 'Cambodia',
                 'China','Fiji', 'Hong Kong', 'India', 'Indonesia', 'Iran', 'Iraq' ,'Israel',
                 'Japan' ,'Jordan' ,'Kazakhstan', 'Kuwait' , 'Kyrgyzstan',
                 'Laos' , 'Lebanon' , 'Malaysia' , 'Mongolia', 'Nepal', 'North Korea' , 'Oman',
                 'Pakistan','Palestine', 'Philippines','Qatar', 'Russia' , 'Saudi Arabia' ,
                 'Singapore','South Korea', 'Sri Lanka','Syria', 'Taiwan', 'Tajikistan', 
                 'Thailand', 'Timor', 'Turkmenistan', 'United Arab Emirates' , 'Uzbekistan'
                 ,'Vietnam','Yemen']   

asie= methanedf.loc[(methanedf.country.isin(liste_zoom_asie))]

liste_zoom_europe= ['Albania','Andorra','Armenia' ,'Austria','Belarus' ,'Belgium' ,'Bosnia and Herzegovina','Bulgaria',
                    'Croatia', 'Cyprus', 'Czechia','Denmark', 'Estonia', 'Faeroe Islands' ,'Falkland Islands' 
                    ,'Finland' ,'France', 'Georgia' ,'Germany', 'Greece' ,'Greenland', 'Hungary' ,'Iceland',
                    'Ireland', 'Italy', 'Kosovo', 'Latvia', 'Lithuania','Luxembourg', 'Malta', 'Moldova', 
                    'Montenegro','Netherlands', 'North Macedonia','Norway', 'Poland' ,'Portugal' ,
                    'Romania' , 'Serbia' , 'Slovakia', 'Slovenia', 'Sweden' ,'Switzerland','Spain',
                    'Turkey' , 'Ukraine','United Kingdom']

europe= methanedf.loc[(methanedf.country.isin(liste_zoom_europe))]

liste_zoom_oceanie= ['Australia','French Polynesia','New Caledonia', 'New Zealand',
                      'Papua New Guinea', 'Samoa','Solomon Islands', 'Tonga','Vanuatu' ]

oceanie= methanedf.loc[(methanedf.country.isin(liste_zoom_oceanie))]

## Graphiques par pays scores indisctincs

sns.set(rc = {'figure.figsize':(40,40)})
sns.relplot(data=afrique, x='methane', y='country', hue='year', height=10)
plt.title('Evolution de la production de co2 par pays en Afrique');
sns.relplot(data=ameriques, x='methane', y='country', hue='year', height=10)
plt.title('Evolution de la production de co2 par pays aux Amériques');
sns.relplot(data=asie, x='methane', y='country', hue='year', height=10)
plt.title('Evolution de la production de co2 par pays en Asie');
sns.relplot(data=europe, x='methane', y='country',hue='year', height=10)
plt.title('Evolution de la production de co2 par pays en Europe');
sns.relplot(data=oceanie, x='methane', y='country', hue='year', height=10)
plt.title('Evolution de la production de co2 par pays en Oceanie');
plt.show()

#  Accumulation co2 par pays

afrique_mth = afrique[['methane', 'year','country']]
afrique_mth = afrique_mth.groupby('country')['methane'].sum().reset_index()

ameriques_mth = ameriques[['methane', 'year','country']]
ameriques_mth = ameriques_mth.groupby('country')['methane'].sum().reset_index()

asie_mth = afrique[['methane', 'year','country']]
asie_mth = asie_mth.groupby('country')['methane'].sum().reset_index()

europe_mth = europe[['methane', 'year','country']]
europe_mth = europe_mth.groupby('country')['methane'].sum().reset_index()

oceanie_mth = oceanie[['methane', 'year','country']]
oceanie_mth = oceanie_mth.groupby('country')['methane'].sum().reset_index()

## Création d'un df des pays les plus émetteurs de co2 à la date de 2019

methane_2019=methanedf.loc[methanedf.year == 2019]
pays_max_2019 = methane_2019.sort_values(by=['methane'], ascending=False)

"""les pays les plus émetteurs à la date de 2021 sont par ordre: China, United States, India,
Russia, Japan, Iran, Germany, Saudi Arabia, Indonesia, South Korea, Canada, Brazil, Turkey,
South Africa, Mexico, Australia, United Kingdom, Italy, Poland, Vietnam, France"""

liste_top15_2019= ['China', 'United States', 'Russia', 'India', 'Brazil', 'Indonesia', 'Iran', 
                   'Pakistan', 'Mexico', 'Nigeria', 'Australia', 'Iraq', 'Argentina', 
                   'Venezuela', 'Saudi Arabia']

top15_2019= methane_2019.loc[(methane_2019.country.isin(liste_top15_2019))]
top15_2019 = top15_2019.sort_values('methane', ascending=False)
sns.barplot(x='methane', y='country', data=top15_2019, palette='rocket')
plt.title('Emissions de methane des 15 pays les plus émetteurs en 2019', fontsize=50)
plt.xlabel('billions de tonnes?', fontsize=35)
plt.xticks(fontsize=30)
plt.ylabel('pays', fontsize=20)
plt.yticks(fontsize=40)
plt.show()

import plotly.graph_objs as go
import plotly.offline as pyo


# Creación de la trace
trace = go.Bar(x=top15_2019['methane'], y=top15_2019['country'], 
               orientation='h', marker=dict(color=top15_2019['methane'], 
               colorscale='rocket'))

# Creación de la lista de traces
data = [trace]

layout = go.Layout(title='Les 15 pays les plus émetteurs de méthane en 2019',
                   xaxis=dict(title='Emissions de méthane'),
                   yaxis=dict(title='Pays'))

# Creación de la figura del gráfico
fig = go.Figure(data=data, layout=layout)

# Mostrar el gráfico
pyo.plot(fig)



## Création d'un df des pays les plus émetteurs de co2 à la date de 1990

methanedf_1990=methanedf.loc[methanedf.year == 1990]
pays_max_1990 = methanedf_1990.sort_values(by=['methane'], ascending=False)

"""les pays les plus émetteurs à la date de 1990 sont par ordre: China, Russia, United States,
India, Indonesia, Brazil, Australia, Venezuela, Nigeria, Argentina, United Kingdom, Iran, Germany,
Mexico, Ukraine, Pakistan, France, Canada, Bangladesh, Turkmenistan, Saudi Arabia"""

liste_top15_1990= ['China', 'Russia', 'United States',
'India', 'Indonesia', 'Brazil', 'Australia', 'Venezuela', 'Nigeria', 'Argentina', 'United Kingdom', 
'Iran', 'Germany', 'Mexico', 'Ukraine', 'Pakistan', 'France', 'Canada', 'Bangladesh', 'Turkmenistan', 
'Saudi Arabia']

top15_1990= methanedf_1990.loc[(methanedf_1990.country.isin(liste_top15_1990))]
top15_1990 = top15_1990.sort_values('methane', ascending=False)
sns.barplot(x='methane', y='country', data=top15_1990, palette='mako')
plt.title('Emissions de methane des 15 pays les plus émetteurs en 1990', fontsize=50)
plt.xlabel('billions de tonnes', fontsize=35)
plt.xticks(fontsize=30)
plt.ylabel('pays', fontsize=20)
plt.yticks(fontsize=40)
plt.show()

## Créer les df par zone géo: continents

liste_continents= ["Africa", "Asia","Europe", "North America", "Oceania", "South America"]

continents= methanedf.loc[(methanedf.country.isin(liste_continents))]


sns.relplot(data=continents, x='year', y='methane', hue='country', kind='line', palette='husl')
plt.title('Evolution des émissions de methane par continent')
plt.xlabel('année')
plt.ylabel('billions de tonnes')
plt.show()

## Créer les df par zone géo: income_zones

liste_income_zones= ['High-income countries','Low-income countries',
'Lower-middle-income countries','Upper-middle-income countries']

income_zones= methanedf.loc[(methanedf.country.isin(liste_income_zones))]

correlation_income_zones = income_zones.corr()
plt.figure(figsize=(10, 10))  # Taille de la figure
sns.heatmap(correlation_income_zones, annot=True, cmap='cividis')
plt.title('Correlation df income zones réduit en variables');

## Graphiques du df income zones

plt.figure(figsize=(50, 80))  # Taille de la figure
g=sns.catplot(data=income_zones, x='country', y='methane', kind='boxen', palette='husl')
g.ax.set_xticklabels(['Elévé', 'Faible', 'Moyen inférieur', 'Moyen supérieur'])
plt.title('Emissions de methane par zone de revenu')
plt.xlabel('zones')
plt.ylabel('tonnes methane')
plt.xticks(rotation=90);


## Créer les df par zone géo: monde

monde = df[df['country']=='World']
monde= monde[monde['year']>=1962]
monde= methanedf[methanedf['country']=='World']

## Graphiques du df monde 

sns.relplot(data=monde, kind='line', x='year', y='methane', height=5)
plt.title('Evolution mondiale des émissions de methane')
plt.xlabel('Année')
plt.ylabel('Tonnes de methane');
