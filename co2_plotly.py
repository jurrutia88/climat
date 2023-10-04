# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:54:48 2023

@author: jt_ur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_theme() # pour modifier le theme

# Pour éviter d'avoir les messages warning
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\jt_ur\OneDrive\Documents\DataScientest\Projet\Datasets\owid-co2-data.csv")
main_indicators= df[['country', 'year','population','gdp','cement_co2', 'co2', 'coal_co2', 
                     'consumption_co2', 'flaring_co2', 'gas_co2', 'land_use_change_co2',
                     'methane', 'nitrous_oxide', 'oil_co2', 'other_industry_co2', 'trade_co2' ]]
main_indicators2= main_indicators.drop('trade_co2', axis=1) 
main_indicators2=main_indicators2.loc[main_indicators2.year >= 1962]

## Créer les df par zone géo: pays

"""Au niveau des pays, nous allons traiter les données par continent afin de rendre plus lisibles
les visualisations"""

"""df par pays selon les main_indicators"""

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

afrique_main_indicators= main_indicators2.loc[(main_indicators2.country.isin(liste_zoom_afrique))]
afrique_main_indicators= afrique_main_indicators.drop(['gdp','other_industry_co2',
                                                              'consumption_co2','methane',
                                                              'nitrous_oxide'], axis=1)

liste_zoom_ameriques=['Antigua and Barbuda','Argentina','Bahamas','Barbados','Belize','Bolivia', 'Bonaire Sint Eustatius and Saba',
                      'Brazil' , 'Canada', 'Chile','Colombia','Costa Rica', 'Cuba', 'Curacao',
                      'Dominica' ,'Dominican Republic' ,'Ecuador', 'El Salvador', 'French Guiana',
                      'Grenada' ,'Guadeloupe', 'Guatemala','Guyana' ,'Haiti', 'Honduras', 'Jamaica',
                      'Martinique' , 'Mexico', 'Montserrat' ,'Netherlands Antilles','Nicaragua'
                      'Panama','Paraguay' ,'Peru', 'Puerto Rico', 'Saint Kitts and Nevis'
                      ,'Saint Lucia','Saint Pierre and Miquelon' ,'Saint Vincent and the Grenadines',
                      'Suriname', 'Trinidad and Tobago', 'United States', 'United States Virgin Islands',
                      'Uruguay', 'Venezuela' ]

ameriques_main_indicators= main_indicators2.loc[(main_indicators2.country.isin(liste_zoom_ameriques))]
ameriques_main_indicators= ameriques_main_indicators.drop(['gdp','other_industry_co2',
                                                              'consumption_co2','methane',
                                                              'nitrous_oxide'], axis=1)

liste_zoom_asie=['Afghanistan', 'Azerbaijan','Bahrain' ,'Bangladesh',  'Bhutan', 'Brunei', 'Cambodia',
                 'China','Fiji', 'Hong Kong', 'India', 'Indonesia', 'Iran', 'Iraq' ,'Israel',
                 'Japan' ,'Jordan' ,'Kazakhstan', 'Kuwait' , 'Kyrgyzstan',
                 'Laos' , 'Lebanon' , 'Malaysia' , 'Mongolia', 'Nepal', 'North Korea' , 'Oman',
                 'Pakistan','Palestine', 'Philippines','Qatar', 'Russia' , 'Saudi Arabia' ,
                 'Singapore','South Korea', 'Sri Lanka','Syria', 'Taiwan', 'Tajikistan', 
                 'Thailand', 'Timor', 'Turkmenistan', 'United Arab Emirates' , 'Uzbekistan'
                 ,'Vietnam','Yemen']   

asie_main_indicators= main_indicators2.loc[(main_indicators2.country.isin(liste_zoom_asie))]
asie_main_indicators= asie_main_indicators.drop(['gdp','other_industry_co2',
                                                              'consumption_co2','methane',
                                                              'nitrous_oxide'], axis=1)  

liste_zoom_europe= ['Albania','Andorra','Armenia' ,'Austria','Belarus' ,'Belgium' ,'Bosnia and Herzegovina','Bulgaria',
                    'Croatia', 'Cyprus', 'Czechia','Denmark', 'Estonia', 'Faeroe Islands' ,'Falkland Islands' 
                    ,'Finland' ,'France', 'Georgia' ,'Germany', 'Greece' ,'Greenland', 'Hungary' ,'Iceland',
                    'Ireland', 'Italy', 'Kosovo', 'Latvia', 'Lithuania','Luxembourg', 'Malta', 'Moldova', 
                    'Montenegro','Netherlands', 'North Macedonia','Norway', 'Poland' ,'Portugal' ,
                    'Romania' , 'Serbia' , 'Slovakia', 'Slovenia', 'Sweden' ,'Switzerland','Spain',
                    'Turkey' , 'Ukraine','United Kingdom']

europe_main_indicators= main_indicators2.loc[(main_indicators2.country.isin(liste_zoom_europe))]
europe_main_indicators= europe_main_indicators.drop(['gdp','other_industry_co2',
                                                              'consumption_co2','methane',
                                                              'nitrous_oxide'], axis=1)  

liste_zoom_oceanie= ['Australia','French Polynesia','New Caledonia', 'New Zealand',
                      'Papua New Guinea', 'Samoa','Solomon Islands', 'Tonga','Vanuatu' ]

oceanie_main_indicators= main_indicators2.loc[(main_indicators2.country.isin(liste_zoom_oceanie))]
oceanie_main_indicators= oceanie_main_indicators.drop(['gdp','other_industry_co2',
                                                              'consumption_co2','methane',
                                                              'nitrous_oxide'], axis=1)  

## Création d'un df des pays les plus émetteurs de co2 à la date de 2021

main_indicators_2021=main_indicators2.loc[main_indicators2.year == 2021]
pays_max_2021 = main_indicators_2021.sort_values(by=['co2'], ascending=False)

"""les pays les plus émetteurs à la date de 2021 sont par ordre: China, United States, India,
Russia, Japan, Iran, Germany, Saudi Arabia, Indonesia, South Korea, Canada, Brazil, Turkey,
South Africa, Mexico, Australia, United Kingdom, Italy, Poland, Vietnam, France"""

liste_top15_2021= ['China', 'United States', 'India', 'Russia', 'Japan', 'Iran', 'Germany', 
                   'Saudi Arabia', 'Indonesia', 'South Korea', 'Canada', 'Brazil', 'Turkey', 
                   'South Africa', 'Mexico']

top15_2021= main_indicators_2021.loc[(main_indicators_2021.country.isin(liste_top15_2021))]
top15_2021 = top15_2021.sort_values('co2', ascending=False)

## Création d'un df des pays les plus émetteurs de co2 à la date de 1962

main_indicators_1962=main_indicators2.loc[main_indicators2.year == 1962]
pays_max_1962 = main_indicators_1962.sort_values(by=['co2'], ascending=False)

"""les pays les plus émetteurs à la date de 2021 sont par ordre: China, United States, India,
Russia, Japan, Iran, Germany, Saudi Arabia, Indonesia, South Korea, Canada, Brazil, Turkey,
South Africa, Mexico, Australia, United Kingdom, Italy, Poland, Vietnam, France"""

liste_top15_1962= ['United States', 'Russia', 'Germany', 'United Kingdom', 'China', 'France', 
                   'Ukraine', 'Japan', 'Poland', 'Canada', 'Italy', 'India', 
                   'Czechia', 'Kazakhstan', 'Belgium']

top15_1962= main_indicators_1962.loc[(main_indicators_1962.country.isin(liste_top15_1962))]
top15_1962 = top15_1962.sort_values('co2', ascending=False)

## Créer les df par zone géo: continents

liste_continents= ["Africa", "Asia","Europe", "North America", "Oceania", "South America"]

continents_main_indicators= main_indicators2.loc[(main_indicators2.country.isin(liste_continents))]
continents_main_indicators= continents_main_indicators.drop(['gdp','other_industry_co2',
                                                              'consumption_co2','methane',
                                                              'nitrous_oxide'], axis=1)

coal_max_2021 = main_indicators_2021[['country', 'year','coal_co2']]
coal_max_2021 = coal_max_2021.sort_values('coal_co2', ascending=False)

"""Les pays les plus émetteurs de co2 par charbon en 2021 sont China, India, United States,
Japan,Russia, South Africa"""

liste_pays_coalmax= ['China', 'India', 'United States','Japan','Russia']
coal_max_pays2O21= coal_max_2021.loc[(coal_max_2021.country.isin(liste_pays_coalmax))]
coal_max_pays2O21 = coal_max_pays2O21.sort_values('coal_co2', ascending=False)

oil_max_2021 = main_indicators_2021[['country', 'year','oil_co2']]
oil_max_2021 = oil_max_2021.sort_values('oil_co2', ascending=False)

liste_pays_oilmax= ['United States', 'China', 'India','Russia','Japan']
oil_max_pays2O21= oil_max_2021.loc[(oil_max_2021.country.isin(liste_pays_oilmax))]
oil_max_pays2O21 = oil_max_pays2O21.sort_values('oil_co2', ascending=False)

gas_max_2021 = main_indicators_2021[['country', 'year','gas_co2']]
gas_max_2021 = gas_max_2021.sort_values('gas_co2', ascending=False)

liste_pays_gasmax= ['United States', 'Russia', 'China','Iran','Saudi Arabia']
gas_max_pays2O21= gas_max_2021.loc[(gas_max_2021.country.isin(liste_pays_gasmax))]
gas_max_pays2O21 = gas_max_pays2O21.sort_values('gas_co2', ascending=False)

cement_max_2021 = main_indicators_2021[['country', 'year','cement_co2']]
cement_max_2021 = cement_max_2021.sort_values('cement_co2', ascending=False)

liste_pays_cementmax= ['China', 'India', 'Vietnam','Turkey','United States']
cement_max_pays2O21= cement_max_2021.loc[(cement_max_2021.country.isin(liste_pays_cementmax))]
cement_max_pays2O21 = cement_max_pays2O21.sort_values('cement_co2', ascending=False)

flaring_max_2021 = main_indicators_2021[['country', 'year','flaring_co2']]
flaring_max_2021 = flaring_max_2021.sort_values('flaring_co2', ascending=False)

liste_pays_flaringmax= ['United States', 'Russia', 'Iraq','Iran','Brazil']
flaring_max_pays2O21= flaring_max_2021.loc[(flaring_max_2021.country.isin(liste_pays_flaringmax))]
flaring_max_pays2O21 = flaring_max_pays2O21.sort_values('flaring_co2', ascending=False)

land_use_change_max_2021 = main_indicators_2021[['country', 'year','land_use_change_co2']]
land_use_change_max_2021 = land_use_change_max_2021.sort_values('land_use_change_co2', ascending=False)

liste_pays_land_use_changemax= ['Indonesia', 'Brazil', 'Russia','Democratic Republic of Congo','China']
land_use_change_max_pays2O21= land_use_change_max_2021.loc[(land_use_change_max_2021.country.isin(liste_pays_land_use_changemax))]
land_use_change_max_pays2O21 = land_use_change_max_pays2O21.sort_values('land_use_change_co2', ascending=False)

liste_income_zones= ['High-income countries','Low-income countries',
'Lower-middle-income countries','Upper-middle-income countries']

income_zones_main_indicators= main_indicators2.loc[(main_indicators2.country.isin(liste_income_zones))]
income_zones_main_indicators= income_zones_main_indicators.drop(['gdp','other_industry_co2',
                                                              'consumption_co2','methane',
                                                              'nitrous_oxide'], axis=1)

monde = df[df['country']=='World']
monde= monde[monde['year']>=1962]
monde_main_indicators= main_indicators2[main_indicators2['country']=='World']
monde_main_indicators= monde_main_indicators.drop(['gdp','other_industry_co2',
                                                              'consumption_co2','methane',
                                                              'nitrous_oxide'], axis=1)

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio


#Figure 1
fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=monde_main_indicators['year'], y=monde_main_indicators['co2'], line=dict(color='red', width=3)))
fig1.update_layout(title='Evolution mondiale des émissions de CO2',
                  xaxis=dict(title='Année', title_font=dict(size=10)),
                  yaxis=dict(title='Tonnes de CO2', title_font=dict(size=10)),
                  font=dict(size=18),
                  legend=dict(x=0, y=1, orientation='h', font=dict(size=10)),
                  width=550, height=550)

#fig1.show()

#Figure 2

color_map = {'Africa': 'green', 'Asia': 'orange', 'Europe': 'magenta', 'North America': 'black', 'Oceania': 'yellow', 'South America' : 'blue'}

fig2 = px.line(continents_main_indicators, x='year', y='co2', color='country', color_discrete_map=color_map)
fig2.update_traces(line=dict(width=3))
fig2.update_layout(title='Evolution des émissions de CO2 par continent',
                  xaxis_title='Année',
                  yaxis_title='Tonnes de CO2',
                  font=dict(size=18),
                  legend=dict(x=0, y=1, orientation='h', font=dict(size=10)),
                  width=550, height=550)

#fig2.show()


#Figure 3
fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=monde_main_indicators['year'], y=monde_main_indicators['coal_co2'], name='Charbon', line=dict(color='green', width=3)))
fig3.add_trace(go.Scatter(x=monde_main_indicators['year'], y=monde_main_indicators['oil_co2'], name='Pétrole', line=dict(color='orange', width=3)))
fig3.add_trace(go.Scatter(x=monde_main_indicators['year'], y=monde_main_indicators['gas_co2'], name='Gas', line=dict(color='magenta', width=3)))
fig3.add_trace(go.Scatter(x=monde_main_indicators['year'], y=monde_main_indicators['land_use_change_co2'], name='Changement affectation des terres', line=dict(color='black', width=3)))
fig3.add_trace(go.Scatter(x=monde_main_indicators['year'], y=monde_main_indicators['cement_co2'], name='Ciment', line=dict(color='yellow', width=3)))
fig3.add_trace(go.Scatter(x=monde_main_indicators['year'], y=monde_main_indicators['flaring_co2'], name='Torchage', line=dict(color='blue', width=3)))


fig3.update_layout(title='Evolution mondiale des émissions de CO2 par cause',
                  xaxis=dict(title='Année', title_font=dict(size=10)),
                  yaxis=dict(title='Tonnes de CO2', title_font=dict(size=10)),
                  font=dict(size=18),
                  legend=dict(x=0, y=1, orientation='h', font=dict(size=10)),
                  width=650, height=700)

#fig3.show()

#Figure 4

rocket_colorscale = [
    (0.0, '#3c3e40'),
    (0.1, '#652c90'),
    (0.2, '#bb3754'),
    (0.3, '#f66d44'),
    (0.4, '#feae65'),
    (0.5, '#fddc73'),
    (1.0, '#fcffa4')
]

top15_1962 = main_indicators_1962.loc[(main_indicators_1962.country.isin(liste_top15_1962))]
top15_1962 = top15_1962.sort_values('co2', ascending=True)

fig4 = px.bar(top15_1962, x='co2', y='country',
             color='co2',
             color_continuous_scale=rocket_colorscale,
             labels={'co2': 'billions de tonnes'},
             title='Emissions de co2 des 15 pays les plus émetteurs en 1962')
fig4.update_layout(showlegend=False)
#fig4.show()


#Figure 5

rocket_colorscale = [
    (0.0, '#3c3e40'),
    (0.1, '#652c90'),
    (0.2, '#bb3754'),
    (0.3, '#f66d44'),
    (0.4, '#feae65'),
    (0.5, '#fddc73'),
    (1.0, '#fcffa4')
]

liste_top15_2021 = ['China', 'United States', 'India', 'Russia', 'Japan', 'Iran', 'Germany',
                   'Saudi Arabia', 'Indonesia', 'South Korea', 'Canada', 'Brazil', 'Turkey',
                   'South Africa', 'Mexico']

top15_2021 = main_indicators_2021.loc[(main_indicators_2021.country.isin(liste_top15_2021))]
top15_2021 = top15_2021.sort_values('co2', ascending=True)

fig5 = px.bar(top15_2021, x='co2', y='country',
             color='co2',
             color_continuous_scale=rocket_colorscale,
             labels={'co2': 'billions de tonnes'},
             title='Emissions de co2 des 15 pays les plus émetteurs en 2021')
fig5.update_layout(showlegend=False)
#fig5.show()

#Figure 6


coal_max_2021 = main_indicators_2021[['country', 'year','coal_co2']]
coal_max_2021 = coal_max_2021.sort_values('coal_co2', ascending=False)

liste_pays_coalmax = ['China', 'India', 'United States','Japan','Russia']
coal_max_pays2O21 = coal_max_2021.loc[(coal_max_2021.country.isin(liste_pays_coalmax))]
coal_max_pays2O21 = coal_max_pays2O21.sort_values('coal_co2', ascending=True)

fig6 = px.bar(coal_max_pays2O21, x='coal_co2', y='country',
             color='coal_co2',
             color_continuous_scale='Viridis',
             labels={'coal_co2': 'tonnes co2'},
             title='Top 5 des pays emetteurs de co2 liées à la production de charbon en 2021')
fig6.update_layout(showlegend=False)
#fig6.show()

#Figure 7

liste_pays_oilmax = ['United States', 'China', 'India','Russia','Japan']
oil_max_pays2O21 = oil_max_2021.loc[(oil_max_2021.country.isin(liste_pays_oilmax))]
oil_max_pays2O21 = oil_max_pays2O21.sort_values('oil_co2', ascending=True)

fig7 = px.bar(oil_max_pays2O21, x='oil_co2', y='country',
             color='oil_co2',
             color_continuous_scale='Viridis',
             labels={'oil_co2': 'tonnes co2'},
             title='Top 5 des pays emetteurs de co2 provenant du pétrole en 2021')
fig7.update_layout(showlegend=False)
#fig7.show()

#Figure 8 

gas_max_2021 = main_indicators_2021[['country', 'year','gas_co2']]
gas_max_2021 = gas_max_2021.sort_values('gas_co2', ascending=False)

liste_pays_gasmax = ['United States', 'Russia', 'China','Iran','Saudi Arabia']
gas_max_pays2O21 = gas_max_2021.loc[(gas_max_2021.country.isin(liste_pays_gasmax))]
gas_max_pays2O21 = gas_max_pays2O21.sort_values('gas_co2', ascending=True)

fig8 = px.bar(gas_max_pays2O21, x='gas_co2', y='country',
             color='gas_co2',
             color_continuous_scale='Viridis',
             labels={'gas_co2': 'tonnes co2'},
             title='Top 5 des pays emetteurs de co2 issu du gaz en 2021')
fig8.update_layout(showlegend=False)
#fig8.show()

#Figure 9

cement_max_2021 = main_indicators_2021[['country', 'year','cement_co2']]
cement_max_2021 = cement_max_2021.sort_values('cement_co2', ascending=False)

liste_pays_cementmax = ['China', 'India', 'Vietnam','Turkey','United States']
cement_max_pays2O21 = cement_max_2021.loc[(cement_max_2021.country.isin(liste_pays_cementmax))]
cement_max_pays2O21 = cement_max_pays2O21.sort_values('cement_co2', ascending=True)

fig9 = px.bar(cement_max_pays2O21, x='cement_co2', y='country',
             color='cement_co2',
             color_continuous_scale='Viridis',
             labels={'cement_co2': 'tonnes co2'},
             title='Top 5 des pays emetteurs de co2 liées à la production de ciment en 2021')
fig9.update_layout(showlegend=False)
#fig9.show()

#Figure 10 

flaring_max_2021 = main_indicators_2021[['country', 'year','flaring_co2']]
flaring_max_2021 = flaring_max_2021.sort_values('flaring_co2', ascending=False)

liste_pays_flaringmax = ['United States', 'Russia', 'Iraq','Iran','Brazil']
flaring_max_pays2O21 = flaring_max_2021.loc[(flaring_max_2021.country.isin(liste_pays_flaringmax))]
flaring_max_pays2O21 = flaring_max_pays2O21.sort_values('flaring_co2', ascending=True)

fig10 = px.bar(flaring_max_pays2O21, x='flaring_co2', y='country',
             color='flaring_co2',
             color_continuous_scale='Viridis',
             labels={'flaring_co2': 'tonnes co2'},
             title='Top 5 des pays emetteurs de co2 liées au torchage en 2021')
fig10.update_layout(showlegend=False)
#fig10.show()

# Figure 11

land_use_change_max_2021 = main_indicators_2021[['country', 'year','land_use_change_co2']]
land_use_change_max_2021 = land_use_change_max_2021.sort_values('land_use_change_co2', ascending=False)

liste_pays_land_use_changemax = ['Indonesia', 'Brazil', 'Russia','Democratic Republic of Congo','China']
land_use_change_max_pays2O21 = land_use_change_max_2021.loc[(land_use_change_max_2021.country.isin(liste_pays_land_use_changemax))]
land_use_change_max_pays2O21 = land_use_change_max_pays2O21.sort_values('land_use_change_co2', ascending=True)

fig11 = px.bar(land_use_change_max_pays2O21, x='land_use_change_co2', y='country',
             color='land_use_change_co2',
             color_continuous_scale='Viridis',
             labels={'land_use_change_co2': 'tonnes co2'},
             title='Top 5 des pays emetteurs de co2 due au changement affectation des terres en 2021')
fig11.update_layout(showlegend=False)
#fig11.show()


