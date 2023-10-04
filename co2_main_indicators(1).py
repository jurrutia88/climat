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

df = pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\owid-co2-data.csv")

df.info()
print(df.shape)
print(df.country.unique())
print(df.columns)
#df.columns.value_counts()

## Création de df par catégorie de variables

main_indicators= df[['country', 'year','population','gdp','cement_co2', 'co2', 'coal_co2', 
                     'consumption_co2', 'flaring_co2', 'gas_co2', 'land_use_change_co2',
                     'methane', 'nitrous_oxide', 'oil_co2', 'other_industry_co2', 'trade_co2' ]]

## Analyse des corrélations 

main_indicators_ = main_indicators.drop('country', axis=1)
correlation_main = main_indicators_.corr()
plt.figure(figsize=(8, 8))  # Taille de la figure
sns.heatmap(correlation_main, annot=True, cmap='viridis')
plt.title('Correlation df main variables');

## Couper les df par variable et par an

main_indicators2= main_indicators.drop('trade_co2', axis=1) 
main_indicators2=main_indicators2.loc[main_indicators2.year >= 1962]

## Créer les df par zone géo 

"""Nous allons travailler à quatre échelles: pays, continents, income_zones, world"""

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
## Graphiques par pays scores indisctincs

sns.set(rc = {'figure.figsize':(40,40)})
sns.relplot(data=afrique_main_indicators, x='co2', y='country', hue='year', height=10)
plt.title('Evolution des émissions de co2 par pays en Afrique');
sns.relplot(data=ameriques_main_indicators, x='co2', y='country', hue='year', height=10)
plt.title('Evolution des émissions de co2 par pays aux Amériques');
sns.relplot(data=asie_main_indicators, x='co2', y='country', hue='year', height=10)
plt.title('Evolution des émissions de co2 par pays en Asie');
sns.relplot(data=europe_main_indicators, x='co2', y='country',hue='year', height=10)
plt.title('Evolution des émissions de co2 par pays en Europe');
sns.relplot(data=oceanie_main_indicators, x='co2', y='country', hue='year', height=10)
plt.title('Evolution des émissions de co2 par pays en Oceanie');
plt.show()

#  Accumulation co2 par pays

afrique_co2 = afrique_main_indicators[['country', 'year','co2']]
afrique_co2 = afrique_co2.groupby('country')['co2'].sum().reset_index()

ameriques_co2 = ameriques_main_indicators[['country', 'year','co2']]
ameriques_co2 = ameriques_co2.groupby('country')['co2'].sum().reset_index()

asie_co2 = asie_main_indicators[['country', 'year','co2']]
asie_co2 = asie_co2.groupby('country')['co2'].sum().reset_index()

europe_co2 = europe_main_indicators[['country', 'year','co2']]
europe_co2 = europe_co2.groupby('country')['co2'].sum().reset_index()

oceanie_co2 = oceanie_main_indicators[['country', 'year','co2']]
oceanie_co2 = oceanie_co2.groupby('country')['co2'].sum().reset_index()



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
sns.barplot(x='co2', y='country', data=top15_2021, palette='rocket')
plt.title('Emissions de co2 des 15 pays les plus émetteurs en 2021', fontsize=50)
plt.xlabel('billions de tonnes', fontsize=35)
plt.xticks(fontsize=30)
plt.ylabel('pays', fontsize=20)
plt.yticks(fontsize=40)
plt.show()

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
sns.barplot(x='co2', y='country', data=top15_1962, palette='mako')
plt.title('Emissions de co2 des 15 pays les plus émetteurs en 1962', fontsize=50)
plt.xlabel('billions de tonnes', fontsize=35)
plt.xticks(fontsize=30)
plt.ylabel('pays', fontsize=20)
plt.yticks(fontsize=40)
plt.show()




## Créer les df par zone géo: continents

liste_continents= ["Africa", "Asia","Europe", "North America", "Oceania", "South America"]

continents_main_indicators= main_indicators2.loc[(main_indicators2.country.isin(liste_continents))]
continents_main_indicators= continents_main_indicators.drop(['gdp','other_industry_co2',
                                                              'consumption_co2','methane',
                                                              'nitrous_oxide'], axis=1)

continents_main_indicators_= continents_main_indicators.drop(['country'], axis=1)
correlation_continents = continents_main_indicators_.corr()
plt.figure(figsize=(10, 10))  # Taille de la figure
sns.heatmap(correlation_continents, annot=True, cmap='viridis')
plt.title('Correlation df continent réduit en variables');

sns.relplot(data=continents_main_indicators, x='year', y='co2', hue='country', kind='line', palette='husl')
plt.title('Evolution des émissions de co2 par continent')
plt.xlabel('année')
plt.ylabel('billions de tonnes')
plt.show()


## Charbon 
sns.relplot(data=continents_main_indicators, x='year', y='coal_co2', hue='country', kind='line', palette='husl')
plt.title('Evolution des émissions de co2 liées à la production de charbon par continent')
plt.xlabel('année')
plt.ylabel('billions de tonnes')
plt.show()


coal_max_2021 = main_indicators_2021[['country', 'year','coal_co2']]
coal_max_2021 = coal_max_2021.sort_values('coal_co2', ascending=False)

"""Les pays les plus émetteurs de co2 par charbon en 2021 sont China, India, United States,
Japan,Russia, South Africa"""

liste_pays_coalmax= ['China', 'India', 'United States','Japan','Russia']
coal_max_pays2O21= coal_max_2021.loc[(coal_max_2021.country.isin(liste_pays_coalmax))]
coal_max_pays2O21 = coal_max_pays2O21.sort_values('coal_co2', ascending=False)
sns.barplot(x='coal_co2', y='country', data=coal_max_pays2O21, palette='rocket')
plt.title('Top 5 des pays emetteurs de co2 liées à la production de charbon en 2021', fontsize=60, fontweight='bold')
plt.xlabel('tonnes co2')
plt.ylabel('pays')
plt.xticks(fontsize=40)
plt.yticks(fontsize=60, fontweight='bold')
plt.show()

##Pétrole

sns.relplot(data=continents_main_indicators, x='year', y='oil_co2', hue='country', kind='line', palette='husl')
plt.title('Evolution des émissions de co2 provenant du pétrole par continent')
plt.xlabel('année')
plt.ylabel('billions de tonnes')
plt.show()

oil_max_2021 = main_indicators_2021[['country', 'year','oil_co2']]
oil_max_2021 = oil_max_2021.sort_values('oil_co2', ascending=False)

liste_pays_oilmax= ['United States', 'China', 'India','Russia','Japan']
oil_max_pays2O21= oil_max_2021.loc[(oil_max_2021.country.isin(liste_pays_oilmax))]
oil_max_pays2O21 = oil_max_pays2O21.sort_values('oil_co2', ascending=False)
sns.barplot(x='oil_co2', y='country', data=oil_max_pays2O21, palette='rocket')
plt.title('Top 5 des pays emetteurs de co2 provenant du pétrole en 2021', fontsize=60, fontweight='bold')
plt.xlabel('tonnes co2')
plt.ylabel('pays')
plt.xticks(fontsize=40)
plt.yticks(fontsize=60, fontweight='bold')
plt.show()

## Gaz

sns.relplot(data=continents_main_indicators, x='year', y='gas_co2', hue='country', kind='line', palette='husl')
plt.title('Evolution des émissions de co2 issues du gaz par continent')
plt.xlabel('année')
plt.ylabel('billions de tonnes')
plt.show()

gas_max_2021 = main_indicators_2021[['country', 'year','gas_co2']]
gas_max_2021 = gas_max_2021.sort_values('gas_co2', ascending=False)

liste_pays_gasmax= ['United States', 'Russia', 'China','Iran','Saudi Arabia']
gas_max_pays2O21= gas_max_2021.loc[(gas_max_2021.country.isin(liste_pays_gasmax))]
gas_max_pays2O21 = gas_max_pays2O21.sort_values('gas_co2', ascending=False)
sns.barplot(x='gas_co2', y='country', data=gas_max_pays2O21, palette='rocket')
plt.title('Top 5 des pays emetteurs de co2 issu du gaz en 2021', fontsize=60, fontweight='bold')
plt.xlabel('tonnes co2')
plt.ylabel('pays')
plt.xticks(fontsize=40)
plt.yticks(fontsize=60, fontweight='bold')
plt.show()

## Ciment

sns.relplot(data=continents_main_indicators, x='year', y='cement_co2', hue='country', kind='line', palette='husl')
plt.title('Evolution des émissions de co2 liées à la production de ciment')
plt.xlabel('année')
plt.ylabel('billions de tonnes')
plt.show()

cement_max_2021 = main_indicators_2021[['country', 'year','cement_co2']]
cement_max_2021 = cement_max_2021.sort_values('cement_co2', ascending=False)

liste_pays_cementmax= ['China', 'India', 'Vietnam','Turkey','United States']
cement_max_pays2O21= cement_max_2021.loc[(cement_max_2021.country.isin(liste_pays_cementmax))]
cement_max_pays2O21 = cement_max_pays2O21.sort_values('cement_co2', ascending=False)
sns.barplot(x='cement_co2', y='country', data=cement_max_pays2O21, palette='rocket')
plt.title('Top 5 des pays emetteurs de co2 liées à la production de ciment en 2021', fontsize=60, fontweight='bold')
plt.xlabel('tonnes co2')
plt.ylabel('pays')
plt.xticks(fontsize=40)
plt.yticks(fontsize=60, fontweight='bold')
plt.show()

## Torchage

sns.relplot(data=continents_main_indicators, x='year', y='flaring_co2', hue='country', kind='line', palette='husl')
plt.title('Evolution des émissions de co2 liées au torchage par continent')
plt.xlabel('année')
plt.ylabel('billions de tonnes')
plt.show()

flaring_max_2021 = main_indicators_2021[['country', 'year','flaring_co2']]
flaring_max_2021 = flaring_max_2021.sort_values('flaring_co2', ascending=False)

liste_pays_flaringmax= ['United States', 'Russia', 'Iraq','Iran','Brazil']
flaring_max_pays2O21= flaring_max_2021.loc[(flaring_max_2021.country.isin(liste_pays_flaringmax))]
flaring_max_pays2O21 = flaring_max_pays2O21.sort_values('flaring_co2', ascending=False)
sns.barplot(x='flaring_co2', y='country', data=flaring_max_pays2O21, palette='rocket')
plt.title('Top 5 des pays emetteurs de co2 liées au torchage en 2021', fontsize=60, fontweight='bold')
plt.xlabel('tonnes co2')
plt.ylabel('pays')
plt.xticks(fontsize=40)
plt.yticks(fontsize=60, fontweight='bold')
plt.show()


## Changement usage de terres 

sns.relplot(data=continents_main_indicators, x='year', y='land_use_change_co2', hue='country', kind='line', palette='husl')
plt.title('Evolution des émissions de co2 due au changement affectation des terres par continent')
plt.xlabel('année')
plt.ylabel('billions de tonnes')
plt.show()

land_use_change_max_2021 = main_indicators_2021[['country', 'year','land_use_change_co2']]
land_use_change_max_2021 = land_use_change_max_2021.sort_values('land_use_change_co2', ascending=False)

liste_pays_land_use_changemax= ['Indonesia', 'Brazil', 'Russia','Democratic Republic of Congo','China']
land_use_change_max_pays2O21= land_use_change_max_2021.loc[(land_use_change_max_2021.country.isin(liste_pays_land_use_changemax))]
land_use_change_max_pays2O21 = land_use_change_max_pays2O21.sort_values('land_use_change_co2', ascending=False)
sns.barplot(x='land_use_change_co2', y='country', data=land_use_change_max_pays2O21, palette='rocket')
plt.title('Top 5 des pays emetteurs de co2 due au changement affectation des terres en 2021', fontsize=60, fontweight='bold')
plt.xlabel('tonnes co2')
plt.ylabel('pays')
plt.xticks(fontsize=40)
plt.yticks(fontsize=60, fontweight='bold')
plt.show()

## Créer les df par zone géo: income_zones

liste_income_zones= ['High-income countries','Low-income countries',
'Lower-middle-income countries','Upper-middle-income countries']

income_zones_main_indicators= main_indicators2.loc[(main_indicators2.country.isin(liste_income_zones))]
income_zones_main_indicators= income_zones_main_indicators.drop(['gdp','other_industry_co2',
                                                              'consumption_co2','methane',
                                                              'nitrous_oxide'], axis=1)

income_zones_main_indicators_= income_zones_main_indicators.drop(['country'], axis=1)
correlation_income_zones = income_zones_main_indicators_.corr()
plt.figure(figsize=(10, 10))  # Taille de la figure
sns.heatmap(correlation_income_zones, annot=True, cmap='cividis')
plt.title('Correlation df income zones réduit en variables');

## Graphiques du df income zones

plt.figure(figsize=(50, 80))  # Taille de la figure
g=sns.catplot(data=income_zones_main_indicators, x='country', y='co2', kind='boxen', palette='husl')
g.ax.set_xticklabels(['Elévé', 'Faible', 'Moyen inférieur', 'Moyen supérieur'])
plt.title('Emissions de co2 par zone de revenu')
plt.xlabel('zones')
plt.ylabel('tonnes co2')
plt.xticks(rotation=90);


## Créer les df par zone géo: monde

monde = df[df['country']=='World']
monde= monde[monde['year']>=1962]
monde_main_indicators= main_indicators2[main_indicators2['country']=='World']
monde_main_indicators= monde_main_indicators.drop(['gdp','other_industry_co2',
                                                              'consumption_co2','methane',
                                                              'nitrous_oxide'], axis=1)
## Graphiques du df monde 

sns.relplot(data=monde_main_indicators, kind='line', x='year', y='co2', height=5, c='g')
plt.title('Evolution mondiale des émissions de co2')
plt.xlabel('Année')
plt.ylabel('Tonnes de co2');


# Graphique sur les causes des emissions de co2 au niveau mondiale
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(13, 8))

monde_main_indicators.plot(x='year', y='coal_co2', ax=axs[0, 0], title='Production de charbon', c='g')
axs[0, 2].set_title('Production\nde charbon')
monde_main_indicators.plot(x='year', y='oil_co2', ax=axs[0, 1], title='Pétrole', c='r')
monde_main_indicators.plot(x='year', y='gas_co2', ax=axs[0, 2], title='Gas', c='m')
monde_main_indicators.plot(x='year', y='land_use_change_co2', ax=axs[1, 0], title='Changement affectation des terres', c='k')
axs[1,1].set_title('Changement affectation\ndes terres')
monde_main_indicators.plot(x='year', y='cement_co2', ax=axs[1, 1], title='Production de cement', c='y')
monde_main_indicators.plot(x='year', y='flaring_co2', ax=axs[1, 2], title='Torchage', c='b')


plt.subplots_adjust(hspace=0.5, wspace=0.3)
fig.suptitle('Evolution des émissions de co2 par cause:', fontsize=15)
plt.show()

monde_main_indicators.plot(x='year', y=['land_use_change_co2','cement_co2', 'gas_co2','coal_co2','flaring_co2','oil_co2'], style=["k-","y-", "m-", "g-", "b-", "r-"], linewidth=7, title = "Evolution mondiale des émissions de co2 par cause")
plt.legend(prop={'size':40})
plt.xlabel('Year', fontsize=30)
plt.xticks(fontsize=30)
plt.ylabel('Tonnes de co2', fontsize=30)
plt.yticks(fontsize=30)
plt.title("Evolution mondiale des émissions de co2 par cause", fontsize=50);
plt.show()

monde_main_indicators_= monde_main_indicators.drop(['country'], axis=1)
correlation_monde = monde_main_indicators_.corr()
plt.figure(figsize=(10, 10))  # Taille de la figure
sns.heatmap(correlation_monde, annot=True, cmap='magma')
plt.title('Correlation df monde réduit en variables');
plt.show()



