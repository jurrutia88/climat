#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:39:25 2023

@author: harry
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as geo
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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




''' Avant de modifier le dataset pour le machine learning, on va revérifier la correlation entre variables.'''

correlation=data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(correlation, annot=True, cmap='viridis')
plt.show()

''' On remarque une forte correlation entre les variables co2, coal_co2, population, oil_co2, gas_co2, land_change_co2 et flaring_co2
. Les émissions de co2 peuvent donc etre prédites sur ses variables en entrée. on remarque cependant une faible correlation entre le co2 et 
la température et l anomalie, mais ici nous sommes à l'échelle nationale. On va ajouter des variables à data de la température et anomalie
mondiale et co2 moyenne par ans, puis refaire une matrice de correlation.'''

temperature_moyenne_mondiale=data[['température', 'year']].groupby('year').agg('mean')
data_corr=data.merge(temperature_moyenne_mondiale, on='year', how='inner')


anomalie_moyenne_mondiale=data[['anomalie', 'year']].groupby('year').agg('mean')
data_corr=data_corr.merge(anomalie_moyenne_mondiale, on='year', how='inner')


co2_moyenne_mondiale=data[['co2', 'year']].groupby('year').agg('mean')
data_corr=data_corr.merge(co2_moyenne_mondiale, on='year', how='inner')

data_corr=data_corr.rename(columns={'température_y': 'température_moy_mon', 'anomalie_y': 'anomalie_moy_mon', 'co2_y': 'co2_moy_mon', 'température_x':'température', 'anomalie_x': 'anomalie', 'co2_x': 'co2'})
correlation_2=data_corr.corr()

plt.figure(figsize=(12,12))
sns.heatmap(correlation_2, annot=True, cmap='viridis')
plt.show()

''' là on remarque beaucoup plus la correlation entre le co2 et la température  ainsi que anomalie
quand nous basculons à l'échelle mondiale. Elle est meme plutot forte. On va donc proceder en deux etapes
premierement predire le co2 a partir des variables causales (charbon, petrole, gas, population, terrain, ciment, torchage) et par année, pays, continent.'''

''' tests de pearson co2/variables causales'''

print(f"co2/population:{pearsonr(data.co2, data.population)}")
print(f"co2/charbon:{pearsonr(data.co2, data.coal_co2)}")
print(f"co2/petrole:{pearsonr(data.co2, data.oil_co2)}")
print(f"co2/ciment:{pearsonr(data.co2, data.cement_co2)}")
print(f"co2/torchage:{pearsonr(data.co2, data.flaring_co2)}")
print(f"co2/gaz:{pearsonr(data.co2, data.gas_co2)}")
print(f"co2/terrain:{pearsonr(data.co2, data.land_use_change_co2)}") 

''' On retouve bien les scores de la matrice de correlation.'''


''' etape un : modele predictif du co2 émis. Comme les correlations sont fortes et positives, un modele lineaire semble approprié.'''

variables_causales=['population', 'coal_co2', 'oil_co2', 'gas_co2', 'land_use_change_co2', 'cement_co2', 'flaring_co2']
df=data_corr[['year', 'country', 'continent', 'co2', 'population', 'coal_co2', 'oil_co2', 'gas_co2', 'land_use_change_co2', 'cement_co2', 'flaring_co2']]
df=pd.get_dummies(df, columns=['country', 'continent']) # on traite les variables categorielle non ordinales.
scaler=StandardScaler() # on normalise les variables car elles ne sont pas a la meme echelle.

training_set=df.loc[df.year<2000,:] # on va évaluer le modele sur sa capacité à predire la moyenne de co2 émise par ans après 2000.
test_set=df.loc[df.year>=2000,:]

X_train=training_set.drop('co2', axis=1)
X_train[variables_causales]=scaler.fit_transform(X_train[variables_causales])

y_train=training_set['co2']

X_test=test_set.drop('co2', axis=1)
X_test[variables_causales]=scaler.transform(X_test[variables_causales])

y_test=test_set['co2']


model=LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test)) # le score semble très bon, 99%...Trop beau pour etre vrai ?
print(mean_absolute_error(y_test, model.predict(X_test))) # erreur absolue moyenne de 1.46.

''' si on teste le modele avec une année antérieure à 2000, on voit que plus l'on avance dans
le temps, plus le decalage se forme entre la moyenne d emmissions de co2 annuelle et les prédictions. Donc ça semble
a priori cohérent. '''

''' on va comparer la prediction du co2 avec celle qui est réelle en graphique sur les 20 dernières années.'''

predictions=model.predict(X_test)
co2_global_annuel=df[['co2', 'year']].groupby('year').agg('mean')
années_test=predictions[0]
co2_global_annuel_test=pd.DataFrame({'year':X_test.year, 'co2': predictions})
co2_global_annuel_test=co2_global_annuel_test[['co2', 'year']].groupby('year').agg('mean')


plt.figure(figsize=(20, 10))
plt.title('évolution globale moyenne annuelle de co2')
plt.plot(co2_global_annuel.index, co2_global_annuel.co2, '--g', label='données')
plt.plot(co2_global_annuel_test.co2, 'r', label='prediction')
plt.legend()
plt.show() # les courbes restent assez proches, le modele a l air bon.

''' etape deux: modele de regression lineaire pour predire la temperature moyenne mondiale annuelle a partir du co2 moyen annuel émis. '''

df_2=data[['co2', 'température', 'year']].groupby('year').agg('mean')

print(pearsonr(df_2.co2, df_2.température)) # coefficient à 90% et pvalue à moins de 0.05 ce qui confirme le résultat précédent de la matrice de correlation.


training_set_2=df_2.loc[df_2.index<2000,:]
test_set_2=df_2.loc[df_2.index>=2000,:]


X_train_2=np.array(training_set_2['co2']).reshape(-1, 1)
y_train_2=training_set_2['température']


X_test_2=np.array(test_set_2['co2']).reshape(-1, 1)
y_test_2=test_set_2['température']


model_2=LinearRegression()
model_2.fit(X_train_2, y_train_2)
print(model_2.score(X_train_2, y_train_2)) # score à 42%, pas terrible.
print(mean_absolute_error(y_test_2, model_2.predict(X_test_2))) # mais erreur absolue a seulement 0.26.

'''  on compare la prediction de la temperature avec les données reelles.'''

predictions_2=model_2.predict(X_test_2) # predictions températures.
ans=[i for i in range(2000, 2020)] # années test du modele.
moyenne=LinearRegression() # va permettre de comparer la prediction de temperature à la tendance moyenne des données plutôt que les données brutes.
moyenne.fit(np.array(df_2.index).reshape(-1, 1), df_2.température )
moyenne=moyenne.predict(np.array(df_2.index).reshape(-1, 1))

plt.figure(figsize=(20, 10))
plt.title('comparaison temperature moyenne annuelle et sa prévision')
plt.plot(df_2.index, df_2.température, '--g', label='données réelles')
plt.plot(df_2.index, moyenne, 'y', label='tendance des données')
plt.plot(ans,predictions_2, 'r',  label='predictions')
plt.legend()
plt.show()

''' le modele suit la tendance, legerement à la baisse.'''

''' maintenant on va tester notre modele à deux étapes, en essayant de faire des prédictions pour 2100.'''

''' hypothèse 1: rien ne change. Pour ce cas de figure on va faire une regression lineaire sur chaque variable causale,
pour chaque pays, pour garder la tendance moyenne sur les années de données de 1960 à 2019, que l'on va garder pour faire evoluer
ces variables sur la meme pente jusqu en 2100 '''


variables_causales=['population', 'coal_co2', 'oil_co2', 'gas_co2', 'land_use_change_co2', 'cement_co2', 'flaring_co2']
dataframe=data[variables_causales+['co2', 'country', 'continent']]
durée_données=[an for an in range(debut, fin+1)]
durée_prédiction=[an for an in range(fin+1, 2101)]

causes_estimées=[] # dans cette liste on recupere les causes d emission de co2 estimées sur 2020/2100.
for pays in dataframe.country.unique():
    predictions_causales=[]
    continent=dataframe.loc[dataframe.country==pays, 'continent'].unique()[0]
    for var in variables_causales:
        model_causal=LinearRegression()
        train=np.array(durée_données).reshape(-1,1)
        estimation=np.array(durée_prédiction).reshape(-1,1)
        test=np.array(dataframe.loc[dataframe.country==pays, var]).reshape(-1, 1)
        model_causal.fit(train, test)
        prediction_causale=model_causal.predict(estimation)
        predictions_causales.append(list(prediction_causale))
    estimation_pays_causes=np.concatenate(predictions_causales, axis=1)
    estimation_pays_causes=pd.DataFrame(estimation_pays_causes, index=durée_prédiction, columns=variables_causales)
    estimation_pays_causes['pays']=pays
    estimation_pays_causes['continent']=continent
    causes_estimées.append(estimation_pays_causes)
    
causes_estimées=pd.concat(causes_estimées) # on a notre dataframe d entrainement pour estimer l emission de co2.
    
''' maintenant on va predire ces estimations sur une regression lineaire pour estimer le co2 moyen émis par ans. '''

variables_causales_year=variables_causales+['year']
co2_model=LinearRegression()
co2_scaler=StandardScaler()

X_train_co2=df.drop('co2', axis=1)

X_train_co2[variables_causales_year]=co2_scaler.fit_transform(X_train_co2[variables_causales_year])
y_train=df['co2']


co2_model.fit(X_train_co2, y_train)
co2_model.predict(X_train_co2) # on a entrainé notre modele predictif du co2.

causes_estimées=causes_estimées.reset_index()
causes_estimées=causes_estimées.rename(columns={'index': 'year'}) # on met les données causales estimées au bon format.
causes_estimées_2=pd.get_dummies(causes_estimées, columns=['pays', 'continent'])
causes_estimées_2[variables_causales_year]=co2_scaler.transform(causes_estimées_2[variables_causales_year])


co2_pred=co2_model.predict(causes_estimées_2)
co2_pred=pd.DataFrame({'year': causes_estimées_2.year, 'co2': co2_pred})
co2_pred=co2_pred.groupby('year').agg('mean') # on obtient un dataframe avec la moyenne de co2 émise estimée par ans de 2020 a 2100.

''' second temps, avec ce resultat prevision de la temperature globale moyenne.'''

temp_model=LinearRegression()

X=data[['year', 'température', 'co2']].groupby('year').agg('mean')
X_train_tempé=X.drop('température', axis=1)
y_train_tempé=X['température']

X_train_tempé=np.array(X_train_tempé).reshape(-1, 1)
temp_model.fit(X_train_tempé, y_train_tempé)

tempé_pred=temp_model.predict(co2_pred) 

''' petite visualisation'''

plt.figure(figsize=(20,10))
plt.title('hypothese 1: évolution de la température moyenne si les émissions de co2 évoluent globalement de la meme façon.')
plt.plot(durée_prédiction, tempé_pred, 'g', label='prédiction')
plt.plot(durée_données, X['température'], 'b', label='données')
plt.legend()
plt.show() # le modele prevoit une augmentation de environ deux degrés, ce qui rejoint l'estimation du modele des forets aleatoires sur le jeu de données des températures uniquement.


''' hypothèse deux: on va modifier les données de certains pays gros producteurs de co2 pour voir l impact dans le temps.
 Chine: -50% sur charbon, pétrole, ciment, torchage.
 USA: -50% sur pétrole, gaz, torchage.
 Russie: -50% sur gaz.
 Indonésie: -50% sur affectation des terres.
 Brésil: -50% sur affectation des terres.
 On considère la baisse comme progressive pour plus de réalisme, et le reste des pays restera axé sur l'évolution estimées calculées précédemment.'''
 
 
''' Chine '''

chine_last_var=df.loc[(df.country_China!=0),['year', 'coal_co2', 'oil_co2', 'cement_co2', 'flaring_co2']]
chine_last_var=chine_last_var.loc[chine_last_var.year==2019, ['coal_co2', 'oil_co2', 'cement_co2', 'flaring_co2']] # dernieres valeurs de 2019 Chine.
chine_decrease_var=chine_last_var.apply(lambda x: x*0.5) # reduction des dernières valeurs de 50%

chine_new_estimation=pd.DataFrame({'coal_co2': np.linspace(float(chine_last_var.coal_co2), float(chine_decrease_var.coal_co2 ), len(durée_prédiction)),
                                   'oil_co2': np.linspace(float(chine_last_var.oil_co2), float(chine_decrease_var.oil_co2), len(durée_prédiction)),
                                   'cement_co2': np.linspace(float(chine_last_var.cement_co2), float(chine_decrease_var.cement_co2), len(durée_prédiction)),
                                   'flaring_co2': np.linspace(float(chine_last_var.flaring_co2), float(chine_decrease_var.flaring_co2), len(durée_prédiction))}, index=durée_prédiction).reset_index()

chine_new_estimation=chine_new_estimation.rename(columns={'index':'year'})   
            
# ce dataframe contient la baisse voulue dans notre seconde hypothèse.

''' USA ''' 

usa_last_var=df.loc[(df['country_United States of America']!=0),['year', 'oil_co2', 'flaring_co2', 'gas_co2']]
usa_last_var=usa_last_var.loc[usa_last_var.year==2019, ['oil_co2', 'flaring_co2', 'gas_co2']] # dernieres valeurs de 2019 usa.
usa_decrease_var=usa_last_var.apply(lambda x: x*0.5) # reduction des dernières valeurs de 50%

usa_new_estimation=pd.DataFrame({
                                   'oil_co2': np.linspace(float(usa_last_var.oil_co2), float(usa_decrease_var.oil_co2), len(durée_prédiction)),
                                   'gas_co2': np.linspace(float(usa_last_var.gas_co2), float(usa_decrease_var.gas_co2), len(durée_prédiction)),
                                   'flaring_co2': np.linspace(float(usa_last_var.flaring_co2), float(usa_decrease_var.flaring_co2), len(durée_prédiction))}, index=durée_prédiction).reset_index()

usa_new_estimation=usa_new_estimation.rename(columns={'index':'year'})   
        
''' Russie '''

russie_last_var=df.loc[(df.country_Russia!=0),['year', 'gas_co2']]
russie_last_var=russie_last_var.loc[russie_last_var.year==2019, ['gas_co2']] # dernieres valeurs de 2019 Russie.
russie_decrease_var=russie_last_var.apply(lambda x: x*0.5) # reduction des dernières valeurs de 50%

russie_new_estimation=pd.DataFrame({'gas_co2': np.linspace(float(russie_last_var.gas_co2), float(russie_decrease_var.gas_co2 ), len(durée_prédiction))}, index= durée_prédiction).reset_index()
russie_new_estimation=russie_new_estimation.rename(columns={'index':'year'})   
                                          
    
''' Indonésie ''' 

indo_last_var=df.loc[(df.country_Indonesia!=0),['year', 'land_use_change_co2']]
indo_last_var=indo_last_var.loc[indo_last_var.year==2019, ['land_use_change_co2']] # dernieres valeurs de 2019 Indonésie.
indo_decrease_var=indo_last_var.apply(lambda x: x*0.5) # reduction des dernières valeurs de 50%

indo_new_estimation=pd.DataFrame({'land_use_change_co2': np.linspace(float(indo_last_var.land_use_change_co2), float(indo_decrease_var.land_use_change_co2), len(durée_prédiction))}, index= durée_prédiction).reset_index()
indo_new_estimation=indo_new_estimation.rename(columns={'index':'year'})   
        

''' Brésil '''   
    
bresil_last_var=df.loc[(df.country_Brazil!=0),['year', 'land_use_change_co2']]
bresil_last_var=bresil_last_var.loc[bresil_last_var.year==2019, ['land_use_change_co2']] # dernieres valeurs de 2019 Brésil.
bresil_decrease_var=bresil_last_var.apply(lambda x: x*0.5) # reduction des dernières valeurs de 50%

bresil_new_estimation=pd.DataFrame({'land_use_change_co2': np.linspace(float(bresil_last_var.land_use_change_co2), float(bresil_decrease_var.land_use_change_co2), len(durée_prédiction))}, index= durée_prédiction).reset_index()   
bresil_new_estimation=bresil_new_estimation.rename(columns={'index':'year'})   
    
''' Inde '''

inde_last_var=df.loc[(df.country_India!=0), ['year', 'coal_co2', 'cement_co2']]
inde_last_var=inde_last_var.loc[inde_last_var.year==2019, ['coal_co2', 'cement_co2']]  
inde_decrease_var=inde_last_var.apply(lambda x: x*0.5) 

inde_new_estimation=pd.DataFrame({'coal_co2': np.linspace(float(inde_last_var.coal_co2), float(inde_decrease_var.cement_co2), len(durée_prédiction)),
                                  'cement_co2': np.linspace(float(inde_last_var.cement_co2), float(inde_decrease_var.cement_co2), len(durée_prédiction))}, index=durée_prédiction).reset_index()
inde_new_estimation=inde_new_estimation.rename(columns={'index': 'year'})

''' on va modifier le dataframe causes estimées pour tester la seconde hypothèse. '''  

''' chine'''

causes_deux=causes_estimées

for an in range(2020, 2101):
    for var in ['coal_co2', 'oil_co2', 'cement_co2', 'flaring_co2']:
        val=chine_new_estimation.loc[chine_new_estimation.year==an, var]
        causes_deux.loc[(causes_deux.pays=='China') & (causes_deux.year==an),var]=float(val)
    for var in ['oil_co2', 'gas_co2', 'flaring_co2']:
        val=usa_new_estimation.loc[usa_new_estimation.year==an, var]
        causes_deux.loc[(causes_deux.pays=='United States of America') & (causes_deux.year==an),var]=float(val)
    for var in ['gas_co2']:
        val=russie_new_estimation.loc[russie_new_estimation.year==an, var]
        causes_deux.loc[(causes_deux.pays=='Russia') & (causes_deux.year==an),var]=float(val)
    for var in ['land_use_change_co2']:
        val=indo_new_estimation.loc[indo_new_estimation.year==an, var]
        causes_deux.loc[(causes_deux.pays=='Indonesia') & (causes_deux.year==an),var]=float(val)
    for var in ['land_use_change_co2']:
        val=bresil_new_estimation.loc[bresil_new_estimation.year==an, var]
        causes_deux.loc[(causes_deux.pays=='Brazil') & (causes_deux.year==an),var]=float(val)
    for var in ['cement_co2', 'coal_co2']:
        val=inde_new_estimation.loc[inde_new_estimation.year==an, var]
        causes_deux.loc[(causes_deux.pays=='India') & (causes_deux.year==an), var]=float(val)



''' on modifie causes deux pour etre utilisé par co2_model '''

causes_deux=pd.get_dummies(causes_deux, columns=['pays', 'continent'])
causes_deux[variables_causales_year]=co2_scaler.transform(causes_deux[variables_causales_year])


hyp2_co2_pred=co2_model.predict(causes_deux)
hyp2_co2_pred=pd.DataFrame({'year': causes_deux.year, 'co2': hyp2_co2_pred})
hyp2_co2_pred=hyp2_co2_pred.groupby('year').agg('mean')


''' on prevoit mainteneant la température pour la seconde hypothese '''

hyp2_tempé_pred=temp_model.predict(hyp2_co2_pred)


''' visualisation finale '''

plt.figure(figsize=(20,10))
plt.title('Comparaison des scenarios selon le modele lineaire.')
plt.plot(durée_prédiction, tempé_pred, 'g', label='Scenario 1')
plt.plot(durée_prédiction, hyp2_tempé_pred, 'y*', label='Scenario 2')
plt.plot(durée_données, X['température'], 'b', label='Evolution température moyenne globale')
plt.legend()
plt.xlabel('Années')
plt.ylabel('Température en degrés Celsius')
plt.show()

''' conclusion: en 2100 la température va continuer à augmenter mais beaucoup moins,
moins de un degré. C est moins pire... '''


print(f'un: {causes_estimées_2.flaring_co2.mean()}')
print(f'deux:{causes_deux.flaring_co2.mean()}')
