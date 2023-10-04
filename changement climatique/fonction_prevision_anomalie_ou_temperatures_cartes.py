#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:40:14 2023

@author: harry
"""

""" on importe les packages et les données."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

data=Dataset('/home/harry/projet datascientest/fichiers csv/climate change knowledge portal/timeseries-tas-annual-mean_cru_annual_cru-ts4.06-timeseries_mean_1901-2021.nc')

print(data.variables.keys())

df_1=data.variables['time'][:]
df_2=data.variables['lat'][:]
df_3=data.variables['lon'][:]
df_4=data.variables['timeseries-tas-annual-mean'][:]
df_5=data.variables['bnds'][:]
df_6=data.variables['lon_bnds'][:]
df_7=data.variables['lat_bnds'][:]

""" ce qui va nous interesser surtout c'est df_2, df_3 et df_4. Ces arrays
en les combinant vont nous permettre de dresser une carte du monde
évolutive concernant les températures. """


échantillon=df_4[120] # une tranche du tenseur df_4 indique la température annuelle moyenne pour une année (ici 2020).

def tenseur_to_carts(array=df_4, dates=[1960, 2021], pas=1, zoom=[-90, 90, -180, 180], prevision=2100, prevision_type='anomalie', machine_learning='forest', params={'poly_deg': 2, 'neighbors_numbers': 5, 'forest_trees': 100, 'svr_kernel': 'rbf', 'svr_C': 1.0, 'svr_degree': 3, 'svr_epsilon': 0.1 }):
    
    """ 
    cette fonction transforme le cube de données tenseur en dataframe
    de quatre variables pour chaque point, a savoir la latitude, la longitude
    et l'anomalie de température ou température et aussi l'année. Ensuite affiche les cartes
    et courbes.Retourne le dataframe aussi.
    
    dates est une liste où le premier chiffre est l'année de départ
    que l'on veut et le second l'année d'arrêt. Le premier ne peut pas être
    inférieur à 1931 et le second supérieur à 2021.
    
    prevision_type indique si on veut bosser sur les températures ou alors les anomalies de température.
    
    pas indique l'espace que l'on veut entre chaque année. Par défaut tout
    les ans. Permet de limiter les calculs pour soulager le pc si besoin.
    Par contre plus on augmente le pas, plus on perd en precision et efficacite.
    
    on commence à 1931 car il nous faut les trente premières années pour calculer
    l'anomalie de température sur les températures enregistrées.
    
    zoom permet de zoomer sur la carte et voir l'évolution sur une zone en particulier.
    Cette liste comprend en premier la latitude basse puis haute, et ensuite la 
    longitude basse et haute.Par defaut à peu prêt l'Europe.
    
    
    
    prevision indique en quelle année on veut prédire les températures.
    
    machine_learning indique quelle technique on veut utiliser pour la prevision ('linear':regression linéaire, 'poly':regression polynomiale, 'forest': foret aléatoire, 'svr': support vecteur, 'neighbors': plus proches voisins, 'poly_by_point' pour faire une reg poly par points géogrphiques et pas au global comme 'poly' tout comme 'linear_by_point')
    
    params: permet de régler les parametres du modele de machine learning selectionné.
    
    
    
    
    """
    
    
    if prevision_type=='anomalie':
        
    
        for an in range(array.shape[0]): # permet de convertir les températures en anomalies de températures.
            if an<30:
                pass
            else:
                array[an,:,:]= array[an,:,:]-np.mean(array[an-30:an,:,:], axis=0)
            
    
    
    elif prevision_type=='température':
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
    

    data_nan=pd.DataFrame({'année': year_nan,'latitude': lat_nan, 'longitude': lon_nan })
    data_nan=data_nan.loc[(data_nan.latitude>=zoom[0]) & (data_nan.latitude<=zoom[1]) & (data_nan.longitude>=zoom[2]) & (data_nan.longitude<=zoom[3])]
    
    data_last=data.loc[data.année==data.année.max(),:]
    
    contraste=data.sort_values(by=['latitude', 'longitude', 'année'], ascending=True)
    
    
    point_lat=[] #ces trois listes vont permettre de faire le dataframe avec les pentes.
    point_lon=[]
    point_coef=[]
    
    point_lat_prevision=[] # ces trois listes vont permettre d afficher la prevision sous forme de carte a partir d'un dataframe.
    point_lon_prevision=[]
    point_anomalie_prevision=[]
    
    score=0
    
    for i in contraste.latitude.unique():
        for j in contraste.loc[contraste.latitude==i,:].longitude.unique():
            évolution_tempé=contraste.loc[(contraste.latitude==i) & (contraste.longitude==j),['année', 'anomalie_température']].sort_values(by='année', ascending=True)
            linear_model=LinearRegression()
            linear_model.fit(np.array(évolution_tempé.année).reshape(-1, 1) , np.array(évolution_tempé.anomalie_température))
            point_lat.append(i)
            point_lon.append(j)
            point_coef.append(linear_model.coef_[0])
            
    X=data.drop(['anomalie_température'], axis=1) # X et Y vont etre utile pour le machine learning.
    Y=data['anomalie_température']
    
    
        
    if machine_learning=='poly_by_point': # applique une regression polynomiale par point géographique.
        score_total=0
        point_total=0
        for i in data.latitude.unique():
            for j in data.loc[data.latitude==i,:].longitude.unique():
                
                X=data.loc[(data.latitude==i) & (data.longitude==j),'année']
                Y=data.loc[(data.latitude==i) & (data.longitude==j), 'anomalie_température']
                
                X_train, X_test, y_train, y_test=train_test_split(X, Y, test_size=1, shuffle=False)
                
                poly=PolynomialFeatures(degree=params['poly_deg'])
                X_train_norm=poly.fit_transform(np.array(X_train).reshape(-1, 1))
                X_test_norm=poly.transform(np.array(X_test).reshape(-1, 1))
                model=LinearRegression()
                model.fit(X_train_norm, y_train)
                point_lat_prevision.append(i)
                point_lon_prevision.append(j)
                point_anomalie_prevision.append(model.predict(poly.transform(np.array(prevision).reshape(1, -1)))[0])
                score_point=mean_absolute_error(y_test, model.predict(X_test_norm))
                score_total+=score_point
                point_total+=1
        score=score_total/point_total
        #score=score**len(range(data.année.max() + pas, prevision, pas))
        
        
    elif machine_learning=='linear_by_point': # applique une regression lineaire par point geographique.
        score_total=0
        point_total=0
        for i in data.latitude.unique():
            for j in data.loc[data.latitude==i,:].longitude.unique():
                
                X=data.loc[(data.latitude==i) & (data.longitude==j),'année']
                Y=data.loc[(data.latitude==i) & (data.longitude==j), 'anomalie_température']
                
                X_train, X_test, y_train, y_test=train_test_split(X, Y, test_size=1, shuffle=False)
                
                model=LinearRegression()
                model.fit(np.array(X_train).reshape(-1, 1), y_train)
                point_lat_prevision.append(i)
                point_lon_prevision.append(j)
                point_anomalie_prevision.append(model.predict(np.array(prevision).reshape(1, -1))[0])
                score_point=mean_absolute_error(y_test, model.predict(np.array(X_test).reshape(-1, 1)))
                score_total+=score_point
                point_total+=1
        score=score_total/point_total
        
    elif (machine_learning=='forest') or (machine_learning=='svr') or (machine_learning=='neighbors'):
        new_data=[]
        for i in data.latitude.unique():
            for j in data.loc[data.latitude==i,:].longitude.unique():
                latitude=i
                longitude=j
                data_arbre=data.loc[(data.latitude==i) & (data.longitude==j),['anomalie_température', 'année']].sort_values(by='année', ascending=True)
                df1=data_arbre.drop('année', axis=1)
                df2=pd.DataFrame(np.transpose(df1))
                df2.columns=list(data_arbre.année.unique())
                df2['latitude']=latitude
                df2['longitude']=longitude
                new_data.append(df2)
                
        
        new_data=pd.concat(new_data)
        
        X_forest=new_data.drop(data_arbre.année.max(), axis=1)
        Y_forest=new_data[data_arbre.année.max()]
        
        X_train, X_test, y_train, y_test=train_test_split(X_forest, Y_forest, test_size=0.1)
        
        if machine_learning=='forest': # via une approche directe, applique randomforestregressor.
        
            model=RandomForestRegressor(n_estimators=params['forest_trees'])
            model.fit(X_train, y_train)
            score=mean_absolute_error(y_test, model.predict(X_test))
            
            #score=score**len(range(data_arbre.année.max()+pas, prevision, pas))
            
        elif machine_learning=='svr': # via une approche directe, applique support vector.
            
            model=SVR(kernel=params['svr_kernel'], degree=params['svr_degree'], C=params['svr_C'], epsilon=params['svr_epsilon'])
            scaler=StandardScaler()
            X_norm_train=scaler.fit_transform(X_train)
            X_norm_test=scaler.transform(X_test)
            
            model.fit(X_norm_train, y_train)
            score=mean_absolute_error(y_test, model.predict(X_norm_test))
            
            #score=score**len(range(data_arbre.année.max()+pas, prevision, pas))
            
        elif machine_learning=='neighbors': # via une approche directe, applique la technique des plus proches voisins.
            
            model=KNeighborsRegressor(n_neighbors=params['neighbors_numbers'])
            scaler=StandardScaler()
            X_norm_train=scaler.fit_transform(X_train)
            X_norm_test=scaler.transform(X_test)
            model.fit(X_norm_train, y_train)
            score=mean_absolute_error(y_test, model.predict(X_norm_test))
            
            #score=score**len(range(data_arbre.année.max()+pas, prevision, pas))
        
        rolling_data=X_forest
        for an in range(data_arbre.année.max()+pas, prevision, pas):
            
            prevision_forest=model.predict(rolling_data)
            latitude=rolling_data['latitude']
            longitude=rolling_data['longitude']
            rolling_data=rolling_data.drop(['latitude', 'longitude'], axis=1)
            rolling_data=rolling_data.drop(rolling_data.columns[0], axis=1)
            rolling_data[an]=prevision_forest
            rolling_data['latitude']=latitude
            rolling_data['longitude']=longitude
        
        final_predict=model.predict(rolling_data)
        
       
    

        point_lat_prevision=rolling_data['latitude']
        point_lon_prevision=rolling_data['longitude']
        point_anomalie_prevision=final_predict
            
        

            
    elif machine_learning=='linear': # applique un modele de regression lineaire basique.
        
        retotal=pd.concat([X, Y], axis=1)
        test_set=retotal.loc[retotal.année==retotal.année.max(),:]
        train_set=retotal.loc[retotal.année!=retotal.année.max(),:]
        X_train=train_set.drop('anomalie_température', axis=1)
        y_train=train_set['anomalie_température']
        X_test=test_set.drop('anomalie_température', axis=1)
        y_test=test_set['anomalie_température']
        
        
        
        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)
        model=LinearRegression()
        model.fit(X_train_scaled, y_train)
        for i in data.latitude.unique():
            for j in data.loc[data.latitude==i,:].longitude.unique():
                point_lat_prevision.append(i)
                point_lon_prevision.append(j)
                point_anomalie_prevision.append(model.predict(scaler.transform(np.array([prevision, i, j]).reshape(1, -1)))[0])
                
        score=model.score(X_test_scaled, y_test)
        
    elif machine_learning=='poly': # applique un modele de regression polynomiale simple.
        
        retotal=pd.concat([X, Y], axis=1)
        test_set=retotal.loc[retotal.année==retotal.année.max(),:]
        train_set=retotal.loc[retotal.année!=retotal.année.max(),:]
        X_train=train_set.drop('anomalie_température', axis=1)
        y_train=train_set['anomalie_température']
        X_test=test_set.drop('anomalie_température', axis=1)
        y_test=test_set['anomalie_température']
        
        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)
        poly=PolynomialFeatures(degree=params['poly_deg'])
        X_train_poly=poly.fit_transform(X_train_scaled)
        X_test_poly=poly.transform(X_test_scaled)
        model=LinearRegression()
        model.fit(X_train_poly, y_train)
        for i in data.latitude.unique():
            for j in data.loc[data.latitude==i,:].longitude.unique():
                point_lat_prevision.append(i)
                point_lon_prevision.append(j)
                point_anomalie_prevision.append(model.predict(poly.transform(scaler.transform(np.array([prevision, i, j]).reshape(1, -1))))[0])
            
        score=model.score(X_test_poly, y_test)
        
        
    df_coef=pd.DataFrame({'latitude': point_lat, "longitude": point_lon, "pente": point_coef})
    
    df_prevision=pd.DataFrame({'latitude': point_lat_prevision, 'longitude': point_lon_prevision, 'anomalie': point_anomalie_prevision})
    
    plt.figure(figsize=(20, 20))
    plt.subplot(5,1,1)
    plt.title(f" évolution {prevision_type} selon la région entre {dates[0]} et {dates[1]}")
    plt.xticks([])
    plt.yticks([])
    #df_coef.pente=pd.cut(df_coef.pente, bins=20, labels=[i for i in range(1, 21)])
    plt.scatter(df_coef.longitude, df_coef.latitude, c=df_coef.pente)
    plt.colorbar()
    plt.scatter(df_coef.nlargest(500, 'pente').longitude, df_coef.nlargest(500, 'pente').latitude, c='r')
    plt.subplot(5,1,2)
    plt.title(f"évolution moyenne de {prevision_type} entre {dates[0]} et {dates[1]}")
    data_moyennes=pd.DataFrame(data.groupby('année').agg({'anomalie_température': 'mean'})).reset_index()
    plt.plot(data_moyennes['année'], data_moyennes['anomalie_température'], '-r')
    scaler_2=PolynomialFeatures(degree=2, include_bias=False)
    X_scaled=scaler_2.fit_transform(np.array(data_moyennes.année).reshape(-1,1))
    linear_2=LinearRegression()
    linear_2.fit(X_scaled , np.array(data_moyennes.anomalie_température))
    plt.plot(data_moyennes.année, linear_2.predict(X_scaled), '--b')
    plt.subplot(5,1,3)
    plt.title(f"{prevision_type} en {data_last.année.max()}.")
    plt.xticks([])
    plt.yticks([])
    plt.scatter(data_last.longitude, data_last.latitude, c=data_last.anomalie_température)
    plt.colorbar()
    plt.subplot(5,1,4)
    plt.title(f"prévision des {prevision_type} pour {prevision}, M_A_E:{round(score*100)}%")
    plt.xticks([])
    plt.yticks([])
    plt.scatter(df_prevision.longitude, df_prevision.latitude, c=df_prevision.anomalie)
    plt.colorbar()
    plt.subplot(5,1,5)
    plt.title(f"distribution des {prevision_type} selon {data_last.année.max()} et {prevision}")
    plt.boxplot([data_last.anomalie_température, df_prevision.anomalie], labels=[data_last.année.max(), prevision])
    plt.text(1, -20, f'moyenne:{round(data_last.anomalie_température.mean(),1)}°C')
    plt.text(2, -20, f'moyenne:{round(df_prevision.anomalie.mean(), 1)}°C')
    plt.show()
    
    return data, df_coef, data_nan, data_last, df_prevision, score # retourne dans l ordre le dataframe des données traitées, dataframe des pentes qui permet la premiere carte, le dataframe des nan, le df de la dernière année de données, et enfin le df des prévisions pour l année choisie et le score du modele predictif. 
    
dataset_global, df_pentes, df_nan, data_2021, data_prev, score = tenseur_to_carts(array=df_4, pas=1, dates=[1960, 2021], prevision=2022, prevision_type='température', machine_learning='neighbors')

print(data_2021.anomalie_température.describe())
print(data_prev.anomalie.describe())
print(score) # le score utilise l'erreur absolue moyenne calculée sur la capacité du modele à prévoir l'année suivant les données d'entrainement.


