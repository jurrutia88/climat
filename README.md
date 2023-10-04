# Climat
Analyse de l’évolution des anomalies de température au monde

# Introduction
Le réchauffement climatique désigne l'augmentation de la température moyenne de la surface terrestre et la modification des régimes météorologiques et des écosystèmes à grande échelle qui en résultent. Bien qu'il y ait eu des périodes précédentes de changements climatiques, depuis la révolution industrielle les activités humaines ont eu un impact sans précédent sur le système climatique de la Terre et ont provoqué des changements à l'échelle mondiale. En 2016, la température moyenne sur la planète Terre était environ 1 à 1.5 degrés au-dessus des températures moyennes de l’ère pré-industrielle (avant 1850). Pour peu qu’il semble, cette augmentation affecte tout, de la géopolitique aux économies en passant par la migration.

Lorsque l’on parle du réchauffement climatique, on parle notamment de l’effet de serre : on parle donc parfois du réchauffement climatique dit « d’origine anthropique » (d’origine humaine). Il s’agit donc d’une forme dont les causes ne sont pas naturelles mais économiques et industrielles.

Notre hypothèse de travail est que les anomalies de température sont, principalement, la conséquence de la production de CO2, et que cette production est notamment en lien avec deux variables : la population et l’activité industrielle. Conséquemment, dans un premier temps, notre parti pris a été de combiner les données de la NASA présentant les anomalies de températures avec les données de Our World in Data présentant l’évolution des émissions de CO2. Une fois les corrélations entre anomalies de température et CO2 et populations établies, il s’agirait, dans un deuxième temps, d’en comprendre les principales causes au niveau des pays, des continents et du monde entier.

D’entrée de jeu, et en fonction d’un premier regard des bases de données, nous nous sommes fixé un certain nombre de critères d’analyse :

# Objectifs

Présenter l’évolution des anomalies de température à différentes échelles géographiques (pays, continents, …)
Présenter l’évolution des émissions de CO2 ainsi que leurs causes à différentes échelles géographiques (pays, continents, …)
Présenter la corrélation entre l’évolution des anomalies de température et l’évolution des émissions de CO2 à différentes échelles géographiques (pays, continents, …) et explorer d’autres corrélations pertinentes.
Réaliser de prédictions concernant l’évolution des anomalies de température sur différentes échelles géographiques, utilisant les méthodes les plus appropriées
Analyser les scenarii d’évolution des anomalies de température (de continuité et de rupture) et leurs conséquences pour la planète à moyen-long terme.

# Datasets
Analyse des évolutions des anomalies de température
L’anomalie de température c'est l'écart entre la température du lieu mesurée en degrés celsius, positive ou négative, au regard de la température moyenne normale. La température moyenne normale n'est rien d'autre que la température locale (du lieu mesuré) moyenne calculée sur au moins trente ans. Concernant l'étude des anomalies et les températures, nous allons utiliser trois datasets issus du site de la NASA (GITSEMP), de Kaggle (Climate change) et du portail web d'accès aux données de changement climatique (Climate Change Knowledge Portal).

GITSEMP

L'analyse de la température de surface globale de la NASA Goddard (GISTEMP) combine les températures de l'air de surface terrestre principalement de la version GHCN-M 4 avec les SST de l'analyse ERSSTv5 dans un ensemble complet de données de température de surface mondiale couvrant 1880 à nos jours à une résolution mensuelle, sur une latitude de 2x2 degrés -grille de longitude. En tant que tel, il s'agit de l'un des principaux ensembles de données utilisés pour surveiller la variabilité et les tendances de la température mondiale et régionale. GISTEMP a une plus grande couverture polaire que MLOST ou HadCRUT4 en raison de la méthode d'interpolation.
Nous avons utilisé ces fichiers csv pour afficher l'évolution de la température de 1880 à 2022 au niveau mondial, puis au niveau des hémisphères, bref on va zoomer au fur et à mesure. Dans un premier temps on va utiliser le fichier csv 'ZonnAnn.Ts+dSST.csv'. Ces fichiers ne présentent pas de nan mais des valeurs aberrantes ('****') au début et à la fin. Nous avons fait le choix d'interpoler avec la valeur suivante pour la première ligne, et la valeur précédente pour la dernière ligne. Ensuite on va utiliser le premier fichier des mesures AIRS de 2002 à 2022.

Climate Change (Kaggle)

Le projet appelé Climate Change interroge la réalité de l'augmentation des températures liée aux activités industrielles et à l'effet de serre pour chaque zone. Sur leur page web on a la possibilité de charger et exploiter un fichier csv qui exprime le changement de température par pays dans le monde entre 1961 et 2019.
Pour le traitement des valeurs manquantes nous avons 1) crée un tableau qui compte les Nan grâce aux fonctions sort_values, groupby et agg, 2) Interpolé grâce à la fonction simpleImputer les données pour les variables dans lesquelles le nombre des Nan était de moins de 30% et viré les variables pour lesquelles il en avait plus que cette limite, 3) Appliqué une régression linéaire pour chaque pays sur ses données de températures, dans le but d'obtenir le coefficient directeur de la droite affine pour lisser ces données de température et classer les pays par ordre d'importance, 4) Crée un nouveau df grâce à une boucle ordonnant les pays en fonction de leur coefficient

Climate Change Knowledge Portal

Ce portail web fournit des données historiques sur le climat ainsi que des prédictions. On peut télécharger les données selon chaque pays, selon une période donnée, ou alors toutes les données.
L'objectif a été de visualiser sur une carte du globe les zones les plus impactées par le changement climatique. Pour ceci, nous avons 1) Importé le package netCDF4, 2) Crée un dictionnaire où pour chaque clé il y a une information, 3) Rédigé une fonction qui va convertir le tenseur en cartes et courbe

Analyse des évolutions des émissions de CO2

Nous allons explorer le dataset “owid-co2-data” du site OurWorldinData afin de comprendre 1) l’évolution des émissions de CO2 dans le monde et ses principales cause, 2) dans quelles mesures les émissions de CO2 sont corrélées avec les anomalies de température.
Le pré-processing a consisté notamment en:

· A l’échelle temporelle, couper les dataframes à partir de 1962, afin de les faire coïncider avec le dataframe des anomalies de température et pouvoir les travailler ensemble a posteriori (cohérence temporelle).

· Au niveau du type d’indicateurs, réaliser des analyses spécifiques à partir de types et natures d’indicateurs différents : les indicateurs qui mesurent l’évolution annuelle des entités géographiques traitées (main indicateurs), les indicateurs qui mesurent l’évolution per capita des entités géographiques traitées (indicateurs per capita), et les indicateurs qui mesurent l’évolution cumulative des entités géographiques traitées (indicateurs cumulatifs)

· Au niveau des indicateurs à retenir, et afin d’assurer des analyses comparatives solides et des prédictions fines, nous avons gardé les variables explicatives, d'après le test de Pearson, de l’augmentation du CO2 ne présentant pas de Nan au delà du 30% : coal_co2, oil_co2, gas_co2, land_use_change_co2, cement_co2, flaring_co2.

Voici à quoi ressemble notre dataframe avant le pré-processing
