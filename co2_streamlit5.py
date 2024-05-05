# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:16:00 2023

@author: jt_ur
"""

import streamlit as st 
from PIL import Image
import pandas as pd
import base64
from streamlit_option_menu import option_menu

from co2_plotly import fig1
from co2_plotly import fig2
from co2_plotly import fig3
from co2_plotly import fig4
from co2_plotly import fig5
from co2_plotly import fig6
from co2_plotly import fig7
from co2_plotly import fig8
from co2_plotly import fig9
from co2_plotly import fig10
from co2_plotly import fig11


# Image fixe 

image = Image.open (r"C:\Users\jt_ur\OneDrive\Documents\DataScientest\Projet\5fc50d4d94402.jpg")
image = image.resize((900, 250))
alpha = Image.new('L', image.size, 188) # Créer un canal alpha avec une opacité de 50%
image.putalpha(alpha)
st.image(image)

# code CSS 
st.markdown(
    """
    <style>
    .sidebar {
        font-family: 'ff-meta-web-pro', sans-serif;
    }
    .css-vk3wp9 {
        background-color: #FFFAF0;
        color: #000000;
    }
    .bi-linkedin {
        color: #063672;
    }
    .css-promotion {
        margin-top: 20px;
    }
    .css-promotion-text {
        color: #000000;
    }
    .datascientest-logo {
        width: 25px;
        height: auto;
        display: inline-block;
        margin-left: 5px;
    }
    </style>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    """,
    
    unsafe_allow_html=True,
)
# Sidebar
st.sidebar.title('Température Terrestre : Evolution du dérèglement climatique global')
options = st.sidebar.radio('Table des matières', options=["Introduction", "Datasets", "Visualisations et analyses", "Prédictions", "Conclusions"])


st.sidebar.markdown(
    """
    <style>
        .sidebar .sidebar-content h1, .sidebar .sidebar-content label {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
    <p class="css-vk3wp9">Thomas BARRALIER  <a href=https://www.linkedin.com/in/thomas-barralier-45282197/><i class="bi bi-linkedin"></i></a></p>
    <p class="css-vk3wp9">José URRUTIA  <a href="https://www.linkedin.com/in/jurrutia88/"><i class="bi bi-linkedin"></i></a></p>
    <p class="css-vk3wp9 css-promotion"><span class="css-promotion-text">Promotion Continue DA - Octobre 2002</span><a href="https://datascientest.com/" target="_blank"><img src="https://datascientest.com/en/wp-content/uploads/sites/9/2020/08/new-logo.png" alt="Datascientest" class="datascientest-logo" /></a></p>
    """,
    unsafe_allow_html=True,
)


def intro ():
    st.title(':green[Introduction]')
    st.write("Le **réchauffement climatique** désigne l'**augmentation de la température moyenne** de la surface terrestre et la **modification des régimes météorologiques et des écosystèmes** à grande échelle qui en résultent. Bien qu'il y ait eu des périodes précédentes de changements climatiques, depuis la révolution industrielle les activités humaines ont eu un impact sans précédent sur le système climatique de la Terre et ont provoqué des changements à l'échelle mondiale. En 2016, la température moyenne sur la planète Terre était environ **1 à 1.5 degrés au-dessus des températures moyennes de l’ère pré-industrielle** (avant 1850). Pour peu qu’il semble, cette augmentation affecte tout, de la géopolitique aux économies en passant par la migration.")
    st.write("Lorsque l’on parle du réchauffement climatique, on parle notamment de l’effet de serre : on parle donc parfois du réchauffement climatique dit « **d’origine anthropique** » (d’origine humaine). Il s’agit donc d’une forme dont **les causes ne sont pas naturelles mais économiques et industrielles**.")
    st.write("Notre **hypothèse de travail** est que les **anomalies de température** sont, principalement, la **conséquence de la production de CO2**, et que cette production est notamment **en lien avec deux variables** : la **population** et l’**activité industrielle**. Conséquemment, dans un premier temps, notre parti pris a été de combiner les données de la **NASA** présentant les anomalies de températures avec les données de **Our World in Data** présentant l’évolution des émissions de CO2. Une fois les corrélations entre anomalies de température et CO2 et populations établies, il s’agirait, dans un deuxième temps, d’en comprendre les principales causes au niveau des pays, des continents et du monde entier.")
    st.write("D’entrée de jeu, et en fonction d’un premier regard des bases de données, nous nous sommes fixé un certain nombre de critères d’analyse :")
    st.title(':green[Objectifs]')
    st.write("1. Présenter **l’évolution des anomalies de température** à différentes échelles géographiques (pays, continents, …)")
    st.write("2. Présenter **l’évolution des émissions de CO2** ainsi que **leurs causes** à différentes échelles géographiques (pays, continents, …)")
    st.write("3. Présenter la **corrélation** entre l’évolution des anomalies de température et l’évolution des émissions de CO2 à différentes échelles géographiques (pays, continents, …) et explorer d’autres corrélations pertinentes.")
    st.write("4. Réaliser de **prédictions** concernant l’évolution des anomalies de température sur différentes échelles géographiques, utilisant les méthodes les plus appropriées")
    st.write("5. Analyser **les scenarii** d’évolution des anomalies de température (de continuité et de rupture) et **leurs conséquences** pour la planète à moyen-long terme.")

def datasets ():
    st.title(':green[Datasets]')
    
    st.header(':green[Analyse des évolutions des anomalies de température]')
    st.write("L’**anomalie de température** c'est l'**écart entre la température du lieu** mesurée en degrés celsius, positive ou négative, **au regard de la température moyenne normale**. La température moyenne normale n'est rien d'autre que la température locale (du lieu mesuré) moyenne calculée sur au moins trente ans. Concernant l'étude des anomalies et les températures, nous allons utiliser trois datasets issus du site de la NASA (**GITSEMP**), de Kaggle (**Climate change**) et du portail web d'accès aux données de changement climatique (**Climate Change Knowledge Portal**).")
    
    st.write("**GITSEMP**")
    st.write("L'analyse de la température de surface globale de la NASA Goddard (GISTEMP) combine les températures de l'air de surface terrestre principalement de la version GHCN-M 4 avec les SST de l'analyse ERSSTv5 dans un ensemble complet de données de température de surface mondiale couvrant 1880 à nos jours à une résolution mensuelle, sur une latitude de 2x2 degrés -grille de longitude. En tant que tel, il s'agit de l'un des principaux ensembles de données utilisés pour surveiller la variabilité et les tendances de la température mondiale et régionale. GISTEMP a une plus grande couverture polaire que MLOST ou HadCRUT4 en raison de la méthode d'interpolation.")
    st.write("Dataframe 1880-2002")
    df1 = pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\ZonAnn.Ts+dSST_1880-2002.csv")
    st.table(df1.head(5))
    st.write("Dataframe 2002-2022")
    df2 = pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\ZonAnn.Ts+dSST_2002-2021.csv")
    st.table(df2.head(5))
    st.write("Nous avons utilisé ces fichiers csv pour afficher l'évolution de la température de 1880 à 2022 au niveau mondial, puis au niveau des hémisphères, bref on va zoomer au fur et à mesure. Dans un premier temps on va utiliser le fichier csv 'ZonnAnn.Ts+dSST.csv'. Ces fichiers ne présentent pas de nan mais des valeurs aberrantes ('****') au début et à la fin. Nous avons fait le choix d'interpoler avec la valeur suivante pour la première ligne, et la valeur précédente pour la dernière ligne. Ensuite on va utiliser le premier fichier des mesures AIRS de 2002 à 2022.")
    
    st.write("**Climate Change (Kaggle)**")
    st.write("Le projet appelé Climate Change interroge la réalité de l'augmentation des températures liée aux activités industrielles et à l'effet de serre pour chaque zone. Sur leur page web  on a la possibilité de charger et exploiter un fichier csv qui exprime le changement de température par pays dans le monde entre 1961 et 2019.")
    df3 = pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\data_temperature_co2_anomalie.csv")
    st.table(df3.head(5))
    st.write("Pour le traitement des valeurs manquantes nous avons 1) crée un tableau qui compte les Nan grâce aux fonctions sort_values, groupby et agg, 2) Interpolé grâce à la fonction simpleImputer les données pour les variables dans lesquelles le nombre des Nan était de moins de 30% et viré les variables pour lesquelles il en avait plus que cette limite, 3) Appliqué une régression linéaire pour chaque pays sur ses données de températures, dans le but d'obtenir le coefficient directeur de la droite affine pour lisser ces données de température et classer les pays par ordre d'importance, 4) Crée un nouveau df grâce à une boucle ordonnant les pays en fonction de leur coefficient")
    
    st.write("**Climate Change Knowledge Portal**")
    st.write("Ce portail web  fournit des données historiques sur le climat ainsi que des prédictions. On peut télécharger les données selon chaque pays, selon une période donnée, ou alors toutes les données.")
    df4 = pd.read_csv(r"C:\Users\jt_ur\DA\datasets\changement climatique\Temperature_change_Data (1).csv")
    st.table(df4.head(5))
    st.write("L'objectif a été de visualiser sur une carte du globe les zones les plus impactées par le changement climatique. Pour ceci, nous avons 1) Importé le package netCDF4, 2) Crée un dictionnaire où pour chaque clé il y a une information, 3) Rédigé une fonction qui va convertir le tenseur en cartes et courbe ")
    
    st.header(':green[Analyse des évolutions des émissions de CO2]')
    st.write('Nous allons explorer le dataset “owid-co2-data” du site OurWorldinData afin de **comprendre** 1) l’**évolution des émissions de CO2** dans le monde et **ses principales cause**, 2) dans quelles mesures les émissions de CO2 sont **corrélées** avec les anomalies de température.')
    st.write('Voici à quoi ressemble notre dataframe avant le pré-processing')
    df5 = pd.read_csv(r"C:\Users\jt_ur\OneDrive\Documents\DataScientest\Projet\Datasets\owid-co2-data.csv") 
    st.table(df5.head(5))
    st.write("Le pré-processing a consisté notamment en:")
    st.write("·	A l’**échelle temporelle**, couper les dataframes à partir de 1962, afin de les faire coïncider avec le dataframe des anomalies de température et pouvoir les travailler ensemble a posteriori (cohérence temporelle).")
    st.write("·	Au niveau du **type d’indicateurs**, réaliser des analyses spécifiques à partir de types et natures d’indicateurs différents : les indicateurs qui mesurent l’**évolution annuelle des entités géographiques** traitées (main indicateurs), les indicateurs qui mesurent l’évolution per capita des entités géographiques traitées (indicateurs per capita), et les indicateurs qui mesurent l’évolution cumulative des entités géographiques traitées (indicateurs cumulatifs)")
    st.write("·	Au niveau des indicateurs à retenir, et afin d’assurer des analyses comparatives solides et des prédictions fines, nous avons gardé les **variables explicatives**, d'après le test de Pearson, de l’augmentation du CO2 ne **présentant pas de Nan au delà du 30%**  : coal_co2, oil_co2, gas_co2, land_use_change_co2, cement_co2, flaring_co2.")

def visualisations_analyses ():
    st.title(':green[Visualisations et analyses]')
    
    st.header(':green[Evolution mondiale des températures entre 1960 et 2019]')
    st.write("Cette premier graphique, faite à partir du dataset GITSEMP, compare l'évolution des températures entre hémisphères. Nous voyons que la **tendance globale est à la hausse**, et ceci plus **particulièrement pour l'hémisphère Nord**. On remarque une augmentation aux alentours de 1940 suivie d'une légère baisse, puis reprise progressive de l'augmentation des températures.")
    image = Image.open (r"C:\Users\jt_ur\DA\projects\streamlit_co2\images\gigi.png")
    st.image(image)
    st.write("L’analyse de dataset CCKP montre que le réchauffement climatique **impacte surtout** les **pays du nord de l'Europe** ainsi que **la Russie**.")
    image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_co2\images\carte temperature pays geo.png")
    st.image(image, width=700)
    st.write("Enfi, en se servant d'une **regréssion lineaire** nous pouvons constater que la **température a augmenté de 1,5 C°** entre 1962-2020.")
    image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_co2\images\dede.png")
    st.image(image)
    
    
    st.header(':green[Analyse des évolutions des émissions de CO2]')
    st.header('Evolution mondiale des émissions de CO2 entre 1962 et 2021')
    st.plotly_chart(fig1)
    st.write("Une première visualisation des données nous permet de constater qu'en soixante ans **les émissions de dioxyde de carbone (co2) ont plus que triplé**. En effet, elles sont passées de 9.751 billions de tonnes en 1962 à plus de 37.123 billions de tonnes en 2021.")
    st.header('Evolution des émissions de CO2 par continent')
    st.plotly_chart(fig2)
    st.write("Si nous regardons ladite évolution au niveau des continents, quatre constats forts en ressortent :")
    st.write("· **L’Asie est de loin le continent le plus émetteur de co2** avec 21.689 billions de tonnes en 2021, soit **60%** des émissions de co2 mondiales.")
    st.write("· **L’Amérique du Nord** est la **deuxième région** la plus émettrice de CO2 avec 6137 billions de tonnes, soit **17%** des émissions de co2 mondiales.")
    st.write("· Bien que **plus équitable dans la répartition** des valeurs, **l’Europe** est le **troisième territoire** le plus émetteur avec 5306 billions de tonnes, soit environ **15%** des émissions de co2 mondiales.")
    st.write("· En ce qui concerne **l'Afrique** (4%), **l'Amérique du Sud** (3%) et **l’Océanie** (1%), elles représentent seulement **8% des émissions de co2 mondiales**.")
    
    st.header('Evolution des pays les plus émetteurs de CO2 en 1962 et en 2021')
    st.write("Maintenant regardons l'évolution de la liste de 15 pays les plus émetteurs au monde entre 1962 et 2021")
    co2_paysmax_=st.selectbox("Les pays les plus émétteurs en 1962 et en 2021 (choisir la date).", ['1962', '2021'])
    if co2_paysmax_ == '1962':
        st.plotly_chart(fig4) 
        
    else:
        st.plotly_chart(fig5)
    st.write("Ce que nous retenons de cette évolution en 5 points:")
    st.write("1.- L'**ordre de grandeur a énormément augmenté**, en passant de 0-3000 tonnes en 1962 à 0-12000 en 2021")
    st.write("2.- Une **très forte tendance à la hausse** des émissions de co2 des **pays d’Asie**.")
    st.write("3.- Une **tendance à la baisse** des émissions de co2 des **pays d’Europe**.")
    st.write("5.- Le **poids disproportionné** dans les émissions de co2 de leur continent de **certains pays**, **notamment aux Amériques**.")
    st.write("6.- La **faible participation** des **pays de l’Afrique et de l’Océanie** dans les émissions de co2.")
        
    st.header('Evolution mondiale des émissions de CO2 par cause')
    st.plotly_chart(fig3)
    st.write("Si nous rentrons dans l’analyse de l’évolution des émissions de co2 par cause, ou type d’activité humaine produisant des émissions de co2, nous pouvons tirer des conclusions principales :")
    st.write("Dans l’ordre de grandeurs, les **variables les plus déterminantes** sont la production de **charbon** (14.979 billions de tonnes en 2021, soit 40,4%), l’activité du **pétrole** (11.837 billions de tonnes en 2021, 31,9%) et le **gaz** (7.921 billions des tonnes, soit 21,3%). La production de ciment (4,5%) et le torchage (1,1%) étant loin derrière comme causes déterminantes.")
    st.write("La seule variable que **ne suit pas l’évolution croissante** des émissions de co2 c’est le **changement d’affectation des terres** (10,6%), qui en effet descend drastiquement après un pic durant la fin des années 90.")
    st.write("Ainsi, nous pouvons en déduire **l’augmentation des émissions de C02** est notamment corrélée avec l’augmentation de production de charbon, avec l’activité pétrolière et avec les émissions des gaz, en même temps qu’elle **n’est pas corrélée avec le changement d’affectation des terres**, bien qu’il demeure la quatrième cause la plus importante des émissions de co2.")
    
    st.header('Les pays les plus émetteurs de CO2 par cause')
    st.write("Nous avons vu précédemment les principales causes des émissions de CO2 au niveau mondial. Dans cet ordre d’idées, il est important de comprendre quels sont les pays qui sont à la tête de ces émissions afin de bien construire nos scenarii et nos modèles de prédiction.")
    st.write("Vous pouvez sélectionner les différents causes pour déployer les graphiques et leur analyse")
    co2_causes_top5pays=st.selectbox('Sélection de la cause', ['Charbon', 'Pétrole', 'Gaz', 'Changement affectation des terres', 'Ciment', 'Torchage'])
    if co2_causes_top5pays == 'Charbon':
        st.plotly_chart(fig6)
        st.write("L’**Asie est de loin le continent le plus producteur** d'émissions de co2 liées à la production de charbon avec 11.959 billions de tonnes en 2021. Elle est par ailleurs **responsable de 80% des émissions** mondiales de co2 liées à la production de charbon en 2021, et suivie de très loin par l’Amérique du Nord avec 2.260 et par l’Europe avec 1.233 billions de tonnes. L’Afrique, l’Océanie et l’Amérique du Sud présentent, dans cet ordre, des émissions considérablement plus basses.")
        st.write("Sans surprise, **la Chine est en tête des pays les plus émetteurs** de co2 par production de charbon avec 7.955 billions de tonnes. Elle est **responsable de 53% des émissions mondiales** de co2 liées à la production de charbon en 2021, suivie de très loin par l’Inde avec 1.802 billions de tonnes et par les Etats-Unis avec 1.002 billions des tonnes en 2021.")
    elif co2_causes_top5pays == 'Pétrole':
        st.plotly_chart(fig7)
        st.write("L’**Asie** est toujours en tête des émissions avec 4.806 billions de tonnes, ce qui représente **41% des émissions mondiales**. Derrière l’on trouve l’Amérique du Nord avec 2.787 et l’Europe avec 1.884 billions de tonnes. L’Amérique du Sud, l’Afrique et l’Océanie contribuent toutes les trois au 11% de ces émissions.")
        st.write("Les **Etats-Unis** sont le **pays le plus émetteur** de co2 provenant du pétrole, avec 2.233 billions de tonnes, ce qui **représente le 18% des émissions mondiales**. Lui-même **contribue plus que l’Europe entière** et que l’Amérique du Sud, l’Afrique et l’Océanie ensemble. Il est suivi par la Chine avec 1.713 et par l’Inde avec 6.22 billions de tonnes.")
    elif co2_causes_top5pays == 'Gaz':
        st.plotly_chart(fig8)
        st.write("L’**Asie** se trouve **encore en tête** de ces émissions avec 3.242 billions de tonnes, soit **41% des émissions** de co2 issues du gaz, suivi par l’Amérique du Nord avec 2.969 billions de tonnes, l’Europe avec 1.936 billions de tonnes. L’Afrique, l’Amérique du Sud et l’Océanie ensemble représentent moins du 9% des émissions.")
        st.write("Le **pays le plus émetteur**, à savoir les **Etats-Unis**, dont la production de 1.637 billions de tonnes représente **20% des émissions mondiales**, ne se trouve pas en Asie mais en Amérique du Nord. Or, il est suivi exclusivement par des pays asiatiques, notamment par la Russie avec 875 et la Chine avec 773 billions de tonnes. Nous pouvons constater la présence des deux nouveaux pays parmi le top 5 : l’Iran et l’Arabie Saoudite.")
    elif co2_causes_top5pays == 'Changement affectation des terres':
        st.plotly_chart(fig11)
        st.write("Contrairement à tous les autres types d’émission elles ont une **tendance à la baisse** et se trouvent en effet à un niveau plus bas qu’elles ne l’étaient en 1962. Il est possible que les émissions de CO2 dues au changement d'affectation des terres aient diminué pour plusieurs raisons : la réduction de la déforestation, le changement des pratiques agricoles, la réduction des incendies et les politiques et réglementations. ")
        st.write("Il s’agit d’un type d’émission fortement lié à la **disponibilité de terres**, c’est pourquoi derrière **l’Asie** qui, avec ces 2098 billions de tonnes, représente **53,3% des émissions mondiales**, nous trouvons pour la première fois **l’Amérique du Sud** avec 1260 billions de tonnes, soit **32% des émissions mondiales** et **l’Afrique**, avec 1160 billions de tonnes, soit **29,4% des émissions** mondiales. Ces trois continents produisent en effet 85,3% des émissions mondiales dues au changement d’affectation des terres.")
        st.write( "Conséquemment, nous trouvons parmi les pays les plus émetteurs des pays avec des vastes zones cultivables ou forestières telles que **l’Indonésie** avec 1030 billions de tonnes qui à elle seule représente **26% des émissions mondiales**, suivie par le Brésil avec 992 billions de tonnes et la Russie avec 437 billions de tonnes. D’ailleurs les cinq pays le plus émetteurs dans cette catégorie représentent 80% des émissions mondiales.")
    elif co2_causes_top5pays == 'Ciment':
        st.plotly_chart(fig9)
        st.write("Encore une **autre catégorie dans laquelle l’Asie est très loin devant**. En effet, à elle seule l’Asie produit **81% des émissions mondiales**. Il s’agit d’un cas pratiquement de monoproduction.")
        st.write("Sans surprises, **la Chine** se trouve loin devant et représente avec 852 billions de tonnes **50% des émissions mondiales** de co2 liées à la production de ciment. Les quatre pays restant ne font ensemble que 17% des émissions. Nous devons constater néanmoins l’apparition de deux nouveaux pays dans le top cinq : le Vietnam et la Turquie.")
    elif co2_causes_top5pays == 'Torchage':
        st.plotly_chart(fig10)
        st.write("La **répartition des valeurs** sur les continents est dans ce cas **beaucoup plus homogène**. **L’Asie** en est le **premier émetteur** avec 121 billions de tonnes, soit 29% des émissions globales. Elle est suivie par l’Amérique du Nord avec 96 billions de tonnes et l’Europe avec 85 billions de tonnes.")
        st.write("Le **pays le plus émetteur** en 2021 est les **Etats-Unis** avec 67 billions de tonnes, soit le **16% des émissions mondiales**, suivi par la Russie 58 billions de tonnes et l’Iraq avec 33 billions de tonnes.")
     
    st.subheader("Schématisation:")
    st.write("Afin de préparer les modèles de prédiction, nous allons identifié les deux pays les plus emétteurs par cause:")
    st.write("·	**Pétrole**: les Etats-Unis, la Chine")
    st.write("·	**Gaz** : les Etats-Unis, la Russie")
    st.write("·	**Changement d'affectation des terres**: l'Indonesie, le Brésil")
    st.write("·	**Ciment**: la Chine, l'Inde")
    st.write("·	**Torchage**: les Etats-Unis, la Russie")

def predictions ():
    st.title(':green[Prédictions]')
    

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from sklearn.metrics import mean_squared_error

    from joblib import load, dump

    st.header(':green[Modélisation de la température moyenne mondiale.]')
    st.write("**Matrice de corrélation des différentes variables:**")
    image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\matrice correlation variables")
    st.image(image)
    st.write(" D'après la matrice de corrélations, nous avons pu modéliser la température moyenne mondiale avec la variable temps ('year'), la variable co2 moyenne mondiale ('co2') et des variables géographiques. On a donc fait trois modèles avec des approches différentes; **une approche directe** avec un modèle prédictif sur le temps/latitude/longitude/températures précédentes, une **approche avec les séries temporelles**, et enfin **une approche avec des modèles de régressions linéaires** car en observant la matrice on distingue bien des variables fortement corrélées linéairement.")
    st.subheader("Deux scénarios")
    st.write("A partir des analyses exposées sur le changement climatique et la part de responsabilité des émissions de co2 dans l’augmentation des températures, nous avons trouvé instructif de travailler sur deux scénarios différents de modélisation :")
    st.write("Un **scénario pessimiste** : Quelle serait l’augmentation de température « **si tout continuait comme maintenant** », , c’est-à-dire, si les pays maintenaient la pente d’évolution des émissions de CO2 d’ici 2100 ?")
    st.write("Pour ce scénario nous avons mobilisé deux modèles : les **forêts aléatoires** (1.1) et les **séries temporelles** (2.1).")
    st.write("Un **scénario optimiste** : Quelle serait l’augmentation de température ** si les principaux émetteurs de CO2 par cause réduisaient de 50% leur émissions** d’ici 2100 ?")
    st.write("Pour ce dernier scénario, nous avons mobilisé les **régressions linéaires**(3.1), en s'inspirant de nos analyses statistiques en deçà de ce qui pourrait être politiquement et économiquement raisonnable ou encore faisable, car nous n’avons évidemment pas tous les éléments pour des telles prédictions. Ainsi, nous avons fixé le **critère de réduction de -50% des deux premiers pays émetteurs par cause**, comme suit :")
    st.write("·	Chine: -50% sur charbon, pétrole, ciment, torchage")
    st.write("·	USA: -50% sur pétrole, gaz, torchage.")
    st.write("·	Inde : -50% sur charbon, ciment.")
    st.write("·	Russie: -50% sur gaz.")
    st.write("·	Indonésie: -50% sur affectation des terres.")
    st.write("·	Brésil: -50% sur affectation des terres.")
        
    
    st.header(":green[Méthodologie.]")
    
    st.subheader("1.1 Approche directe, forêts aléatoires.")
    st.write("Le jeu de données se présente sous forme de tenseur en trois dimensions (temps, latitude, longitude), chaque nombre est donc une température moyenne annuelle à un endroit et une année précise. Il a fallu rédiger un programme pour réagencer ce tenseur sous forme de Dataframe, et en transposant ce dataframe avoir pour chaque échantillon la latitude, longitude et les températures par années gardées dans notre jeu d'entrainement.")
    st.subheader("1.2 Approche linéaire générale, régression linéaire.")
    st.write("Cette fois le jeu de données est réagencé de façon plus classique en un Dataframe avec quatre dimensions (année, latitude, longitude, température). On va ensuite appliquer une régression linéaire pour obtenir la température selon l'année et la zone géographique. On a également essayé la régression polynomiale pour tenir compte de la forme de la courbe de températures qui n'est en réalité pas exactement linéaire.")
    st.subheader("1.3 Approche linéaire par point géographique.")
    st.write("on a appliqué ici un modèle de régression différent pour chaque point géographique. Donc cette stratégie par rapport à la précédente prédit l'évolution pour chaque point géographique, mais indépendemment des autres points même à proximité. La régression polynomiale a aussi été utilisée.")
    st.subheader("1.4 Métriques utilisées.")
    st.write("L'erreur moyenne absolue a été utilisée. Comme il est difficile avec l'approche directe de séparer un dataset en un jeu d'entrainement, puis un jeu test 80 ans dans le futur (les données avant 1960 étant disparates), les différents modèles ont été testés sur leur capacité à prévoir la température de l'année suivante (année suivant la dernière année du jeu d'entrainement).")
    st.write("Pour les modèles on a arrêté les données traités jusqu'à 2019, puis on a séparé en un train set à 80% et un test set à 20%. Ensuite on a estimé l'erreur entre les données prédites du jeu test et les données réelles pour 2020.")
    st.write("**Résultats des différents modèles dans leur capacité à prévoir la température de l'année suivante:**")
    fig, ax=plt.subplots()
    ax.bar(["Forêts aléatoires", "Approche linéaire\ngénérale", "Approche linéaire\npolynomiale\ngénérale", "Approche linéaire\npar point", "Approche linéaire\npolynomiale\npar point"], [0.06, 0.62, 0.88, 0.69, 0.64], color=['red', 'orange', 'grey', 'yellow', 'green'])
    plt.xticks(rotation=75)
    plt.title("Erreur moyenne absolue selon les modèles.")
    plt.ylabel('Erreur absolue moyenne.')
    st.pyplot(fig)
    st.subheader("1.5 Limites.")
    st.write("D'après notre métrique la modélisation la plus efficace semble être l'approche directe avec les forêts aléatoires. Cependant la métrique ne donne pas la capacité du modèle à prédire 80 années dans le futur, de plus l'erreur augmente d'années en années donc ces résultats sont à prendre avec du recul.")
    st.write("De plus le dataset d'origine bien qu'assez complet n'intègre pas certaines zones géographiques (l'Antarctique par exemple) donc cela peut biaiser à la longue les valeurs des températures.")
    st.write("Le volume de données est ici conséquent, c'est pourquoi nous n'avons pas pu faire une recherche d'optimisation des paramètres de nos modèles avec gridsearchcv(chaque session d'entrainement dure au moins 30 minutes pour une configuration de paramètres donnée), ou même un randomizedsearchcv. Avec une puissance de calcul supplémentaire on aurait probablement pu optimiser le modèle des forêts aléatoires.")

    st.subheader("2.1 Les séries temporelles.")
    st.write("Notre matrice de corrélations indique une forte corrélation linéaire entre le temps et la température moyenne annuelle. Le temps étant ici mesuré par ans, il n'y a pas de saisonnalité à traiter à priori, nous avons donc exploité le modèle ARIMA des séries temporelles afin de prédire les températures futures, notamment la température moyenne en 2100.")
    st.subheader("2.2 Métriques utilisées.")
    st.write("Pour optimiser les paramètres et vérifier la stationnarité du modèle, on a utilisé la décomposition de fonction, l'analyse des auto-corrélations simples et partielles, le test statistique Ad-Fuller et l'erreur moyenne absolue calculées sur les 20 dernières années.")
    st.write("Nous avons trouvé que pour stationnariser le modèle on devait régler le paramètre de différentiation à 1. Pour avoir une optimisation des paramètres, le paramètre AR doit être réglé à 1 et le paramètre MA à 0. L'erreur quadratique moyenne est alors de seulement 0.013.")
    image = Image.open (r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\courbe erreur arima 110 erreur 0.0128")
    st.image(image)
    st.subheader("2.3 Limites.")
    st.write("Ce modèle ne prend en compte que la variable temps pour estimer sa prédiction.")
    st.subheader("3.1 Les régressions linéaires.")
    st.write("En observant la matrice de corrélations, on constate de fortes corrélations linéaires entre le gaz, le charbon, le pétrole, le ciment, l'expoitation des sols et le co2 émis à l'échelle nationale.")
    st.write("A l'échelle mondiale, on distingue une forte corrélation linéaire entre le co2 moyen mondial émis(co2_moy_mon) et la température moyenne mondiale (temperature_moy_mon)")
    st.write("Donc ici l'objectif sera de faire un modèle capable de prédire la température moyenne mondiale selon l'exploitation du charbon, du gaz, du pétrole...De certains pays. On va procéder en trois étapes.")
    st.subheader("3.2 Régressions linéaires sur les variables causales.")
    st.write("On va effectuer une régression linéaire différente pour chaque pays sur ses variables causales(gaz, charbon...). Les droites obtenues vont nous permettre d'estimer l'évolution de ces variables dans le futur, selon chaque pays et chaque variable. On considère donc dans un premier temps que ces variables vont évoluer linéairement au même rythme qu'entre 1960 et 2020.")
    st.subheader("3.3 Régression linéaire pour estimer le co2 émis par pays.")
    st.write("Cette fois on va entrainer un nouveau modèle de régression sur les données prédites précédemment, afin de prédire le co2 émis par pays. Ce modèle est entrainé sur tout les pays. Une fois que l'on obtient le co2 émis par pays, on calcule le co2 émis moyen mondial.")
    st.subheader("3.4 Régression linéaire pour estimer la température moyenne mondiale.")
    st.write("Enfin, on met au point un modèle de régression linéaire pour prédire la température moyenne mondiale à partir du co2 émis moyen mondial, et de l'année.")
    st.subheader("3.5 Métriques.")
    st.write("La métrique s'est faite en deux étapes, en premier le modèle de prédiction du co2 émis. Comme pour Arima, on a testé ce modèle sur les vingt dernières années.")
    image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\perf modele co2 20 dernières années")
    st.image(image)
    st.write("On constate que le modèle est assez précis, ça reste cohérent avec notre matrice de corrélations. L'erreur moyenne absolue est de 0.99.")
    st.write("Pour la seconde métrique, on a estimé le modèle de prédiction des températures également sur les vingt dernières années.")
    image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\perf modele temperatures sur les 20 dernières années")
    st.image(image)
    st.write("Le modèle suit la même direction que la tendance, mais légèrement à la baisse. L'erreur moyenne absolue est de 0.26.")
    st.subheader("3.6 Limites.")
    st.write("Cette stratégie part du postulat que les pays du monde vont exploiter les variables causant l'émission du co2 de façon linéaire comme entre 1960 et 2020...Alors que ce n'est pas vraiment conforme à la réalité.")
    st.write("Certaines variables comme le méthane n'ont pas été intégrées, alors que nous savons qu'elles sont en partie aussi responsables de l'effet de serre.")


    st.header(":green[Modélisation.]")
    
    st.subheader('Evolution mondiale des températures pour 2100, 2500 et 2100.')
    st.write("Les prédictions sont faites avec différents modèles de machine learning au choix.")
    machine_learning_model=st.selectbox('Sélection du modèle de machine learning.', ['forets aléatoires', 'régression linéaire générale', 'régression polynomiale générale', 'régression linéaire par point géographique', 'régression polynomiale par point géographique'])
    year_prediction=st.selectbox("Sélection de l'année à prédire.", ['2100', '2500', '3000'])

    if (machine_learning_model=='forets aléatoires') and (year_prediction=='2100'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prevision foret 2100")
        st.image(image)

    elif (machine_learning_model=='forets aléatoires') and (year_prediction=='2500'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prediction 2500 foret")
        st.image(image)
        
    elif (machine_learning_model=='forets aléatoires') and (year_prediction=='3000'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prediction foret 3000")
        st.image(image)
        
    elif (machine_learning_model=='régression linéaire par point géographique') and (year_prediction=='2100'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte regression lineaire 2100")
        st.image(image)
        
    elif (machine_learning_model=='régression linéaire par point géographique') and (year_prediction=='2500'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte linearbypoint prediction 2500")
        st.image(image)
        
    elif (machine_learning_model=='régression linéaire par point géographique') and (year_prediction=='3000'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prediction lineaire 3000")
        st.image(image)
        
    elif (machine_learning_model=='régression polynomiale par point géographique') and (year_prediction=='2100'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prediction poly 2100")
        st.image(image)
        
    elif (machine_learning_model=='régression polynomiale par point géographique') and (year_prediction=='2500'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prediction poly 2500")
        st.image(image)
        
    elif (machine_learning_model=='régression polynomiale par point géographique') and (year_prediction=='3000'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prevision poly 3000")
        st.image(image)
        
    elif (machine_learning_model=='régression linéaire générale') and (year_prediction=='2100'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prevision linear gene2100")
        st.image(image)
        
    elif (machine_learning_model=='régression linéaire générale') and (year_prediction=='2500'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prevision linear gene 2500")
        st.image(image)
        
    elif (machine_learning_model=='régression linéaire générale') and (year_prediction=='3000'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prevision linear gene 3000")
        st.image(image)
        
    elif (machine_learning_model=='régression polynomiale générale') and (year_prediction=='2100'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prediciotn poly gene2100")
        st.image(image)
        
    elif (machine_learning_model=='régression polynomiale générale') and (year_prediction=='2500'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prediction temperature poly gene2500")
        st.image(image)
        
    elif (machine_learning_model=='régression polynomiale générale') and (year_prediction=='3000'):
        
        image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\carte prediction 3000 poly gene")
        st.image(image)
        
    st.subheader("Evolution de la température avec ARIMA (série temporelle)")
    st.write("Le modele ARIMA permet d'estimer dans le temps l'évolution de la température.")
    st.subheader('A propos de la température, paramètres:')
    annee_prediction=st.number_input("Année de prédiction", 2050, 2500, step=50)
    differentiation=st.number_input("Ordre de différentiation", 0, 1)
    AR=st.number_input("Paramètre du processus AR", 0, 5)
    MA=st.number_input("Paramètre du processus MA", 0, 5)

    data=pd.read_csv(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\data_temperature_co2_anomalie.csv", index_col=0)
    temperature=data[['year', 'température']].groupby('year').agg('mean')
    temperature.index=pd.date_range('1960-06-01', '2020-06-01', freq='y')



    if differentiation==0:
        diff=temperature
    elif differentiation==1:
        diff=temperature.diff().dropna()

        
    ax2=seasonal_decompose(diff).plot()
    st.subheader("Décomposition des données selon nos paramètres sur la température:")
    st.pyplot(ax2)
    result2=adfuller(diff.température)

    st.subheader("Statistiques AD-Fuller.")
    st.write(f"Statistiques ADF: {round(result2[0], 2)}")
    st.write(f"P-value: {round(result2[1], 2)}")
    if result2[1]<0.05:
        st.write("Le modèle est stationnaire.")
    elif result2[1]>=0.05:
        st.write("Le modèle n'est pas stationnaire.")


    st.subheader("Autocorrélation partielle de la série.")
    ax3=plot_pacf(diff)
    st.pyplot(ax3)
    st.write("Doit tendre vers 0. Permet surtout visuellement de régler le paramètre AR. S'annule généralement après le paramètre AR.")

    st.subheader("Autocorrélation simple de la série.")
    ax4=plot_acf(diff)
    st.pyplot(ax4)
    st.write("Doit également tendre vers 0. S'annule en général après le paramètre MA.")
    
    st.subheader("Prédictions Arima de la température.")
    image = Image.open(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\arima_momo.png")
    st.image(image)
    st.write("Prédictions du modèle avec paramètres AR à 1, differentiation à 1 et MA à 0. La différence de température constatée entre 2020 et 2100 est d'environ 1.5 degrés celsius.")


    st.subheader("Prédictions des modèles de régression linéaire pour 2100.")
    scaler=load(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\scaler_modele_1\scaler_modele_1")
    model_1=load(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\modele_lineaire_1\modele_lineaire_1")
    model_2=load(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\modèle_linéaire_2\modèle_linéaire_2")
    data=load(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\données_estimées.csv")
    data_normal=load(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\données_estimées.csv")
    data_changed=load(r"C:\Users\jt_ur\DA\projects\streamlit_changement_climatique\images 2\données_estimées.csv")
    st.write("On va dans un premier temps modifier le dataset des variables prédites par le modèle linéaire du carbone pour ensuite prévoir la température moyenne annuelle avec une seconde régression linéaire.")
    liste_pays=st.multiselect("Choix des pays", ["Chine", "USA", "Russie", "Indonésie", "Brésil", "Inde", "France"])
    liste_variables_causales=st.multiselect("Choix des variables causales", ["population", "Charbon", "Pétrole", "Gaz", "Changement_affectation_terres", "Ciment", "Torchage"])
    réduction=st.slider("Choix de la baisse en pourcentage", 0, 100)
    variables_causales=["population", "coal_co2", "oil_co2", "gas_co2", "land_use_change_co2", "cement_co2", "flaring_co2"]

    new_liste_pays=[]
    dict_translate_pays={"Chine":"China", "USA": "United States of America", "Russie": "Russia", "Indonésie": "Indonesia", "Brésil": "Brazil", "Inde": "India"}
    for pays in liste_pays:
        if pays in dict_translate_pays.keys():
            pays=dict_translate_pays[pays]
            new_liste_pays.append(pays)
        else:
            new_liste_pays.append(pays)

    new_liste_var=[]
    dict_translate_var={'Charbon': "coal_co2", "Pétrole": "oil_co2", "Gaz": "gas_co2", "Ciment": "cement_co2", "Changement_affectation_terres":"land_use_change_co2", "Torchage":"flaring_co2" }
    for var in liste_variables_causales:
        if var in dict_translate_var.keys():
            var=dict_translate_var[var]
            new_liste_var.append(var)
        else:
            new_liste_var.append(var)
            
    liste_pays=new_liste_pays
    liste_variables_causales=new_liste_var

    data_normal[variables_causales]=scaler.transform(data_normal[variables_causales])
    data_normal=pd.get_dummies(data_normal, columns=['pays', 'continent'])

    co2_normal=model_1.predict(data_normal)

    df_co2_normal=pd.DataFrame({'year': data.year, 'co2': co2_normal})
    co2_moy_an_normal=df_co2_normal.groupby('year').agg('mean')

    temperature_prevision_normal=model_2.predict(co2_moy_an_normal)

    difference_normal=temperature_prevision_normal[-1]-temperature_prevision_normal[0]



    liste_traitement=[]
    durée=[an for an in range(2020, 2101)]
    for pays in liste_pays:
        
        df=data.loc[data.pays==pays,:]
        df_first_var=df.loc[df.year==2020, liste_variables_causales]
        df_dec_var=df_first_var.apply(lambda x: x*(réduction/100)) 
        df_change_var=df_first_var-df_dec_var
        new_df=pd.DataFrame({}, index=durée)

        for var in liste_variables_causales:
        
            new_df[var]=np.linspace(float(df_first_var[var]), float(df_change_var[var]), len(durée))
            new_df['pays']=pays
        
        liste_traitement.append(new_df)

    try:

        df_changed=pd.concat(liste_traitement)



        for pays in df_changed.pays.unique():
        
            for var in df_changed.columns:
            
                for line in durée:
                
                    transit=df_changed[df_changed.pays==pays].loc[line, var]
                    data_changed.loc[(data_changed.year==line) & (data_changed.pays==pays), var]=transit
       



        data_changed[variables_causales]=scaler.transform(data_changed[variables_causales])
        data_changed=pd.get_dummies(data_changed, columns=['pays', 'continent'])    

        co2_emission=model_1.predict(data_changed)

        df_co2_emission=pd.DataFrame({'year': data_changed.year, 'co2': co2_emission})
        co2_moy_an=df_co2_emission.groupby('year').agg('mean')

        temperature_prevision=model_2.predict(co2_moy_an)

        difference_changements=temperature_prevision[-1]-temperature_prevision[0]


        plt.clf()
        fig8, ax8=plt.subplots() 
        ax8.plot(durée, temperature_prevision, c='r',label='Prévisions avec modifications.')
        ax8.plot(durée, temperature_prevision_normal, c='b', label='Prévisions sans modifications.')
        ax8.legend()



        st.pyplot(fig8)
        st.write(f"Différence de température pour une évolution sans changements: {round(difference_normal,2)} degrés Celsius.")
        st.write(f"Différence de température pour une évolution avec changements: {round(difference_changements, 2)} degrés Celsius.")

    except:
        
        plt.clf()
        fig8, ax8=plt.subplots()
        ax8.plot(durée, temperature_prevision_normal, c='b', label='Prévisions sans modifications.')
        ax8.legend()
        st.pyplot(fig8)
        st.write(f"Différence de température pour une évolution sans changements: {round(difference_normal, 2)} degrés Celsius.")

    st.header(":green[Mise en perspective.]")
    st.write(" Utilisant l'**approche directe des forêts aléatoires** et la **prédiction Arima** pour modéliser notre **scenario 1**,nous constatons une augmentation de la température moyenne mondiale de presque **1.5 degrés d'ici l'année 2100**.")
    st.write("Ces deux modèles, tel qu'explicité sur le scenario 1, prédisent la température si le CO2 évolue conjointement de la même façon que pendant la période 1960/2020.")
    st.write("Le scenario 2 découle d'un modèle  intégrant le CO2 comme variable et permet de prédire la température future au cas où des variables causales changeraient d'évolution.")
    st.write("Si la Chine, la Russie, l'Inde, l'Indonésie, les USA et le Brésil venaient à réduire de 50% leurs émissions de CO2 sur les causes dans lesquels ils sont les principaux émetteurs, le modèle nous prédit **une stabilisation de la température moyenne mondiale d'ici 2100**.")
    st.write("Si la France n'émet plus de CO2, en 2100 la température moyenne mondiale baissera d'un centième de degrés environ selon notre modèle.")
    st.write("Il va donc falloir réduire de façon drastique les émissions CO2 pour équilibrer au plus vite la température moyenne mondiale.")


def conclusions ():
    st.title(':green[Conclusions]')
    st.header("Que va-t-il se passer si les pays maintenaient la pente d’évolution des émissions de CO2 d’ici 2100 ?")
    st.write("Selon le modèle des **forêts aléatoires**, la température mondiale moyenne **augmentera de presque 1,5°C d’ici 2100**. Selon le modèle **ARIMA** elle grimpera également de **presque 1,5°C d'ici 2100**. Il s’agit d’un **scénario assez pessimiste au regard du rapport du GIEC** selon lequel si l’augmentation des températures **dépasse 1,5 °C** nous verrons des conséquences catastrophiques, telles que:")
    st.write("•	Des **phénomènes climatiques aggravés** : l’évolution du climat modifie la fréquence, l’intensité, la répartition géographique et la durée des événements météorologiques extrêmes (**tempêtes, inondations, sécheresses**).")
    st.write("•	Un **bouleversement des écosystèmes** : avec l’**extinction de 20 à 30 % des espèces animales et végétales**, et des conséquences importantes pour les implantations humaines.")
    st.write("•	Des **crises liées aux ressources alimentaires** : dans de nombreuses parties du globe (Asie, Afrique, zones tropicales et subtropicales), les productions agricoles pourraient chuter, provoquant de graves crises alimentaires, **sources de conflits et de migrations**.")
    st.write("•	Des **dangers sanitaires** : le changement climatique aura vraisemblablement des impacts directs sur le fonctionnement des écosystèmes et sur la **transmission des maladies animales**, susceptibles de présenter des éléments pathogènes potentiellement dangereux pour l’Homme.")
    st.write("•	L’**acidification des eaux** : l’augmentation de la concentration en CO2 (dioxyde de carbone) dans l’atmosphère entraîne une plus forte concentration du CO2 dans l’océan. Une acidification des eaux en découle, ce qui représente un **risque majeur pour les récifs coralliens et certains types de plancton** menaçant l’équilibre de nombreux écosystèmes.")
    st.write("•	Des **déplacements de population** : l’augmentation du niveau de la mer (26 à 98 cm d’ici 2100, selon les scénarios) devrait provoquer l’**inondation de certaines zones côtières** (notamment les deltas en Afrique et en Asie), voire la **disparition de pays insulaires entiers** (Maldives, Tuvalu), provoquant d’importantes migrations.")
    st.write("Enfin, ce rapport averit que **les impacts du changement climatique peuvent être très différents d’une région à une autre**, mais ils concerneront toute la planète. A ceci nous pouvons rajouter que d'après nos analyses ce ne sont pas les principaux émetteurs de CO2 qui risquent les plus grands dommages. Ce qui invite à réfléchir au sens de la responsabilité écologique et humaine que ces pays portent.")
    st.header("Que se passerait-il si les principaux émetteurs de CO2 par cause réduisaient de 50% leur émissions d’ici 2100 ?")
    st.write("Selon le **modèle de la régression linéaire**, la température moyenne mondiale va seulement **augmenter d'environ 0.5° pour 2100**. C’est un **scénario assez optimiste** ! ")
    st.write("D’après le **rapport du GIEC** si nous souhaitons limiter le réchauffement à 1,5 °C -chiffre stipulé par l’accord de Paris- à la fin du siècle, nous devons réduire les émissions dans tous les secteurs d’activité, et rapidement. Selon leur meilleure estimation, fondée sur des données historiques et des modèles climatiques, **le monde atteindra la limite de 1,5 °C d’ici 2030-2035**. ")
    st.write("**Avec notre modèle, ce seuil n’est même pas atteint pour 2100 et cela sans compter la réduction d’un autre gaz à effet de serre : le méthane**. Gaz à effet de serre environ 80 fois plus puissant que le CO2 sur 20 ans et qui atteint actuellement un tel niveau qu’il aggrave l’accélération du réchauffement planétaire.")
    st.write("Par ailleurs, il avertit que même si certaines technologies, telles que le captage et le stockage du carbone (**CCS**) et l’élimination du dioxyde de carbone (**CDR**), sont mises en place intégralement cela **ne suffira pour atteindre la neutralité carbone d’ici 2050**. **Pas sans la réduction de nos émissions**, et ce, dès maintenant. ")
    st.write("Notre modèle de prédiction s'appuie totalement, et exclusivement, sur la réduction des émissions de CO2', et cela marche. Or, il n’est pas capable de prendre en compte tous les aléas économiques et politiques qui comprendraient des mesures aussi drastiques que la réduction de 50% des émissions de CO2 des plus gros émetteurs par cause d’ici 2100. Néanmoins, il permet, en effet, de penser que **le dérèglement climatique et ses conséquences dévastatrices** pour la planète n’est pas un fait inévitable. Bien au contraire, il **dépend totalement du modèle de développement qu' adopteront les principales puissances mondiales dans les prochaines décennies**. ")
    


#Options
if options == 'Introduction':
    intro() 
if options == 'Datasets':
    datasets()
if options == 'Visualisations et analyses':
    visualisations_analyses()
if options == 'Prédictions':
    predictions()
if options == 'Conclusions':
    conclusions()

    
