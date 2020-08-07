# Investigación COVID-19
 Proyecto de investigación estadístico sobre el virus actual covid-19

Dentro de este archivo se explicara el proceso que hemos llevado para realizar nuestra investigacion sobre la cusas de muerte por COVID-19

Tema: Modelo de an ́alisis para predecir el total de muertesa causa del COVID-19 usando t ́ecnicas de regresionmultivariante y clustering

Objetivo: Determinar las variables que mas se correlacionan segun el total de muertes por países desde que la enfermedad comenzó a ser una pandemia.

Dataset: https://ourworldindata.org/covid-deaths

Tipo de Problema de Aprendizaje: regresión, clustering, kmeans.

Medida de calidad: varianza, covarianza, media, correlación. 

Método propuesto: 

1.	Limpieza de datos usando el metodo replace
2.	Estandarización usando el metodo Fit Transform
3.	Medidas de calidad entre ellas media, varianza y correlacion de variables
4.	Agrupamiento se utilizo el metodo Group By
5.	Transformación de datos categóricos a numéricos (Preprocesamiento) con los procesos de StandarScaler (variables numericas) y OneHotEncoder (variables categoricas o texto)
6.	Reducción de dimensionalidad (PCA) para reducir el uso de recursos y mejorar el rendimiento de procesamiento
7.	Métodos de aprendizaje (KNN, Regresión Multilineal) para determinar variables y grupos que nos lleven un correcto analisis y resultados esperados.
8.	Visualización usando gráficas y tablas de datos

¿Qué hemos obtenido? 

Hemos descubierto con este dataset datos que nos parecieron interesantes y que cave mencionar como la media de muertes a nivel mundial con una correlación del 0.952852 como por cada país (Estados Unidos con: 0.235051, Reino Unido: 0.081315, Italia: 0.076147, etc.), también tomamos una media de datos por muertes que corresponde al 2468.412, desde que inicio a extenderse el virus desde Wuhan hasta la fecha del 22 de Julio del 2020.
Cabe mencionar que cuando se realizo el calculo de error de la media utilizando la regresión lineal obtuvimos un 0.13373741142399884 gracias a los métodos propuestos.
