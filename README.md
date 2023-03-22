# Investigación COVID-19
### Proyecto de investigación estadístico sobre el virus actual covid-19

Dentro de este archivo se explicará el proceso que hemos llevado a cabo para realizar nuestra investigación sobre las causas de muerte por COVID-19 y cuáles son los factores de riesgo.

##

Tema: Modelo de análisis para predecir el total de muertes a causa del COVID-19 usando técnicas de regresión multivariante y clustering.

Objetivo: Determinar las variables que mas se correlacionan segun el total de muertes por países desde que la enfermedad comenzó a ser una pandemia.

Dataset: https://ourworldindata.org/covid-deaths

Tipo de Problema de Aprendizaje: regresión, clustering, kmeans.

Medida de calidad: varianza, covarianza, media, correlación.

Método propuesto:

1. Limpieza de datos usando el metodo replace.
2. Estandarización usando el metodo Fit Transform.
3. Medidas de calidad entre ellas media, varianza y correlacion de variables.
4. Agrupamiento mediante el método "Group By".
5. Transformación de datos categóricos a numéricos (Preprocesamiento) con los procesos de StandarScaler (variables numéricas) y OneHotEncoder (variables categóricas o texto).
6. Reducción de dimensionalidad (PCA) para reducir el uso de recursos y mejorar el rendimiento de procesamiento.
7. Métodos de aprendizaje (KNN, Regresión Multilineal) para determinar variables y grupos que nos lleven un correcto análisis y resultados esperados.
8. Visualización mediante el uso de gráficas y tablas de datos.

##

### ¿Qué hemos obtenido?

Hemos descubierto, mediante el análisis del dataset, datos que podrían resultar interesantes para el lector, como por ejemplo la Media de muertes a nivel mundial, con una correlación del 0.952852. Valores específicos por país (Estados Unidos con: 0.235051, Reino Unido: 0.081315, Italia: 0.076147, etc.). También tomamos una media de datos de fallecimientos de aproximadamente 2468.412, cifra que data desde el inicio de la propagación del virus desde Wuhan, hasta la fecha de elaborado éste trabajo, 22 de Julio del 2020. Cabe mencionar que cuando se realizó el cálculo de error de la media utilizando la regresión lineal obtuvimos un 0.13373741142399884 gracias a los métodos propuestos.
