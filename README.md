# Investigación COVID-19
 Proyecto de investigación estadístico sobre el virus actual covid-19
 Estadisticas\ para\ las\ ciencias\ de\ la\ computacion

<img src="https://www.pequeciencia.ups.edu.ec/imgcontenidos/2-3_Logo%20UPS.png" alt="UPS" width = 500px>

Integrantes: Pedro Orellana Jaramillo, David Cornejo Bravo

Tema: ANÁLISIS DE DATOS MEDIANTE ANÁLISIS MULTIVARIANTE Y PRINICPIOS DE APRENDIZAJE DE MÁQUINA

Actividades desarrolladas

1. Revisar el contenido teórico del tema

2. Profundizar los conocimientos revisando los libros guías, los enlaces contenidos en los objetos de aprendizaje y la documentación disponible en fuentes académicas en línea

3. Realizar el siguiente proceso:

3.1 Revisar los datasets y enlaces del directorio “4.DatasetsCOVID19”. Pueden conseguir otro referente a COVID19.

3.2 Seleccionar y cargar un dataset

import pandas as pd
import numpy as np
df = pd.read_csv('owid-covid-data.csv', header=0)
df



df['iso_code'] = df['iso_code'].replace(np.nan, 'NoIso')
df['date'] = df['date'].replace(np.nan, 'NoDate')
df['continent'] = df['continent'].replace(np.nan, 'NoContinent')
df['location'] = df['location'].replace(np.nan, 'NoLocation')
df['total_cases'] = df['total_cases'].replace(np.nan, 0)
df['new_cases'] = df['new_cases'].replace(np.nan, 0)
df['total_deaths'] = df['total_deaths'].replace(np.nan, 0)
df['new_deaths'] = df['new_deaths'].replace(np.nan, 0)
df['total_cases_per_million'] = df['total_cases_per_million'].replace(np.nan, 0)
df['new_cases_per_million'] = df['new_cases_per_million'].replace(np.nan, 0)
df['total_deaths_per_million'] = df['total_deaths_per_million'].replace(np.nan, 0)
df['new_deaths_per_million'] = df['new_deaths_per_million'].replace(np.nan, 0)
df['total_tests'] = df['total_tests'].replace(np.nan, 0)
df['new_tests'] = df['new_tests'].replace(np.nan, 0)
df['total_tests_per_thousand'] = df['total_tests_per_thousand'].replace(np.nan, 0)
df['new_tests_per_thousand'] = df['new_tests_per_thousand'].replace(np.nan, 0)
df['new_tests_smoothed'] = df['new_tests_smoothed'].replace(np.nan, 0)
df['new_tests_smoothed_per_thousand'] = df['new_tests_smoothed_per_thousand'].replace(np.nan, 0)
df['tests_units'] = df['tests_units'].replace(np.nan, 'NoTest')
df['stringency_index'] = df['stringency_index'].replace(np.nan, 0)
df['population'] = df['population'].replace(np.nan, 0)
df['population_density'] = df['population_density'].replace(np.nan, 0)
df['median_age'] = df['median_age'].replace(np.nan, 0)
df['aged_65_older'] = df['aged_65_older'].replace(np.nan, 0)
df['aged_70_older'] = df['aged_70_older'].replace(np.nan, 0)
df['gdp_per_capita'] = df['gdp_per_capita'].replace(np.nan, 0)
df['extreme_poverty'] = df['extreme_poverty'].replace(np.nan, 0)
df['cardiovasc_death_rate'] = df['cardiovasc_death_rate'].replace(np.nan, 0)
df['diabetes_prevalence'] = df['diabetes_prevalence'].replace(np.nan, 0)
df['female_smokers'] = df['female_smokers'].replace(np.nan, 0)
df['male_smokers'] = df['male_smokers'].replace(np.nan, 0)
df['handwashing_facilities'] = df['handwashing_facilities'].replace(np.nan, 0)
df['hospital_beds_per_thousand'] = df['hospital_beds_per_thousand'].replace(np.nan, 0)
df['life_expectancy'] = df['life_expectancy'].replace(np.nan, 0)
df


import statistics as stats
df['total_cases'] = df['total_cases']
df['total_deaths'] = df['total_deaths']
print("Media de Casos por Covid-19: " + str(stats.mean(df['total_cases'])))
print("Media de Muertes por Covid-19: " + str(stats.mean(df['total_deaths'])))
print("Varianza de casos: " + str(df['total_cases'].var()))
print("Varianza de muertes: " + str(df['total_deaths'].var()))

Media de Casos por Covid-19: 45734.80360332132
Media de Muertes por Covid-19: 2468.4121886260377
Varianza de casos: 216501321431.14407
Varianza de muertes: 561568357.140213

import seaborn as sns
from matplotlib import pyplot as plt
df_cor = df.corr()
plt.figure(figsize = (15,10))
sns.heatmap(df_cor, center=0, cmap='Reds_r', annot=False)

df_cord = pd.DataFrame(df.corr()['total_deaths'].sort_values(ascending=False))
df_cord

sns.heatmap(df_cord, center=0, cmap='Reds_r', annot=False)

eliminacion = df.drop(['iso_code','continent','date','tests_units','new_cases','new_deaths','total_cases_per_million','new_cases_per_million','total_deaths_per_million','new_deaths_per_million','total_tests','new_tests','total_tests_per_thousand','new_tests_per_thousand','new_tests_smoothed','new_tests_smoothed_per_thousand','stringency_index','population','population_density','median_age','aged_65_older','aged_70_older','gdp_per_capita','extreme_poverty','cardiovasc_death_rate','diabetes_prevalence','female_smokers','male_smokers','handwashing_facilities','hospital_beds_per_thousand','life_expectancy'], axis=1)
agrupamiento = eliminacion.groupby(['total_deaths']).mean()
agrupamiento = eliminacion.groupby(['total_cases']).mean()
agrupamiento = eliminacion.groupby(['location']).mean()
agrupamiento.to_csv("datasetAgrupado.csv", sep=",",index = True) #sep es el separado, por defector es ","
agrupamiento = pd.read_csv('datasetAgrupado.csv', header=0)
agrupamiento

Xsubset = agrupamiento[['location','total_cases','total_deaths']]
y = df.total_deaths.values
Xsubset


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
#PRIMERA FORMA DE PREPROCESAR

preprocesador1 = make_column_transformer(
    (StandardScaler(),['total_cases','total_deaths']),
   (OneHotEncoder(),['location']),)

XProcesado = preprocesador1.fit_transform(Xsubset)

categorical_features = ['location']
cnamesDataset1 = ['total_cases','total_deaths']
cnamesDataset2 = preprocesador1.transformers_[1][1].get_feature_names(categorical_features)

cnamesDataset1.extend(cnamesDataset2)
print(cnamesDataset1)
cnamesDataset1 = np.array(cnamesDataset1)

DatasetPreprocesado = pd.DataFrame(XProcesado.toarray(),columns=cnamesDataset1)

DatasetPreprocesado.to_csv("datasetPreprocesado.csv", sep=";",index = False) #sep es el separado, por defector es ","
DatasetPreprocesado.head()


#Estandarizacion
from sklearn.preprocessing import StandardScaler
x=DatasetPreprocesado
x=StandardScaler().fit_transform(x)
x

array([[-0.1026423 , -0.11861563, 14.52583905, ..., -0.06884284,
        -0.06884284, -0.06884284],
       [-0.13245301, -0.13155919, -0.06884284, ..., -0.06884284,
        -0.06884284, -0.06884284],
       [-0.11727577, -0.10834718, -0.06884284, ..., -0.06884284,
        -0.06884284, -0.06884284],
       ...,
       [-0.13517865, -0.12378754, -0.06884284, ..., 14.52583905,
        -0.06884284, -0.06884284],
       [-0.13402437, -0.13308996, -0.06884284, ..., -0.06884284,
        14.52583905, -0.06884284],
       [-0.13617458, -0.13378692, -0.06884284, ..., -0.06884284,
        -0.06884284, 14.52583905]])
        
#Reducción de Dimensionalidad con PCA: un dataset con menores dimensiones necesita menos costo computacional (CPU) y memoria.
from sklearn.decomposition import PCA

#Si no se especifica el número de componentes en PCA, se intenta con todas las características del dataset. 
#num_components = 2
pca = PCA()
principalComponents = pca.fit_transform(x)
principalComponents

#La varianza explicada dice cuanta información (varianza) puede ser obtenida a cada componente principal.
num_components=principalComponents.shape[1]
explained_variance_ratio_=pca.explained_variance_ratio_
explained_variance_ratio_

print(explained_variance_ratio_.sum())  

a = range(num_components)
num_pc= a[::1]

principalDf = pd.DataFrame(data = principalComponents
             , columns = num_pc)
principalDf=round(principalDf, 2)
print(principalDf)

1.0
       0     1      2     3     4     5     6     7     8     9    ...   202  \
0    -0.19 -0.13  14.10 -0.00  0.00 -0.00 -0.00  0.00 -0.00 -0.00  ... -0.00   
1    -0.23  0.01  -0.15 -3.14 -1.87 -1.78 -0.82  0.44  0.08  0.53  ...  0.13   
2    -0.20  0.07  -0.11 -0.64 -1.45  0.14  0.20 -1.12 -0.14  0.81  ...  0.01   
3    -0.23  0.03  -0.11  1.96  7.38 -3.89 -2.03 -1.35 -0.89  5.81  ...  0.03   
4    -0.23  0.03  -0.11 -0.08 -5.10 -0.65  1.72 -4.02 -0.11  2.31  ...  0.11   
..     ...   ...    ...   ...   ...   ...   ...   ...   ...   ...  ...   ...   
207  -0.23  0.02   0.02  0.32 -0.12  0.35  0.05  0.71  0.14  0.38  ... -2.30   
208  24.03 -0.19  -0.55  0.15  0.04 -0.07  0.11 -0.01  0.08  0.12  ... -0.09   
209  -0.22  0.10   0.03 -0.07 -0.33  0.11  0.21  0.14  0.92 -0.03  ...  0.88   
210  -0.23  0.01   0.03 -0.10 -0.74 -0.44 -0.41 -0.13  0.15 -0.54  ... -2.86   
211  -0.23  0.02   0.02 -0.27 -0.65  0.01  0.07  0.24 -0.33  0.96  ... -3.02   

      203   204   205   206   207   208   209   210  211  
0    0.00 -0.00  0.00 -3.62 -0.00 -0.00 -0.00 -0.00  0.0  
1    0.09  0.36 -0.14 -0.29 -0.72 -0.15 -0.11  0.01  0.0  
2    0.02 -0.03  0.04 -0.16 -0.08  0.00 -0.09 -0.03  0.0  
3    0.00  0.03 -0.05 -0.16  0.13  0.01 -0.25 -0.08  0.0  
4   -0.12 -0.08  0.10 -0.16 -0.23 -0.04 -0.18 -0.13  0.0  
..    ...   ...   ...   ...   ...   ...   ...   ...  ...  
207  0.04 -0.57 -1.18  0.35  3.34 -1.47 -3.80 -1.13  0.0  
208 -0.10  0.03  0.07 -2.29 -0.38 -0.06 -0.11 -0.11  0.0  
209 -0.27  0.10 -1.00  0.39  2.30  0.07 -3.16  7.67  0.0  
210  0.35  0.04  0.87  0.39 -5.08  0.51 -0.69 -0.27  0.0  
211  0.15 -0.21  0.47  0.36  0.04  0.26  3.67  0.31  0.0  

#Metodo Elbow
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
X=principalDf
K_range=range(1,20,1)
distortions=[]

for i in K_range:
    kmeanModel = KMeans(n_clusters=i,init='k-means++')
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'),axis=1)) / X.shape[0])

print('Kmeans terminado')

Kmeans terminado

#Datamining: Clustering (método no supervisado de Machine Learning)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
nclust=20
#Kmeans Clustering 
def doKmeans(X, nclust, init='k-means++',max_iter=100, tol=0.0001, random_state=10, algorithm='full'):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(principalDf, nclust, init='k-means++',max_iter=100, tol=0.0001, random_state=10, algorithm='full' )
#kmeans = pd.DataFrame(clust_labels)
kmeans = pd.DataFrame(clust_labels,columns=['Grupos'])
#kmeans
print('Usuarios agrupados')

Usuarios agrupados

kmeans
UserGrupoK=kmeans.groupby(kmeans.Grupos).Grupos.count()

UserGrupoK=UserGrupoK.sort_values(ascending=False, inplace=False, kind='quicksort')
#UserGrupoK.sort_values(by='values', axis=1, ascending=False, inplace=False, kind='quicksort')
UserGrupoK

Grupos
2     39
4     37
11    22
15    15
18    14
17    13
14    11
12     9
8      9
5      9
3      8
9      6
6      6
7      4
1      3
19     3
10     1
13     1
16     1
0      1
Name: Grupos, dtype: int64

# Plot the elbow
plt.plot(K_range, distortions, 'bx-')
plt.xlabel('cantidad de grupos (K)')
plt.ylabel('distortion (intra-cluster)')
plt.title('Clustering con ' + str(nclust) +  ' grupos')
plt.grid(True)
plt.show()

df_cor1 = pd.DataFrame(x).corr()
df_cor1

df_cord1 = pd.DataFrame(DatasetPreprocesado.corr()['total_deaths'].sort_values(ascending=False))
df_cord1[:22]

X = DatasetPreprocesado.drop('total_deaths', 1)  
y = DatasetPreprocesado['total_deaths']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_test)

#Training
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()

regressor.fit(X_train, y_train)
#El modelo de regresión intenta encontrar los coeficientes más óptimos para todos los atributos.
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df


#Predicciones
y_pred = regressor.predict(X_test)

#Comparacion Manual
datasetPreprocesado = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  

#Evaluación
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

Mean Absolute Error: 0.13373741142399884
Mean Squared Error: 0.18639442464710487
Root Mean Squared Error: 0.43173420601928786
