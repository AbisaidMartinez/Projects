# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 16:55:23 2026

@author: qbo28
"""

import pandas as pd

df = pd.read_csv('dataset_ecommerce.csv')
df

#%% Verificar datos faltantes

df.isna().sum()

#%% Visualizacion de datos con diagrama de cajas

import seaborn as sns
import matplotlib.pyplot as plt

num_cols = df.select_dtypes(include=['number']).columns

plt.figure(figsize=(10,7))

for i, col in enumerate(num_cols, 1):
    plt.subplot(2,3, i)
    sns.boxplot(data=df,x=col)
    plt.title(f"Boxplot of {col}")
plt.tight_layout()

#%% Filtrado de valores negativos

df[df[num_cols]<0].count()

#%% Limpieza del dataset

df_clean = df.dropna()
print(f'Dataset size before eliminate remain data: {df.shape}')
print(f'Dataset size after eliminate remain data: {df_clean.shape}')

#%% Eliminar la columna de ID de cliente

df_clean.drop(columns=["ID"], inplace=True)

#%%
print(f'Dataset size before eliminate outliers: {df_clean.shape}')
df_clean = df_clean[df_clean[num_cols].ge(0).all(axis=1)]
print(f'Dataset size after eliminate outliers: {df_clean.shape}')

#%% Matriz de correlacion

plt.figure(figsize=(8,8))
corr = df_clean.corr() 
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.title("Correlation Matrix")

#%% Segmentacion con K-means

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
Xs = scaler.fit_transform(df_clean)

#%% Scree plot para observar el numero ideal de clusters requeridos

from sklearn.cluster import KMeans
import numpy as np

K = np.arange(1,11)

inercias = []
for k in K:
    kmeans = KMeans(n_clusters = k, init='k-means++', random_state=23)
    kmeans.fit(Xs)
    inercias.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(K, inercias, '--ro')
ax.grid(axis='both')
ax.set_xlabel('K')
ax.set_ylabel('Inercia')

#%%

kmeans = KMeans(n_clusters = 3, random_state=32)
kmeans.fit(Xs)

#%%
kmeans.inertia_

#%% Agregar columna de labels en el dataset limpio

df_clean['segmento'] = kmeans.labels_
df_clean

#%% 

features = df_clean.columns.tolist()[:-1]

#%%
plt.figure(figsize=(15,10))
for i, col in enumerate(features, 1):
    plt.subplot(3,2, i)
    sns.boxplot(x='segmento', y=col, data=df_clean)
    plt.title(f'{col} for each segmenter')
plt.tight_layout()
plt.show()

#%%

segmentos = df_clean.copy()
segmentos.drop(columns=['dias_primera_compra', 'info_perfil'], inplace=True)

#%% Caracterizar segmentos de clientes

radar = segmentos.groupby("segmento").mean()
radar

#%%

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"   # abre en navegador

fig = go.Figure()

#Grafico de radar para cada segmento
fig.add_trace(go.Scatterpolar(
    r = radar.iloc[0],
    theta = radar.columns,
    fill = 'toself', 
    name = 'Segmento 0'
    )) 

fig.add_trace(go.Scatterpolar(
    r = radar.iloc[1],
    theta = radar.columns,
    fill = 'toself', 
    name = 'Segmento 1'
    )) 

fig.add_trace(go.Scatterpolar(
    r = radar.iloc[2],
    theta = radar.columns,
    fill = 'toself', 
    name = 'Segmento 2'
    ))     

fig.show()

#%%

segmentos_s = pd.DataFrame(Xs, columns=df_clean.columns[:-1])
segmentos_s['segmento'] = kmeans.labels_

#%%
segmentos_s.drop(columns=['dias_primera_compra', 'info_perfil'], inplace=True)
segmentos_s

#%%

radar_s = segmentos_s.groupby("segmento").mean()
radar_s

#%%

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"   # abre en navegador

fig = go.Figure()

#Grafico de radar para cada segmento
fig.add_trace(go.Scatterpolar(
    r = radar_s.iloc[0],
    theta = radar.columns,
    fill = 'toself', 
    name = 'Segmento 0'
    )) 

fig.add_trace(go.Scatterpolar(
    r = radar_s.iloc[1],
    theta = radar.columns,
    fill = 'toself', 
    name = 'Segmento 1'
    )) 

fig.add_trace(go.Scatterpolar(
    r = radar_s.iloc[2],
    theta = radar.columns,
    fill = 'toself', 
    name = 'Segmento 2'
    ))     

fig.show()