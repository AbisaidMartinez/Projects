# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 03:05:51 2026

@author: qbo28
"""

# SARIMA (Seasonal AutoRegressive Integrated Moving Average)

import pandas as pd

df = pd.read_csv('demanda_electricidad_california_2019_2025.csv')
df

#%%

df.info()

#%%

df['period'] = pd.to_datetime(df['period'])
df['period']

#%%

df_gr = df.groupby(by='period')['value'].sum()
df_gr

#%%

df_gr = df_gr.sort_index(ascending=True)
df_gr

#%%

#Create ideal index
complete_idx = pd.date_range(start=df_gr.index.min(), end=df_gr.index.max(), freq='h')

#Reindex
df_gr_full = df_gr.reindex(complete_idx)

#Count Nans
faltantes = df_gr_full[df_gr_full.isna()]
print(f'Marcas de tiempo faltantes : {len(faltantes)}')

#%%

import matplotlib.pyplot as plt
import seaborn as sns

mask = df_gr_full.isna()

fig, ax = plt.subplots(1, figsize=(12,5))
sns.lineplot(df_gr_full, ax=ax)

ymin, ymax = df_gr_full.min(), df_gr_full.max()
for ts in faltantes.index:
    if ts == faltantes.index[0]:
        ax.vlines(ts, ymin, ymax, label='NaN Value', colors='red', linestyles='dashed', linewidth=1)
    else:         
        ax.vlines(ts, ymin, ymax, colors='red', linestyles='dashed', linewidth=1)

ax.legend()
ax.set_xlabel('Fecha-Hora')
ax.set_ylabel('Demanda energética')

#%% Interpolación para relleno de valores nulos

df_interp = df_gr_full.interpolate(method='linear')

fig, ax = plt.subplots(1, figsize=(12,4))
ax.plot(df_interp)
ax.grid()
ax.set_xlabel('Fecha-Hora')
ax.set_ylabel('Demanda energética')

#%% Comprobación

df_interp.isna().sum()

#%% Verificar marcas de tiempo consecutivas (1 Hora)

time_diffs = df_interp.index.to_series().diff().dt.total_seconds()
time_diffs.describe()

#%% IQR y metodo de Tukey para detección de outliers

q1 = df_interp.quantile(0.25)
q3 = df_interp.quantile(0.75)
iqr = q3 - q1

outliers = df_interp[ (df_interp < (q1 - 2*iqr))]
outliers

#%% Eliminacion de outliers

# Outliers como NaN
idxs = df_interp.index.isin(outliers.index)
df_interp.iloc[idxs] = None

# Interpolacion lineal
serie = df_interp.interpolate(method = 'linear')

# Verificar limpieza de outliers
plt.figure(3)
serie.plot(figsize=(12,4))
plt.grid()
plt.xlabel('Fecha-Hora')
plt.ylabel('Demanda energética')

#%%

from statsmodels.graphics.tsaplots import plot_acf
import numpy as np

fig, ax = plt.subplots(1, figsize=(12,5))
fig = plot_acf(serie, lags=100,ax=ax)

xticks = np.arange(0,101,4) #Each 4 hours
ax.grid()
ax.set_ylim([-.25, 1.2])
ax.set_xticks(xticks)
ax.set_xlabel('Hora (lag)')
ax.set_ylabel('Autocorrelación')
ax.set_title('Función de autocorrelación (ACF)')

#%% Preparar dataset para graficos estacionales

ts = serie.copy().to_frame()

ts['hora'] = ts.index.hour
ts['mes'] = ts.index.month
ts['año'] = ts.index.year

#%% Grafico estacional demanda vs horas para diferentes meses

fig, ax = plt.subplots(figsize=(12,4))
sns.lineplot(data=ts, x='hora', y='value', hue='mes', ax=ax)

#%% Grafico estacional demanda vs horas para diferentes años

fig, ax = plt.subplots(figsize=(12,4))
sns.lineplot(data=ts, x='hora', y='value', hue='año', ax=ax)

#%% Si queremos observar un solo mes

fig, ax = plt.subplots(figsize=(12,4))
sns.lineplot(data=ts.loc['2020-01'], x='hora', y='value', hue='mes', ax=ax)
ax.grid()
ax.set_xlabel('Hora')
ax.set_ylabel('Demanda energetica')

#%% Construccion modelo SARIMA

H = 24

#Train/Test

train = serie.iloc[:-H]
test = serie.iloc[-H:]

#%%

y_pred = train.shift(H)

#%%

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

#Evaluación

mape_base = mape(train[H:], y_pred[H:])
rmse_base = rmse(train[H:], y_pred[H:])

print(f'Modelo base (Seasonal Naive) - MAPE: {100*mape_base:.2f}% and RMSE: {rmse_base:.4f} MW')

#%% Crear funcion para probar estacionariedad

def prueba_estacionariedad(serie, serie_name):
    from statsmodels.tsa.stattools import adfuller
    
    adf = adfuller(serie.dropna())
    
    p = adf[1]
    
    print(f'Serie: {serie_name}')
    if p < 0.05:
        #H0 rechazada
        print(f'La serie es estacionaria porque p = {p:.4f} < 0.05')
    else:
        #H0 aceptada
        print(f'La serie es NO estacionaria porque p = {p:.4f} >= 0.05')
    print('-'*20)
    
#%% Evaluar estacionariedad de serie de tiempo

prueba_estacionariedad(ts['value'], 'Serie_original')

#%% Determinar P mediante PACF

from statsmodels.graphics.tsaplots import plot_pacf

fig, ax = plt.subplots(1, figsize=(12,5))
fig = plot_pacf(serie, lags=100, ax=ax)

xticks = np.arange(0,101,4) #Each 4 hours
ax.grid()
ax.set_ylim([-1, 1.2])
ax.set_xticks(xticks)
ax.set_xlabel('Hora (lag)')
ax.set_ylabel('Autocorrelación')
ax.set_title('Función de autocorrelación parcial (PACF)')

#%%

from statsmodels.tsa.statespace.sarimax import SARIMAX

VENTANAS_HORAS = [7*H, 15*H, 30*H, 60*H, 90*H, 120*H]
# 1 semana, 2 semanas, 1 mes, 2 meses, 3 meses, 4 meses

res_mape = []
res_rmse = []
serie = train.copy()

for ventana in VENTANAS_HORAS:
    print(f'\n Evaluando ventana: {ventana} horas ({ventana // 24} días)')
    
    #Train/Val
    train_ventana = train[-(ventana + H):-H]
    val_ventana = train[-H:]
    
    #SARIMA
    model = SARIMAX(
        train_ventana,
        order=(1,0,0),
        seasonal_order=(1,0,1,24)
        )

    res = model.fit(disp=False)
    
    forecast = res.forecast(steps=H)
    
    #Evaluación
    error_mape = mape(val_ventana, forecast)
    error_rmse = rmse(val_ventana, forecast)
    
    res_mape.append((ventana, error_mape))
    res_rmse.append((ventana, error_rmse))
    
#%% Grafico de errores (MAPE & RMSE)

values, errors_mape =  zip(*res_mape)

plt.figure(figsize=(12,5))
plt.plot(values, errors_mape)
plt.grid()
plt.xlabel('Horas (h)')
plt.ylabel('MAPE (%)')


valores, errors_rmse =  zip(*res_rmse)

plt.figure(figsize=(12,5))
plt.plot(valores, errors_rmse)
plt.grid()
plt.xlabel('Horas (h)')
plt.ylabel('RMSE (MW)')

#%% Entrenamiento modelo final

print(res.summary())

#%% Diagnostico

res.plot_diagnostics(figsize=(12,8))
plt.show()


#%% Analisis de residuales

p, d, q = 1, 0, 0
P, D, Q, s = 1, 0, 1, 24
VENTANA = 24*7

train_ventana = train[-(VENTANA + H):-H]
val_ventana = train[-H:]

modelo = SARIMAX(train_ventana, 
                 order=(p,d,q), 
                 seasonal_order=(P,D,Q,s)).fit(disp=False)

pred_is = modelo.predict()

residuales = train_ventana[1:] - pred_is[1:]

print("Media de los residuos: ", residuales.mean())
print("Desviacón Estandar: ", residuales.std())

#%% Graficos residuales 

fig, axs = plt.subplots(3,1,figsize=(12,6))
sns.lineplot(residuales, ax=axs[0])
sns.histplot(residuales, ax=axs[1])
plot_acf(residuales, zero=False, auto_ylims=True, ax=axs[2])
plt.tight_layout()

#%% Pronostico set de prueba

train_ventana = train[-VENTANA:]
modelo = SARIMAX(train_ventana,
               order=(p,d,q), 
               seasonal_order=(P,D,Q,s),
               enforce_stationarity=False).fit(disp=False)

#%% DataFrame para pronosticos 

prons = modelo.get_forecast(steps=H)
prons_df = prons.summary_frame(alpha=0.05)
prons_df

#%% Graficar serie de prueba, pronostico 
#   e intervalo de prediccion

fig, ax = plt.subplots(figsize=(12,5))
sns.lineplot(test, linestyle='--', label='Test Set')
sns.lineplot(prons_df, x=prons_df.index, y='mean', 
             linestyle='--', color='r', label=f'Pronostico {H} horas con SARIMA')
plt.fill_between(x=prons_df.index, y1=prons_df["mean_ci_lower"], 
                 y2=prons_df["mean_ci_upper"],
                 color='red',
                 alpha=0.2,
                 label='Intervalo de Confianza (95%)')
ax.set_xlabel('Fecha')
ax.set_ylabel('Demanda energetica (MW)')
ax.grid(True)
ax.legend(loc="lower left")

#%% Ajustar parametros ideales para modelo SARIMA

import pmdarima as pm

# Supongamos que 'serie' es tu DataFrame o Series de Pandas
model = pm.auto_arima(
    train_ventana,#serie,
    seasonal=True,      # Activa la búsqueda estacional (P,D,Q)
    m=24,               # Frecuencia estacional (ej. 24 para dia)
    start_p=0, start_q=0, # Valores iniciales para p, q
    max_p=2, max_q=2,   # Valores máximos para p, q
    d=None,             # Deja que auto_arima lo calcule (o pon 0, 1, 2)
    D=None,             # Deja que auto_arima lo calcule (o pon 0, 1)
    start_P=0, start_Q=0,
    max_P=2, max_Q=2,
    trace=True,         # Muestra el proceso de ajuste
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True       # Búsqueda rápida (algoritmo Hyndman-Khandakar)
)

print(model.summary())

#%%

p, d, q = 1, 1, 0
P, D, Q, s = 1, 0, 2, 24

modelo = SARIMAX(train_ventana, 
                 order=(p,d,q), 
                 seasonal_order=(P,D,Q,s)).fit(disp=False)

pred_is = modelo.predict()

residuales = train_ventana[1:] - pred_is[1:]

print("Media de los residuos: ", residuales.mean())
print("Desviacón Estandar: ", residuales.std())
