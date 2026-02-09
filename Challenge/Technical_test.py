# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 19:49:24 2026

@author: qbo28
"""
# Actividad 01
# Crear DataFrame

import pandas as pd

df = pd.read_csv('Municipal-Delitos-2015-2025_dic2025.csv', encoding="latin-1")

#%% Informacion general de datos

df.info()

#%%

# Filtrado de datos
df_robo = df[
    (df["Tipo de delito"] == "Robo") &
    (df["Bien jurídico afectado"] == "El patrimonio")
]

#%% Creacion Series de Tiempo

meses = {"Enero":1, "Febrero":2, "Marzo":3, "Abril":4, "Mayo":5, "Junio":6,
         "Julio":7, "Agosto":8, "Septiembre":9, "Octubre":10, "Noviembre":11, "Diciembre":12}

cols_meses = list(meses.keys())

df_long = df_robo.melt(id_vars=["Año", "Entidad", "Municipio", "Bien jurídico afectado", "Tipo de delito", "Modalidad"], 
                  value_vars = cols_meses, var_name="Mes", value_name="Total")

df_long["Año"] = (
    df_long["Año"].astype(str).str.strip().astype(int)
)

df_long["Mes_num"] = df_long["Mes"].map(meses)

df_long["Fecha"] = pd.to_datetime(dict(year=df_long["Año"], month=df_long["Mes_num"], day=1))

df_long = df_long.sort_values("Fecha")

#%% Comprobar Intervalo de Series de Tiempo

df_long[["Fecha", "Total"]].head()
df_long["Fecha"].min(), df_long["Fecha"].max()

#Debe ser cero, confirmando la creacion adecuada de Series de Tiempo
df_long["Fecha"].isna().sum()

#%% Gráfica de Robos vs Fechas 

import matplotlib.pyplot as plt

#Registro ultimos 10 años
plt.figure(1)
ax = df_long.groupby("Fecha")["Total"].sum().plot()

ax.set_title('Historial de Robos durante ultimos 10 años', fontsize=16)
ax.set_ylabel('Robos', fontsize=16)
ax.set_xlabel('Fechas', fontsize=16)

#Registro ultimos 5 años
plt.figure(2)
ax = df_long.groupby("Fecha")["Total"].sum().plot()

ax.set_title('Historial de Robos durante ultimos 5 años', fontsize=16)
ax.set_ylabel('Robos', fontsize=16)
ax.set_xlabel('Fechas', fontsize=16)
plt.xlim(pd.Timestamp('2020-01-01'), pd.Timestamp('2025-12-01'))

plt.show()

#%% Modelo 01 (Considerando datos completos)

serie = (df_long.groupby("Fecha")["Total"].sum().asfreq("MS"))

from statsmodels.tsa.stattools import adfuller

result = adfuller(serie)

print("Estadistica ADF ", result[0])
print("p-value: ", result[1]) #Analizar si medidas de tendencia central 
                              #son Estacionarias o No Estacionarias
#%%
serie_diff = serie.diff().dropna()
serie_diff_season = serie.diff(1).diff(12).dropna()

result_diff = adfuller(serie_diff_season)

print("Estadistica ADF ", result_diff[0])
print("p-value: ", result_diff[1])

#%% Comparativas Autocorrelación

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(serie_diff_season, lags=36)
plot_pacf(serie_diff_season, lags=36)
plt.show()

#%% SARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    serie,
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
    )

res = model.fit()
print(res.summary())

#%% Diagnostico

res.plot_diagnostics(figsize=(12,8))
plt.show()

#%% Predicción

forecast = res.get_forecast(steps=3)

pred = forecast.predicted_mean
conf_int = forecast.conf_int()


#%% Visualización

plt.figure(figsize=(12,5))

plt.plot(serie, label="Historico")
plt.plot(pred, label="Forecast", color="red")

plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1],
                 color="pink", alpha=0.3)

plt.legend()
plt.title("Forecast SARIMA - Robos Patrimonio")
plt.show()

##########################################################
#%% Actividad 02 (Tres Municipios)

serie_municipio = (df_long.groupby(["Fecha","Municipio"])["Total"].sum().reset_index())

top_municipios = (df_long.groupby("Municipio")["Total"]
                                   .sum()
                                   .sort_values(ascending=False)
                                   .head(20)
                                   .index)

#%% Serie de Tiempo por municipio

series_num = {}

for num in top_municipios:
    
    serie = (serie_municipio[serie_municipio["Municipio"] == num]
    .set_index("Fecha")["Total"]
    .asfreq("MS")
    )

    series_num[num] = serie

#%% Forecast por municipio

forecast_mun = {}

for mun, serie in series_num.items():
    
    modelo = SARIMAX(
        serie,
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
        )

    res2 = modelo.fit(disp=False)
    
    pred = res.get_forecast(steps=3).predicted_mean
    
    forecast_mun[mun] = pred.mean()
#%%
ranking_mun = (pd.Series(forecast_mun).sort_values())
municipios_desc = ranking_mun.head(3)

print(municipios_desc)

#%% Actividad 2.5 (3 Modalidades de Robo)

serie_modalidad = (df_long.groupby(["Fecha", "Modalidad"])["Total"]
                   .sum().reset_index())

top_mod = (df_long.groupby("Modalidad")["Total"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index)


series_mod = {}

for mod in top_mod:
    
    serie = (serie_modalidad[serie_modalidad["Modalidad"] == mod]
          .set_index("Fecha")["Total"]
          .asfreq("MS"))

    series_mod[mod] = serie

#%% Forecast Modalidades

forecast_mod = {}

for mod, serie in series_mod.items():
    
    model = SARIMAX(
        serie,
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    )
    
    res = model.fit(disp=False)
    
    pred = res.get_forecast(steps=3).predicted_mean
    
    forecast_mod[mod] = pred.mean()


#%%
ranking_mod = (pd.Series(forecast_mod).sort_values())

modalidades_descuento = ranking_mod.head(3)

print(modalidades_descuento)