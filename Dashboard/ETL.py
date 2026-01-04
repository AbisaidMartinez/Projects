# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 21:35:20 2026

@author: qbo28
"""

import requests 
import pandas as pd

#%% Fase de Extraccion (E)

APP_ID = 'api_bikepoint'
PRIMARY_KEY = 'f8c31cdae7f349958e8be3e690390ec5'
ENDPOINT = "https://api.tfl.gov.uk/BikePoint/"#'https://api-portal.tfl.gov.uk/profile'

params = {"app_id": APP_ID, 
          "app_key": PRIMARY_KEY}

try:
    response = requests.get(ENDPOINT, params = params)
    
    # Mostrar excepcion si hay algun error
    response.raise_for_status()
    
    data = response.json()
    
    print(f"Se descargaron correctamente los datos de {len(data)} estaciones")
    
except requests.exceptions.RequestException as e:
    print(f"Error descargando los datos: {e}")
    
#%% Fase de Transformacion (T)

print(data[0]["commonName"]) #Relevant Feature
print(data[0]["lat"]) #Relevant Feature
print(data[0]["lon"]) #Relevant Feature

for item in data[0]['additionalProperties']:
    print(f"key: {item['key']} - value: {item['value']}")
    
"""
Datos de importancia para manipulacion:
    CommonName: Nombre de la estacion
    Locked: Indica si la estacion esta fuera de servicio (True) o en funcionamiento (False)
    NbBikes: Numero total de bicicletas disponibles
    NbEmptyDocks: Numero de puestos para bicicletas disponibles
    NbDocks: Numero total de puestos para bicicletas
    NbStandardBikes: Numero total de bicicletas estandar disponibles
    NbEBikes: Numero total de bicicletas electricas disponibles
"""
#%% Creacion DataFrame
info_stations = {
    'nombre_estacion': [], 
    'disponible': [],
    'Num_Bicis': [],
    'Num_EspaciosDisp': [],
    'Num_Espacios': [],
    'Num_BicisEstandar': [],
    'Num_BicisElectricas': [],
    'ultima_actualizacion': []
    }

for station in data:
    info_stations['nombre_estacion'].append(station['commonName'])
    
    for item in station['additionalProperties']:
        if item['key'] == 'NbBikes':
            info_stations['Num_Bicis'].append(int(item['value']))
            
            #Extraer marca de tiempo de ultima actualizacion
            datetime_update = pd.to_datetime(item['modified'])
            
            #Presevar unicamente hasta horas, minutos y segundos
            datetime_update = datetime_update.strftime("%Y-%m-%d %H:%M:%S")
            info_stations['ultima_actualizacion'].append(datetime_update)
        if item['key'] == 'NbEmptyDocks':
            info_stations['Num_EspaciosDisp'].append(int(item['value']))
        if item['key'] == 'NbDocks':
            info_stations['Num_Espacios'].append(int(item['value']))
        if item['key'] == 'NbStandardBikes':
            info_stations['Num_BicisEstandar'].append(int(item['value']))
        if item['key'] == 'NbEBikes':
            info_stations['Num_BicisElectricas'].append(int(item['value']))
        if item['key'] == 'Locked':
            if item['value']=="false":
                info_stations['disponible'].append('Si')
            else:
                info_stations['disponible'].append('No')
            
df = pd.DataFrame(info_stations)
#%%
print(df)


#%% Fase de Carga (L)

df.to_parquet('estaciones.parquet')


