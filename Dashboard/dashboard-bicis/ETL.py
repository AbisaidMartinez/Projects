# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 23:51:15 2026

@author: qbo28
"""

import streamlit as st
import requests 
import pandas as pd

#%% Fase de Extraccion (E)

APP_ID = 'api_bikepoint'
PRIMARY_KEY = st.secrets["api"]["API_PRIMARY_KEY"]
ENDPOINT = "https://api.tfl.gov.uk/BikePoint/"

params = {"app_id": APP_ID, 
          "app_key": PRIMARY_KEY}

def get_api_data(endpoint, params):
    try:
        response = requests.get(endpoint, params = params)
        
        # Mostrar excepcion si hay algun error
        response.raise_for_status()
        
        data = response.json()
        return data
        
        print(f"Se descargaron correctamente los datos de {len(data)} estaciones")
        
    except requests.exceptions.RequestException as e:
        print(f"Error descargando los datos: {e}")
        return None
    
#%% Fase de Transformacion (T)

def transform_load_data(data):
    info_stations = {
        'nombre_estacion': [],
        'latitud': [],
        'longitud': [],
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
        info_stations['latitud'].append(station['lat'])
        info_stations['longitud'].append(station['lon'])
        
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
    return df

def run_ETL():
    data = get_api_data(ENDPOINT, params)
    df = transform_load_data(data)
    return df

