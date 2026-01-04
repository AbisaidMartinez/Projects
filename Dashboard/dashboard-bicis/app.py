# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 00:07:14 2026

@author: qbo28
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from ETL import run_ETL
import locale
from datetime import datetime

#Crear pagina y asignar titulo de la pagina
st.set_page_config(page_title="Bikepoints Londres", layout="wide")
st.title("Availability of Bikepoints in London - On Live")

if "df" not in st.session_state:
    try:
        #Ejecutar y filtrar solo estaciones disponibles
        df = run_ETL()
        df = df[df['disponible'] == 'Si']
        st.session_state.df = df
        
        #Ultima fecha de actualizacion
        locale.setlocale(locale.LC_TIME, "es_ES.UTF-8")
        st.session_state.last_updated = datetime.now().strftime("%d %b %Y, %H:%M")
        
    except Exception:
        st.session_state.df = None
        st.session_state.last_updated = "Never"

#Boton uptaded e info de actualizacion
col1, col2 = st.columns([1,4])
with col1:
    if st.button("Update"):
        with st.spinner("Updating info stations..."):
            df = run_ETL()
            st.session_state.df = df
            st.session_state.last_updated = datetime.now().strftime("%d %b %Y, %H %M")
        st.success("Done!")
        
with col2:
    st.markdown(f"**Last updated** {st.session_state.last_updated}")
    
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Metrica general
    c1, c2, c3 = st.columns(3)
    c1.metric("Total stations available ", int(len(df)))
    
    #Crear mapa centrado en la primera estacion
    lat_in, lon_in = df.iloc[0][['latitud', 'longitud']]
    m = folium.Map(location=[lat_in, lon_in],zoom_start=12)

    #Crear grupo de marcadores
    fg = folium.FeatureGroup(name="Stations in London")
    
    for est in df.itertuples(index=False):
        nombre = est.nombre_estacion
        lat, lon = est.latitud, est.longitud
        espacios = est.Num_EspaciosDisp
        bicis_est = est.Num_BicisEstandar
        bicis_el = est.Num_BicisElectricas
        
        # Color del popup y del marcador según disponibilidad total
        total_bicis = bicis_est + bicis_el
        if total_bicis == 0:
            bg_color = "#ffe6e6"  # rojo
            color = "red"
        elif total_bicis < 5:
            bg_color = "#fff3cd"  # amarillo
            color = "orange"
        else:
            bg_color = "#e6ffe6"  # verde
            color = "green"
    
        # Código HTML del popup
        popup_str = f"""
        <div style="
            background-color:{bg_color};
            border-radius:8px;
            padding:8px 12px;
            font-size:13px;
            line-height:1.5;
            color:#333;
        ">
            <b style="font-size:14px;">{nombre}</b><br>
            🚲 {bicis_est} standard<br>
            ⚡ {bicis_el} electrical<br>
            🅿️ {espacios} slots
        </div>
        """
    
        # Añadir marcador
        fg.add_child(
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_str, max_width=250),
                tooltip=nombre,
                icon=folium.Icon(color=color)
            )
        )
        
    #Ajustar zoom automaticamente del mapa del area total 
    m.fit_bounds(df[['latitud', 'longitud']].values.tolist())
        
    st_folium(
        m,
        feature_group_to_add=fg,
        width="100%",
        height=600,
        returned_objects=[]            
        )
else:
    st.info("No data. Please click updated")    